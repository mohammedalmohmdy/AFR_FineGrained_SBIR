"""AFR-SBIR full model (Section 3).

Pipeline
--------
1. Spatial Feature Encoding (Section 3.3): shared ResNet-50 -> F_s, F_p.
2. Feature normalization (Section 3.3, last sentence) before FFT.
3. Frequency Decomposition (Section 3.4): FFT -> radial masks -> IFFT,
   producing K = 3 complementary bands (low / mid / high).
4. Instance-Adaptive Frequency Weighting (Section 3.5): sketch-driven softmax
   over K hypotheses -> alpha_s; shared with the photo branch (Section 3.6).
5. Confidence-Guided Fusion (Section 3.7): sketch-driven confidence c in [0,1];
   F_final = c * F_freq + (1 - c) * F_spatial.
6. Metric Embedding (Section 3.8): GAP + Linear(2048, 256) + L2 normalisation.

Routing symmetry (Sections 3.2 and 3.6)
---------------------------------------
The instance-adaptive frequency weights and the confidence score are predicted
*exclusively from the sketch representation* ("adaptive frequency weighting is
guided only by the sketch representation", Section 3.2) and the *same* weights
are applied symmetrically to the photo branch ("The same adaptive weights
predicted from the sketch branch are symmetrically applied to the photo branch
to preserve cross-modal consistency", Section 3.6). This holds both during
paired training (``forward(s, p)``) and at test time:

* ``encode_sketch(s)`` computes the query embedding together with the
  sketch-conditioned weights alpha_s and confidence c.
* The gallery photo embedding is *sketch-conditioned*: for query q and
  gallery photo g, the photo fused representation is built with q's alpha and
  c. Because GAP and the embedding projection are linear, the fused photo
  embedding equals the L2-normalised linear combination of per-band
  projections -- ``c_q * sum_k alpha_qk * u_band[k][g] + (1-c_q) * u_spatial[g]``
  -- so ``gallery_components`` precomputes the unnormalised projections once
  and ``evaluation/evaluate.py`` assembles the exact per-query conditioned
  embeddings without re-running the backbone.

``encode_self`` exists only for the two ablation baselines whose definition is
per-modality rather than sketch-conditioned: the spatial-only baseline (no
frequency branch) and the conventional channel attention/gating baseline
(Section 5.5, Table 6). The AFR-SBIR retrieval protocol itself never uses
self-conditioned photo routing.

Confidence-disabled ablation variants (Table 6 rows without confidence fusion)
use the balanced convex fusion c = 0.5, i.e. equal weighting of the
frequency-aware and spatial representations; the paper specifies the convex
fusion of Eq. (15) and does not define an alternative fusion for the
no-confidence ablation rows, so the constant 0.5 (the regularizer target of
Eq. 21) is the least-arbitrary choice. See PAPER_CODE_CONSISTENCY.md.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import SpatialEncoder
from .frequency import FrequencyBank, select_bands
from .reasoning import InstanceReasoner, ConfidenceEstimator


def _weighted_bands(bands, weights):
    """bands: (B, K, C, H, W), weights: (B, K) -> (B, C, H, W)."""
    return (bands * weights.view(-1, weights.shape[1], 1, 1, 1)).sum(dim=1)


class ChannelAttentionGate(nn.Module):
    """Conventional SE-style channel attention (used only for the ablation
    baseline ``Spatial + Conventional Attention/Gating``). This reweights
    feature *channels*, in contrast to the frequency-hypothesis softmax."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        g = self.fc(self.pool(x).flatten(1))   # (B, C)
        return g.view(-1, x.shape[1], 1, 1)


class AFRSBIR(nn.Module):
    def __init__(
        self,
        backbone="resnet50",
        embed_dim=256,
        k=3,
        frequency_hidden=512,
        confidence_hidden=256,
        use_frequency=True,
        weight_mode="adaptive",      # "adaptive" | "fixed" | "attention"
        use_confidence=True,
        feature_norm=True,
        pretrained=True,
        frequency_bands=None,        # None -> all bands; or band-name subset
    ):
        super().__init__()
        assert weight_mode in ("adaptive", "fixed", "attention")
        self.backbone_name = backbone
        self.embed_dim = embed_dim
        self.weight_mode = weight_mode
        self.use_frequency = use_frequency
        self.use_confidence = use_confidence and use_frequency
        self.feature_norm = feature_norm

        # Frequency hypothesis space (Section 3.4): band subset selection is
        # exposed only for the frequency-band ablation (Supplementary B.3).
        self.bands = select_bands(frequency_bands)
        if use_frequency:
            self.k = len(self.bands)
            if k is not None and self.k != k:
                raise ValueError(
                    f"num_freq_hypotheses={k} conflicts with frequency_bands="
                    f"{[b[0] for b in self.bands]} -> K={self.k}"
                )
        else:
            self.k = 0

        self.encoder = SpatialEncoder(backbone, pretrained=pretrained)
        C = self.encoder.out_dim

        if self.use_frequency:
            self.freq_bank = FrequencyBank(bands=self.bands)
            if weight_mode == "adaptive":
                self.reasoner = InstanceReasoner(C, self.k, hidden=frequency_hidden)
            elif weight_mode == "attention":
                self.attn_gate = ChannelAttentionGate(C)
        if self.use_confidence:
            self.conf = ConfidenceEstimator(C, hidden=confidence_hidden)

        # Embedding head f_emb (Eq. 16): GAP -> Flatten -> Linear(C, embed_dim).
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(C, embed_dim),
        )

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _normalize(self, x):
        """Feature normalization before frequency-domain processing (Sec 3.3).

        Per-sample standardisation over all elements keeps FFT inputs bounded
        and consistent. The manuscript states that normalization is applied but
        does not name the operator; see PAPER_CODE_CONSISTENCY.md.
        """
        if not self.feature_norm:
            return x
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        return (x - mean) / (std + 1e-5)

    def _routing_weights(self, Fs):
        """Returns (weights, logits) where weights is (B, K) softmax scores for
        the adaptive mode and uniform weights for the fixed mode. The
        'attention' mode uses channel gating instead (handled by _fuse)."""
        if self.weight_mode == "adaptive":
            return self.reasoner(Fs)
        # fixed: uniform weights over K hypotheses
        w = torch.ones(Fs.shape[0], self.k, device=Fs.device) / self.k
        return w, None

    def _confidence(self, Fs, n):
        """Predicted sketch-conditioned confidence c in [0, 1] (B, 1), or the
        constant 0.5 (balanced convex fusion) for no-confidence ablations."""
        if self.use_confidence:
            return self.conf(Fs)
        return torch.full((n, 1), 0.5, device=Fs.device)

    def _fuse(self, F, bands, weights, conf_score):
        """Convex spatial-frequency fusion (Eq. 15):
        F_final = c * F_freq + (1 - c) * F_spatial.
        bands: (B, K, C, H, W) for adaptive/fixed modes; for 'attention' mode
        bands is the stacked band maps and weights is ignored (channel gate)."""
        if self.weight_mode == "attention":
            freq = self.attn_gate(F) * bands.mean(dim=1)
        else:
            freq = _weighted_bands(bands, weights)
        c = conf_score.view(-1, 1, 1, 1)
        return c * freq + (1.0 - c) * F

    # ------------------------------------------------------------------ #
    # paired forward (training) -- sketch-conditioned routing for both
    # ------------------------------------------------------------------ #
    def forward(self, s, p):
        Fs = self._normalize(self.encoder(s))
        Fp = self._normalize(self.encoder(p))

        if not self.use_frequency:
            zs = F.normalize(self.proj(Fs), dim=1)
            zp = F.normalize(self.proj(Fp), dim=1)
            return {
                "zs": zs, "zp": zp,
                "alphas": None, "alpha_logits": None, "confidence": None,
                "Fs": Fs, "Fp": Fp,
            }

        bands_s = torch.stack(self.freq_bank(Fs), dim=1)   # (B, K, C, H, W)
        bands_p = torch.stack(self.freq_bank(Fp), dim=1)

        weights, logits = self._routing_weights(Fs)         # sketch-driven
        conf = self._confidence(Fs, Fs.shape[0])

        Fs_final = self._fuse(Fs, bands_s, weights, conf)
        Fp_final = self._fuse(Fp, bands_p, weights, conf)

        zs = F.normalize(self.proj(Fs_final), dim=1)
        zp = F.normalize(self.proj(Fp_final), dim=1)
        return {
            "zs": zs, "zp": zp,
            "alphas": weights if self.weight_mode == "adaptive" else None,
            "alpha_logits": logits,
            "confidence": conf if self.use_confidence else None,
            "Fs": Fs, "Fp": Fp,
        }

    # ------------------------------------------------------------------ #
    # test-time retrieval (sketch-conditioned, Sections 3.2/3.6)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def encode_sketch(self, s):
        """Query-sketch embedding with sketch-conditioned routing.

        Returns a dict with 'z' (B, D) L2-normalised embedding, 'alphas'
        (B, K) softmax frequency weights (None for attention/spatial-only
        modes) and 'confidence' (B, 1) in [0, 1].
        """
        Fs = self._normalize(self.encoder(s))
        if not self.use_frequency:
            return {
                "z": F.normalize(self.proj(Fs), dim=1),
                "alphas": None, "confidence": None,
            }

        bands = torch.stack(self.freq_bank(Fs), dim=1)
        weights, _ = self._routing_weights(Fs)              # sketch-driven
        conf = self._confidence(Fs, Fs.shape[0])
        Fs_final = self._fuse(Fs, bands, weights, conf)
        return {
            "z": F.normalize(self.proj(Fs_final), dim=1),
            "alphas": weights if self.weight_mode == "adaptive" else None,
            "confidence": conf if self.use_confidence else None,
        }

    @torch.no_grad()
    def gallery_components(self, x):
        """Unnormalised embedding projections of a gallery photo.

        Returns a dict with ``u_spatial`` (B, D) = proj(GAP(F)) and ``u_bands``
        (B, K, D) = proj(GAP(H_k(F))), or None for the spatial-only and
        conventional-attention baselines (their gallery embedding is a pure
        function of the input by definition; see ``encode_self``).

        Because proj(GAP(.)) is linear, the sketch-conditioned fused photo
        embedding for query q is exactly
            z_p(q, g) = L2norm( c_q * sum_k alpha_qk * u_bands[k, g]
                                + (1 - c_q) * u_spatial[g] ).
        """
        if not self.use_frequency or self.weight_mode == "attention":
            return None
        Fx = self._normalize(self.encoder(x))
        bands = torch.stack(self.freq_bank(Fx), dim=1)      # (B, K, C, H, W)
        u_spatial = self.proj(Fx)                            # (B, D)
        u_bands = torch.stack(
            [self.proj(bands[:, k]) for k in range(self.k)], dim=1)  # (B, K, D)
        return {"u_spatial": u_spatial, "u_bands": u_bands}

    @torch.no_grad()
    def encode_self(self, x):
        """Self-conditioned embedding, used ONLY by the spatial-only and
        conventional-attention ablation baselines (Table 6). The AFR-SBIR
        retrieval protocol never uses this for the photo branch (see module
        docstring)."""
        Fx = self._normalize(self.encoder(x))
        if not self.use_frequency:
            return F.normalize(self.proj(Fx), dim=1)
        bands = torch.stack(self.freq_bank(Fx), dim=1)
        weights, _ = self._routing_weights(Fx)
        conf = self._confidence(Fx, Fx.shape[0])
        return F.normalize(self.proj(self._fuse(Fx, bands, weights, conf)), dim=1)

    # ------------------------------------------------------------------ #
    # frequency-weight / confidence analysis helpers (Appendix B, C)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def sketch_frequency_weights(self, s):
        """Per-instance softmax frequency weights alpha_s for a batch of
        sketches (used to reproduce Appendix B weight statistics)."""
        Fs = self._normalize(self.encoder(s))
        alphas, _ = self.reasoner(Fs)
        return alphas

    @torch.no_grad()
    def sketch_confidence(self, s):
        Fs = self._normalize(self.encoder(s))
        return self._confidence(Fs, Fs.shape[0])


def build_model(cfg):
    m = cfg.get("model", {})
    bands = m.get("frequency_bands")
    return AFRSBIR(
        backbone=m.get("backbone", "resnet50"),
        embed_dim=m.get("embedding_dim", 256),
        k=m.get("num_freq_hypotheses", 3),
        frequency_hidden=m.get("frequency_hidden", 512),
        confidence_hidden=m.get("confidence_hidden", 256),
        use_frequency=m.get("use_frequency", True),
        weight_mode=m.get("weight_mode", "adaptive"),
        use_confidence=m.get("use_confidence", True),
        feature_norm=m.get("feature_norm", True),
        pretrained=m.get("pretrained", True),
        frequency_bands=bands,
    )