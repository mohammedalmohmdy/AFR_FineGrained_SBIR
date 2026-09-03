"""Frequency Hypothesis Space (Section 3.4 of the manuscript).

The spatial feature maps are transformed into the spectral domain using a
two-dimensional Fourier transform and decomposed into complementary spectral
bands using *predefined* radial masks:

    H_k(F) = F^{-1}( M_k odot F(F) )                     (Eq. 5)

    M_k = 1  if r_k^min <= r < r_k^max, else 0           (Eq. 6)

The paper defines three complementary bands over the normalized radial
frequency interval. Eq. (6) states the masks are defined using normalized
radial frequency intervals; combined with the paper's claim that the bands sum
to a complete reconstruction of the spectrum, the natural exact partition is

    Low-frequency    0   <= r <  1/3
    Mid-frequency    1/3 <= r <  2/3
    High-frequency   2/3 <= r <= 1

The normalized radial frequency r is measured in the *shifted* spectrum (DC at
the center) and is normalised so that the farthest corner equals 1.0. The
three half-open intervals above form a complete, non-overlapping partition of
[0, 1] (the upper inclusive bound of the high band is implemented as an
epsilon margin so the corner frequencies are included).

This module performs real 2-D Fourier processing on the deep feature maps; it
does not approximate the spectral bands with spatial convolutions.
"""

import torch
import torch.nn as nn

# (band_name, r_min, r_max) with exact one-third interval boundaries. r_max of
# the high band is slightly above 1.0 so that the corner frequencies (r == 1.0)
# are included, keeping the partition exhaustive.
DEFAULT_BANDS = (
    ("low", 0.0, 1.0 / 3.0),
    ("mid", 1.0 / 3.0, 2.0 / 3.0),
    ("high", 2.0 / 3.0, 1.0 + 1e-3),
)

BAND_NAMES = tuple(name for name, _, _ in DEFAULT_BANDS)


def select_bands(names=None):
    """Return the (name, r_min, r_max) tuples for the requested band names.

    ``names`` may be None (all bands), a single band name, or an iterable of
    band names. Unknown names raise so a typo in a config cannot silently
    change the frequency hypothesis space.
    """
    if names is None:
        return DEFAULT_BANDS
    if isinstance(names, str):
        names = [names]
    known = {name: (name, lo, hi) for (name, lo, hi) in DEFAULT_BANDS}
    out = []
    for n in names:
        if n not in known:
            raise ValueError(
                f"Unknown frequency band '{n}'. Available: {list(known)}"
            )
        out.append(known[n])
    return tuple(out)


def build_radial_mask(H, W, r_min, r_max, device=None, dtype=None):
    """Return a boolean mask selecting normalized radial frequencies in [r_min, r_max).

    The origin (DC component) sits at the FFT-shifted center. Radii are min-max
    normalised so the largest radius (the corners) equals 1.0, matching the
    normalized intervals used to define M_k in Eq. (6).
    """
    xs = torch.arange(W, device=device, dtype=torch.float64) - (W // 2)
    ys = torch.arange(H, device=device, dtype=torch.float64) - (H // 2)
    Y, X = torch.meshgrid(ys, xs, indexing="ij")
    R = torch.sqrt(X * X + Y * Y)
    R = R / R.max()
    return (R >= r_min) & (R < r_max)


class FrequencyHypothesis(nn.Module):
    """A single spectral band applied via 2-D FFT masking + inverse FFT.

    ``forward`` returns a real spatial map of the same shape as its input,
    containing only the input's energy within the band ``[r_min, r_max)``.
    """

    def __init__(self, band, r_min, r_max):
        super().__init__()
        self.band = band
        self.r_min = float(r_min)
        self.r_max = float(r_max)

    def forward(self, x):
        # x: (B, C, H, W) real feature map.
        F = torch.fft.fft2(x, norm="ortho")
        mask = build_radial_mask(
            x.shape[-2], x.shape[-1], self.r_min, self.r_max, device=x.device
        ).to(dtype=x.dtype)
        # Mask the *shifted* spectrum so the low band is at the center.
        F_shifted = torch.fft.fftshift(F, dim=(-2, -1))
        F_band = F_shifted * mask
        F_band = torch.fft.ifftshift(F_band, dim=(-2, -1))
        return torch.fft.ifft2(F_band, norm="ortho").real


class FrequencyBank(nn.Module):
    """Produces the K complementary spectral representations H_k(F)."""

    def __init__(self, bands=DEFAULT_BANDS):
        super().__init__()
        self.bands = list(bands)
        self.bank = nn.ModuleList(
            [FrequencyHypothesis(name, lo, hi) for (name, lo, hi) in self.bands]
        )

    def forward(self, x):
        """Return a list [H_1(x), ..., H_K(x)] of real band reconstructions."""
        return [h(x) for h in self.bank]

    def __len__(self):
        return len(self.bank)