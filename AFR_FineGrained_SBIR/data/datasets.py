"""Benchmark dataset loaders and protocols (Section 4.1 / Table 1).

Four benchmarks are supported with their paper-stated splits and positive-sample
definitions:

    Sketchy (Extended)   category-level  -> mAP, P@100
    TU-Berlin (Extended) category-level  -> mAP, P@100
    QMUL-Chair-V2        instance-level  -> Top-1/5/10/20, mAP, CMC
    QMUL-Shoe-V2         instance-level  -> Top-1/5/10/20, mAP, CMC

The module reads *real* files from disk; it never fabricates samples. Two
indexing backends are provided:

1. Explicit list files (deterministic; recommended for Sketchy / TU-Berlin
   instance-pairing where the exact sketch->photo pairing is not inferable from
   the raw download). Format per line (whitespace separated):
       train   : <sketch_path> <photo_path> <photo_id>
       query   : <image_path> <label>
       gallery : <image_path> <label>

2. Automatic directory discovery:
   * QMUL (instance-level) -- reads ``root/photo/{train,test}`` and
     ``root/sketch/{train,test}``. Instance ids are read either from the
     sketch sub-folder name (``root/sketch/train/<id>/...``) or, in flat
     layout, from the file stem (the prefix before the last ``-``/``_``).
   * Sketchy / TU-Berlin (category-level) -- reads class sub-folders under
     ``root/sketch/{train,test}` and ``root/photo/{train,test}`; the class of
     each image is its sub-folder name.

Train pairs always carry the *photo identity* (unique per photo) as their label
so that hard-negative mining excludes only the matched photo. Query/gallery
samples carry the benchmark label (category for category-level, instance id
for instance-level) used to build the retrieval positive mask.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

PROTOCOLS = {
    "QMUL-Shoe-V2": "instance",
    "QMUL-Chair-V2": "instance",
    "Sketchy-Extended": "category",
    "TU-Berlin-Extended": "category",
}


@dataclass
class Sample:
    path: str
    label: str


@dataclass
class TrainPair:
    sketch_path: str
    photo_path: str
    photo_id: str


@dataclass
class Index:
    protocol: str
    train_pairs: List[TrainPair] = field(default_factory=list)
    query: List[Sample] = field(default_factory=list)
    gallery: List[Sample] = field(default_factory=list)

    def stats(self) -> Dict[str, int]:
        return {
            "train_pairs": len(self.train_pairs),
            "query": len(self.query),
            "gallery": len(self.gallery),
        }


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _list_images(dirpath: str) -> List[str]:
    if not os.path.isdir(dirpath):
        return []
    out = []
    for f in sorted(os.listdir(dirpath)):
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
            out.append(os.path.join(dirpath, f))
    return out


def _instance_id_from_stem(stem: str) -> str:
    """Default flat-layout instance id: prefix before the last separator."""
    for sep in ("-", "_"):
        if sep in stem:
            head, _tail = stem.rsplit(sep, 1)
            if _tail.isdigit():
                return head
    return stem


def _read_list_file(path: str, three_col: bool) -> List[Tuple[str, ...]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if three_col and len(parts) != 3:
                raise ValueError(f"train list expects 3 columns, got: {line!r}")
            if not three_col and len(parts) != 2:
                raise ValueError(f"query/gallery list expects 2 columns, got: {line!r}")
            rows.append(tuple(parts))
    return rows


# --------------------------------------------------------------------------- #
# QMUL (instance-level) discovery
# --------------------------------------------------------------------------- #
def _discover_qmul(root: str, phase: str) -> Tuple[List[TrainPair], List[Sample], List[Sample]]:
    """phase in {'train','test'}. Returns (train_pairs or [], query, gallery)."""
    part = "train" if phase == "train" else "test"
    photo_dir = os.path.join(root, "photo", part)
    sketch_dir = os.path.join(root, "sketch", part)

    photos = _list_images(photo_dir) if os.path.isdir(photo_dir) else []
    photo_by_id = {_stem(p): p for p in photos}

    sketch_files = []
    for sub in sorted(os.listdir(sketch_dir)) if os.path.isdir(sketch_dir) else []:
        sub_path = os.path.join(sketch_dir, sub)
        if os.path.isdir(sub_path):                                   # subfolder layout
            for sk in _list_images(sub_path):
                sketch_files.append((sk, sub))
        elif os.path.splitext(sub)[1].lower() in IMAGE_EXTS:          # flat layout
            sketch_files.append((sub_path, _instance_id_from_stem(_stem(sub_path))))

    pairs: List[TrainPair] = []
    queries: List[Sample] = []
    gallery: List[Sample] = [Sample(p, _stem(p)) for p in photos] if photos else []

    for sk_path, sk_id in sketch_files:
        if phase == "train":
            matched = photo_by_id.get(sk_id)
            if matched is not None:
                pairs.append(TrainPair(sk_path, matched, sk_id))
            else:
                # sketch without matching photo: still valid as a training
                # sketch? no -- triplet requires a matched photo, so skip.
                continue
        else:
            queries.append(Sample(sk_path, sk_id))
    return pairs, queries, gallery


# --------------------------------------------------------------------------- #
# Category-level discovery (Sketchy / TU-Berlin)
# --------------------------------------------------------------------------- #
def _discover_category(root: str, phase: str) -> Tuple[List[TrainPair], List[Sample], List[Sample]]:
    part = "train" if phase == "train" else "test"
    sketch_base = os.path.join(root, "sketch", part)
    photo_base = os.path.join(root, "photo", part)

    pairs: List[TrainPair] = []
    queries: List[Sample] = []
    gallery: List[Sample] = []

    classes = set()
    if os.path.isdir(photo_base):
        classes.update(os.listdir(photo_base))
    if os.path.isdir(sketch_base):
        classes.update(os.listdir(sketch_base))

    for cls in sorted(classes):
        sk_cls_dir = os.path.join(sketch_base, cls)
        ph_cls_dir = os.path.join(photo_base, cls)
        sk_files = _list_images(sk_cls_dir)
        ph_files = _list_images(ph_cls_dir)

        if phase == "train":
            # Deterministic pairing: each sketch is paired with the first photo
            # of its class. NOTE: for true instance-level FG-SBIR training, use
            # an explicit instance-pairing list file instead.
            for sk in sk_files:
                if ph_files:
                    pairs.append(TrainPair(sk, ph_files[0], ph_files[0]))
        else:
            queries.extend(Sample(sk, cls) for sk in sk_files)
            gallery.extend(Sample(p, cls) for p in ph_files)
    return pairs, queries, gallery


# --------------------------------------------------------------------------- #
# public index builder
# --------------------------------------------------------------------------- #
def build_index(dataset_name: str, root: str, list_files: Optional[Dict[str, str]] = None) -> Index:
    protocol = PROTOCOLS.get(dataset_name)
    if protocol is None:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Choose from {list(PROTOCOLS)}"
        )

    idx = Index(protocol=protocol)
    if list_files and list_files.get("train"):
        idx.train_pairs = [
            TrainPair(sk, ph, ph) for (sk, ph, _phid) in _read_list_file(list_files["train"], three_col=True)
        ]
    else:
        for part in ("train", "test"):
            if protocol == "instance":
                p, q, g = _discover_qmul(root, part)
            else:
                p, q, g = _discover_category(root, part)
            idx.train_pairs.extend(p)
            idx.query.extend(q)
            idx.gallery.extend(g)

        if list_files:
            if list_files.get("query"):
                idx.query = [Sample(a, b) for (a, b) in _read_list_file(list_files["query"], three_col=False)]
            if list_files.get("gallery"):
                idx.gallery = [Sample(a, b) for (a, b) in _read_list_file(list_files["gallery"], three_col=False)]

    return idx


# --------------------------------------------------------------------------- #
# PyTorch datasets
# --------------------------------------------------------------------------- #
def _pil_loader(path: str, transform) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return transform(img) if transform is not None else img


class LabelEncoder:
    """Deterministic string -> int encoder shared between query and gallery."""

    def __init__(self):
        self._ids: Dict[str, int] = {}

    def fit(self, labels: List[str]):
        for lab in sorted(set(labels)):
            if lab not in self._ids:
                self._ids[lab] = len(self._ids)
        return self

    def encode(self, label: str) -> int:
        if label not in self._ids:
            self._ids[label] = len(self._ids)
        return self._ids[label]

    def __len__(self):
        return len(self._ids)


class PairDataset(Dataset):
    """Training dataset yielding (sketch, photo, photo_identity)."""

    def __init__(self, pairs: List[TrainPair], transform=None):
        self.pairs = pairs
        self.transform = transform
        self.encoder = LabelEncoder().fit([p.photo_path for p in pairs])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        p = self.pairs[i]
        s = _pil_loader(p.sketch_path, self.transform)
        ph = _pil_loader(p.photo_path, self.transform)
        return s, ph, self.encoder.encode(p.photo_path)


class ImageDataset(Dataset):
    """Query/gallery dataset yielding (image, label)."""

    def __init__(self, samples: List[Sample], transform=None, encoder: Optional[LabelEncoder] = None):
        self.samples = samples
        self.transform = transform
        self.encoder = encoder if encoder is not None else LabelEncoder().fit([s.label for s in samples])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        return _pil_loader(s.path, self.transform), self.encoder.encode(s.label)


def build_datasets(cfg, transform):
    """Return (train_ds, query_ds, gallery_ds, protocol, label_encoder)."""
    ds_cfg = cfg["dataset"]
    idx = build_index(ds_cfg["name"], ds_cfg["root"], ds_cfg.get("list_files"))
    query_enc = LabelEncoder().fit([s.label for s in idx.query] + [s.label for s in idx.gallery])
    train_ds = PairDataset(idx.train_pairs, transform) if idx.train_pairs else None
    query_ds = ImageDataset(idx.query, transform, query_enc)
    gallery_ds = ImageDataset(idx.gallery, transform, query_enc)
    return train_ds, query_ds, gallery_ds, idx.protocol, idx.stats()