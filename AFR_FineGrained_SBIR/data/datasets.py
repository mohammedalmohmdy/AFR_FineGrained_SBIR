
from torch.utils.data import Dataset
from PIL import Image
import os

class PairDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.items = []  # populate with (sketch_path, pos_path, neg_path)
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        s,p,n = self.items[idx]
        s = Image.open(os.path.join(self.root, s)).convert("RGB")
        p = Image.open(os.path.join(self.root, p)).convert("RGB")
        n = Image.open(os.path.join(self.root, n)).convert("RGB")
        if self.transform:
            s,p,n = self.transform(s), self.transform(p), self.transform(n)
        return s,p,n
