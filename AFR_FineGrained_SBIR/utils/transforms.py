"""Input transforms.

The manuscript states both sketch and photo inputs are resized to 224x224
(Section 4.1). No data augmentation is specified, so the default transform is a
plain resize + ToTensor + ImageNet normalisation.
"""

from torchvision import transforms


def build_transform(cfg):
    size = cfg.get("dataset", {}).get("image_size", 224)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        normalize,
    ])


def build_train_transform(cfg):
    # Kept identical to the eval transform: the paper specifies no augmentation.
    return build_transform(cfg)