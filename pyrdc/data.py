"""Datasets and transforms for the classification / segmentation experiments."""
import os
import random
from functools import partial

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TVF
from PIL import Image


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def oxford_mask_extractor(mask):
    return ((mask == 3) | (mask == 1)).long()


def sbdataset_mask_extractor(mask):
    return (mask > 0).long()


def kvasir_mask_extractor(mask):
    # Kvasir masks are white-on-black RGB JPEGs; pil_to_tensor -> (3, H, W)
    return (mask[0:1] > 127).long()


class KvasirSEG(torch.utils.data.Dataset):
    """Kvasir-SEG polyp segmentation (Jha et al. 2020): 1000 image/mask pairs
    with large object-scale variation. Deterministic split: filenames sorted,
    first 80% train, last 20% val.

    Expects `root` to contain Kvasir-SEG/images/*.jpg and Kvasir-SEG/masks/*.jpg
    (the layout of the official kvasir-seg.zip from datasets.simula.no).
    """

    def __init__(self, root, split, transform):
        base = os.path.join(root, "Kvasir-SEG")
        self.img_dir = os.path.join(base, "images")
        self.mask_dir = os.path.join(base, "masks")
        names = sorted(os.listdir(self.img_dir))
        n_train = int(len(names) * 0.8)
        self.names = names[:n_train] if split == "train" else names[n_train:]
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        img = Image.open(os.path.join(self.img_dir, name)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, name))
        return self.transform(img, mask)


class SegmentationTrainTransform:
    def __init__(self, mask_extractor, size=(64, 64)):
        self.size = size
        self.color_jitter = transforms.ColorJitter(
            brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02
        )
        self.mask_extractor = mask_extractor

    def __call__(self, img, mask):
        if random.random() > 0.5:
            img = TVF.hflip(img)
            mask = TVF.hflip(mask)
        angle = random.uniform(-7, 7)
        img = TVF.rotate(img, angle, interpolation=TVF.InterpolationMode.BILINEAR)
        mask = TVF.rotate(mask, angle, interpolation=TVF.InterpolationMode.NEAREST)
        img = TVF.resize(img, self.size, interpolation=TVF.InterpolationMode.BILINEAR)
        mask = TVF.resize(mask, self.size, interpolation=TVF.InterpolationMode.NEAREST)
        img = self.color_jitter(img)
        img = TVF.to_tensor(img)
        mask = torch.as_tensor(TVF.pil_to_tensor(mask), dtype=torch.long)
        mask = self.mask_extractor(mask)
        return img, mask.squeeze(0)


class SegmentationTestTransform:
    def __init__(self, mask_extractor, size=(64, 64)):
        self.size = size
        self.mask_extractor = mask_extractor

    def __call__(self, img, mask):
        img = TVF.resize(img, self.size, interpolation=TVF.InterpolationMode.BILINEAR)
        mask = TVF.resize(mask, self.size, interpolation=TVF.InterpolationMode.NEAREST)
        img = TVF.to_tensor(img)
        mask = torch.as_tensor(TVF.pil_to_tensor(mask), dtype=torch.long)
        mask = self.mask_extractor(mask)
        return img, mask.squeeze(0)


class _WrappedSegmentation(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]
        return self.transform(img, mask)


def build_segmentation_datasets(name, size=(64, 64), root=None, target_class_id=13):
    """Returns (train_dataset, test_dataset) for 'oxford_pet', 'sbd' or 'urbansys'."""
    if name == "oxford_pet":
        root = root or "oxford_iiit_pet"
        make = lambda split: torchvision.datasets.OxfordIIITPet(
            root=root, split=split, target_types="segmentation", download=True
        )
        train_raw, test_raw = make("trainval"), make("test")
        extractor = oxford_mask_extractor
    elif name == "sbd":
        root = root or "sbdataset"
        make = lambda split: torchvision.datasets.SBDataset(
            root=root, image_set=split, mode="segmentation", download=False
        )
        train_raw, test_raw = make("train"), make("val")
        extractor = sbdataset_mask_extractor
    elif name == "kvasir":
        root = root or "kvasir_seg_raw"
        train_ds = KvasirSEG(
            root, "train", SegmentationTrainTransform(kvasir_mask_extractor, size)
        )
        test_ds = KvasirSEG(
            root, "val", SegmentationTestTransform(kvasir_mask_extractor, size)
        )
        return train_ds, test_ds
    elif name == "urbansys":
        # UrbanSYSDataset applies the transform itself, so no wrapping needed.
        from urbansys_dataset import UrbanSYSDataset, urbansys_mask_extractor

        root = root or "urbansys_ds"
        extractor = partial(urbansys_mask_extractor, target_class_id)
        train_ds = UrbanSYSDataset(
            root=root, split="train",
            transform=SegmentationTrainTransform(extractor, size),
        )
        test_ds = UrbanSYSDataset(
            root=root, split="val",
            transform=SegmentationTestTransform(extractor, size),
        )
        return train_ds, test_ds
    else:
        raise ValueError(f"unknown segmentation dataset: {name}")

    train_ds = _WrappedSegmentation(train_raw, SegmentationTrainTransform(extractor, size))
    test_ds = _WrappedSegmentation(test_raw, SegmentationTestTransform(extractor, size))
    return train_ds, test_ds


# ---------------------------------------------------------------------------
# Cityscapes (multi-class, 19 train classes, ignore=255)
# ---------------------------------------------------------------------------

# labelId -> trainId mapping (Cityscapes convention); everything else -> 255
_CITYSCAPES_ID_TO_TRAIN = torch.full((256,), 255, dtype=torch.long)
for _label_id, _train_id in {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9,
    23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18,
}.items():
    _CITYSCAPES_ID_TO_TRAIN[_label_id] = _train_id

CITYSCAPES_NUM_CLASSES = 19
CITYSCAPES_CLASS_NAMES = (
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic_light",
    "traffic_sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle",
)
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class CityscapesTrainTransform:
    """Resize to `base_size` (W, H), random crop `crop_size` (H, W), hflip.

    `label_map` is a 256-entry id->trainId lookup; defaults to the Cityscapes
    mapping, ADE20K passes `ADE20K_LABEL_MAP`.
    """

    def __init__(self, base_size=(1024, 512), crop_size=(256, 512), label_map=None):
        self.base_size = base_size
        self.crop_size = crop_size
        self.label_map = _CITYSCAPES_ID_TO_TRAIN if label_map is None else label_map

    def __call__(self, img, target):
        img = img.resize(self.base_size, Image.BILINEAR)
        target = target.resize(self.base_size, Image.NEAREST)
        ch, cw = self.crop_size
        x = random.randint(0, self.base_size[0] - cw)
        y = random.randint(0, self.base_size[1] - ch)
        img = img.crop((x, y, x + cw, y + ch))
        target = target.crop((x, y, x + cw, y + ch))
        if random.random() > 0.5:
            img = TVF.hflip(img)
            target = TVF.hflip(target)
        img = TVF.normalize(TVF.to_tensor(img), IMAGENET_MEAN, IMAGENET_STD)
        ids = torch.as_tensor(TVF.pil_to_tensor(target), dtype=torch.long).squeeze(0)
        return img, self.label_map[ids]


class CityscapesValTransform:
    def __init__(self, base_size=(1024, 512), label_map=None):
        self.base_size = base_size
        self.label_map = _CITYSCAPES_ID_TO_TRAIN if label_map is None else label_map

    def __call__(self, img, target):
        img = img.resize(self.base_size, Image.BILINEAR)
        target = target.resize(self.base_size, Image.NEAREST)
        img = TVF.normalize(TVF.to_tensor(img), IMAGENET_MEAN, IMAGENET_STD)
        ids = torch.as_tensor(TVF.pil_to_tensor(target), dtype=torch.long).squeeze(0)
        return img, self.label_map[ids]


class CityscapesSeg(torch.utils.data.Dataset):
    """Cityscapes semantic segmentation via torchvision, with trainId mapping.

    `root` must contain leftImg8bit/{split}/... and gtFine/{split}/... in the
    official layout (the extracted leftImg8bit_trainvaltest.zip +
    gtFine_trainvaltest.zip).
    """

    def __init__(self, root, split, transform):
        self.dataset = torchvision.datasets.Cityscapes(
            root, split=split, mode="fine", target_type="semantic"
        )
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        return self.transform(img, target)


# ---------------------------------------------------------------------------
# ADE20K / SceneParse150 (multi-class, 150 classes, ignore=255)
# ---------------------------------------------------------------------------

# annotation value 0 = ignore, 1..150 -> train ids 0..149
ADE20K_LABEL_MAP = torch.full((256,), 255, dtype=torch.long)
ADE20K_LABEL_MAP[1:151] = torch.arange(150)

ADE20K_NUM_CLASSES = 150
# objectInfo150 order; cosmetic (analysis keys on class id + pixel frequency)
ADE20K_CLASS_NAMES = (
    "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed",
    "windowpane", "grass", "cabinet", "sidewalk", "person", "earth", "door",
    "table", "mountain", "plant", "curtain", "chair", "car", "water",
    "painting", "sofa", "shelf", "house", "sea", "mirror", "rug", "field",
    "armchair", "seat", "fence", "desk", "rock", "wardrobe", "lamp",
    "bathtub", "railing", "cushion", "base", "box", "column", "signboard",
    "chest_of_drawers", "counter", "sand", "sink", "skyscraper", "fireplace",
    "refrigerator", "grandstand", "path", "stairs", "runway", "case",
    "pool_table", "pillow", "screen_door", "stairway", "river", "bridge",
    "bookcase", "blind", "coffee_table", "toilet", "flower", "book", "hill",
    "bench", "countertop", "stove", "palm", "kitchen_island", "computer",
    "swivel_chair", "boat", "bar", "arcade_machine", "hovel", "bus", "towel",
    "light", "truck", "tower", "chandelier", "awning", "streetlight",
    "booth", "television", "airplane", "dirt_track", "apparel", "pole",
    "land", "bannister", "escalator", "ottoman", "bottle", "buffet",
    "poster", "stage", "van", "ship", "fountain", "conveyer_belt", "canopy",
    "washer", "plaything", "swimming_pool", "stool", "barrel", "basket",
    "waterfall", "tent", "bag", "minibike", "cradle", "oven", "ball",
    "food", "step", "tank", "trade_name", "microwave", "pot", "animal",
    "bicycle", "lake", "dishwasher", "screen", "blanket", "sculpture",
    "hood", "sconce", "vase", "traffic_light", "tray", "ashcan", "fan",
    "pier", "crt_screen", "plate", "monitor", "bulletin_board", "shower",
    "radiator", "glass", "clock", "flag",
)


class Ade20kSeg(torch.utils.data.Dataset):
    """ADE20K scene parsing benchmark in the ADEChallengeData2016 layout:

        <root>/images/{training,validation}/*.jpg
        <root>/annotations/{training,validation}/*.png

    Reuses the Cityscapes transforms with `label_map=ADE20K_LABEL_MAP`.
    """

    SPLITS = {"train": "training", "val": "validation"}

    def __init__(self, root, split, transform):
        sub = self.SPLITS[split]
        img_dir = os.path.join(root, "images", sub)
        ann_dir = os.path.join(root, "annotations", sub)
        self.items = [
            (os.path.join(img_dir, fn), os.path.join(ann_dir, fn[:-4] + ".png"))
            for fn in sorted(os.listdir(img_dir)) if fn.endswith(".jpg")
        ]
        if not self.items:
            raise FileNotFoundError(f"no .jpg images under {img_dir}")
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, ann_path = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        target = Image.open(ann_path)
        return self.transform(img, target)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

CIFAR10_MEAN, CIFAR10_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)


def build_cifar_datasets(name="cifar10", root=None):
    """Returns (train_dataset, test_dataset, num_classes) for cifar10/cifar100."""
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    if name == "cifar10":
        cls, root, n = torchvision.datasets.CIFAR10, root or "./cifar_data", 10
    elif name == "cifar100":
        cls, root, n = torchvision.datasets.CIFAR100, root or "./cifar_data100", 100
    else:
        raise ValueError(f"unknown classification dataset: {name}")
    train_ds = cls(root=root, train=True, download=True, transform=train_tf)
    test_ds = cls(root=root, train=False, download=True, transform=test_tf)
    return train_ds, test_ds, n


def build_loaders(train_ds, test_ds, batch_size=128, num_workers=4):
    """Test loader keeps every sample (the notebook used drop_last=True for the
    test set, which silently biased the reported metrics)."""
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False,
    )
    return train_loader, test_loader
