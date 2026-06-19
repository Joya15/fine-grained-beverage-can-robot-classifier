# ============================================================
# utils/dataset_utils.py
# Dataset loading, splitting, augmentation, class-imbalance
# handling via WeightedRandomSampler.
# ============================================================

import os
import json
import shutil
import random
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


# ------------------------------------------------------------------
# Transforms
# ------------------------------------------------------------------

def get_train_transform(image_size=224):
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.2, hue=0.05),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_val_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ------------------------------------------------------------------
# Split dataset into train / val / test
# ------------------------------------------------------------------

def split_dataset(source_dir, output_dir,
                  train_ratio=0.70, val_ratio=0.15,
                  test_ratio=0.15, seed=42):
    """
    Copy images from source_dir (one folder per class) into
    output_dir/{train,val,test}/{class_name}/ splits.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    random.seed(seed)
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    for class_dir in sorted(source_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        images = [f for f in class_dir.iterdir()
                  if f.suffix.lower() in IMG_EXTS]
        if not images:
            print(f"  [skip] {class_dir.name} — no images")
            continue

        random.shuffle(images)
        n       = len(images)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)

        splits = {
            'train': images[:n_train],
            'val':   images[n_train:n_train + n_val],
            'test':  images[n_train + n_val:],
        }
        for split_name, split_imgs in splits.items():
            dest = output_dir / split_name / class_dir.name
            dest.mkdir(parents=True, exist_ok=True)
            for img in split_imgs:
                shutil.copy2(img, dest / img.name)

        print(f"  {class_dir.name}: "
              f"train={len(splits['train'])} "
              f"val={len(splits['val'])} "
              f"test={len(splits['test'])}")


# ------------------------------------------------------------------
# DataLoader factory
# ------------------------------------------------------------------

def make_dataloaders(data_dir, image_size=224, batch_size=16,
                     num_workers=2, use_weighted_sampler=True):
    """
    Build train/val/test DataLoaders from a pre-split directory.
    Returns loaders dict, class_to_idx, idx_to_class.
    """
    data_dir = Path(data_dir)
    datasets_dict = {}

    for split in ('train', 'val', 'test'):
        split_dir = data_dir / split
        if not split_dir.exists():
            print(f"  [warning] {split_dir} not found — skipping")
            continue
        transform = (get_train_transform(image_size)
                     if split == 'train'
                     else get_val_transform(image_size))
        datasets_dict[split] = datasets.ImageFolder(
            root=str(split_dir), transform=transform)

    ref = datasets_dict.get('train') or next(iter(datasets_dict.values()))
    class_to_idx = ref.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    loaders = {}
    for split, dataset in datasets_dict.items():
        if split == 'train' and use_weighted_sampler:
            sampler = _weighted_sampler(dataset)
            loaders[split] = DataLoader(
                dataset, batch_size=batch_size,
                sampler=sampler, num_workers=num_workers,
                pin_memory=True)
        else:
            loaders[split] = DataLoader(
                dataset, batch_size=batch_size,
                shuffle=(split == 'train'),
                num_workers=num_workers, pin_memory=True)

    return loaders, class_to_idx, idx_to_class


def _weighted_sampler(dataset):
    counts  = Counter(dataset.targets)
    total   = sum(counts.values())
    weights = [total / counts[l] for l in dataset.targets]
    return WeightedRandomSampler(weights, len(weights), replacement=True)


# ------------------------------------------------------------------
# Class map helpers
# ------------------------------------------------------------------

def save_class_map(class_to_idx, save_path):
    with open(save_path, 'w') as f:
        json.dump(class_to_idx, f, indent=2)
    print(f"  Class map saved → {save_path}")


def load_class_map(json_path):
    with open(json_path) as f:
        class_to_idx = json.load(f)
    idx_to_class = {int(v): k for k, v in class_to_idx.items()}
    return class_to_idx, idx_to_class
