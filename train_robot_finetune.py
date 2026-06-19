#!/usr/bin/env python3
"""
Fine-tune the all-class classifier on robot-camera images.

Expected input:
  data/robot_split/
    train/<class_name>/*.jpg
    val/<class_name>/*.jpg
    test/<class_name>/*.jpg
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from utils.dataset_utils import make_dataloaders, save_class_map
from utils.eval_utils import save_training_curves
from utils.model_utils import (
    build_model,
    freeze_backbone,
    get_device,
    save_checkpoint,
    unfreeze_all,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune the 20-class model on robot-camera images."
    )
    parser.add_argument("--data-dir", default="data/robot_split")
    parser.add_argument("--base-model", default="outputs/models/all_class_model.pth")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--head-lr", type=float, default=0.001)
    parser.add_argument("--backbone-lr", type=float, default=0.0001)
    parser.add_argument("--arch", default="mobilenet_v3_large")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args()


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for images, labels in tqdm(loader, desc="  train" if train else "  val  ", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            if train:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_correct += outputs.argmax(dim=1).eq(labels).sum().item()
            total_seen += images.size(0)

    return total_loss / total_seen, total_correct / total_seen


def load_matching_weights(model, model_path):
    if not os.path.exists(model_path):
        print(f"  [warning] Base model not found: {model_path}")
        return

    checkpoint = torch.load(model_path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    current = model.state_dict()
    matched = {
        key: value
        for key, value in state.items()
        if key in current and current[key].shape == value.shape
    }
    current.update(matched)
    model.load_state_dict(current)
    print(f"  Loaded {len(matched)} matching tensors from {model_path}")


def main():
    args = parse_args()
    config = yaml.safe_load(open(args.config)) if os.path.exists(args.config) else {}

    for directory in ["outputs/models", "outputs/plots", "outputs/logs"]:
        os.makedirs(directory, exist_ok=True)

    device = get_device()

    print("\n[1/4] Building robot DataLoaders ...")
    loaders, class_to_idx, _ = make_dataloaders(
        args.data_dir,
        config.get("image_size", 224),
        args.batch_size,
        args.workers,
    )
    num_classes = len(class_to_idx)
    print(f"  Classes: {num_classes}")

    map_path = f"outputs/models/{config.get('robot_finetuned_map_name', 'robot_finetuned_class_to_idx.json')}"
    save_class_map(class_to_idx, map_path)

    print(f"\n[2/4] Building {args.arch} and loading base weights ...")
    model = build_model(
        num_classes,
        arch=args.arch,
        pretrained=True,
        dropout=config.get("dropout", 0.3),
    )
    load_matching_weights(model, args.base_model)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc = 0.0
    save_path = f"outputs/models/{config.get('robot_finetuned_model_name', 'robot_finetuned_model.pth')}"

    head_epochs = min(3, max(1, args.epochs // 4))

    print(f"\n[3/4] Phase A: classifier head fine-tuning ({head_epochs} epochs) ...")
    freeze_backbone(model, args.arch)
    optimizer = optim.Adam(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=args.head_lr,
        weight_decay=config.get("weight_decay", 0.0001),
    )

    for epoch in range(1, head_epochs + 1):
        start = time.time()
        train_loss, train_acc = run_epoch(model, loaders["train"], criterion, optimizer, device, True)
        val_loss, val_acc = run_epoch(model, loaders["val"], criterion, None, device, False)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        print(
            f"  Epoch {epoch:03d}/{args.epochs} | "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} | {time.time() - start:.1f}s"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, class_to_idx, save_path, args.arch, {"epoch": epoch, "val_acc": val_acc})

    print(f"\n[4/4] Phase B: full-network fine-tuning ({args.epochs - head_epochs} epochs) ...")
    unfreeze_all(model)
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.backbone_lr,
        weight_decay=config.get("weight_decay", 0.0001),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs - head_epochs),
        eta_min=1e-7,
    )

    for epoch in range(head_epochs + 1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = run_epoch(model, loaders["train"], criterion, optimizer, device, True)
        val_loss, val_acc = run_epoch(model, loaders["val"], criterion, None, device, False)
        scheduler.step()
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        print(
            f"  Epoch {epoch:03d}/{args.epochs} | "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} | {time.time() - start:.1f}s"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, class_to_idx, save_path, args.arch, {"epoch": epoch, "val_acc": val_acc})

    save_training_curves(history, "outputs/plots", "robot_finetune_train")
    print(f"\nDone. Best validation accuracy: {best_acc:.4f}")
    print(f"Model: {save_path}")
    print(f"Class map: {map_path}")


if __name__ == "__main__":
    main()

