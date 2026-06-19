#!/usr/bin/env python3
# ============================================================
# train_all_classes.py
# STEP 1 — Train MobileNetV3-Large on ALL 20 classes using
# the Phase 2 clean_dataset. This is the baseline model.
#
# Usage:
#   python train_all_classes.py \
#     --data-dir data/raw/clean_dataset \
#     --epochs 20 --batch-size 16
# ============================================================

import os, sys, argparse, time, yaml
import torch, torch.nn as nn, torch.optim as optim
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from utils.dataset_utils import split_dataset, make_dataloaders, save_class_map
from utils.model_utils   import build_model, get_device, save_checkpoint
from utils.eval_utils    import save_training_curves


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir',   default='data/raw/clean_dataset')
    p.add_argument('--split-dir',  default='data/processed/clean_dataset_split')
    p.add_argument('--epochs',     type=int,   default=20)
    p.add_argument('--batch-size', type=int,   default=16)
    p.add_argument('--lr',         type=float, default=0.001)
    p.add_argument('--arch',       default='mobilenet_v3_large')
    p.add_argument('--workers',    type=int,   default=2)
    p.add_argument('--config',     default='config.yaml')
    return p.parse_args()


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss = correct = total = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, labels in tqdm(loader,
                desc='  train' if train else '  val  ', leave=False):
            images, labels = images.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            if train:
                loss.backward(); optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct    += outputs.max(1)[1].eq(labels).sum().item()
            total      += images.size(0)
    return total_loss / total, correct / total


def main():
    args = parse_args()
    cfg  = yaml.safe_load(open(args.config)) if os.path.exists(args.config) else {}

    for d in ['outputs/models','outputs/plots','outputs/logs']:
        os.makedirs(d, exist_ok=True)

    device = get_device()

    print('\n[1/4] Splitting clean_dataset into train/val/test ...')
    split_dataset(args.data_dir, args.split_dir,
                  cfg.get('train_ratio', 0.70),
                  cfg.get('val_ratio',   0.15),
                  cfg.get('test_ratio',  0.15))

    print('\n[2/4] Building DataLoaders ...')
    loaders, class_to_idx, idx_to_class = make_dataloaders(
        args.split_dir, cfg.get('image_size', 224),
        args.batch_size, args.workers)
    num_classes = len(class_to_idx)
    print(f'  Classes: {num_classes}')

    save_class_map(class_to_idx,
                   f"outputs/models/{cfg.get('all_class_map_name','all_class_class_to_idx.json')}")

    print(f'\n[3/4] Building {args.arch} ({num_classes} classes) ...')
    model     = build_model(num_classes, args.arch, pretrained=True,
                            dropout=cfg.get('dropout', 0.3)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=cfg.get('weight_decay', 0.0001))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    print(f'\n[4/4] Training {args.epochs} epochs ...')
    save_path = f"outputs/models/{cfg.get('all_class_model_name','all_class_model.pth')}"
    history   = {'train_loss':[],'val_loss':[],'train_acc':[],'val_acc':[]}
    best_acc  = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tl, ta = run_epoch(model, loaders['train'], criterion, optimizer, device, True)
        vl, va = run_epoch(model, loaders['val'],   criterion, None,      device, False)
        scheduler.step()
        history['train_loss'].append(tl); history['val_loss'].append(vl)
        history['train_acc'].append(ta);  history['val_acc'].append(va)
        print(f'  Epoch {epoch:3d}/{args.epochs} | '
              f'train_acc={ta:.4f} val_acc={va:.4f} | {time.time()-t0:.1f}s')
        if va > best_acc:
            best_acc = va
            save_checkpoint(model, class_to_idx, save_path, args.arch,
                            {'epoch': epoch, 'val_acc': va})
            print(f'    ✓ Best saved (val_acc={va:.4f})')

    save_training_curves(history, 'outputs/plots', 'all_class_train')
    print(f'\nDone. Best val acc: {best_acc:.4f}')
    print(f'Model → {save_path}')


if __name__ == '__main__':
    main()
