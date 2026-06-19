#!/usr/bin/env python3
# ============================================================
# train_three_classes.py
# STEP 4 — Final fine-tune on 3 selected classes only.
# Classes: Cocacola_classic, Sprite, Redbull_Classic
# This is the model deployed to the robot for actions.
#
# Usage:
#   python train_three_classes.py \
#     --robot-dir data/robot_split \
#     --output-dir data/processed/three_class_robot \
#     --base-model outputs/models/robot_finetuned_model.pth \
#     --epochs 15 --batch-size 16
# ============================================================

import os, sys, argparse, time, yaml, shutil
import torch, torch.nn as nn, torch.optim as optim
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from utils.dataset_utils import (split_dataset, make_dataloaders,
                                  save_class_map, IMG_EXTS)
from utils.model_utils   import (build_model, get_device, save_checkpoint,
                                  freeze_backbone, unfreeze_all)
from utils.eval_utils    import save_training_curves

THREE_CLASSES = ['Cocacola_classic', 'Sprite', 'Redbull_Classic']


def parse_args():
    p = argparse.ArgumentParser(
        description='Final 3-class fine-tune for robot deployment')
    p.add_argument('--robot-dir',   default='data/robot_split',
                   help='Robot split directory (all 20 classes)')
    p.add_argument('--output-dir',  default='data/processed/three_class_robot',
                   help='Where to write the 3-class subset')
    p.add_argument('--base-model',  default='outputs/models/robot_finetuned_model.pth')
    p.add_argument('--epochs',      type=int,   default=15)
    p.add_argument('--batch-size',  type=int,   default=16)
    p.add_argument('--lr',          type=float, default=0.0005)
    p.add_argument('--arch',        default='mobilenet_v3_large')
    p.add_argument('--workers',     type=int,   default=2)
    p.add_argument('--config',      default='config.yaml')
    return p.parse_args()


def prepare_three_class_subset(robot_dir, output_dir):
    """Extract only 3 classes from robot_split into a fresh folder."""
    robot_dir  = Path(robot_dir)
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)

    print(f'  Extracting 3 classes from {robot_dir} ...')
    for split in ('train', 'val', 'test'):
        for cls in THREE_CLASSES:
            src = robot_dir / split / cls
            dst = output_dir / split / cls
            if src.exists():
                shutil.copytree(src, dst)
                n = sum(1 for f in dst.rglob('*') if f.suffix.lower() in IMG_EXTS)
                print(f'    {split}/{cls}: {n} images')
            else:
                print(f'    [warn] {src} not found — skipping')


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

    for d in ['outputs/models', 'outputs/plots']:
        os.makedirs(d, exist_ok=True)

    device = get_device()

    print(f'\n[1/4] Preparing 3-class subset: {THREE_CLASSES}')
    prepare_three_class_subset(args.robot_dir, args.output_dir)

    print('\n[2/4] Building DataLoaders ...')
    loaders, class_to_idx, idx_to_class = make_dataloaders(
        args.output_dir, cfg.get('image_size', 224),
        args.batch_size, args.workers)
    num_classes = len(class_to_idx)
    print(f'  Classes: {list(class_to_idx.keys())}')

    map_path = f"outputs/models/{cfg.get('three_class_map_name','three_class_class_to_idx.json')}"
    save_class_map(class_to_idx, map_path)

    print(f'\n[3/4] Loading base model: {args.base_model}')
    model = build_model(num_classes, args.arch, pretrained=True,
                        dropout=cfg.get('dropout', 0.3))

    if os.path.exists(args.base_model):
        import torch as _torch
        ckpt    = _torch.load(args.base_model, map_location='cpu')
        state   = ckpt.get('state_dict', ckpt)
        m_state = model.state_dict()
        matched = {k: v for k, v in state.items()
                   if k in m_state and 'classifier' not in k
                   and m_state[k].shape == v.shape}
        m_state.update(matched)
        model.load_state_dict(m_state)
        print(f'  Loaded {len(matched)} backbone layers.')

    model     = model.to(device)
    criterion = nn.CrossEntropyLoss()
    save_path = f"outputs/models/{cfg.get('three_class_model_name','three_class_robot_model.pth')}"
    history   = {'train_loss':[],'val_loss':[],'train_acc':[],'val_acc':[]}
    best_acc  = 0.0

    print(f'\n[4/4] Two-phase fine-tuning ({args.epochs} epochs) ...')

    head_epochs = min(3, args.epochs // 4)
    freeze_backbone(model, args.arch)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=cfg.get('weight_decay', 0.0001))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7)

    print(f'  Phase A — head only ({head_epochs} epochs)')
    for epoch in range(1, head_epochs + 1):
        t0 = time.time()
        tl, ta = run_epoch(model, loaders['train'], criterion, optimizer, device, True)
        vl, va = run_epoch(model, loaders['val'],   criterion, None,      device, False)
        scheduler.step()
        history['train_loss'].append(tl); history['val_loss'].append(vl)
        history['train_acc'].append(ta);  history['val_acc'].append(va)
        print(f'  Epoch {epoch:3d}/{args.epochs} | train_acc={ta:.4f} val_acc={va:.4f} | {time.time()-t0:.1f}s')
        if va > best_acc:
            best_acc = va
            save_checkpoint(model, class_to_idx, save_path, args.arch,
                            {'epoch': epoch, 'val_acc': va})
            print(f'    ✓ Best saved')

    remaining = args.epochs - head_epochs
    if remaining > 0:
        print(f'\n  Phase B — full fine-tune ({remaining} epochs)')
        unfreeze_all(model)
        optimizer = optim.Adam(model.parameters(), lr=args.lr * 0.1,
                               weight_decay=cfg.get('weight_decay', 0.0001))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=remaining, eta_min=1e-8)
        for epoch in range(head_epochs + 1, args.epochs + 1):
            t0 = time.time()
            tl, ta = run_epoch(model, loaders['train'], criterion, optimizer, device, True)
            vl, va = run_epoch(model, loaders['val'],   criterion, None,      device, False)
            scheduler.step()
            history['train_loss'].append(tl); history['val_loss'].append(vl)
            history['train_acc'].append(ta);  history['val_acc'].append(va)
            print(f'  Epoch {epoch:3d}/{args.epochs} | train_acc={ta:.4f} val_acc={va:.4f} | {time.time()-t0:.1f}s')
            if va > best_acc:
                best_acc = va
                save_checkpoint(model, class_to_idx, save_path, args.arch,
                                {'epoch': epoch, 'val_acc': va})
                print(f'    ✓ Best saved')

    save_training_curves(history, 'outputs/plots', 'three_class_train')
    print(f'\nDone. Best val acc: {best_acc:.4f}')
    print(f'Model → {save_path}')
    print(f'Class map → {map_path}')


if __name__ == '__main__':
    main()
