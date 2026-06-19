#!/usr/bin/env python3
# ============================================================
# prepare_robot_dataset.py
# STEP 2 — Split robot-captured images (all 20 classes) into
# train / val / test sets for fine-tuning and evaluation.
#
# Put your robot-captured images here first:
#   data/robot_captured/
#       Cocacola_classic/  image0.jpg  image1.jpg ...
#       Sprite/            ...
#       Redbull_Classic/   ...
#       ... (all 20 classes)
#
# Usage:
#   python prepare_robot_dataset.py \
#     --source-dir data/robot_captured \
#     --output-dir data/robot_split
# ============================================================

import os, sys, argparse, yaml

sys.path.insert(0, os.path.dirname(__file__))
from utils.dataset_utils import split_dataset


def parse_args():
    p = argparse.ArgumentParser(
        description='Split robot-captured images into train/val/test')
    p.add_argument('--source-dir', default='data/robot_captured',
                   help='Folder with one subfolder per class (robot images)')
    p.add_argument('--output-dir', default='data/robot_split',
                   help='Output folder for train/val/test split')
    p.add_argument('--train-ratio', type=float, default=0.70)
    p.add_argument('--val-ratio',   type=float, default=0.15)
    p.add_argument('--test-ratio',  type=float, default=0.15)
    p.add_argument('--config',      default='config.yaml')
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = yaml.safe_load(open(args.config)) if os.path.exists(args.config) else {}

    print(f'\nSource : {args.source_dir}')
    print(f'Output : {args.output_dir}')
    print(f'Split  : train={args.train_ratio} val={args.val_ratio} test={args.test_ratio}\n')

    split_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    # Print summary
    print('\n--- Dataset Summary ---')
    from pathlib import Path
    for split in ('train', 'val', 'test'):
        split_dir = Path(args.output_dir) / split
        if not split_dir.exists():
            continue
        total = sum(
            sum(1 for f in cls_dir.iterdir()
                if f.suffix.lower() in {'.jpg','.jpeg','.png'})
            for cls_dir in split_dir.iterdir() if cls_dir.is_dir()
        )
        print(f'  {split}: {total} images')

    print(f'\nDone! Robot dataset ready at: {args.output_dir}')


if __name__ == '__main__':
    main()
