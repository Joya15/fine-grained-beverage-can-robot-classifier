#!/usr/bin/env python3
# ============================================================
# evaluate.py
# Evaluate a saved model on a test set.
# Works for both 20-class and 3-class models.
# Outputs: accuracy, F1, confusion matrix PNG, predictions CSV.
#
# Usage:
#   # Before robot fine-tune (all 20 classes):
#   python evaluate.py \
#     --data-dir data/robot_split/test \
#     --model-path outputs/models/all_class_model.pth \
#     --tag before_robot_finetune
#
#   # After robot fine-tune (all 20 classes):
#   python evaluate.py \
#     --data-dir data/robot_split/test \
#     --model-path outputs/models/robot_finetuned_model.pth \
#     --tag after_robot_finetune
#
#   # Three-class robot model:
#   python evaluate.py \
#     --data-dir data/processed/three_class_robot/test \
#     --model-path outputs/models/three_class_robot_model.pth \
#     --tag three_class_eval
# ============================================================

import os, sys, argparse, yaml
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from utils.model_utils   import load_checkpoint, get_device
from utils.dataset_utils import get_val_transform
from utils.eval_utils    import (compute_metrics, save_metrics,
                                  save_confusion_matrix,
                                  save_predictions_csv)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir',    required=True)
    p.add_argument('--model-path',  required=True)
    p.add_argument('--batch-size',  type=int, default=16)
    p.add_argument('--workers',     type=int, default=2)
    p.add_argument('--tag',         default='eval')
    p.add_argument('--config',      default='config.yaml')
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = yaml.safe_load(open(args.config)) if os.path.exists(args.config) else {}

    for d in ['outputs/logs', 'outputs/plots', 'outputs/predictions']:
        os.makedirs(d, exist_ok=True)

    device = get_device()

    print(f'\nLoading model : {args.model_path}')
    model, class_to_idx, idx_to_class = load_checkpoint(
        args.model_path, device=str(device))

    print(f'Loading data  : {args.data_dir}')
    transform   = get_val_transform(cfg.get('image_size', 224))
    test_dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers)

    img_paths   = [s[0] for s in test_dataset.samples]

    print('\nRunning inference ...')
    model.eval()
    true_names = []
    pred_names = []
    records = []
    idx = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='  evaluating'):
            images  = images.to(device)
            outputs = model(images)
            probs   = torch.softmax(outputs, dim=1)
            confs, preds = probs.max(1)

            for i in range(len(labels)):
                true_name = test_dataset.classes[labels[i].item()]
                pred_name = idx_to_class[preds[i].item()]
                true_names.append(true_name)
                pred_names.append(pred_name)
                records.append({
                    'image_path':      img_paths[idx],
                    'true_label':      true_name,
                    'predicted_label': pred_name,
                    'confidence':      round(confs[i].item(), 4),
                    'correct':         (true_name == pred_name),
                })
                idx += 1

    # Evaluate on the union of true labels and predicted labels. This keeps
    # out-of-subset predictions visible instead of remapping them silently.
    class_names = list(test_dataset.classes)
    for name in sorted(set(pred_names)):
        if name not in class_names:
            class_names.append(name)
    eval_class_to_idx = {name: i for i, name in enumerate(class_names)}
    all_labels = [eval_class_to_idx[name] for name in true_names]
    all_preds = [eval_class_to_idx[name] for name in pred_names]

    # Metrics
    metrics = compute_metrics(all_labels, all_preds, class_names)
    print(f'\n--- Results ({args.tag}) ---')
    print(f'  Accuracy       : {metrics["accuracy"]:.4f}')
    print(f'  F1 Macro       : {metrics["f1_macro"]:.4f}')
    print(f'  F1 Weighted    : {metrics["f1_weighted"]:.4f}')
    print(f'  Precision Macro: {metrics["precision_macro"]:.4f}')
    print(f'  Recall Macro   : {metrics["recall_macro"]:.4f}')
    print('\n  Per-class accuracy:')
    for cls, acc in metrics['per_class_accuracy'].items():
        print(f'    {cls}: {acc:.4f}')

    save_metrics(metrics, 'outputs/logs', args.tag)
    save_confusion_matrix(
        all_labels, all_preds, class_names,
        f'outputs/plots/{args.tag}_confusion_matrix.png',
        title=f'Confusion Matrix — {args.tag}')
    save_predictions_csv(records,
        f'outputs/predictions/{args.tag}_predictions.csv')
    print('\nEvaluation complete.')


if __name__ == '__main__':
    main()
