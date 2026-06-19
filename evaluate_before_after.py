#!/usr/bin/env python3
# ============================================================
# evaluate_before_after.py
# Run BOTH evaluations in one command and generate the
# before vs after comparison table and plot.
# This is exactly what the assignment asks for:
#   1. Test all-class model on robot test images (before fine-tune)
#   2. Test robot-fine-tuned model on same test images (after)
#   3. Compare and report the difference
#
# Usage:
#   python evaluate_before_after.py \
#     --test-dir data/robot_split/test \
#     --before-model outputs/models/all_class_model.pth \
#     --after-model  outputs/models/robot_finetuned_model.pth
# ============================================================

import os, sys, argparse, json, yaml
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from utils.model_utils   import load_checkpoint, get_device
from utils.dataset_utils import get_val_transform
from utils.eval_utils    import (compute_metrics, save_metrics,
                                  save_confusion_matrix,
                                  save_predictions_csv,
                                  save_comparison_plot)


def parse_args():
    p = argparse.ArgumentParser(
        description='Before vs after robot fine-tune evaluation')
    p.add_argument('--test-dir',      required=True,
                   help='Test set directory (robot_split/test)')
    p.add_argument('--before-model',  required=True,
                   help='all_class_model.pth (Phase 2 trained)')
    p.add_argument('--after-model',   required=True,
                   help='robot_finetuned_model.pth (Phase 3 fine-tuned)')
    p.add_argument('--batch-size',    type=int, default=16)
    p.add_argument('--workers',       type=int, default=2)
    p.add_argument('--config',        default='config.yaml')
    return p.parse_args()


def run_inference(model, test_loader, test_dataset,
                  idx_to_class, class_to_idx, device):
    img_paths = [s[0] for s in test_dataset.samples]
    all_preds = []; all_labels = []; records = []
    idx = 0
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='  evaluating', leave=False):
            images  = images.to(device)
            outputs = model(images)
            probs   = torch.softmax(outputs, dim=1)
            confs, preds = probs.max(1)
            for i in range(len(labels)):
                true_name      = test_dataset.classes[labels[i].item()]
                pred_name      = idx_to_class[preds[i].item()]
                true_model_idx = class_to_idx.get(true_name, labels[i].item())
                all_labels.append(true_model_idx)
                all_preds.append(preds[i].item())
                records.append({
                    'image_path':      img_paths[idx],
                    'true_label':      true_name,
                    'predicted_label': pred_name,
                    'confidence':      round(confs[i].item(), 4),
                    'correct':         (true_name == pred_name),
                })
                idx += 1
    return all_labels, all_preds, records


def main():
    args = parse_args()
    cfg  = yaml.safe_load(open(args.config)) if os.path.exists(args.config) else {}

    for d in ['outputs/logs', 'outputs/plots', 'outputs/predictions']:
        os.makedirs(d, exist_ok=True)

    device    = get_device()
    transform = get_val_transform(cfg.get('image_size', 224))

    test_dataset = datasets.ImageFolder(args.test_dir, transform=transform)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers)

    results = {}

    for tag, model_path in [('before_robot_finetune', args.before_model),
                             ('after_robot_finetune',  args.after_model)]:
        print(f'\n=== {tag} ===')
        print(f'  Model: {model_path}')
        model, class_to_idx, idx_to_class = load_checkpoint(
            model_path, device=str(device))
        class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

        all_labels, all_preds, records = run_inference(
            model, test_loader, test_dataset,
            idx_to_class, class_to_idx, device)

        metrics = compute_metrics(all_labels, all_preds, class_names)
        print(f'  Accuracy  : {metrics["accuracy"]:.4f}')
        print(f'  F1 Macro  : {metrics["f1_macro"]:.4f}')
        print(f'  F1 Weighted: {metrics["f1_weighted"]:.4f}')

        save_metrics(metrics, 'outputs/logs', tag)
        save_confusion_matrix(
            all_labels, all_preds, class_names,
            f'outputs/plots/{tag}_confusion_matrix.png',
            title=f'Confusion Matrix — {tag}')
        save_predictions_csv(records,
            f'outputs/predictions/{tag}_predictions.csv')

        results[tag] = {
            'metrics':     metrics,
            'class_names': class_names,
        }

    # --- Comparison table ---
    print('\n=== Before vs After Comparison ===')
    before_m = results['before_robot_finetune']['metrics']
    after_m  = results['after_robot_finetune']['metrics']

    comp = {
        'Metric': ['Accuracy', 'F1 Macro', 'F1 Weighted',
                   'Precision Macro', 'Recall Macro'],
        'Before fine-tune': [
            before_m['accuracy'], before_m['f1_macro'],
            before_m['f1_weighted'], before_m['precision_macro'],
            before_m['recall_macro']],
        'After fine-tune': [
            after_m['accuracy'], after_m['f1_macro'],
            after_m['f1_weighted'], after_m['precision_macro'],
            after_m['recall_macro']],
    }
    df = pd.DataFrame(comp)
    df['Change'] = (df['After fine-tune'] - df['Before fine-tune']).round(4)
    print(df.to_string(index=False))
    df.to_csv('outputs/logs/before_after_comparison.csv', index=False)
    print('\n  Comparison CSV → outputs/logs/before_after_comparison.csv')

    # --- Per-class comparison plot ---
    class_names = results['before_robot_finetune']['class_names']
    save_comparison_plot(
        before_m, after_m, class_names,
        'outputs/plots/before_after_per_class.png')

    print('\nAll done.')


if __name__ == '__main__':
    main()
