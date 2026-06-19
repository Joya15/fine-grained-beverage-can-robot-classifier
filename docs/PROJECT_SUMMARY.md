# Project Summary

## Problem

The project addresses fine-grained product recognition for beverage cans. Unlike generic object detection, every class has the same basic cylindrical shape, so the model must learn small differences in logos, colour layouts, fonts, and packaging patterns.

## Dataset

- Planned group dataset: 20 beverage-can categories.
- Joya's initial contribution: `Kirks`, `V`, `Solo`, and `Calm&Stormy`.
- Robot deployment subset: `Cocacola_classic`, `Sprite`, and `Redbull_Classic`.
- Robot-camera dataset in the Phase 3 report: 163 images split into 113 fine-tuning, 23 validation, and 27 test images.

## Methods

- PyTorch transfer learning with ImageNet-pretrained CNNs.
- Phase 1-2: frozen classifier-head training with ResNet18 and MobileNetV3-Small.
- Phase 2: fine-tuning experiments with ResNet50 and MobileNetV2 using augmentation, weighted loss, and scheduler variants.
- Phase 3/4: MobileNetV3-Large, robot-camera fine-tuning, OpenCV ROI extraction, and ROS2 deployment.

## Main Results

- Phase 1-2 individual subset: MobileNetV3-Small achieved 90.00% test accuracy and macro F1 0.889.
- Phase 2 individual subset: best MobileNetV2 and ResNet50 runs achieved 100% test accuracy on the held-out split.
- Final 20-class group model: 97.20% accuracy, macro F1 0.9617, weighted F1 0.9716 on 1,927 test images.
- Robot deployment subset: 100% accuracy in the reported target-search evaluation.

## Deployment

The ROS2 pipeline subscribes to the RGB camera stream, converts frames with `cv_bridge`, extracts can-like regions with OpenCV contour filtering, classifies crops with MobileNetV3-Large, publishes prediction messages, and drives robot behaviour through `SEARCH`, `ALIGN`, `APPROACH`, and `ARRIVED` states.

