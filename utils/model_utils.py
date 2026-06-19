# ============================================================
# utils/model_utils.py
# MobileNetV3-Large build, save, load, freeze helpers.
# ============================================================

import os
import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes, arch='mobilenet_v3_large',
                pretrained=True, dropout=0.3):
    """Build model with replaced classifier head."""
    if arch == 'mobilenet_v3_large':
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        model   = models.mobilenet_v3_large(weights=weights)
        in_f    = model.classifier[3].in_features
        model.classifier[3] = nn.Sequential(
            nn.Dropout(p=dropout), nn.Linear(in_f, num_classes))

    elif arch == 'efficientnet_b0':
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model   = models.efficientnet_b0(weights=weights)
        in_f    = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=dropout), nn.Linear(in_f, num_classes))

    elif arch == 'resnet18':
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model   = models.resnet18(weights=weights)
        in_f    = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout), nn.Linear(in_f, num_classes))
    else:
        raise ValueError(f"Unknown arch: {arch}")
    return model


def freeze_backbone(model, arch='mobilenet_v3_large'):
    for name, param in model.named_parameters():
        head = 'classifier' if arch != 'resnet18' else 'fc'
        if head not in name:
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Backbone frozen. Trainable params: {trainable:,}")


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  All layers unfrozen. Trainable params: {trainable:,}")


def save_checkpoint(model, class_to_idx, save_path,
                    arch='mobilenet_v3_large', extra_info=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    payload = {
        'arch':        arch,
        'num_classes': len(class_to_idx),
        'class_to_idx': class_to_idx,
        'state_dict':  model.state_dict(),
    }
    if extra_info:
        payload.update(extra_info)
    torch.save(payload, save_path)
    print(f"  Checkpoint saved → {save_path}")


def load_checkpoint(model_path, num_classes=None,
                    arch=None, device='cpu'):
    """Load checkpoint. Returns model, class_to_idx, idx_to_class."""
    ckpt         = torch.load(model_path, map_location=device)
    _arch        = arch or ckpt.get('arch', 'mobilenet_v3_large')
    _num_classes = num_classes or ckpt.get('num_classes')
    class_to_idx = ckpt.get('class_to_idx', {})
    idx_to_class = {int(v): k for k, v in class_to_idx.items()}
    model = build_model(_num_classes, arch=_arch,
                        pretrained=False)
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(device)
    model.eval()
    print(f"  Loaded: {model_path} | arch={_arch} | classes={_num_classes}")
    return model, class_to_idx, idx_to_class


def get_device():
    if torch.cuda.is_available():
        d = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        d = torch.device('mps')
    else:
        d = torch.device('cpu')
    print(f"  Device: {d}")
    return d
