# ============================================================
# utils/inference_utils.py
#
# Detection pipeline:
#   Step 1 — YOLOv8n finds ALL objects in the frame
#   Step 2 — Filter to only keep can-like categories:
#             bottle, cup, wine glass, vase
#             (these are the COCO classes closest to drink cans)
#   Step 3 — Crop each detected region (tight box)
#   Step 4 — Run YOUR MobileNetV3 classifier on each crop
#   Step 5 — Return label + confidence + tight bounding box
#
# Result: tight boxes ONLY around cans, nothing else.
# ============================================================

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage
from torchvision import transforms

# YOLO is imported lazily so the file still loads even if
# ultralytics is not yet installed
_yolo_model = None

# COCO class names that correspond to can-like objects
# YOLOv8 pretrained on COCO — these are the relevant classes
CAN_YOLO_CLASSES = {
    'bottle',
    'cup',
    'wine glass',
    'vase',
}


def _get_yolo():
    """Load YOLOv8n once and cache it."""
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            # yolov8n = nano — fastest, runs well on CPU
            # Downloads ~6MB model on first run automatically
            _yolo_model = YOLO('yolov8n.pt')
            print('  YOLOv8n loaded.')
        except ImportError:
            print('  ERROR: ultralytics not installed.')
            print('  Run: pip install ultralytics')
            raise
    return _yolo_model


def get_inference_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ------------------------------------------------------------------
# Detect cans using YOLOv8 — returns tight (x,y,w,h) boxes
# ------------------------------------------------------------------

def detect_rois(frame_bgr,
                conf_yolo=0.25,
                # legacy params kept so old callers don't break
                blur_kernel=7, canny_low=30, canny_high=100,
                dilate_iters=3, min_area=3000,
                max_area_ratio=0.50, padding=8):
    """
    Use YOLOv8 to detect can-like objects in the frame.
    Returns list of (x, y, w, h) tight bounding boxes.
    """
    yolo = _get_yolo()
    h, w = frame_bgr.shape[:2]

    # Run YOLO inference — verbose=False suppresses console spam
    results = yolo(frame_bgr, verbose=False, conf=conf_yolo)[0]

    rois = []
    for box in results.boxes:
        cls_id   = int(box.cls[0])
        cls_name = results.names[cls_id]

        # Only keep can-like classes
        if cls_name not in CAN_YOLO_CLASSES:
            continue

        # Get tight bounding box in pixel coords
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1 = max(0, int(x1) - padding)
        y1 = max(0, int(y1) - padding)
        x2 = min(w, int(x2) + padding)
        y2 = min(h, int(y2) + padding)

        bw = x2 - x1
        bh = y2 - y1
        if bw > 0 and bh > 0:
            rois.append((x1, y1, bw, bh))

    return rois


# ------------------------------------------------------------------
# Classify a single crop with YOUR MobileNetV3 model
# ------------------------------------------------------------------

def classify_crop(crop_bgr, model, idx_to_class, transform, device):
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img  = PILImage.fromarray(crop_rgb)
    tensor   = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1)
        conf, pred_idx = probs.max(dim=1)
    return idx_to_class[pred_idx.item()], conf.item()


# ------------------------------------------------------------------
# Full pipeline: YOLO detect + classify each can crop
# ------------------------------------------------------------------

def detect_and_classify(frame_bgr, model, idx_to_class,
                         transform, device,
                         confidence_threshold=0.75,
                         # legacy params
                         blur_kernel=7, canny_low=30,
                         canny_high=100, dilate_iters=3,
                         min_area=3000, max_area_ratio=0.50,
                         padding=8):
    """
    Returns list of dicts:
      { bbox:(x,y,w,h), label, confidence, above_threshold }
    """
    rois       = detect_rois(frame_bgr, padding=padding)
    detections = []

    for (x, y, w, h) in rois:
        crop = frame_bgr[y:y+h, x:x+w]
        if crop.size == 0:
            continue
        label, conf = classify_crop(
            crop, model, idx_to_class, transform, device)
        detections.append({
            'bbox':            (x, y, w, h),
            'label':           label,
            'confidence':      conf,
            'above_threshold': conf >= confidence_threshold,
        })

    return detections


# ------------------------------------------------------------------
# Draw tight bounding boxes on frame
# ------------------------------------------------------------------

def draw_detections(frame_bgr, detections,
                    confidence_threshold=0.75):
    """
    Green box  = confidence above threshold
    Orange box = below threshold (YOLO found a can but classifier
                 is not confident — show it anyway so you can see)
    """
    out = frame_bgr.copy()
    for det in detections:
        x, y, w, h = det['bbox']
        label = det['label']
        conf  = det['confidence']
        above = det['above_threshold']

        # Green if confident, orange if not
        color = (0, 200, 0) if above else (0, 140, 255)

        # Tight rectangle around the can
        cv2.rectangle(out, (x, y), (x+w, y+h), color, 2)

        # Label background + text
        text = f"{label}  {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        thick = 1
        (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
        label_y = max(y - 6, th + 6)
        cv2.rectangle(out,
                      (x, label_y - th - bl - 4),
                      (x + tw + 6, label_y + bl - 2),
                      color, cv2.FILLED)
        cv2.putText(out, text,
                    (x + 3, label_y - bl - 1),
                    font, scale, (255, 255, 255),
                    thick, cv2.LINE_AA)
    return out


# ------------------------------------------------------------------
# Debug view — show raw YOLO detections before classification
# ------------------------------------------------------------------

def draw_debug_yolo(frame_bgr):
    """
    Show ALL objects YOLO finds, coloured by whether they are
    in the can-class filter or not.
    Blue  = detected AND in can-class list (will be classified)
    Grey  = detected but NOT a can class (ignored)
    """
    yolo    = _get_yolo()
    results = yolo(frame_bgr, verbose=False, conf=0.20)[0]
    out     = frame_bgr.copy()

    for box in results.boxes:
        cls_id   = int(box.cls[0])
        cls_name = results.names[cls_id]
        conf_val = float(box.conf[0])
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

        is_can = cls_name in CAN_YOLO_CLASSES
        color  = (255, 140, 0) if is_can else (120, 120, 120)
        label  = f"{'CAN:' if is_can else ''}{cls_name} {conf_val:.2f}"

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        (tw, th), bl = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.rectangle(out,
                      (x1, y1 - th - bl - 4),
                      (x1 + tw + 4, y1 + bl - 2),
                      color, cv2.FILLED)
        cv2.putText(out, label, (x1 + 2, y1 - bl - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(out,
                'DEBUG — blue=can class  grey=ignored',
                (10, out.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (200, 200, 200), 1, cv2.LINE_AA)
    return out