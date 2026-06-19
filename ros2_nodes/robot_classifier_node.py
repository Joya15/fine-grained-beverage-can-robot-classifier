#!/usr/bin/env python3
# ============================================================
# robot_classifier_node.py
# DEMO 1 — Camera only. Robot does NOT move.
# Uses OpenCV contour detection — NO ultralytics/YOLO needed.
# ============================================================

import sys, os, json, argparse
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2
import torch
import torch.nn.functional as F
from cv_bridge import CvBridge
from PIL import Image as PILImage
from torchvision import transforms, models
import torch.nn as nn
import numpy as np

# ------------------------------------------------------------------
# Transform
# ------------------------------------------------------------------
def _get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

# ------------------------------------------------------------------
# Load MobileNetV3 checkpoint
# ------------------------------------------------------------------
def _load_model(model_path, device):
    ckpt         = torch.load(model_path, map_location=device)
    num_classes  = ckpt.get('num_classes')
    class_to_idx = ckpt.get('class_to_idx', {})
    idx_to_class = {int(v): k for k, v in class_to_idx.items()}
    model = models.mobilenet_v3_large(weights=None)
    in_f  = model.classifier[3].in_features
    model.classifier[3] = nn.Sequential(
        nn.Dropout(p=0.3), nn.Linear(in_f, num_classes))
    model.load_state_dict(ckpt['state_dict'])
    model.to(device).eval()
    return model, class_to_idx, idx_to_class

# ------------------------------------------------------------------
# Contour-based ROI detection — no YOLO needed
# ------------------------------------------------------------------
def _detect_rois(frame_bgr, min_area=3000, max_area_ratio=0.60, padding=10):
    h, w = frame_bgr.shape[:2]
    gray    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges   = cv2.Canny(blurred, 40, 120)
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rois = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > h * w * max_area_ratio:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bh == 0:
            continue
        ratio = bw / bh
        if ratio > 1.2 or ratio < 0.15:
            continue
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w, x + bw + padding)
        y2 = min(h, y + bh + padding)
        rois.append((x1, y1, x2 - x1, y2 - y1))
    return rois

# ------------------------------------------------------------------
# Detect + classify
# ------------------------------------------------------------------
def _detect_and_classify(frame_bgr, model, idx_to_class,
                          transform, device, conf_thresh=0.70):
    rois = _detect_rois(frame_bgr)
    detections = []
    for (x, y, w, h) in rois:
        crop = frame_bgr[y:y+h, x:x+w]
        if crop.size == 0:
            continue
        rgb    = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = transform(PILImage.fromarray(rgb)).unsqueeze(0).to(device)
        with torch.no_grad():
            probs      = F.softmax(model(tensor), dim=1)
            conf, pred = probs.max(1)
        label = idx_to_class[pred.item()]
        c     = conf.item()
        detections.append({
            'bbox':            (x, y, w, h),
            'label':           label,
            'confidence':      c,
            'above_threshold': c >= conf_thresh,
        })
    return detections

# ------------------------------------------------------------------
# Draw
# ------------------------------------------------------------------
def _draw(frame, detections, conf_thresh):
    out = frame.copy()
    for d in detections:
        x, y, w, h = d['bbox']
        color = (0, 200, 0) if d['above_threshold'] else (0, 140, 255)
        cv2.rectangle(out, (x, y), (x+w, y+h), color, 2)
        text = f"{d['label']}  {d['confidence']:.2f}"
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ly = max(y-6, th+6)
        cv2.rectangle(out, (x, ly-th-bl-4), (x+tw+6, ly+bl-2), color, cv2.FILLED)
        cv2.putText(out, text, (x+3, ly-bl-1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
    return out

# ------------------------------------------------------------------
# ROS2 Node
# ------------------------------------------------------------------
class RobotClassifierNode(Node):

    def __init__(self, model_path, conf_threshold=0.70):
        super().__init__('robot_classifier')
        self.cv_bridge      = CvBridge()
        self.conf_threshold = conf_threshold
        self.device         = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info('Loading model ...')
        self.model, self.class_to_idx, self.idx_to_class = \
            _load_model(model_path, self.device)
        self.transform = _get_transform()
        self.cam_subscription = self.create_subscription(
            Image, '/depth_cam/rgb/image_raw', self.image_callback, 1)
        self.pred_publisher = self.create_publisher(
            String, '/object_classifier/prediction', 10)
        self.get_logger().info(
            f'Ready | {len(self.class_to_idx)} classes | No movement.')

    def image_callback(self, msg):
        image_bgr  = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
        detections = _detect_and_classify(
            image_bgr, self.model, self.idx_to_class,
            self.transform, self.device, self.conf_threshold)
        annotated = _draw(image_bgr, detections, self.conf_threshold)
        cv2.putText(annotated, 'DEMO 1 — Camera only | Robot stationary',
                    (10, annotated.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200,200,200), 1, cv2.LINE_AA)
        cv2.imshow('Demo 1', annotated)
        cv2.waitKey(1)
        above   = [d for d in detections if d['above_threshold']]
        best    = max(above, key=lambda d: d['confidence']) if above else None
        payload = ({'label': best['label'],
                    'confidence': round(best['confidence'], 4),
                    'bbox': list(best['bbox'])}
                   if best else {'label': '', 'confidence': 0.0, 'bbox': []})
        msg_out      = String()
        msg_out.data = json.dumps(payload)
        self.pred_publisher.publish(msg_out)
        if best:
            self.get_logger().info(f"{best['label']} conf={best['confidence']:.3f}")


def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path',  required=True)
    parser.add_argument('--class-map',   required=True)
    parser.add_argument('--conf-thresh', type=float, default=0.70)
    parsed, _ = parser.parse_known_args()
    node = RobotClassifierNode(parsed.model_path, parsed.conf_thresh)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()