#!/usr/bin/env python3
# ============================================================
# camera_classifier_action_node.py
# DEMO 2 — SCAN: robot rotates, classifies all cans.
# DEMO 3 — TARGET: robot finds target can, moves forward.
# Uses OpenCV contour detection — NO ultralytics/YOLO needed.
# ============================================================

import sys, os, json, argparse, time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import cv2
import torch
import torch.nn.functional as F
from cv_bridge import CvBridge
from PIL import Image as PILImage
from torchvision import transforms, models
import torch.nn as nn
import numpy as np

DEPLOYMENT_CLASSES = ['Cocacola_classic', 'Sprite', 'Redbull_Classic']
STATE_SEARCH   = 'SEARCH'
STATE_ALIGN    = 'ALIGN'
STATE_APPROACH = 'APPROACH'
STATE_ARRIVED  = 'ARRIVED'

def _get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

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
        rois.append((x1, y1, x2-x1, y2-y1))
    return rois

def _detect_and_classify(frame_bgr, model, idx_to_class,
                          transform, device, conf_thresh=0.75):
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


class CameraClassifierActionNode(Node):

    def __init__(self, model_path, mode='scan',
                 target_class=None, conf_threshold=0.75,
                 consecutive_needed=5, frame_width=640):
        super().__init__('camera_classifier_action')

        self.mode               = mode
        self.target_class       = target_class
        self.conf_threshold     = conf_threshold
        self.consecutive_needed = consecutive_needed
        self.frame_center       = frame_width / 2.0

        self.search_angular_vel  = 0.3
        self.align_angular_gain  = 0.003
        self.approach_linear_vel = 0.1
        self.approach_ang_gain   = 0.001
        self.center_tolerance    = 80.0
        self.arrival_bbox_area   = 40000
        self.max_approach_time   = 8.0

        self.state               = STATE_SEARCH
        self.consecutive_count   = 0
        self.approach_start_time = None

        self.device    = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.cv_bridge = CvBridge()

        self.get_logger().info('Loading model ...')
        self.model, self.class_to_idx, self.idx_to_class = \
            _load_model(model_path, self.device)
        self.transform = _get_transform()

        self.cam_subscription = self.create_subscription(
            Image, '/depth_cam/rgb/image_raw', self.image_callback, 1)
        self.cmd_vel_publisher = self.create_publisher(
            Twist, '/cmd_vel', 10)
        self.pred_publisher = self.create_publisher(
            String, '/object_classifier/prediction', 10)

        if self.mode == 'scan':
            self.get_logger().info(
                '\n  DEMO 2 — SCAN MODE\n'
                '  Robot rotates and classifies all cans.\n'
                '  Ctrl+C to stop.')
        else:
            self.get_logger().info(
                f'\n  DEMO 3 — TARGET MODE\n'
                f'  Target: {self.target_class}\n'
                f'  Robot will search, align, move forward, stop.')

    def image_callback(self, msg):
        image_bgr  = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
        detections = _detect_and_classify(
            image_bgr, self.model, self.idx_to_class,
            self.transform, self.device, self.conf_threshold)
        annotated  = _draw(image_bgr, detections, self.conf_threshold)

        label_text = ('DEMO 2 — SCAN | rotating and classifying'
                      if self.mode == 'scan'
                      else f'DEMO 3 | Target: {self.target_class} | {self.state}')
        cv2.putText(annotated, label_text,
                    (10, annotated.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200,200,200), 1, cv2.LINE_AA)
        cv2.imshow('Phase 3 Robot Demo', annotated)
        cv2.waitKey(1)

        above = [d for d in detections if d['above_threshold']]
        best  = max(above, key=lambda d: d['confidence']) if above else None
        self._publish_prediction(best)

        if self.mode == 'scan':
            self._publish_twist(0.0, self.search_angular_vel)
        else:
            self._target_mode(detections)

    def _target_mode(self, detections):
        target_dets    = [d for d in detections
                          if d['label'] == self.target_class
                          and d['above_threshold']]
        target_visible = len(target_dets) > 0
        best           = (max(target_dets, key=lambda d: d['confidence'])
                          if target_visible else None)

        if self.state == STATE_SEARCH:
            if target_visible:
                self.consecutive_count += 1
                self.get_logger().info(
                    f'[SEARCH] {self.target_class} '
                    f'conf={best["confidence"]:.3f} '
                    f'{self.consecutive_count}/{self.consecutive_needed}')
                if self.consecutive_count >= self.consecutive_needed:
                    self.state = STATE_ALIGN
                    self.consecutive_count = 0
                    self.get_logger().info('→ ALIGN')
            else:
                self.consecutive_count = 0
            self._publish_twist(0.0, self.search_angular_vel)

        elif self.state == STATE_ALIGN:
            if not target_visible:
                self.get_logger().info('Lost → SEARCH')
                self.state = STATE_SEARCH
                self.consecutive_count = 0
                self._publish_stop()
                return
            x, y, w, h = best['bbox']
            error = (x + w/2.0) - self.frame_center
            if abs(error) <= self.center_tolerance:
                self.state = STATE_APPROACH
                self.approach_start_time = time.time()
                self.get_logger().info('Centred → APPROACH')
            else:
                az = max(-0.4, min(0.4, -self.align_angular_gain * error))
                self._publish_twist(0.0, az)

        elif self.state == STATE_APPROACH:
            elapsed = time.time() - self.approach_start_time
            if elapsed > self.max_approach_time:
                self.get_logger().info('Timeout → STOP')
                self.state = STATE_ARRIVED
                self._publish_stop()
                return
            if not target_visible:
                self.get_logger().info('Lost → SEARCH')
                self.state = STATE_SEARCH
                self.consecutive_count = 0
                self._publish_stop()
                return
            x, y, w, h = best['bbox']
            area  = w * h
            error = (x + w/2.0) - self.frame_center
            if area >= self.arrival_bbox_area:
                self.get_logger().info(f'ARRIVED near {self.target_class}')
                self.state = STATE_ARRIVED
                self._publish_stop()
                return
            az = max(-0.15, min(0.15, -self.approach_ang_gain * error))
            self.get_logger().info(
                f'[APPROACH] area={area} elapsed={elapsed:.1f}s')
            self._publish_twist(self.approach_linear_vel, az)

        elif self.state == STATE_ARRIVED:
            self._publish_stop()
            self.get_logger().info(f'DONE — stopped near {self.target_class}')

    def _publish_twist(self, linear_x=0.0, angular_z=0.0):
        msg           = Twist()
        msg.linear.x  = linear_x
        msg.linear.y  = 0.0
        msg.linear.z  = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = angular_z
        self.cmd_vel_publisher.publish(msg)

    def _publish_stop(self):
        self._publish_twist(0.0, 0.0)

    def _publish_prediction(self, best):
        payload = ({'label': best['label'],
                    'confidence': round(best['confidence'], 4),
                    'bbox': list(best['bbox'])}
                   if best else {'label': '', 'confidence': 0.0, 'bbox': []})
        msg      = String()
        msg.data = json.dumps(payload)
        self.pred_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path',   required=True)
    parser.add_argument('--class-map',    required=True)
    parser.add_argument('--mode',         default='scan',
                        choices=['scan', 'target'])
    parser.add_argument('--target-class', default=None,
                        choices=DEPLOYMENT_CLASSES)
    parser.add_argument('--conf-thresh',  type=float, default=0.75)
    parser.add_argument('--consecutive',  type=int,   default=5)
    parser.add_argument('--frame-width',  type=int,   default=640)
    parsed, _ = parser.parse_known_args()

    if parsed.mode == 'target' and parsed.target_class is None:
        print('ERROR: --target-class required when --mode target')
        print('Choose: Cocacola_classic | Sprite | Redbull_Classic')
        return

    node = CameraClassifierActionNode(
        model_path=parsed.model_path,
        mode=parsed.mode,
        target_class=parsed.target_class,
        conf_threshold=parsed.conf_thresh,
        consecutive_needed=parsed.consecutive,
        frame_width=parsed.frame_width,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node._publish_stop()
        node.get_logger().info('Stopped.')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()