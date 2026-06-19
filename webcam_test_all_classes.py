#!/usr/bin/env python3
# ============================================================
# webcam_test_all_classes.py
#
# How it works:
#   1. YOLOv8n detects cans (bottle/cup class) in the frame
#   2. Draws a TIGHT box exactly around each can
#   3. Crops each can region
#   4. Runs YOUR trained classifier on each crop
#   5. Shows the drink brand label + confidence on the box
#
# Only cans are detected. Faces, bodies, furniture = ignored.
#
# First run: YOLOv8n downloads ~6MB model automatically.
#
# Usage:
#   python webcam_test_all_classes.py \
#     --model-path outputs/models/all_class_model.pth \
#     --class-map  outputs/models/all_class_class_to_idx.json
#
# Keys:
#   q  — quit
#   s  — save current frame
#   d  — debug mode (shows ALL yolo detections)
#   +  — raise classifier confidence threshold
#   -  — lower classifier confidence threshold
# ============================================================

import os, sys, argparse, time, yaml
import cv2, torch

sys.path.insert(0, os.path.dirname(__file__))
from utils.model_utils     import load_checkpoint, get_device
from utils.inference_utils import (get_inference_transform,
                                   detect_and_classify,
                                   draw_detections,
                                   draw_debug_yolo)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model-path',  default='outputs/models/all_class_model.pth')
    p.add_argument('--class-map',   default='outputs/models/all_class_class_to_idx.json')
    p.add_argument('--camera-id',   type=int,   default=0)
    p.add_argument('--conf-thresh', type=float, default=0.60,
                   help='Classifier confidence threshold (0.0-1.0)')
    p.add_argument('--yolo-conf',   type=float, default=0.25,
                   help='YOLO detection confidence (lower = find more cans)')
    p.add_argument('--width',       type=int,   default=640)
    p.add_argument('--height',      type=int,   default=480)
    p.add_argument('--config',      default='config.yaml')
    return p.parse_args()


def open_camera(camera_id, width, height):
    """Try to open the MacBook camera — handles Continuity Camera."""
    # Try AVFoundation backend first (macOS native)
    for idx in [camera_id, 0, 1, 2]:
        for backend in [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                print(f'  Camera {idx} opened.')
                return cap
            cap.release()
    return None


def main():
    args = parse_args()
    cfg  = yaml.safe_load(open(args.config)) \
           if os.path.exists(args.config) else {}
    os.makedirs('outputs/predictions', exist_ok=True)

    # ---- Load YOUR classifier ----
    device = get_device()
    print(f'\nLoading classifier: {args.model_path}')
    model, class_to_idx, idx_to_class = load_checkpoint(
        args.model_path, device=str(device))
    num_classes = len(class_to_idx)
    print(f'  {num_classes} classes loaded.')

    # ---- Load YOLO (triggers download on first run) ----
    print('\nLoading YOLOv8n detector ...')
    from utils.inference_utils import _get_yolo
    _get_yolo()

    transform   = get_inference_transform(cfg.get('image_size', 224))
    conf_thresh = args.conf_thresh
    yolo_conf   = args.yolo_conf
    debug_mode  = False

    # ---- Open camera ----
    print(f'\nOpening camera ...')
    cap = open_camera(args.camera_id, args.width, args.height)
    if cap is None:
        print('ERROR: No camera found.')
        print('  Disconnect iPhone or disable Continuity Camera.')
        print('  System Settings → General → AirPlay & Handoff → Continuity Camera OFF')
        sys.exit(1)

    print('\n--- Webcam running ---')
    print('  Point camera at any drink can')
    print('  YOLO finds the can → your model names the brand')
    print('  q=quit  s=save  d=debug  +=conf up  -=conf down\n')

    frame_count = 0
    fps_timer   = time.time()
    fps_display = 0.0
    last_frame  = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print('Camera read failed.')
            break

        frame_count += 1

        # ---- Debug mode — see raw YOLO output ----
        if debug_mode:
            display = draw_debug_yolo(frame)
            cv2.putText(display,
                        'd=exit debug  q=quit',
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 1)

        # ---- Normal mode — YOLO detect + classify ----
        else:
            detections = detect_and_classify(
                frame, model, idx_to_class,
                transform, device,
                confidence_threshold=conf_thresh,
                padding=8)

            display    = draw_detections(frame, detections, conf_thresh)
            last_frame = display.copy()

            # FPS every 30 frames
            if frame_count % 30 == 0:
                elapsed     = time.time() - fps_timer
                fps_display = 30.0 / elapsed if elapsed > 0 else 0.0
                fps_timer   = time.time()

            above = [d for d in detections if d['above_threshold']]
            total = len(detections)

            # HUD bar at top
            cv2.rectangle(display, (0, 0),
                          (display.shape[1], 48),
                          (20, 20, 20), cv2.FILLED)
            hud = (f'FPS:{fps_display:.1f}  '
                   f'Model:{num_classes}cls  '
                   f'Cans found:{total}  '
                   f'Confident:{len(above)}  '
                   f'Thresh:{conf_thresh:.2f}  '
                   f'+/-  d=debug  s=save  q=quit')
            cv2.putText(display, hud, (8, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                        (200, 200, 200), 1, cv2.LINE_AA)

            # If nothing found — show hint
            if total == 0:
                cv2.putText(
                    display,
                    'No can detected — point at a drink can',
                    (display.shape[1]//2 - 180,
                     display.shape[0]//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (80, 80, 80), 1, cv2.LINE_AA)

        cv2.imshow('Phase 3 — Can Detector (YOLO + MobileNetV3)',
                   display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print('Quit.')
            break

        elif key == ord('s'):
            if last_frame is not None:
                ts   = int(time.time())
                path = f'outputs/predictions/webcam_{ts}.jpg'
                cv2.imwrite(path, last_frame)
                print(f'  Saved → {path}')

        elif key in (ord('+'), ord('=')):
            conf_thresh = min(0.99, round(conf_thresh + 0.05, 2))
            print(f'  Classifier threshold → {conf_thresh}')

        elif key == ord('-'):
            conf_thresh = max(0.05, round(conf_thresh - 0.05, 2))
            print(f'  Classifier threshold → {conf_thresh}')

        elif key == ord('d'):
            debug_mode = not debug_mode
            print(f'  Debug mode: {"ON — showing raw YOLO" if debug_mode else "OFF"}')

    cap.release()
    cv2.destroyAllWindows()
    print('Done.')


if __name__ == '__main__':
    main()