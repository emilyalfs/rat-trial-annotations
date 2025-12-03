#/usr/bin/env python3
"""
Streamed inference with temporal aggregation (no intermediate frames on disk).
Outputs:
  - Per model size: raw JSON (all probs) + CSV (top label per frame)
  - Per (method, window, size): aggregated JSON + CSV
"""

import cv2
import csv
import sys

import numpy as np
import torch
from ultralytics import YOLO


def load_model(device: str):
    model = YOLO("./models/behavior-model.pt")
    # move to device if possible (YOLO supports .to)
    if device.startswith('cuda') and torch.cuda.is_available():
        model.to(device)
    return model


@torch.inference_mode()
def stream_inference(video_path: str,  device: str, batch_size: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")

    # Load once
    model = load_model(device)

    # pull class names from model metadata via a dummy call OR from model.model.names (cls task)
    # safer: run one dummy on a tiny black frame to acquire names from Results
    dummy = np.zeros((8, 8, 3), dtype=np.uint8)
    res = model([dummy], verbose=False)[0]
    if isinstance(res.names, dict):
        names = [res.names[k] for k in sorted(res.names)]
    else:
        names = list(res.names)

    raw = {}
    batch_frames, batch_idxs = [], []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        batch_frames.append(frame)
        batch_idxs.append(frame_idx)

        frame_idx += 1

        # Process full batches
        if len(batch_frames) == batch_size:
            results = model(batch_frames, verbose=False)

            for i, res in enumerate(results):
                idx = batch_idxs[i]
                probs = np.asarray(res.probs.data.cpu().numpy(), dtype=float)
                top = int(res.probs.top1)
                conf = float(res.probs.top1conf)
                raw[str(idx)] = {
                    'frame_id': str(idx),
                    'pred_class': top,
                    'pred_class_name': names[top] if 0 <= top < len(names) else str(top),
                    'confidence': conf,
                    'probs': probs.tolist()
                }

            batch_frames.clear()
            batch_idxs.clear()

    # Flush leftovers
    if batch_frames:
        results = model(batch_frames, verbose=False)

        for i, res in enumerate(results):
            idx = batch_idxs[i]
            probs = np.asarray(res.probs.data.cpu().numpy(), dtype=float)
            top = int(res.probs.top1)
            conf = float(res.probs.top1conf)
            raw[str(idx)] = {
                'frame_id': str(idx),
                'pred_class': top,
                'pred_class_name': names[top] if 0 <= top < len(names) else str(top),
                'confidence': conf,
                'probs': probs.tolist()
            }

    cap.release()
    return raw


# ------------------------- Save -------------------------

def save_results(raw_results, short_v_name, proj):
    print(f"Behavior model processed {len(raw_results)} frames")
    
    # Raw CSV (top-1 label per frame)
    with open(f"./{proj}/results/{short_v_name}_2_model.csv", 'w', newline='') as cf:
        w = csv.writer(cf)
        w.writerow(['frame_id', 'label'])
        for fid in sorted(raw_results, key=lambda x: int(x)):
            w.writerow([fid, raw_results[fid]['pred_class_name'].lower()])


# ------------------------- CLI -------------------------

def main():
    

    # device fallback
    dev = sys.argv[2]
    if dev.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU.")
        dev = 'cpu'
    video = sys.argv[1]
    proj = sys.argv[3]
    raw = stream_inference(video, dev, 64)
    short_name = ((video).split("/")[-1])[:-4]
    save_results(raw, short_name, proj)


if __name__ == '__main__':
    main()# generate behaviors (standing, object interaction, none)
