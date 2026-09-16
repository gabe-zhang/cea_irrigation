#!/usr/bin/env python3
"""YOLO Edge TPU CLI Inference Tool for Raspberry Pi 5 + Google Coral Edge TPU."""

from __future__ import annotations
import argparse
from pathlib import Path
import time
import cv2
import numpy as np

try:
    import ai_edge_litert.interpreter as tflite
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        import tensorflow.lite as tflite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO inference on Google Coral Edge TPU")
    parser.add_argument("--model", type=str, default="gui/models/yolo26n_e100.tflite", help="Path to .tflite Edge TPU model")
    parser.add_argument("--source", type=str, required=True, help="Path to input image or directory of images")
    parser.add_argument("--conf", type=float, default=0.15, help="Confidence threshold (default: 0.15)")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold (default: 0.45)")
    parser.add_argument("--out-dir", type=str, default="runs/edgetpu_detect", help="Output directory for annotated images")
    parser.add_argument("--save", action="store_true", default=True, help="Save annotated detection images")
    return parser.parse_args()


class YoloEdgeTPUDetector:
    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        try:
            delegate = tflite.load_delegate("libedgetpu.so.1")
            self.interpreter = tflite.Interpreter(
                model_path=str(self.model_path),
                experimental_delegates=[delegate],
            )
        except Exception as e:
            print(f"[EdgeTPU] Warning: Failed to load libedgetpu.so.1 ({e}), attempting standard interpreter...")
            self.interpreter = tflite.Interpreter(model_path=str(self.model_path))

        self.interpreter.allocate_tensors()
        self.in_det = self.interpreter.get_input_details()[0]
        self.out_det = self.interpreter.get_output_details()[0]

        # Model input shape (typically [1, 640, 640, 3])
        self.in_h = self.in_det["shape"][1]
        self.in_w = self.in_det["shape"][2]
        self.in_scale, self.in_zp = self.in_det["quantization"]
        self.out_scale, self.out_zp = self.out_det["quantization"]
        self.is_int8 = self.in_det["dtype"] == np.int8

    def predict(self, image_bgr: np.ndarray, conf_thresh: float = 0.15, iou_thresh: float = 0.45):
        orig_h, orig_w = image_bgr.shape[:2]

        # Resize & color convert
        resized = cv2.resize(image_bgr, (self.in_w, self.in_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Quantize input
        if self.is_int8:
            input_tensor = ((rgb / 255.0) / self.in_scale + self.in_zp).astype(np.int8)
        else:
            input_tensor = (rgb / 255.0).astype(np.float32)
        input_tensor = np.expand_dims(input_tensor, axis=0)

        self.interpreter.set_tensor(self.in_det["index"], input_tensor)

        t0 = time.perf_counter()
        self.interpreter.invoke()
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        # Parse raw output
        raw_out = self.interpreter.get_tensor(self.out_det["index"])[0]
        if self.out_scale > 0:
            out_float = (raw_out.astype(np.float32) - self.out_zp) * self.out_scale
        else:
            out_float = raw_out.astype(np.float32)

        # Transpose if output is [num_channels, num_anchors]
        if out_float.shape[0] < out_float.shape[1]:
            out_float = out_float.T

        # Format: [cx, cy, w, h, score] or [cx, cy, w, h, class_scores...]
        cx = out_float[:, 0]
        cy = out_float[:, 1]
        w = out_float[:, 2]
        h = out_float[:, 3]

        if out_float.shape[1] == 5:
            scores = out_float[:, 4]
            class_ids = np.zeros_like(scores, dtype=int)
        else:
            class_scores = out_float[:, 4:]
            class_ids = np.argmax(class_scores, axis=1)
            scores = np.max(class_scores, axis=1)

        mask = scores >= conf_thresh
        cx, cy, w, h, scores, class_ids = cx[mask], cy[mask], w[mask], h[mask], scores[mask], class_ids[mask]

        # Convert normalized coordinates to pixel coords
        boxes = []
        for c_x, c_y, b_w, b_h in zip(cx, cy, w, h):
            x1 = int((c_x - b_w / 2.0) * orig_w)
            y1 = int((c_y - b_h / 2.0) * orig_h)
            w_px = int(b_w * orig_w)
            h_px = int(b_h * orig_h)
            boxes.append([x1, y1, w_px, h_px])

        indices = cv2.dnn.NMSBoxes(boxes, scores.tolist(), conf_thresh, iou_thresh)
        final_boxes = []
        if len(indices) > 0:
            for i in indices.flatten():
                final_boxes.append({
                    "bbox": tuple(boxes[i]),
                    "score": float(scores[i]),
                    "class_id": int(class_ids[i]),
                })

        return final_boxes, latency_ms


def main():
    args = parse_args()
    detector = YoloEdgeTPUDetector(args.model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src_path = Path(args.source)
    if src_path.is_file():
        image_files = [src_path]
    elif src_path.is_dir():
        image_files = sorted(list(src_path.glob("*.jpg")) + list(src_path.glob("*.png")))
    else:
        raise FileNotFoundError(f"Source not found: {src_path}")

    print(f"\n🚀 Running YOLO Edge TPU Inference with {args.model}")
    print(f"📁 Processing {len(image_files)} image(s)...")

    for img_p in image_files:
        img = cv2.imread(str(img_p))
        if img is None:
            print(f"⚠️ Failed to read {img_p}")
            continue

        boxes, latency = detector.predict(img, conf_thresh=args.conf, iou_thresh=args.iou)
        print(f"\n📷 {img_p.name} -> {len(boxes)} plants detected in {latency:.2f} ms ({1000.0/latency:.1f} FPS)")

        vis = img.copy()
        for idx, item in enumerate(boxes):
            bx, by, bw, bh = item["bbox"]
            score = item["score"]
            print(f"   Plant #{idx+1}: BBox=({bx}, {by}, {bw}, {bh}), Confidence={score:.2f}")

            cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
            label = f"Plant {score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(vis, (bx, max(0, by - th - 6)), (bx + tw + 6, max(th + 6, by)), (0, 255, 0), -1)
            cv2.putText(vis, label, (bx + 3, max(th, by - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        if args.save:
            out_file = out_dir / f"pred_{img_p.name}"
            cv2.imwrite(str(out_file), vis)
            print(f"   💾 Saved visualization to: {out_file}")


if __name__ == "__main__":
    main()
