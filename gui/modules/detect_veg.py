"""Vegetation detection and segmentation module using color spaces and morphology."""

from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np


def rgb2yiq(im: np.ndarray) -> np.ndarray:
    """Convert an RGB image to YIQ color space."""
    yiq_matrix = np.array([
        [0.299, 0.587, 0.114],
        [0.595716, -0.274453, -0.321263],
        [0.211456, -0.522591, 0.311135],
    ], dtype=np.float32)
    return (im.astype(np.float32) / 255.0) @ yiq_matrix.T


def create_green_mask(image: np.ndarray, min_veg_px: int = 1000, Min_veg_px: int | None = None) -> np.ndarray | None:
    """Generate binary green vegetation mask using LAB and YIQ color spaces."""
    min_px = Min_veg_px if Min_veg_px is not None else min_veg_px
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    mask = cv2.inRange(lab[..., 1], 0, 120)

    if np.sum(mask == 255) <= min_px:
        return None

    yiq = rgb2yiq(image)
    q = cv2.normalize(src=yiq[:, :, 2], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    mask = cv2.inRange(q, 0, 120)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((10, 10), np.uint8))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((20, 20), np.uint8))


def get_bbox(img_mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Extract bounding boxes (x, y, w, h) for detected contour blobs."""
    blobs, _ = cv2.findContours(img_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(blob) for blob in blobs]


def detect_veg(image: np.ndarray, min_veg_px: int = 1000, Min_veg_px: int | None = None) -> list[tuple[int, int, int, int]]:
    """Detect vegetation regions, annotate bounding boxes, and return sorted bboxes."""
    min_px = Min_veg_px if Min_veg_px is not None else min_veg_px
    img_mask = create_green_mask(image, min_px)
    if img_mask is None:
        return []

    cv2.imwrite("img_mask.jpg", img_mask)
    bboxes = get_bbox(img_mask)

    for x, y, w, h in bboxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imwrite("img_bbox.jpg", image)

    return sorted(bboxes, reverse=True, key=lambda b: b[1] + b[3])


if __name__ == "__main__":
    sample_path = Path(__file__).resolve().parent / "test2.jpg"
    if sample_path.exists():
        img = cv2.imread(str(sample_path))
        print("Detected bboxes:", detect_veg(img, min_veg_px=1000))