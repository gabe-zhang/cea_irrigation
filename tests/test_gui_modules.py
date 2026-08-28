"""Unit tests for arducam Camera wrapper and detect_veg module."""

from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest

from gui.modules.arducam import Camera
from gui.modules.detect_veg import create_green_mask, detect_veg, get_bbox, rgb2yiq


# --- Arducam Camera Wrapper Tests ---

def test_camera_picamera2_missing():
    with patch("gui.modules.arducam.Picamera2", None):
        cam = Camera(width=640, height=480)
        assert not cam.is_available
        assert cam.capture_array() is None
        cam.stop()


def test_camera_lifecycle_mocked():
    mock_picam = MagicMock()
    mock_picam_cls = MagicMock(return_value=mock_picam)
    mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_picam.capture_array.return_value = mock_frame

    with patch("gui.modules.arducam.Picamera2", mock_picam_cls):
        with Camera(width=640, height=480) as cam:
            assert cam.is_available
            frame = cam.capture_array()
            assert frame is not None
            assert frame.shape == (480, 640, 3)

        # Context manager exit should have called stop
        mock_picam.stop.assert_called_once()
        assert not cam.is_available


def test_camera_capture_error():
    mock_picam = MagicMock()
    mock_picam_cls = MagicMock(return_value=mock_picam)
    mock_picam.capture_array.side_effect = RuntimeError("Hardware failure")

    with patch("gui.modules.arducam.Picamera2", mock_picam_cls):
        cam = Camera(width=640, height=480)
        assert cam.is_available
        frame = cam.capture_array()
        assert frame is None
        cam.stop()


# --- Detect Veg Module Tests ---

def test_rgb2yiq():
    # Test RGB image to YIQ color space matrix multiplication
    img = np.full((10, 10, 3), 128, dtype=np.uint8)
    yiq = rgb2yiq(img)
    assert yiq.shape == (10, 10, 3)
    assert yiq.dtype == np.float32 or yiq.dtype == np.float64


def test_create_green_mask_and_detect_veg(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    # Synthetic image (black background with a vibrant green box)
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    # BGR format: Green is (0, 255, 0)
    cv2.rectangle(img, (50, 50), (150, 150), (0, 255, 0), -1)

    mask = create_green_mask(img, Min_veg_px=50)
    assert mask is not None
    assert mask.shape == (200, 200)
    assert np.sum(mask == 255) > 50

    bboxes = get_bbox(mask)
    assert len(bboxes) >= 1
    x, y, w, h = bboxes[0]
    assert w > 0 and h > 0

    # Test complete detect_veg pipeline
    result_bboxes = detect_veg(img.copy(), Min_veg_px=50)
    assert len(result_bboxes) >= 1
    assert (tmp_path / "img_mask.jpg").exists()
    assert (tmp_path / "img_bbox.jpg").exists()


def test_detect_veg_no_vegetation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    # Pure black image - no vegetation
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = create_green_mask(img, Min_veg_px=100)
    assert mask is None

    result_bboxes = detect_veg(img, Min_veg_px=100)
    assert result_bboxes == []
