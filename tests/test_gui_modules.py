"""Unit tests for the arducam Camera wrapper."""

from unittest.mock import MagicMock, patch
import numpy as np

from gui.modules.arducam import Camera


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


