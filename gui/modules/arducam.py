"""Picamera2 camera wrapper for Raspberry Pi."""

from __future__ import annotations
import numpy as np

try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None


class Camera:
    """Lightweight, context-managed camera interface wrapping Picamera2."""

    def __init__(self, width: int = 1536, height: int = 864) -> None:
        self.picam2: Picamera2 | None = None
        if Picamera2 is None:
            print("[Camera] Picamera2 is not installed.")
            return

        try:
            self.picam2 = Picamera2()
            self.picam2.preview_configuration.main.size = (int(width), int(height))
            self.picam2.preview_configuration.main.format = "RGB888"
            self.picam2.preview_configuration.align()
            self.picam2.configure("preview")
            self.picam2.start()
        except Exception as e:
            print(f"[Camera] Camera not available: {e}")
            self.picam2 = None

    @property
    def is_available(self) -> bool:
        return self.picam2 is not None

    def capture_array(self) -> np.ndarray | None:
        """Capture and return the current frame as a numpy array."""
        try:
            return self.picam2.capture_array() if self.picam2 else None
        except Exception as e:
            print(f"[Camera] Capture error: {e}")
            return None

    def stop(self) -> None:
        """Release camera hardware."""
        if self.picam2:
            try:
                self.picam2.stop()
            except Exception:
                pass
            finally:
                self.picam2 = None

    def __enter__(self) -> Camera:
        return self

    def __exit__(self, *args) -> None:
        self.stop()


if __name__ == "__main__":
    import signal
    import sys
    from PyQt5.QtCore import QTimer
    from PyQt5.QtGui import QImage, QPixmap
    from PyQt5.QtWidgets import QApplication, QLabel

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    w, h = 1536, 864
    app = QApplication(sys.argv)
    label = QLabel()
    label.setWindowTitle("Camera Live Stream")
    label.resize(w, h)
    label.setScaledContents(True)

    with Camera(width=w, height=h) as camera:
        if not camera.is_available:
            print("[Camera] Offline or not detected.")
            sys.exit(1)

        def update() -> None:
            frame = camera.capture_array()
            if frame is not None:
                label.setPixmap(QPixmap.fromImage(QImage(frame.data, w, h, w * 3, QImage.Format_RGB888)))

        timer = QTimer(timeout=update)
        timer.start(30)
        label.show()
        sys.exit(app.exec_())
