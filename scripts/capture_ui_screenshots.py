"""Script to launch CEA Irrigation GUI with realistic mock telemetry and capture UI screenshots."""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys
import time
import tkinter as tk
from unittest.mock import MagicMock, patch

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import cv2
import numpy as np
from PIL import Image

import gui.psc_irr_gui as psc_mod
from gui.psc_irr_gui import MainWindow, TELEMETRY_DIR

ARTIFACT_DIR = Path("/home/pi50/.gemini/antigravity-ide/brain/fa3253c6-77af-4176-a3e6-eedefd18f949")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_IMG = Path("tests/samples/20260909_172410_497.jpg")


def capture_window_crop(win: tk.Tk | tk.Toplevel, out_path: Path) -> None:
    """Capture screen via grim and crop precisely to window geometry."""
    win.deiconify()
    win.lift()
    win.update()
    win.update_idletasks()
    time.sleep(0.8)

    env = {
        **os.environ,
        "DISPLAY": ":0",
        "WAYLAND_DISPLAY": os.environ.get("WAYLAND_DISPLAY", "wayland-0"),
        "XDG_RUNTIME_DIR": os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"),
    }

    x = max(0, win.winfo_rootx())
    y = max(0, win.winfo_rooty())
    w = win.winfo_width()
    h = win.winfo_height()

    geom = f"{x},{y} {w}x{h}"
    subprocess.run(["grim", "-g", geom, str(out_path)], env=env, check=True)
    print(f"[Screenshot] Captured {out_path.name} via grim ({geom})")


def main():
    print("[Screenshot Harness] Starting GUI in controlled screenshot mode...")

    with patch("gui.psc_irr_gui.Camera") as mock_cam_cls:
        mock_cam = mock_cam_cls.return_value
        mock_cam.is_available = True
        
        # Load sample image as camera feed
        sample_bgr = cv2.imread(str(SAMPLE_IMG))
        if sample_bgr is None:
            sample_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
            sample_bgr[:, :] = (35, 180, 45)
        mock_cam.capture_array.return_value = sample_bgr

        with patch.object(MainWindow, "_init_serial"):
            with patch.object(MainWindow, "_start_periodic_loggers"):
                with patch.object(MainWindow, "_start_scheduled_ticker"):
                    app = MainWindow()
                    app.geometry(f"{app.scr_w}x{app.scr_h}+0+0")
                    app.update()

                    # Set realistic live telemetry
                    mock_data = {
                        "soil": [420, 395, 430, 410],
                        "soil_temp": 22.4,
                        "temp": 24.1,
                        "humidity": 58.2,
                        "light": 1240,
                        "relays": "0000",
                        "pan": 55,
                        "tilt": 30,
                    }
                    moist = [42.1, 38.5, 51.0, 44.2]
                    app.telemetry = {**mock_data, "moisture_pct": moist}
                    app._update_telemetry_ui(mock_data, moist)
                    app.display_image(sample_bgr)
                    app.update()

                    # 1. Capture Main Window
                    main_screenshot = ARTIFACT_DIR / "gui_main_window.png"
                    capture_window_crop(app, main_screenshot)

                    # 2. Open Plot Window in "min" mode
                    app.plot_var.set(1)
                    app.plot_range_var.set("min")
                    app.plotter.toggle(True)
                    app.update()
                    time.sleep(0.5)

                    # Feed some points to live plot
                    for sec in range(1, 15):
                        t_data = (sec * 2.0, [42.1 + sec*0.1, 38.5 - sec*0.05, 51.0, 44.2], 22.4, 24.1, 58.2)
                        app.plotter._update(t_data)
                    app.plotter.canvas_widget.draw()
                    app.update()

                    plot_min_screenshot = ARTIFACT_DIR / "gui_plot_min.png"
                    capture_window_crop(app.plotter.window, plot_min_screenshot)

                    # 3. Switch to "day" mode
                    app.plot_range_var.set("day")
                    app.update()
                    time.sleep(0.5)
                    plot_day_screenshot = ARTIFACT_DIR / "gui_plot_day.png"
                    capture_window_crop(app.plotter.window, plot_day_screenshot)

                    # 4. Switch to "week" mode
                    app.plot_range_var.set("week")
                    app.update()
                    time.sleep(0.5)
                    plot_week_screenshot = ARTIFACT_DIR / "gui_plot_week.png"
                    capture_window_crop(app.plotter.window, plot_week_screenshot)

                    # 5. Switch to "month" mode
                    app.plot_range_var.set("month")
                    app.update()
                    time.sleep(0.5)
                    plot_month_screenshot = ARTIFACT_DIR / "gui_plot_month.png"
                    capture_window_crop(app.plotter.window, plot_month_screenshot)

                    # Clean shutdown
                    app.plotter.toggle(False)
                    app.destroy()
                    print("[Screenshot Harness] Complete! All screenshots captured.")


if __name__ == "__main__":
    main()
