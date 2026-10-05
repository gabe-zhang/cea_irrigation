"""Capture the GUI on the Pi desktop with isolated telemetry and mocked hardware."""

from contextlib import ExitStack
from datetime import date, timedelta
import csv
import os
from pathlib import Path
import random
import subprocess
import tempfile
import time
import tkinter as tk
from unittest.mock import patch

import cv2
import numpy as np

import gui.psc_irr_gui as gui
from tests.mock_telemetry import generate_mock_data


def capture_window_crop(widget: tk.Misc, out_path: Path) -> None:
    """Capture rendered widgets using the existing Pi/Wayland grim workflow."""
    window = widget.winfo_toplevel()
    window.deiconify()
    window.lift()
    window.update()
    time.sleep(0.3)
    # Process compositor Configure events before reading desktop coordinates.
    window.update()
    env = {
        **os.environ,
        "WAYLAND_DISPLAY": os.environ.get("WAYLAND_DISPLAY", "wayland-0"),
        "XDG_RUNTIME_DIR": os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"),
    }
    geometry = (f"{max(0, widget.winfo_rootx())},{max(0, widget.winfo_rooty())} "
                f"{widget.winfo_width()}x{widget.winfo_height()}")
    subprocess.run(["grim", "-g", geometry, str(out_path)], env=env, check=True)
    print(f"[Screenshot] {out_path.name} ({geometry})")


def main():
    artifact_dir = Path("Data/script_output")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    sample = cv2.imread("tests/samples/20260909_172410_497.jpg")
    if sample is None:
        sample = np.full((480, 640, 3), (35, 180, 45), dtype=np.uint8)
    today = date.today()
    with tempfile.TemporaryDirectory(prefix="cea-ui-telemetry-") as temp, ExitStack() as stack:
        telemetry_dir = Path(temp)
        random.seed(42)
        for days_ago in (0, 2, 6, 15, 29):
            path = generate_mock_data(today - timedelta(days=days_ago), telemetry_dir)
            with path.open() as file:
                rows = list(csv.reader(file))
            # A missing reading followed by a logging gap must remain empty.
            modified = [rows[0]]
            for row in rows[1:]:
                if " 13:" in row[0]:
                    continue
                if " 11:30:" in row[0]:
                    row[5:13] = ["null"] * 8
                modified.append(row)
            with path.open("w", newline="") as file:
                csv.writer(file).writerows(modified)
        stack.enter_context(patch.object(gui, "TELEMETRY_DIR", telemetry_dir))
        stack.enter_context(patch.object(gui, "Camera"))
        stack.enter_context(patch.object(gui, "PlantAIDetector"))
        for name in ("_init_serial", "_camera_loop", "_start_periodic_loggers", "_start_auto_ticker"):
            stack.enter_context(patch.object(gui.MainWindow, name))
        app = gui.MainWindow()
        try:
            app.geometry(f"{app.scr_w}x{app.scr_h}+0+0")
            app.update()
            visible_height = min(app.scr_h, app.winfo_screenheight() - app.winfo_rooty())
            app.geometry(f"{app.scr_w}x{visible_height}+0+0")
            app.plant_ai.is_available = False
            app._update_tpu_status_label()
            data = {"soil": [420, 395, 430, 410], "soil_temp": 22.4,
                    "temp": 24.1, "humidity": 58.2, "light": 1240,
                    "relays": "0000", "pan": 55, "tilt": 30}
            moistures = [42.1, 38.5, 51.0, 44.2]
            app.telemetry = {**data, "moisture_pct": moistures}
            app._update_telemetry_ui(data, moistures)
            app.display_image(sample)
            capture_window_crop(app, artifact_dir / "gui_main_window.png")
            app.plot_var.set(1)
            app.plotter.toggle(True)
            app.update()
            app.plotter.ani.event_source.stop()
            for second in (0, 4, 8, 12, 16, 20, 44, 48, 52, 56):
                values = (second, moistures, 22.4, 24.1, 58.2, 1240)
                if second == 16:
                    values = (second, [None] * 4, None, None, None, None)
                app.plotter._update(values)
            app.plotter.canvas_widget.draw()
            capture_window_crop(app.plotter.window, artifact_dir / "gui_plot_min.png")
            for mode in ("min", "day", "week", "month", "period"):
                if mode != "min":
                    if mode == "period":
                        app.period_start = today - timedelta(days=2)
                        app.period_end = today
                    app.plot_range_var.set(mode)
                    app.update()
                    capture_window_crop(app.plotter.window, artifact_dir / f"gui_plot_{mode}.png")
                capture_window_crop(app.card_data, artifact_dir / f"gui_data_{mode}.png")
            app._open_period_picker()
            picker = app._period_picker
            picker.select_date(today - timedelta(days=2))
            picker.select_date(today)
            capture_window_crop(picker, artifact_dir / "gui_period_calendar.png")
            picker.apply_button.invoke()
            capture_window_crop(app.plotter.window, artifact_dir / "gui_plot_period_applied.png")
            app.period_start = app.period_end = today - timedelta(days=90)
            app.plotter._render_current_mode()
            capture_window_crop(app.plotter.window, artifact_dir / "gui_plot_empty.png")
        finally:
            app.plotter.toggle(False)
            app.destroy()


if __name__ == "__main__":
    main()
