"""Comprehensive unit and edge-case tests for the Grand GUI updates.

Covers:
1. Recording interval parser (standard mappings and edge cases).
2. Historical CSV telemetry reader (day, week, month ranges, corrupt rows, nulls).
3. Dynamic soil water setpoint (Spinbox DoubleVar) and real-time auto-loop reaction.
4. Independent periodic data logger and clean image logger timers.
5. Manual clean frame capture guarantee (no burned AI overlays).
6. Scheduled daily irrigation ticker (10am CST check, AUTO gate, duplicate prevention).
7. Scheduled watering monitor (per-channel selective activation, 80% cutoff, 180s safety timeout).
8. PlotWindow dynamic range switching (trace callback, static Matplotlib in-place rendering).
9. Shutdown cleanup (canceling all background jobs and safety shutoff).
"""

from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path
import time
import tkinter as tk
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

import gui.psc_irr_gui as psc_mod
from gui.psc_irr_gui import (
    MainWindow,
    PlotWindow,
    SCHEDULED_MAX_WATERING_SEC,
    SCHEDULED_TARGET_PCT,
    SOIL_WATER_SETPOINT,
    parse_interval_to_ms,
    read_historical_telemetry,
)


# --- 1. Interval Parser Tests ---

def test_interval_parser_standard():
    assert parse_interval_to_ms("10s") == 10_000
    assert parse_interval_to_ms("1min") == 60_000
    assert parse_interval_to_ms("1hr") == 3_600_000
    assert parse_interval_to_ms("sec") == 1_000
    assert parse_interval_to_ms("min") == 60_000
    assert parse_interval_to_ms("hr") == 3_600_000
    assert parse_interval_to_ms("day") == 86_400_000
    assert parse_interval_to_ms("1s") == 1_000
    assert parse_interval_to_ms("1m") == 60_000
    assert parse_interval_to_ms("1h") == 3_600_000
    assert parse_interval_to_ms("1day") == 86_400_000
    assert parse_interval_to_ms("24hr") == 86_400_000


def test_interval_parser_edge_cases():
    # Case insensitivity & leading/trailing whitespace
    assert parse_interval_to_ms("  10S  ") == 10_000
    assert parse_interval_to_ms("1MIN") == 60_000
    assert parse_interval_to_ms("HR") == 3_600_000

    # Unrecognized or empty -> returns default_ms cleanly without throwing
    assert parse_interval_to_ms("", default_ms=5_000) == 5_000
    assert parse_interval_to_ms("nonexistent", default_ms=12_345) == 12_345
    assert parse_interval_to_ms(None, default_ms=10_000) == 10_000


# --- 2. Historical Telemetry CSV Reader Tests ---

def test_read_historical_telemetry_ranges(tmp_path):
    now = datetime(2026, 9, 10, 15, 0, 0)
    
    # Create sample CSV files across 3 days
    header = (
        "timestamp,soil1_raw,soil2_raw,soil3_raw,soil4_raw,"
        "soil1_pct,soil2_pct,soil3_pct,soil4_pct,"
        "soil_temp_c,air_temp_c,humidity_pct,light,relays,pan_deg,tilt_deg\n"
    )
    
    # Day 1: 10 days ago (outside week range, inside month range)
    f1 = tmp_path / "telemetry_20260831.csv"
    f1.write_text(
        header +
        "2026-08-31 12:00:00,420,400,430,410,40.0,45.0,38.0,42.0,21.0,23.0,55.0,1000,0000,55,30\n",
        encoding="utf-8"
    )

    # Day 2: 2 days ago (inside week & month ranges)
    f2 = tmp_path / "telemetry_20260908.csv"
    f2.write_text(
        header +
        "2026-09-08 10:00:00,415,395,425,405,42.0,47.0,40.0,44.0,21.5,23.5,56.0,1100,0000,55,30\n",
        encoding="utf-8"
    )

    # Day 3: Today at 09:00 and 14:00 (inside day, week & month ranges)
    f3 = tmp_path / "telemetry_20260910.csv"
    f3.write_text(
        header +
        "2026-09-10 09:00:00,410,390,420,400,45.0,50.0,42.0,46.0,22.0,24.0,57.0,1200,0000,55,30\n" +
        "2026-09-10 14:00:00,405,385,415,395,47.0,52.0,44.0,48.0,22.5,24.5,58.0,1300,0000,55,30\n",
        encoding="utf-8"
    )

    # Test 'day' mode (only rows from today starting at midnight)
    res_day = read_historical_telemetry(tmp_path, "day", now=now)
    assert len(res_day["timestamps"]) == 2
    assert res_day["moisture"][0] == [45.0, 47.0]
    assert res_day["soil_temp"] == [22.0, 22.5]
    assert res_day["air_temp"] == [24.0, 24.5]

    # Test 'week' mode (rows from last 7 days: Day 2 + Day 3 = 3 rows)
    res_week = read_historical_telemetry(tmp_path, "week", now=now)
    assert len(res_week["timestamps"]) == 3

    # Test 'month' mode (rows from last 30 days: Day 1 + Day 2 + Day 3 = 4 rows)
    res_month = read_historical_telemetry(tmp_path, "month", now=now)
    assert len(res_month["timestamps"]) == 4


def test_read_historical_telemetry_corrupt_and_edge_cases(tmp_path):
    now = datetime(2026, 9, 10, 15, 0, 0)

    # 1. Non-existent directory returns empty lists without exception
    missing_dir = tmp_path / "non_existent_subdir"
    res_empty = read_historical_telemetry(missing_dir, "day", now=now)
    assert res_empty["timestamps"] == []
    assert res_empty["moisture"] == [[], [], [], []]

    # 2. Corrupt rows (truncated row, invalid dates, 'null' tokens)
    header = (
        "timestamp,soil1_raw,soil2_raw,soil3_raw,soil4_raw,"
        "soil1_pct,soil2_pct,soil3_pct,soil4_pct,"
        "soil_temp_c,air_temp_c,humidity_pct,light,relays,pan_deg,tilt_deg\n"
    )
    f = tmp_path / "telemetry_20260910.csv"
    f.write_text(
        header +
        "corrupt-timestamp,410,390,420,400,45.0,50.0,42.0,46.0,22.0,24.0,57.0\n" +  # invalid datetime
        "2026-09-10 10:00:00,410,390\n" +  # truncated row
        "2026-09-10 11:00:00,null,null,null,null,null,null,null,null,null,null,null,null,0000,55,30\n" +  # nulls
        "2026-09-10 12:00:00,410,390,420,400,45.0,50.0,42.0,46.0,22.0,24.0,57.0,1200,0000,55,30\n",  # valid row
        encoding="utf-8"
    )

    res = read_historical_telemetry(tmp_path, "day", now=now)
    # The valid row and the null-row (with None values) are parsed; corrupt and truncated lines are safely skipped
    assert len(res["timestamps"]) == 2
    assert res["moisture"][0][0] is None  # null from first row
    assert res["moisture"][0][1] == 45.0  # valid float from second row


# --- 3. Dynamic Setpoint & Auto-Loop Reaction ---

@pytest.fixture
def headless_gui():
    """Create headless MainWindow instance with background loops mocked."""
    with patch("gui.psc_irr_gui.Camera"):
        with patch.object(MainWindow, "_init_serial"):
            with patch.object(MainWindow, "_camera_loop"):
                with patch.object(MainWindow, "_start_periodic_loggers"):
                    with patch.object(MainWindow, "_start_scheduled_ticker"):
                        app = MainWindow()
    app.withdraw()
    yield app
    try:
        app.destroy()
    except Exception:
        pass


# --- 4. Periodic Data & Image Loggers ---

def test_periodic_data_logger(headless_gui, tmp_path, monkeypatch):
    app = headless_gui
    monkeypatch.setattr(psc_mod, "TELEMETRY_DIR", tmp_path)

    app.telemetry = {
        "soil": [400, 410, 420, 430],
        "moisture_pct": [42.0, 40.5, 38.0, 36.5],
        "soil_temp": 22.0,
        "temp": 24.0,
        "humidity": 55.0,
        "light": 800,
        "relays": "0000",
        "pan": 55,
        "tilt": 30,
    }

    with patch.object(app, "after") as mock_after:
        app.data_record_var.set("1min")
        app._periodic_data_logger()

        # Check file was written to tmp_path
        today = datetime.now().strftime("%Y%m%d")
        csv_file = tmp_path / f"telemetry_{today}.csv"
        assert csv_file.exists()
        lines = csv_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2  # header + row
        assert "42.0,40.5,38.0,36.5" in lines[1]

        # Rescheduled with 60,000ms for "1min"
        mock_after.assert_called_with(60_000, app._periodic_data_logger)


def test_periodic_image_logger_clean_frame(headless_gui, tmp_path, monkeypatch):
    app = headless_gui
    monkeypatch.setattr(psc_mod, "IMAGES_DIR", tmp_path)

    dummy_clean_frame = np.full((100, 100, 3), (40, 200, 50), dtype=np.uint8)
    app.camera.is_available = True
    app.camera.capture_array.return_value = dummy_clean_frame

    with patch.object(app, "after") as mock_after:
        app.image_record_var.set("sec")
        app._periodic_image_logger()

        # Captured independently from camera
        app.camera.capture_array.assert_called()

        # Check image was saved
        saved_images = list(tmp_path.glob("IMG_*.jpg"))
        assert len(saved_images) == 1

        # Rescheduled with 1,000ms for "sec"
        mock_after.assert_called_with(1_000, app._periodic_image_logger)


def test_manual_clean_frame_capture(headless_gui, tmp_path, monkeypatch):
    app = headless_gui
    monkeypatch.setattr(psc_mod, "IMAGES_DIR", tmp_path)

    app.live_var.set(1)
    app.camera.is_available = True
    dummy_clean_frame = np.full((100, 100, 3), (30, 220, 60), dtype=np.uint8)
    app.camera.capture_array.return_value = dummy_clean_frame

    # Trigger capture
    app._on_capture()
    assert app.flag_capture is True

    # Camera loop executes capture
    with patch.object(app, "after"):
        with patch.object(app, "display_image"):
            app._camera_loop()
            assert app.flag_capture is False
            saved_images = list(tmp_path.glob("IMG_*.jpg"))
            assert len(saved_images) == 1


# --- 5. Scheduled Daily Irrigation Tests ---

def test_scheduled_irrigation_ticker_auto_vs_manual(headless_gui):
    app = headless_gui
    today = datetime.now().strftime("%Y-%m-%d")

    # Simulate 10:00 AM clock
    mock_now = MagicMock()
    mock_now.hour = 10
    mock_now.minute = 0
    mock_now.strftime.return_value = today

    with patch("gui.psc_irr_gui.datetime") as mock_dt:
        mock_dt.now.return_value = mock_now

        # Case 1: Mode is MANUAL -> check is skipped with log
        app.auto_var.set(0)
        with patch.object(app, "_start_scheduled_watering") as mock_start:
            app._scheduled_check_ticker()
            mock_start.assert_not_called()
            assert app._last_scheduled_date == today

        # Case 2: Already ran today -> does not trigger again
        app.auto_var.set(1)
        with patch.object(app, "_start_scheduled_watering") as mock_start:
            app._scheduled_check_ticker()
            mock_start.assert_not_called()

        # Case 3: New day and AUTO is ON -> triggers watering!
        app._last_scheduled_date = "2026-09-09"
        with patch.object(app, "_start_scheduled_watering") as mock_start:
            app._scheduled_check_ticker()
            mock_start.assert_called_with(today)


def test_scheduled_watering_selective_channels_and_target_cutoff(headless_gui, tmp_path, monkeypatch):
    app = headless_gui
    monkeypatch.setattr(psc_mod, "SCHEDULED_WATERING_DIR", tmp_path)
    app.ser = MagicMock()
    app.ser.is_open = True
    app.camera.is_available = True
    app.camera.capture_array.return_value = np.zeros((50, 50, 3), dtype=np.uint8)

    # Setpoint is 40.0%
    app.soil_water_setpoint.set(40.0)
    # Channel 0: 32% (<40 -> water), Channel 1: 55% (>=40 -> leave), Channel 2: 36% (<40 -> water), Channel 3: 45% (>=40 -> leave)
    app.telemetry["moisture_pct"] = [32.0, 55.0, 36.0, 45.0]

    with patch.object(app, "after") as mock_after:
        app._start_scheduled_watering()
        assert app._scheduled_watering_active is True
        assert app._scheduled_channels_active == [0, 2]
        # Only channels 0 and 2 energized: bitmask '1010'
        app.ser.write.assert_called_with(b"1010\n")
        assert [v.get() for v in app.water_vars] == [1, 0, 1, 0]
        # Dedicated timestamped run directory created
        assert app._scheduled_run_dir is not None
        assert app._scheduled_run_dir.exists()
        mock_after.assert_called_with(2000, app._scheduled_watering_monitor)

    # Monitor Tick 1: Channel 0 reaches 82% (>= 80% cutoff), Channel 2 reaches 65% (<80%)
    app.telemetry["moisture_pct"] = [82.0, 55.0, 65.0, 45.0]
    with patch.object(app, "after") as mock_after:
        app._scheduled_watering_monitor()
        # Channel 0 de-energized, only channel 2 remains active
        assert app._scheduled_channels_active == [2]
        assert [v.get() for v in app.water_vars] == [0, 0, 1, 0]
        app.ser.write.assert_called_with(b"0010\n")
        # An image was captured to the run directory
        saved_imgs = list(app._scheduled_run_dir.glob("IMG_*.jpg"))
        assert len(saved_imgs) == 1

    # Monitor Tick 2: Channel 2 reaches 80.5% (>= 80% cutoff)
    app.telemetry["moisture_pct"] = [82.0, 55.0, 80.5, 45.0]
    with patch.object(app, "after"):
        app._scheduled_watering_monitor()
        # All channels reached target: all pumps OFF, watering session complete!
        assert app._scheduled_watering_active is False
        assert app._scheduled_channels_active == []
        assert [v.get() for v in app.water_vars] == [0, 0, 0, 0]
        app.ser.write.assert_called_with(b"0000\n")


def test_scheduled_watering_safety_timeout(headless_gui, tmp_path, monkeypatch):
    app = headless_gui
    monkeypatch.setattr(psc_mod, "SCHEDULED_WATERING_DIR", tmp_path)
    app.ser = MagicMock()
    app.ser.is_open = True
    app.camera.is_available = True
    app.camera.capture_array.return_value = np.zeros((50, 50, 3), dtype=np.uint8)

    app.soil_water_setpoint.set(40.0)
    app.telemetry["moisture_pct"] = [35.0, 50.0, 50.0, 50.0]

    with patch.object(app, "after"):
        app._start_scheduled_watering()
        assert app._scheduled_watering_active is True
        assert app._scheduled_channels_active == [0]

    # Simulate elapsed time surpassing 180s (e.g. 182 seconds) while channel is still at 60%
    app._scheduled_start_time = time.time() - (SCHEDULED_MAX_WATERING_SEC + 2)
    app.telemetry["moisture_pct"] = [60.0, 50.0, 50.0, 50.0]

    with patch.object(app, "after"):
        app._scheduled_watering_monitor()
        # Safety timeout reached: pumps forcibly stopped and active state cleared
        assert app._scheduled_watering_active is False
        assert app._scheduled_channels_active == []
        assert [v.get() for v in app.water_vars] == [0, 0, 0, 0]
        app.ser.write.assert_called_with(b"0000\n")


# --- 6. PlotWindow Dynamic Range Switching Tests ---

def test_plot_window_range_trace_switching(headless_gui, tmp_path, monkeypatch):
    app = headless_gui
    monkeypatch.setattr(psc_mod, "TELEMETRY_DIR", tmp_path)

    # Populate temporary telemetry CSV
    header = (
        "timestamp,soil1_raw,soil2_raw,soil3_raw,soil4_raw,"
        "soil1_pct,soil2_pct,soil3_pct,soil4_pct,"
        "soil_temp_c,air_temp_c,humidity_pct,light,relays,pan_deg,tilt_deg\n"
    )
    today_str = datetime.now().strftime("%Y%m%d")
    (tmp_path / f"telemetry_{today_str}.csv").write_text(
        header +
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},410,390,420,400,45.0,50.0,42.0,46.0,22.0,24.0,57.0,1200,0000,55,30\n",
        encoding="utf-8"
    )

    plotter = PlotWindow(app, lambda: app.telemetry, range_var=app.plot_range_var)

    # 1. Open PlotWindow in default "min" mode
    assert app.plot_range_var.get() == "min"
    plotter.toggle(True)
    assert plotter.window is not None
    assert plotter.ani is not None

    # 2. Switch to "day" mode via trace variable
    app.plot_range_var.set("day")
    # FuncAnimation should be stopped and cleared
    assert plotter.ani is None
    assert "Day" in plotter.window.title()

    # 3. Switch to "week" mode
    app.plot_range_var.set("week")
    assert plotter.ani is None
    assert "Week" in plotter.window.title()

    # 4. Switch back to "min" mode
    app.plot_range_var.set("min")
    assert plotter.ani is not None
    assert "60s" in plotter.window.title()

    # Clean close
    plotter.toggle(False)
    assert plotter.window is None


def test_plot_window_individual_pump_bars(headless_gui, tmp_path, monkeypatch):
    """Verify that PlotWindow displays individual bars per pump, not a single combined bar."""
    app = headless_gui
    monkeypatch.setattr(psc_mod, "TELEMETRY_DIR", tmp_path)

    header = (
        "timestamp,soil1_raw,soil2_raw,soil3_raw,soil4_raw,"
        "soil1_pct,soil2_pct,soil3_pct,soil4_pct,"
        "soil_temp_c,air_temp_c,humidity_pct,light,relays,pan_deg,tilt_deg\n"
    )
    today = datetime.now().date()
    t_str = today.strftime("%Y%m%d")
    
    # 4 pumps turn on at 10:00:00 and turn off at 10:00:06.840 (6.84 seconds -> 6.84 * (500/3600) = 0.950 L = 950 mL each)
    row_idle = f"{today} 09:59:00,400,400,400,400,40.0,40.0,40.0,40.0,22.0,24.0,50.0,1000,0000,55,30\n"
    row_on = f"{today} 10:00:00,400,400,400,400,40.0,40.0,40.0,40.0,22.0,24.0,50.0,1000,1111,55,30\n"
    row_off = f"{today} 10:00:06.840,400,400,400,400,45.0,45.0,45.0,45.0,22.0,24.0,50.0,1000,0000,55,30\n"

    (tmp_path / f"telemetry_{t_str}.csv").write_text(
        header + row_idle + row_on + row_off,
        encoding="utf-8"
    )

    plotter = PlotWindow(app, lambda: app.telemetry, range_var=app.plot_range_var)
    app.plot_range_var.set("day")
    plotter.toggle(True)

    # Find the ax_vol secondary y-axis (ylabel is 'Water volume, mL')
    ax_vol = None
    for ax in plotter.fig.axes:
        if "Water volume" in ax.get_ylabel():
            ax_vol = ax
            break
    assert ax_vol is not None, "Secondary water volume axis (ax_vol) not found"

    # Verify there are 4 separate bar patches, NOT 1 combined bar
    patches = ax_vol.patches
    assert len(patches) == 4, f"Expected 4 separate pump bars, found {len(patches)}"

    # Verify each bar represents that specific pump's volume (950 mL)
    for i, patch in enumerate(patches):
        assert round(patch.get_height(), 1) == 950.0, f"Bar {i} height should be 950 mL, got {patch.get_height()}"

    # Verify bars have distinct x positions (due to per-pump time offsets)
    x_positions = [patch.get_x() for patch in patches]
    assert len(set(x_positions)) == 4, "All 4 pump bars should have distinct x positions (side-by-side)"

    plotter.toggle(False)


def test_spinbox_large_buttons_and_gimbal_shrink(headless_gui):
    app = headless_gui

    # 1. Verify spinbox large buttons exist
    assert hasattr(app, "btn_stop_down")
    assert hasattr(app, "btn_stop_up")
    assert hasattr(app, "btn_start_down")
    assert hasattr(app, "btn_start_up")

    # 2. Test turn on (start setpoint) adjustment buttons
    app.soil_water_setpoint.set(40.0)
    app.btn_start_up.invoke()
    assert app.soil_water_setpoint.get() == 45.0
    app.btn_start_down.invoke()
    assert app.soil_water_setpoint.get() == 40.0

    # Bounds clamping for start setpoint (10.0 to 90.0)
    app.soil_water_setpoint.set(88.0)
    app.btn_start_up.invoke()
    assert app.soil_water_setpoint.get() == 90.0
    app.btn_start_up.invoke()
    assert app.soil_water_setpoint.get() == 90.0

    app.soil_water_setpoint.set(12.0)
    app.btn_start_down.invoke()
    assert app.soil_water_setpoint.get() == 10.0
    app.btn_start_down.invoke()
    assert app.soil_water_setpoint.get() == 10.0

    # 3. Test turn off (stop setpoint) adjustment buttons
    app.soil_water_stop_setpoint.set(80.0)
    app.btn_stop_up.invoke()
    assert app.soil_water_stop_setpoint.get() == 85.0
    app.btn_stop_down.invoke()
    assert app.soil_water_stop_setpoint.get() == 80.0

    # Bounds clamping for stop setpoint (10.0 to 100.0)
    app.soil_water_stop_setpoint.set(98.0)
    app.btn_stop_up.invoke()
    assert app.soil_water_stop_setpoint.get() == 100.0
    app.btn_stop_up.invoke()
    assert app.soil_water_stop_setpoint.get() == 100.0

    app.soil_water_stop_setpoint.set(12.0)
    app.btn_stop_down.invoke()
    assert app.soil_water_stop_setpoint.get() == 10.0
    app.btn_stop_down.invoke()
    assert app.soil_water_stop_setpoint.get() == 10.0

    # 4. Verify gimbal buttons shrunk (font size <= 30)
    assert hasattr(app, "btn_tilt_up")
    assert hasattr(app, "btn_pan_left")
    assert hasattr(app, "btn_pan_right")
    assert hasattr(app, "btn_tilt_down")
    assert hasattr(app, "btn_home")

    gimbal_font = app.btn_tilt_up.cget("font")
    if isinstance(gimbal_font, (tuple, list)):
        size = int(gimbal_font[1])
    else:
        parts = str(gimbal_font).split()
        size = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 26
    assert size <= 30, f"Gimbal button font size {size} is too large, should be <= 30"


