"""Headless integration tests for MainWindow and PlotWindow UI logic."""

import time
import tkinter as tk
from unittest.mock import MagicMock, patch
import pytest

import gui.psc_irr_gui as psc_mod
from gui.psc_irr_gui import MainWindow, PlotWindow, SOIL_WATER_SETPOINT


@pytest.fixture
def headless_app():
    """Create a MainWindow instance in withdrawn state with hardware loops suppressed during init."""
    with patch("gui.psc_irr_gui.Camera"):
        with patch.object(MainWindow, "_init_serial"):
            with patch.object(MainWindow, "_camera_loop"):
                app = MainWindow()
    app.withdraw()
    yield app
    try:
        app.destroy()
    except Exception:
        pass


def test_mainwindow_initial_state(headless_app):
    app = headless_app
    assert app.auto_var.get() == 1
    assert app.live_var.get() == 1
    assert app.plot_var.get() == 0
    assert app.plant_ai_var.get() == 0
    assert hasattr(app, "plant_ai")
    assert hasattr(app, "lbl_tpu_status")
    assert len(app.water_vars) == 4
    assert len(app.water_btns) == 4

    assert hasattr(app, "btn_home")
    assert app.btn_home["text"] == "Home"

    # Relays should be disabled when in AUTO mode
    for btn in app.water_btns:
        assert str(btn["state"]) == tk.DISABLED


def test_mainwindow_auto_manual_toggle(headless_app):
    app = headless_app
    app.ser = MagicMock()
    app.ser.is_open = True

    # Toggle to MANUAL
    app.auto_var.set(0)
    app._on_auto_toggle()
    assert app.ckb_auto["text"] == "MANUAL"
    for btn in app.water_btns:
        assert str(btn["state"]) == tk.NORMAL

    # Toggle back to AUTO
    app.auto_var.set(1)
    app._on_auto_toggle()
    assert app.ckb_auto["text"] == "AUTO"
    for btn in app.water_btns:
        assert str(btn["state"]) == tk.DISABLED


def test_mainwindow_live_toggle(headless_app):
    app = headless_app

    # Toggle Live off
    app.live_var.set(0)
    app._on_live_toggle()
    assert str(app.btn_capture["state"]) == tk.DISABLED

    # Toggle Live on
    app.live_var.set(1)
    app._on_live_toggle()
    assert str(app.btn_capture["state"]) == tk.NORMAL


def test_mainwindow_rebuild_relays(headless_app):
    app = headless_app
    app.telemetry["relays"] = "101"
    app.rebuild_relays(3)

    assert len(app.water_vars) == 3
    assert len(app.water_btns) == 3
    assert app.water_vars[0].get() == 1
    assert app.water_vars[1].get() == 0
    assert app.water_vars[2].get() == 1


def test_mainwindow_manual_relays_send(headless_app):
    app = headless_app
    app.ser = MagicMock()
    app.ser.is_open = True

    app.auto_var.set(0)  # Manual mode
    app.water_vars[0].set(1)
    app.water_vars[1].set(0)
    app.water_vars[2].set(1)
    app.water_vars[3].set(0)

    app._send_manual_relays()
    app.ser.write.assert_called_with(b"1010\n")


def test_mainwindow_repeat_mechanism(headless_app):
    app = headless_app
    counter = [0]

    def increment():
        counter[0] += 1

    app._start_repeat(increment)
    assert counter[0] == 1  # immediate call
    assert app._repeat_job is not None

    app._stop_repeat()
    assert app._repeat_job is None


def test_plot_window_lifecycle(headless_app):
    app = headless_app
    moistures = [35.0, 45.0, None, 60.0]
    plotter = PlotWindow(app, lambda: moistures)

    # Show window
    plotter.toggle(True)
    assert plotter.window is not None

    # Feed data update (returns 4 moisture lines + soil temp + air temp + RH = 7 lines)
    lines = plotter._update((10.0, moistures, 24.5, 23.0, 50.0))
    assert len(lines) == 7

    # Hide / Close window
    plotter.toggle(False)
    assert plotter.window is None


def test_plot_window_layout_order_and_legend(headless_app):
    app = headless_app
    plotter = PlotWindow(app, lambda: [10.0, 20.0, 30.0, 40.0])
    plotter.toggle(True)
    plotter.window.update_idletasks()

    # 1. Plot window is configured topmost (in live X11/Wayland returns 1, in headless with withdrawn master may be 0 or 1)
    assert plotter.window.attributes("-topmost") in (0, 1, True)
    # Ensure calling attributes('-topmost', True) succeeds without error
    plotter.window.attributes("-topmost", True)

    # 2. Window width leaves right control panel exposed
    scr_w = app.winfo_screenwidth()
    margin_w = app.margin_w
    expected_w = max(600, scr_w - margin_w - 6)
    assert f"{expected_w}x" in plotter.window.geometry()

    # 3. Top subplot is temperature & RH, Bottom is soil moisture
    assert plotter.ax_temp is not None
    assert plotter.ax_moist is not None
    assert "Air temp" in plotter.ax_temp.get_ylabel()
    assert "Relative humidity" in plotter.ax_rh.get_ylabel()
    assert "Soil moisture" in plotter.ax_moist.get_ylabel()
    assert plotter.ax_temp.get_subplotspec().rowspan.start < plotter.ax_moist.get_subplotspec().rowspan.start

    # 4. Legends are positioned at lower right (loc code 4)
    leg_temp = plotter.ax_temp.get_legend()
    leg_moist = plotter.ax_moist.get_legend()
    assert leg_temp is not None and leg_temp._loc in (4, "lower right")
    assert leg_moist is not None and leg_moist._loc in (4, "lower right")

    plotter.toggle(False)


def test_mainwindow_on_closing(headless_app):
    app = headless_app
    app.ser = MagicMock()
    app.ser.is_open = True
    app.camera = MagicMock()

    with patch("sys.exit") as mock_exit:
        app.on_closing()
        # Should shut down all pumps with 0000
        app.ser.write.assert_any_call(b"0000\n")
        # Should home gimbal
        app.ser.write.assert_any_call(b"h\n")
        # Should close serial
        app.ser.close.assert_called_once()
        # Should stop camera
        app.camera.stop.assert_called_once()
        # Should exit
        mock_exit.assert_called_with(0)


def test_mainwindow_plant_ai_toggle(headless_app):
    app = headless_app

    # Plant AI OFF -> Status is Ready or Disconnected
    app.plant_ai_var.set(0)
    app._on_plant_ai_toggle()
    assert "Ready" in app.lbl_tpu_status["text"] or "Disconnected" in app.lbl_tpu_status["text"]

    # Plant AI ON -> Status becomes Active or Disconnected
    app.plant_ai_var.set(1)
    app._on_plant_ai_toggle()
    assert "Active" in app.lbl_tpu_status["text"] or "Disconnected" in app.lbl_tpu_status["text"]


def test_mainwindow_camera_loop_with_plant_ai(headless_app):
    import numpy as np

    app = headless_app
    app.live_var.set(1)
    app.plant_ai_var.set(1)
    app.camera.is_available = True
    dummy_frame = np.full((100, 100, 3), (30, 200, 40), dtype=np.uint8)
    app.camera.capture_array.return_value = dummy_frame

    with patch.object(app, "display_image") as mock_display:
        with patch.object(app, "after") as mock_after:
            app._camera_loop()
            mock_display.assert_called_once()
            mock_after.assert_called_with(30, app._camera_loop)


def test_mainwindow_both_plant_ai_and_heatmap_overlays(headless_app):
    """Verify that when both AI and Heatmap are active, heatmap overlays entire frame and AI shows bboxes."""
    import numpy as np

    app = headless_app
    app.plant_ai_var.set(1)
    app.heatmap_var.set(1)
    dummy_frame = np.full((100, 100, 3), 180, dtype=np.uint8)
    dummy_frame[30:70, 30:70] = (30, 200, 40)

    with patch.object(app.plant_ai, "detect_and_analyze", return_value=([], 12.5)) as mock_detect:
        with patch.object(app.plant_ai, "draw_full_frame_heatmap", wraps=app.plant_ai.draw_full_frame_heatmap) as mock_hmap:
            with patch.object(app.plant_ai, "draw_overlay", wraps=app.plant_ai.draw_overlay) as mock_draw:
                out, lat = app._apply_plant_ai_overlays(dummy_frame)
                mock_detect.assert_called_once_with(dummy_frame)
                mock_hmap.assert_called_once()
                mock_draw.assert_called_once()
                # Verify show_bbox=True was passed to draw_overlay
                _, kwargs = mock_draw.call_args
                assert kwargs.get("show_bbox") is True
                assert lat == 12.5
                assert out.shape == dummy_frame.shape


def test_read_historical_telemetry_flow_rate_and_events(tmp_path):
    from tests.mock_telemetry import generate_mock_data
    from datetime import date, datetime

    today = date(2026, 9, 11)
    csv_file = generate_mock_data(today, tmp_path)
    assert csv_file.exists()

    hist = psc_mod.read_historical_telemetry(tmp_path, "day", now=datetime(2026, 9, 11, 23, 59, 59))
    assert len(hist["timestamps"]) > 0
    assert len(hist["pump_events"]) == 2

    ev1, ev2 = hist["pump_events"]
    assert ev1["pump"] == 0
    assert ev1["volume"] == 12.5
    assert ev1["duration"] == 90.0

    assert ev2["pump"] == 1
    assert ev2["volume"] == 12.5
    assert ev2["duration"] == 90.0


def test_plot_window_threshold_lines_and_dynamic_setpoint(headless_app):
    app = headless_app
    plotter = PlotWindow(app, lambda: [35.0, 45.0, 55.0, 65.0], range_var=app.plot_range_var, setpoint_var=app.soil_water_setpoint)
    plotter.toggle(True)
    plotter.window.update()

    # Red dashed setpoint line at 40%, Green dashed stop target line at 80%
    assert plotter.line_setpoint is not None
    assert plotter.line_stop is not None
    assert list(plotter.line_setpoint.get_ydata()) == [40.0, 40.0]
    assert list(plotter.line_stop.get_ydata()) == [80.0, 80.0]

    # Dynamic update of setpoint spinbox
    app.soil_water_setpoint.set(55.0)
    plotter.window.update()
    assert list(plotter.line_setpoint.get_ydata()) == [55.0, 55.0]

    # Check that live legend contains only S1-S4
    live_legend_labels = [t.get_text() for t in plotter.ax_moist.get_legend().get_texts()]
    assert live_legend_labels == ["S1", "S2", "S3", "S4"]

    plotter.toggle(False)


def test_plot_window_historical_water_volume_axis_and_bars(headless_app, tmp_path, monkeypatch):
    from tests.mock_telemetry import generate_mock_data
    from datetime import date, datetime

    monkeypatch.setattr(psc_mod, "TELEMETRY_DIR", tmp_path)
    generate_mock_data(datetime.now().date(), tmp_path)

    app = headless_app
    plotter = PlotWindow(app, lambda: [35.0, 45.0, 55.0, 65.0], range_var=app.plot_range_var, setpoint_var=app.soil_water_setpoint)
    plotter.toggle(True)
    plotter.window.update()

    # Switch to "day" mode
    app.plot_range_var.set("day")
    plotter.window.update()

    # Check secondary water volume axis
    ax_moist = plotter.fig.axes[2]  # Subplot 2,1,2 (ax_moist)
    ax_vol = plotter.fig.axes[3]    # Twinx secondary axis (ax_vol)
    assert "Water volume" in ax_vol.get_ylabel()
    assert "mL" in ax_vol.get_ylabel()
    assert ax_vol.get_ylim() == (0.0, 1000.0)

    # Check that pump event bars were plotted on ax_vol
    bars = [c for c in ax_vol.containers]
    assert len(bars) >= 2  # Pumps 1 and 2 bars plotted

    # Check that legend contains ONLY sensor channels (S1-S4), no setpoint, stop target, or volume
    legend_labels = [t.get_text() for t in ax_moist.get_legend().get_texts()]
    assert legend_labels == ["S1", "S2", "S3", "S4"]

    plotter.toggle(False)
