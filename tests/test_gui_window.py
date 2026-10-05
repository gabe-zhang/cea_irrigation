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
    with patch("gui.psc_irr_gui.Camera"), patch("gui.psc_irr_gui.PlantAIDetector"):
        with patch.object(MainWindow, "_init_serial"):
            with patch.object(MainWindow, "_camera_loop"), \
                    patch.object(MainWindow, "_start_periodic_loggers"), \
                    patch.object(MainWindow, "_start_auto_ticker"):
                app = MainWindow()
    app.withdraw()
    app.plant_ai.is_available = False
    app.plant_ai.last_latency_ms = 0.0
    app.plant_ai.detect_and_analyze.return_value = ([], 0.0)
    app.plant_ai.draw_overlay.side_effect = lambda frame, *_args, **_kwargs: frame
    app.plant_ai.draw_full_frame_heatmap.side_effect = lambda frame, *_args, **_kwargs: frame
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

    # Feed data update (returns 4 moisture artists + soil temp + air temp + RH + light)
    lines = plotter._update((10.0, moistures, 24.5, 23.0, 50.0))
    assert len(lines) == 8

    # Hide / Close window
    plotter.toggle(False)
    assert plotter.window is None


def test_plot_window_layout_order_and_legend(headless_app):
    app = headless_app
    plotter = PlotWindow(app, lambda: [10.0, 20.0, 30.0, 40.0])
    topmost_requests = []
    original_attributes = tk.Toplevel.attributes

    def record_attributes(window, *args):
        topmost_requests.append(args)
        return original_attributes(window, *args)

    with patch.object(tk.Toplevel, "attributes", record_attributes):
        plotter.toggle(True)
    plotter.window.update_idletasks()

    assert ("-topmost", True) not in topmost_requests
    assert all(child.winfo_class() != "Button" for child in plotter.window.winfo_children())

    # 2. Window width leaves right control panel exposed
    scr_w = app.winfo_screenwidth()
    margin_w = app.margin_w
    expected_w = max(600, scr_w - margin_w - 6)
    assert f"{expected_w}x" in plotter.window.geometry()

    # 3. Top subplot is temperature & RH, Bottom is soil moisture
    assert plotter.ax_temp is not None
    assert plotter.ax_moist is not None
    assert plotter.ax_temp.get_ylabel() == "Air T/RH, °C/%"
    assert plotter.ax_light.get_ylabel() == "Light, Lux"
    assert "Soil moisture" in plotter.ax_moist.get_ylabel()
    assert plotter.ax_temp.get_subplotspec().rowspan.start < plotter.ax_moist.get_subplotspec().rowspan.start

    # 4. Legends are positioned at lower right (loc code 4)
    leg_temp = plotter.ax_temp.get_legend()
    leg_moist = plotter.ax_moist.get_legend()
    assert leg_temp is not None and leg_temp._loc in (4, "lower right")
    assert leg_moist is not None and leg_moist._loc in (4, "lower right")
    plotter.toggle(False)


def test_plot_stays_above_main_and_calendar_stays_above_plot(headless_app):
    app = headless_app
    app.deiconify()
    app.update_idletasks()
    app.plotter.toggle(True)
    assert str(app.plotter.window.transient()) == str(app)
    app.plot_range_var.set("period")
    with patch.object(tk.Toplevel, "attributes", autospec=True) as attributes:
        app._open_period_picker()
    popup = app._period_picker
    app.update_idletasks()
    assert popup.winfo_ismapped()
    assert str(popup.transient()) == str(app.plotter.window)
    assert not any(call.args[1:] == ("-topmost", True) for call in attributes.call_args_list)
    app._open_period_picker()
    assert app._period_picker is popup
    app.plotter.toggle(False)


@pytest.mark.parametrize("geometry", ["1920x1005+0+0", "1000x800+0+0"])
def test_range_calendar_fits_inside_main_window(headless_app, geometry):
    app = headless_app
    app.geometry(geometry)
    app.plot_range_var.set("period")
    app.deiconify()
    app.update_idletasks()
    app._open_period_picker()
    app.update_idletasks()
    popup = app._period_picker
    assert popup.winfo_ismapped()
    anchor = app.range_summary_button
    assert popup.winfo_rootx() == max(app.winfo_rootx(), min(
        anchor.winfo_rootx(), app.winfo_rootx() + app.winfo_width() - popup.winfo_width()))
    assert popup.winfo_rooty() == max(app.winfo_rooty(), min(
        anchor.winfo_rooty() + anchor.winfo_height(),
        app.winfo_rooty() + app.winfo_height() - popup.winfo_height()))
    assert popup.winfo_rootx() >= app.winfo_rootx()
    assert popup.winfo_rooty() >= app.winfo_rooty()
    assert popup.winfo_rootx() + popup.winfo_width() <= app.winfo_rootx() + app.winfo_width()
    assert popup.winfo_rooty() + popup.winfo_height() <= app.winfo_rooty() + app.winfo_height()
    assert popup.winfo_rootx() + popup.winfo_width() <= app.winfo_screenwidth()


def test_data_card_order_and_reserved_dates(headless_app):
    from datetime import date, datetime

    app = headless_app
    menu = app.plot_range_dropdown["menu"]
    assert [menu.entrycget(i, "label") for i in range(menu.index("end") + 1)] == [
        "min", "day", "week", "month", "period"
    ]
    assert app.card_data["text"].strip() == "DATA"
    assert app.ckb_plot.master is app.record_label.master is app.card_data
    assert app.ckb_plot.grid_info()["row"] == 0
    assert app.record_label.grid_info()["row"] == 2
    assert app.record_label.grid_info()["column"] == app.ckb_plot.grid_info()["column"] == 0
    assert app.data_record_frame.grid_info()["row"] == 2
    assert app.image_record_frame.grid_info()["row"] == 3
    assert app.data_record_frame.grid_info()["column"] == app.image_record_frame.grid_info()["column"] == 1
    now = datetime(2026, 10, 2, 12)
    record_positions = []
    for mode, expected in (
        ("min", ""), ("day", "10/02/2026"),
        ("week", "09/25/2026 – 10/02/2026"),
        ("month", "09/02/2026 – 10/02/2026"),
        ("period", f"{date.today().strftime("%m/%d/%Y")} – {date.today().strftime("%m/%d/%Y")}"),
    ):
        app.plot_range_var.set(mode)
        app._refresh_range_summary(now)
        app.update_idletasks()
        assert app.range_summary_var.get() == expected
        assert app._date_range_frame.winfo_manager() == "grid"
        record_positions.append(app.data_record_frame.winfo_y())
    assert len(set(record_positions)) == 1
    assert app.range_summary_button.winfo_manager() == "grid"
    assert app.range_summary_label.winfo_manager() == ""
    assert app.range_summary_button.grid_info()["row"] == 0
    assert app.range_summary_button.grid_info()["column"] == 0
    assert app.range_summary_button["text"] == expected
    assert app.record_label["anchor"] == "center"
    assert app.record_label.grid_info()["sticky"] == "nesw"
    assert app.record_label.grid_info()["rowspan"] == 2
    dropdowns = (app.plot_range_dropdown, app.data_record_dropdown, app.image_record_dropdown)
    assert len({dropdown.winfo_rootx() for dropdown in dropdowns}) == 1
    assert len({dropdown.winfo_width() for dropdown in dropdowns}) == 1
    assert app.range_summary_button.winfo_width() >= app.range_summary_button.winfo_reqwidth()
    app.plot_range_var.set("min")
    assert app.range_summary_button.winfo_manager() == ""
    assert app.range_summary_label.winfo_manager() == "grid"


def test_period_draft_apply_cancel_and_reversed_dates(headless_app):
    from datetime import date

    app = headless_app
    original = (app.period_start, app.period_end)
    app.plot_range_var.set("period")
    app.range_summary_button.invoke()
    picker = app._period_picker
    assert picker.calendar.cget("date_pattern").lower() == "mm/dd/yyyy"
    assert str(picker.apply_button["state"]) == tk.DISABLED
    picker.select_date(date(2026, 10, 3))
    picker.select_date(date(2026, 9, 28))
    assert (picker.draft_start, picker.draft_end) == (date(2026, 9, 28), date(2026, 10, 3))
    assert picker.status.get() == "09/28/2026 – 10/03/2026"
    assert (app.period_start, app.period_end) == original
    picker.destroy()  # window-manager Cancel equivalent
    assert (app.period_start, app.period_end) == original
    app._open_period_picker()
    picker = app._period_picker
    picker.select_date(date(2026, 12, 31))
    picker.select_date(date(2027, 1, 2))
    with patch.object(app.plotter, "_render_current_mode") as redraw:
        app.plotter.window = MagicMock()
        picker.apply_button.invoke()
        redraw.assert_called_once_with()
        app.plotter.window = None
    assert (app.period_start, app.period_end) == (date(2026, 12, 31), date(2027, 1, 2))
    assert app.range_summary_var.get() == "12/31/2026 – 01/02/2027"
    assert not picker.winfo_exists()
    app.plot_range_var.set("day")
    app.plot_range_var.set("period")
    assert app.period_start == date(2026, 12, 31)


def test_period_same_day_clicks_and_restart(headless_app):
    from datetime import date

    app = headless_app
    app.plot_range_var.set("period")
    app._open_period_picker()
    picker = app._period_picker
    calendar = picker.calendar
    # Exercise the actual day-label mouse binding twice, rather than only
    # calling the range state method. tkcalendar must report repeated clicks.
    app.update()
    target = date.today()
    label = next(label for row in calendar._calendar for label in row
                 if str(label.cget("text")) == str(target.day)
                 and "_om." not in str(label.cget("style")))
    label.event_generate("<Button-1>")
    app.update()
    assert picker.draft_start == target and picker.draft_end is None
    label.event_generate("<Button-1>")
    app.update()
    assert picker.draft_start == picker.draft_end == target
    picker.select_date(date(2026, 11, 1))
    assert picker.draft_end is None
    assert str(picker.apply_button["state"]) == tk.DISABLED
    picker.select_date(date(2026, 11, 1))
    picker.apply_button.invoke()
    assert app.period_start == app.period_end == date(2026, 11, 1)


def test_period_mode_exit_escape_and_missing_calendar(headless_app, monkeypatch):
    app = headless_app
    original = (app.period_start, app.period_end)
    app.plot_range_var.set("period")
    app._open_period_picker()
    picker = app._period_picker
    picker.select_date(app.period_start)
    picker.focus_force()
    app.update()
    picker.event_generate("<Escape>")
    app.update()
    assert not picker.winfo_exists()
    assert (app.period_start, app.period_end) == original
    app._open_period_picker()
    picker = app._period_picker
    app.plot_range_var.set("week")
    assert not picker.winfo_exists()
    monkeypatch.setattr(psc_mod, "Calendar", None)
    app.plot_range_var.set("period")
    assert str(app.range_summary_button["state"]) == tk.DISABLED
    assert app.range_summary_var.get() == "Calendar unavailable"
    assert app.range_summary_button["text"] == "Calendar unavailable"
    app._open_period_picker()
    assert app._period_picker is None


def test_plot_mode_labels_and_window_manager_close(headless_app, tmp_path, monkeypatch):
    from datetime import date, timedelta
    from tests.mock_telemetry import generate_mock_data

    app = headless_app
    monkeypatch.setattr(psc_mod, "TELEMETRY_DIR", tmp_path)
    generate_mock_data(date.today(), tmp_path)
    app.plot_var.set(1)
    app.plotter.toggle(True, lambda: app.plot_var.set(0))
    assert app.plotter.window.winfo_exists()

    for mode, title, fmt in (
        ("day", "Day", "%H:%M"),
        ("week", "Week", "%m/%d/%Y"),
        ("month", "Month", "%m/%d/%Y"),
    ):
        app.plot_range_var.set(mode)
        assert title in app.plotter.window.title()
        assert app.plotter.fig.axes[2].xaxis.get_major_formatter().fmt == fmt

    app.period_start = date.today() - timedelta(days=1)
    app.period_end = date.today()
    app.plot_range_var.set("period")
    assert "Period" in app.plotter.window.title()
    assert app.plotter.fig.axes[2].xaxis.get_major_formatter().fmt == "%m/%d/%Y"

    close_command = app.plotter.window.protocol("WM_DELETE_WINDOW")
    app.tk.call(close_command)
    assert app.plotter.window is None
    assert app.plotter.ani is None
    assert app.plot_var.get() == 0

    app.plot_var.set(1)
    app.plotter.toggle(True, lambda: app.plot_var.set(0))
    app.plot_var.set(0)
    app.plotter.toggle(False)
    assert app.plotter.window is None


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
    # Historical events use the existing 60-second safety cap, even for
    # older logs containing a longer run.
    assert ev1["volume"] == 8.33
    assert ev1["duration"] == 60.0

    assert ev2["pump"] == 1
    assert ev2["volume"] == 8.33
    assert ev2["duration"] == 60.0


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
