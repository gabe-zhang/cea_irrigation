"""Headless integration tests for MainWindow and PlotWindow UI logic."""

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
                with patch.object(MainWindow, "_auto_loop"):
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
    assert len(app.water_vars) == 4
    assert len(app.water_btns) == 4

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


def test_mainwindow_auto_loop_decision(headless_app):
    app = headless_app
    app.ser = MagicMock()
    app.ser.is_open = True

    app.auto_var.set(1)
    # Channel 0: 30% (< 40 -> 1)
    # Channel 1: 50% (>= 40 -> 0)
    # Channel 2: None (disconnected -> safe 0)
    # Channel 3: 20% (< 40 -> 1)
    app.telemetry["moisture_pct"] = [30.0, 50.0, None, 20.0]

    with patch.object(app, "after") as mock_after:
        app._auto_loop()
        assert [v.get() for v in app.water_vars] == [1, 0, 0, 1]
        app.ser.write.assert_called_with(b"1001\n")
        mock_after.assert_called_with(500, app._auto_loop)


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


def test_mainwindow_on_closing(headless_app):
    app = headless_app
    app.ser = MagicMock()
    app.ser.is_open = True
    app.camera = MagicMock()

    with patch("sys.exit") as mock_exit:
        app.on_closing()
        # Should shut down all pumps with 0000
        app.ser.write.assert_any_call(b"0000\n")
        # Should recenter gimbal
        app.ser.write.assert_any_call(b"c\n")
        # Should close serial
        app.ser.close.assert_called_once()
        # Should stop camera
        app.camera.stop.assert_called_once()
        # Should exit
        mock_exit.assert_called_with(0)
