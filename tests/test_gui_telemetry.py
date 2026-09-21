"""Unit tests for GUI telemetry parsing, calibration, serial commands, and core logic."""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from gui.psc_irr_gui import (
    BAUDRATE,
    DATA_DIR,
    DRY_BASELINES,
    GIMBAL_STEP,
    PAN_HOME,
    PAN_MAX,
    PAN_MIN,
    SOIL_WATER_SETPOINT,
    TILT_HOME,
    TILT_MAX,
    TILT_MIN,
    WET_BASELINES,
    _safe_float,
    _safe_int,
    find_arduino_port,
    parse_telemetry_line,
    raw_to_moisture,
)
from scripts.controller import format_telemetry_compact



# --- Helper Conversion Tests ---

def test_safe_float():
    assert _safe_float("23.5") == 23.5
    assert _safe_float("-4.2") == -4.2
    assert _safe_float("0") == 0.0
    assert _safe_float("null") is None
    assert _safe_float("none") is None
    assert _safe_float("nan") is None
    assert _safe_float("") is None
    assert _safe_float("abc") is None


def test_safe_int():
    assert _safe_int("450") == 450
    assert _safe_int("0") == 0
    assert _safe_int("350.5") == 350
    assert _safe_int("null") is None
    assert _safe_int("none") is None
    assert _safe_int("nan") is None
    assert _safe_int("") is None
    assert _safe_int("corrupt") is None


# --- Telemetry Parsing Tests ---

def test_parse_tagged_full():
    raw = "soil,450,430,460,440,soil_temp,21.5,temp,24.2,humi,55.0,light,350,relays,0000,pan,65,tilt,60"
    data = parse_telemetry_line(raw)
    assert data is not None
    assert data["soil"] == [450, 430, 460, 440]
    assert data["soil_temp"] == 21.5
    assert data["temp"] == 24.2
    assert data["humidity"] == 55.0
    assert data["light"] == 350
    assert data["relays"] == "0000"
    assert data["pan"] == 65
    assert data["tilt"] == 60


def test_parse_tagged_partial_null():
    raw = "soil,380,null,null,null,soil_temp,null,temp,null,humi,null,light,null,relays,1000,pan,90,tilt,45"
    data = parse_telemetry_line(raw)
    assert data is not None
    assert data["soil"] == [380, None, None, None]
    assert data["soil_temp"] is None
    assert data["temp"] is None
    assert data["humidity"] is None
    assert data["light"] is None
    assert data["relays"] == "1000"
    assert data["pan"] == 90
    assert data["tilt"] == 45


def test_parse_tagged_all_null():
    raw = "soil,null,null,null,null,soil_temp,null,temp,null,humi,null,light,null,relays,0000,pan,65,tilt,60"
    data = parse_telemetry_line(raw)
    assert data is not None
    assert data["soil"] == [None, None, None, None]
    assert data["soil_temp"] is None
    assert data["relays"] == "0000"


def test_parse_tagged_short_soil():
    # Only 2 soil values provided before next tag
    raw = "soil,410,420,soil_temp,22.0,relays,0011"
    data = parse_telemetry_line(raw)
    assert data is not None
    assert data["soil"] == [410, 420, None, None]
    assert data["soil_temp"] == 22.0
    assert data["relays"] == "0011"


def test_parse_ignore_system_and_invalid():
    assert parse_telemetry_line("") is None
    assert parse_telemetry_line("   ") is None
    assert parse_telemetry_line("soil1,soil2,soil3,soil4,soil_temp,temp,humidity,light,relays,pan,tilt") is None
    assert parse_telemetry_line("ACK: Pan rotated to 65 deg") is None
    assert parse_telemetry_line("ERR: Unknown command 'x'") is None
    assert parse_telemetry_line("STATUS: Controller booted") is None
    assert parse_telemetry_line("random junk text without soil") is None


# --- Soil Moisture Calibration Tests ---

def test_raw_to_moisture_channels():
    # Dry baseline reading should be ~0.0%
    for ch, base in enumerate(DRY_BASELINES):
        val = raw_to_moisture(base, ch)
        assert pytest.approx(val, 0.1) == 0.0

    # Wet baseline reading should be ~100.0%
    for ch, base in enumerate(WET_BASELINES):
        val = raw_to_moisture(base, ch)
        assert pytest.approx(val, 0.1) == 100.0

    # Halfway (midpoint between dry and wet) should be ~50.0%
    mid_ch0 = (DRY_BASELINES[0] + WET_BASELINES[0]) / 2.0
    assert pytest.approx(raw_to_moisture(mid_ch0, 0), 0.1) == 50.0

    # Out of bounds clamping
    assert raw_to_moisture(0.0, 0) == 100.0  # wetter than wet baseline clamped to 100%
    assert raw_to_moisture(550.0, 0) == 0.0  # drier than dry baseline clamped to 0%

    # Disconnected sensor (None) returns None
    assert raw_to_moisture(None, 0) is None
    assert raw_to_moisture(None, 3) is None

    # Channel beyond baseline lists uses average baselines
    avg_dry = sum(DRY_BASELINES) / len(DRY_BASELINES)
    avg_wet = sum(WET_BASELINES) / len(WET_BASELINES)
    assert pytest.approx(raw_to_moisture(avg_dry, 5), 0.1) == 0.0
    assert pytest.approx(raw_to_moisture(avg_wet, 5), 0.1) == 100.0


# --- Telemetry Format Tests ---

def test_format_telemetry_compact():
    data = {
        "soil": [450, 430, 460, 440],
        "soil_temp": 21.5,
        "temp": 24.2,
        "humidity": 55.0,
        "light": 350,
        "relays": "0000",
        "pan": 65,
        "tilt": 60,
    }
    summary = format_telemetry_compact(data)
    assert "450" in summary
    assert "21.5°C" in summary
    assert "24.2°C" in summary
    assert "55.0%" in summary
    assert "350" in summary
    assert "0000" in summary
    assert "65°" in summary
    assert "60°" in summary

    # Null / Missing values
    partial_data = {
        "soil": [380, None, None, None],
        "soil_temp": None,
        "temp": None,
        "humidity": None,
        "light": None,
        "relays": "1000",
        "pan": None,
        "tilt": None,
    }
    partial_summary = format_telemetry_compact(partial_data)
    assert "380" in partial_summary
    assert "---" in partial_summary
    assert "N/A" in partial_summary


# --- Serial Port Detection Tests ---

def test_find_arduino_port_vid():
    mock_port = MagicMock()
    mock_port.vid = 0x2341
    mock_port.device = "/dev/ttyACM0"
    mock_port.description = "Arduino Uno"

    with patch("serial.tools.list_ports.comports", return_value=[mock_port]):
        assert find_arduino_port() == "/dev/ttyACM0"


def test_find_arduino_port_keyword():
    mock_port = MagicMock()
    mock_port.vid = None
    mock_port.device = "/dev/ttyUSB0"
    mock_port.description = "USB-Serial CH340"

    with patch("serial.tools.list_ports.comports", return_value=[mock_port]):
        assert find_arduino_port() == "/dev/ttyUSB0"


def test_find_arduino_port_none():
    with patch("serial.tools.list_ports.comports", return_value=[]):
        assert find_arduino_port() is None


# --- CSV Logging Tests ---

def test_log_telemetry_csv(tmp_path, monkeypatch):
    import gui.psc_irr_gui as psc_mod
    monkeypatch.setattr(psc_mod, "TELEMETRY_DIR", tmp_path / "telemetry")
    (tmp_path / "telemetry").mkdir(parents=True, exist_ok=True)

    # Use MainWindow instance method or helper
    with patch("gui.psc_irr_gui.Camera"):
        with patch.object(psc_mod.MainWindow, "_init_serial"):
            with patch.object(psc_mod.MainWindow, "_camera_loop"):
                app = psc_mod.MainWindow()
                app.withdraw()

                data = {
                    "soil": [450, 430, None, None],
                    "soil_temp": 21.5,
                    "temp": 24.2,
                    "humidity": 55.0,
                    "light": 100,
                    "relays": "1000",
                    "pan": 65,
                    "tilt": 60,
                }
                moist = [15.2, 22.4, None, None]

                # First log creates file and header
                app._log_telemetry_csv(data, moist)

                today_str = datetime.now().strftime("%Y%m%d")
                csv_file = tmp_path / "telemetry" / f"telemetry_{today_str}.csv"
                assert csv_file.exists()

                lines = csv_file.read_text(encoding="utf-8").strip().split("\n")
                assert len(lines) == 2
                assert lines[0].startswith("timestamp,soil1_raw")
                assert "450,430,null,null,15.2,22.4,null,null,21.5,24.2,55.0,100,1000,65,60" in lines[1]

                # Second log appends row without re-writing header
                app._log_telemetry_csv(data, moist)
                lines_after = csv_file.read_text(encoding="utf-8").strip().split("\n")
                assert len(lines_after) == 3

                app.destroy()


# --- Gimbal and Safety Bounds Tests ---

def test_constants_and_bounds():
    assert PAN_MIN == 0 and PAN_MAX == 130
    assert TILT_MIN == 0 and TILT_MAX == 60
    assert PAN_HOME == 55 and TILT_HOME == 30
    assert GIMBAL_STEP == 5
    assert BAUDRATE == 9600
    assert SOIL_WATER_SETPOINT == 40.0


def test_gimbal_nudging():
    import gui.psc_irr_gui as psc_mod
    with patch("gui.psc_irr_gui.Camera"):
        with patch.object(psc_mod.MainWindow, "_init_serial"):
            with patch.object(psc_mod.MainWindow, "_camera_loop"):
                app = psc_mod.MainWindow()
                app.withdraw()
                app.ser = MagicMock()
                app.ser.is_open = True

                # Pan limits
                app._last_servo_cmd = 0.0
                app.current_pan = 129
                app.nudge_pan(5)  # should cap at PAN_MAX (130)
                assert app.current_pan == 130
                app.ser.write.assert_called_with(b"p 130\n")

                app._last_servo_cmd = 0.0
                app.current_pan = 1
                app.nudge_pan(-5)  # should cap at PAN_MIN (0)
                assert app.current_pan == 0
                app.ser.write.assert_called_with(b"p 0\n")

                # Tilt limits
                app._last_servo_cmd = 0.0
                app.current_tilt = 58
                app.nudge_tilt(10)  # should cap at TILT_MAX (60)
                assert app.current_tilt == 60
                app.ser.write.assert_called_with(b"t 60\n")

                # Home
                app._last_servo_cmd = 0.0
                app.home_gimbal()
                assert app.current_pan == PAN_HOME
                assert app.current_tilt == TILT_HOME
                app.ser.write.assert_called_with(b"h\n")

                app.destroy()

