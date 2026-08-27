"""Unit tests for CSV telemetry parsing and controller CLI logic."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.controller import (
    parse_telemetry_line,
    format_telemetry_compact,
    PAN_MIN,
    PAN_MAX,
    TILT_MIN,
    TILT_MAX,
)


def test_full_telemetry():
    raw = "450,430,460,440,21.5,24.2,55.0,null,0000,65,60"
    data = parse_telemetry_line(raw)
    assert data is not None, "Failed to parse full valid telemetry"
    assert data["soil"] == [450, 430, 460, 440]
    assert data["soil_temp"] == 21.5
    assert data["temp"] == 24.2
    assert data["humidity"] == 55.0
    assert data["light"] is None
    assert data["relays"] == "0000"
    assert data["pan"] == 65
    assert data["tilt"] == 60

    formatted = format_telemetry_compact(data)
    assert "21.5°C" in formatted
    assert "24.2°C" in formatted
    assert "65°" in formatted
    assert "60°" in formatted
    print("[PASS] test_full_telemetry")


def test_partial_null_telemetry():
    # Only soil1 and relays connected; soil temp, DHT, and soil 2-4 disconnected
    raw = "380,null,null,null,null,null,null,null,1000,90,45"
    data = parse_telemetry_line(raw)
    assert data is not None, "Failed to parse partial telemetry with nulls"
    assert data["soil"] == [380, None, None, None]
    assert data["soil_temp"] is None
    assert data["temp"] is None
    assert data["humidity"] is None
    assert data["light"] is None
    assert data["relays"] == "1000"
    assert data["pan"] == 90
    assert data["tilt"] == 45

    formatted = format_telemetry_compact(data)
    assert "380" in formatted
    assert "---" in formatted
    assert "N/A" in formatted
    assert "1000" in formatted
    assert "90°" in formatted
    assert "45°" in formatted
    print("[PASS] test_partial_null_telemetry")


def test_all_sensors_missing():
    # Board booted with zero sensors attached
    raw = "null,null,null,null,null,null,null,null,0000,65,60"
    data = parse_telemetry_line(raw)
    assert data is not None
    assert data["soil"] == [None, None, None, None]
    assert data["soil_temp"] is None
    assert data["temp"] is None
    assert data["humidity"] is None
    assert data["relays"] == "0000"
    assert data["pan"] == 65
    assert data["tilt"] == 60

    formatted = format_telemetry_compact(data)
    assert "N/A" in formatted
    print("[PASS] test_all_sensors_missing")


def test_ignore_headers_and_acks():
    # Header should return None (not treated as telemetry frame)
    header = "soil1,soil2,soil3,soil4,soil_temp,temp,humidity,light,relays,pan,tilt"
    assert parse_telemetry_line(header) is None

    # ACKs and ERRs should return None
    assert parse_telemetry_line("ACK: Pan rotated to 65 deg") is None
    assert parse_telemetry_line("ACK: Relays set to 0000") is None
    assert parse_telemetry_line("ERR: Unknown command 'foo'") is None

    # Incomplete or corrupt lines
    assert parse_telemetry_line("") is None
    assert parse_telemetry_line("450,430,null") is None
    assert parse_telemetry_line("corrupt,data,here") is None
    print("[PASS] test_ignore_headers_and_acks")


def test_bounds():
    assert PAN_MIN == 0 and PAN_MAX == 130
    assert TILT_MIN == 0 and TILT_MAX == 90
    print("[PASS] test_bounds")


def test_tagged_csv_full():
    raw = "soil,450,430,460,440,soil_temp,21.5,temp,24.2,humi,55.0,light,null,relays,0000,pan,65,tilt,60"
    data = parse_telemetry_line(raw)
    assert data is not None, "Failed to parse full tagged telemetry"
    assert data["soil"] == [450, 430, 460, 440]
    assert data["soil_temp"] == 21.5
    assert data["temp"] == 24.2
    assert data["humidity"] == 55.0
    assert data["light"] is None
    assert data["relays"] == "0000"
    assert data["pan"] == 65
    assert data["tilt"] == 60

    formatted = format_telemetry_compact(data)
    assert "21.5°C" in formatted
    assert "24.2°C" in formatted
    assert "65°" in formatted
    assert "60°" in formatted
    print("[PASS] test_tagged_csv_full")


def test_tagged_csv_partial_null():
    raw = "soil,380,null,null,null,soil_temp,null,temp,null,humi,null,light,null,relays,1000,pan,90,tilt,45"
    data = parse_telemetry_line(raw)
    assert data is not None, "Failed to parse partial tagged telemetry"
    assert data["soil"] == [380, None, None, None]
    assert data["soil_temp"] is None
    assert data["temp"] is None
    assert data["humidity"] is None
    assert data["light"] is None
    assert data["relays"] == "1000"
    assert data["pan"] == 90
    assert data["tilt"] == 45

    formatted = format_telemetry_compact(data)
    assert "380" in formatted
    assert "---" in formatted
    assert "N/A" in formatted
    assert "1000" in formatted
    assert "90°" in formatted
    assert "45°" in formatted
    print("[PASS] test_tagged_csv_partial_null")


def test_tagged_csv_all_null():
    raw = "soil,null,null,null,null,soil_temp,null,temp,null,humi,null,light,null,relays,0000,pan,65,tilt,60"
    data = parse_telemetry_line(raw)
    assert data is not None
    assert data["soil"] == [None, None, None, None]
    assert data["soil_temp"] is None
    assert data["temp"] is None
    assert data["humidity"] is None
    assert data["relays"] == "0000"
    assert data["pan"] == 65
    assert data["tilt"] == 60
    print("[PASS] test_tagged_csv_all_null")


def test_tagged_csv_with_light():
    raw = "soil,450,430,460,440,soil_temp,21.5,temp,24.2,humi,55.0,light,350.5,relays,0000,pan,65,tilt,60"
    data = parse_telemetry_line(raw)
    assert data is not None, "Failed to parse tagged telemetry with light"
    assert data["soil"] == [450, 430, 460, 440]
    assert data["soil_temp"] == 21.5
    assert data["temp"] == 24.2
    assert data["humidity"] == 55.0
    assert data["light"] == 350
    assert data["relays"] == "0000"
    assert data["pan"] == 65
    assert data["tilt"] == 60

    formatted = format_telemetry_compact(data)
    assert "Light: 350" in formatted
    print("[PASS] test_tagged_csv_with_light")


if __name__ == "__main__":
    test_tagged_csv_full()
    test_tagged_csv_with_light()
    test_tagged_csv_partial_null()
    test_tagged_csv_all_null()
    test_full_telemetry()
    test_partial_null_telemetry()
    test_all_sensors_missing()
    test_ignore_headers_and_acks()
    test_bounds()
    print("\nAll 9 test suites passed successfully!")

