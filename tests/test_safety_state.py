"""Persistent lock, controller synchronization, and explicit recovery contracts."""

import json
from unittest.mock import MagicMock

import pytest

from scripts.safety_state import SafetyState, safety_metadata
from scripts.controller import send_command, parse_telemetry_line as cli_parse
from gui.psc_irr_gui import parse_telemetry_line as gui_parse


def status(locked=0, generation=0, source=0, channels=0):
    return {"safety_version": "1", "safety_locked": str(locked),
            "safety_generation": str(generation), "safety_source": str(source),
            "safety_channels": str(channels), "relays": "0000"}


def test_startup_requires_supported_status(tmp_path):
    state = SafetyState(tmp_path / "fault.json")
    assert not state.can_water
    assert state.observe({"relays": "0000"}) == "invalid"
    assert not state.can_water
    assert state.observe(status()) == "ready"
    assert state.can_water


@pytest.mark.parametrize("tag,value", [
    ("safety_version", "2"), ("safety_locked", "2"), ("safety_generation", "-1"),
    ("safety_generation", "4294967296"), ("safety_generation", "1.5"),
    ("safety_source", "4"), ("safety_channels", "16"), ("safety_locked", "nan"),
])
def test_invalid_metadata_blocks_watering(tmp_path, tag, value):
    state = SafetyState(tmp_path / "fault.json")
    state.observe(status())
    data = {**status(), tag: value}
    assert safety_metadata(data) is None
    state.observe(data)
    assert not state.can_water


def test_host_trip_persists_and_off_status_cannot_clear_it(tmp_path):
    path = tmp_path / "fault.json"
    state = SafetyState(path)
    state.observe(status())
    state.trip({"reason": "safety_timeout", "channel_mask": 3})
    restored = SafetyState(path)
    assert restored.locked
    assert restored.observe(status()) == "trip"
    assert restored.trip_command() == "safety trip 3"
    assert restored.resolve_command() is None  # Must acknowledge the trip first.
    assert not restored.can_water
    restored.observe(status(1, 1, 2, 3))
    assert restored.resolve_command() == "safety resolve 1"
    assert restored.locked
    restored.observe(status(1, 1, 2, 3))
    assert restored.locked
    assert restored.observe(status(0, 2)) == "resolved"
    assert restored.can_water
    assert not SafetyState(path).locked


def test_firmware_only_trip_and_cli_reset_survive_app_restart(tmp_path):
    path = tmp_path / "fault.json"
    state = SafetyState(path)
    assert state.observe(status(1, 9, 1, 4)) == "locked"
    assert state.state["context"]["reason"] == "firmware_timeout"
    assert not state.can_water
    reopened = SafetyState(path)
    assert reopened.observe(status(0, 10)) == "resolved"
    assert reopened.can_water


def test_stale_reset_does_not_clear_new_trip(tmp_path):
    state = SafetyState(tmp_path / "fault.json")
    state.observe(status(1, 3, 1, 1))
    assert state.resolve_command() == "safety resolve 3"
    state.observe(status(1, 5, 1, 2))
    assert state.observe(status(0, 4)) == "stale"
    assert state.locked and not state.can_water
    assert state.resolve_command() is None


def test_generation_rollover(tmp_path):
    state = SafetyState(tmp_path / "fault.json")
    state.observe(status(1, 0xffffffff, 1, 1))
    assert state.observe(status(0, 0)) == "resolved"
    assert state.can_water


def test_controller_storage_recovery_with_lost_generation_is_resolvable(tmp_path):
    state = SafetyState(tmp_path / "fault.json")
    state.observe(status(1, 99, 1, 1))
    assert state.observe(status(1, 1, 3, 0)) == "locked"
    assert state.resolve_command() == "safety resolve 1"
    assert state.observe(status(0, 2)) == "resolved"
    assert state.can_water


def test_reset_unavailable_when_disconnected_or_stale(tmp_path, monkeypatch):
    state = SafetyState(tmp_path / "fault.json")
    state.observe(status(1, 1, 1, 1))
    monkeypatch.setattr("scripts.safety_state.time.monotonic", lambda: state.last_status + 5)
    assert state.resolve_command() is None
    state.disconnect()
    assert state.locked and not state.can_water


@pytest.mark.parametrize("content", ["{bad", "[]", '{"version":1}',
    '{"version":1,"locked":false,"pending_trip":true,"generation":0,"context":{}}'])
def test_corrupt_host_record_inhibits_pumps(tmp_path, content):
    path = tmp_path / "fault.json"
    path.write_text(content)
    state = SafetyState(path)
    assert state.locked
    assert state.observe(status()) == "trip"
    assert not state.can_water


def test_stop_precedes_disk_writes_and_save_failure_stays_locked(tmp_path, monkeypatch):
    state = SafetyState(tmp_path / "fault.json")
    events = []
    original_save = state._save

    def save():
        events.append("save")
        return original_save()

    monkeypatch.setattr(state, "_save", save)
    state.trip({"reason": "safety_timeout"}, stop=lambda: events.append("stop"))
    assert events == ["stop", "save"]
    state.observe(status(1, 1, 2, 1))
    monkeypatch.setattr("scripts.safety_state.os.replace", MagicMock(side_effect=OSError("disk full")))
    assert state.observe(status(0, 2)) == "storage_failed"
    assert state.locked and not state.can_water
    assert json.loads(state.path.read_text())["locked"]


def test_cli_blocks_on_but_allows_off_and_servos_while_locked(tmp_path):
    state = SafetyState(tmp_path / "fault.json")
    state.observe(status(1, 1, 1, 1))
    serial = MagicMock()
    for command in ("1", "01", "1111"):
        assert not send_command(serial, command, state)
    serial.write.assert_not_called()
    assert send_command(serial, "0000", state)
    assert send_command(serial, "h", state)


@pytest.mark.parametrize("parser", [cli_parse, gui_parse])
def test_both_parsers_preserve_complete_safety_status(parser):
    data = parser("soil,432,408,427,424,relays,0000,pan,55,tilt,30,"
                  "safety_version,1,safety_locked,1,safety_generation,7,safety_source,1,safety_channels,3")
    assert safety_metadata(data)["safety_generation"] == 7
    assert data["soil"] == [432, 408, 427, 424]
