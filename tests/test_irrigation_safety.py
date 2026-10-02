"""Exercise complete watering runs with deterministic time and no live hardware."""

from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
import subprocess
from unittest.mock import MagicMock, patch

import pytest

import gui.psc_irr_gui as gui


class Clock:
    def __init__(self):
        self.now = 0.0
        self.jobs = {}
        self.sequence = 0

    def after(self, delay_ms, callback):
        self.sequence += 1
        job = f"test-job-{self.sequence}"
        self.jobs[job] = (self.now + delay_ms / 1000, callback)
        return job

    def cancel(self, job):
        self.jobs.pop(job, None)

    def advance(self, seconds):
        target = self.now + seconds
        while self.jobs:
            job = min(self.jobs, key=lambda key: self.jobs[key][0])
            deadline, callback = self.jobs[job]
            if deadline > target:
                break
            self.now = deadline
            del self.jobs[job]
            callback()
        self.now = target


@pytest.fixture
def watering(tmp_path, monkeypatch):
    with ExitStack() as stack:
        for name in ("Camera", "PlantAIDetector"):
            stack.enter_context(patch.object(gui, name))
        for name in ("_init_serial", "_camera_loop", "_start_periodic_loggers", "_start_scheduled_ticker"):
            stack.enter_context(patch.object(gui.MainWindow, name))
        app = gui.MainWindow()
    app.withdraw()
    app.camera.is_available = False
    app.ser = MagicMock()
    app.ser.is_open = True
    clock = Clock()
    monkeypatch.setattr(gui, "SCHEDULED_WATERING_DIR", tmp_path)
    monkeypatch.setattr(gui.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(app, "after", clock.after)
    monkeypatch.setattr(app, "after_cancel", clock.cancel)
    app.telemetry["moisture_pct"] = [35.8, 37.0, 100.0, 100.0]
    yield app, clock
    app.destroy()


def mask(app):
    return app.ser.write.call_args.args[0]


def test_recorded_sw_run_stops_channel_two_then_hard_stops_channel_one(watering):
    app, clock = watering
    app._start_scheduled_watering("2026-09-30")
    assert mask(app) == b"1100\n"
    # Values from the September 30 10:01:12 telemetry record.
    clock.advance(59)
    app.telemetry["moisture_pct"] = [37.2, 84.2, 100.0, 100.0]
    clock.advance(1)
    assert mask(app) == b"1000\n"
    assert app._scheduled_channels_active == [0]
    clock.advance(119)
    assert mask(app) == b"1000\n"
    clock.advance(1)
    assert mask(app) == b"0000\n"
    assert not app._scheduled_watering_active
    assert not clock.jobs


@pytest.mark.parametrize("duration", [1, 5, 60, 180, 900])
def test_time_mode_stops_at_duration_bounded_by_three_minutes(watering, duration):
    app, clock = watering
    app.off_method_var.set("Time")
    app.pump_duration_var.set(duration)
    app._start_scheduled_watering()
    app.telemetry["moisture_pct"] = [100.0] * 4
    deadline = min(duration, 180)
    clock.advance(deadline - 0.01)
    assert mask(app) == b"1100\n"  # SW target does not stop a Time run.
    clock.advance(0.01)
    assert mask(app) == b"0000\n"
    assert not app._scheduled_watering_active
    assert not clock.jobs


def test_exact_sw_target_and_missing_sensor_stop_individually(watering):
    app, clock = watering
    app.soil_water_stop_setpoint.set(70)
    app._start_scheduled_watering()
    app.telemetry["moisture_pct"] = [70.0, 69.9, 100.0, 100.0]
    clock.advance(2)
    assert mask(app) == b"0100\n"
    app.telemetry["moisture_pct"][1] = None
    clock.advance(2)
    assert mask(app) == b"0000\n"
    assert not clock.jobs


def test_manual_switch_cancels_run_and_prevents_old_monitor_restarting(watering):
    app, clock = watering
    app._start_scheduled_watering()
    app.auto_var.set(0)
    app._on_auto_toggle()
    assert mask(app) == b"0000\n"
    assert not clock.jobs
    app._scheduled_watering_monitor()
    clock.advance(200)
    assert mask(app) == b"0000\n"
    app._start_scheduled_watering()
    assert not app._scheduled_watering_active


def test_no_channels_at_or_above_start_setpoint_are_watered(watering):
    app, clock = watering
    app.telemetry["moisture_pct"] = [40.0, 80.0, None, 100.0]
    app._start_scheduled_watering()
    app.ser.write.assert_not_called()
    assert not clock.jobs


def test_timers_exist_before_first_on_command(watering):
    app, clock = watering
    writes = []

    def record(payload):
        writes.append(payload)
        if payload == b"1100\n":
            assert app._scheduled_safety_job in clock.jobs
            assert app._scheduled_time_job in clock.jobs
            assert app._scheduled_monitor_job in clock.jobs

    app.ser.write.side_effect = record
    app.off_method_var.set("Time")
    app._start_scheduled_watering()
    assert writes == [b"1100\n"]


def test_timer_setup_failure_never_energizes_pumps(watering, monkeypatch):
    app, clock = watering
    original_after = app.after

    def fail_monitor(delay, callback):
        if callback == app._scheduled_watering_monitor:
            raise RuntimeError("Timer setup failure")
        return original_after(delay, callback)

    monkeypatch.setattr(app, "after", fail_monitor)
    with pytest.raises(RuntimeError, match="Timer setup"):
        app._start_scheduled_watering()
    assert mask(app) == b"0000\n"
    assert all(call.args[0] == b"0000\n" for call in app.ser.write.call_args_list)
    assert not app._scheduled_watering_active
    assert not clock.jobs


def test_control_error_stops_pumps_and_camera_error_cannot_disable_safety(watering, monkeypatch):
    app, clock = watering
    app.camera.is_available = True
    app.camera.capture_array.side_effect = RuntimeError("camera disconnected")
    app._start_scheduled_watering()
    clock.advance(2)
    assert app._scheduled_watering_active
    clock.advance(178)
    assert mask(app) == b"0000\n"
    app._start_scheduled_watering()
    monkeypatch.setattr(app.soil_water_stop_setpoint, "get", MagicMock(side_effect=ValueError("invalid target")))
    clock.advance(2)
    assert mask(app) == b"0000\n"
    assert not clock.jobs


def test_safety_callback_stops_without_monitor_and_wall_clock_is_irrelevant(watering):
    app, clock = watering
    app._start_scheduled_watering()
    clock.cancel(app._scheduled_monitor_job)
    with patch.object(gui.time, "time", return_value=-1000000):
        clock.advance(180)
    assert mask(app) == b"0000\n"
    assert not app._scheduled_watering_active


def test_schedule_checks_once_at_ten_am(watering):
    app, clock = watering
    with patch.object(gui, "datetime", wraps=datetime) as date:
        date.now.return_value = datetime(2026, 9, 30, 9, 59, 59)
        app._scheduled_check_ticker()
        app.ser.write.assert_not_called()
        date.now.return_value = datetime(2026, 9, 30, 10, 0, 0)
        app._scheduled_check_ticker()
        assert mask(app) == b"1100\n"
        app._finish_scheduled_watering()
        app.ser.write.reset_mock()
        app._scheduled_check_ticker()
        app.ser.write.assert_not_called()


def test_firmware_safety_runtime_repeated_commands_and_millis_rollover(tmp_path):
    """Compile and run the actual safety code with the same 32-bit AVR clock."""
    source = tmp_path / "pump_safety.cpp"
    source.write_text(r'''
#include <cassert>
#include "PumpSafety.h"
int main() {
    PumpSafety pumps[4];
    pumps[0].request(true, 1000);
    pumps[1].request(true, 61000);
    pumps[0].request(true, 180999); // Repeated ON must preserve original start.
    assert(pumps[0].on);
    pumps[0].check(181000);
    pumps[1].check(181000);
    assert(!pumps[0].on && pumps[0].timedOut);
    assert(pumps[1].on);
    pumps[0].request(true, 181001); // Cannot restart a timed-out pump.
    assert(!pumps[0].on);
    pumps[1].request(true, 241000); // Deadline coincides with an ON command.
    assert(!pumps[1].on && pumps[1].timedOut);
    assert(!pumps[2].on && !pumps[3].on);
    pumps[0].request(false, 250000); // Explicit OFF permits a new run.
    pumps[0].request(true, 250001);
    assert(pumps[0].on && !pumps[0].timedOut);
    pumps[0].check(430001);
    assert(!pumps[0].on);
    PumpSafety rollover;
    const uint32_t start = UINT32_MAX - 1000;
    rollover.request(true, start);
    rollover.check(uint32_t(start + 179999UL));
    assert(rollover.on);
    rollover.check(uint32_t(start + 180000UL));
    assert(!rollover.on && rollover.timedOut);
}
''')
    firmware = Path(__file__).resolve().parents[1] / "firmware" / "controller"
    executable = tmp_path / "pump_safety"
    subprocess.run([
        "g++", "-std=c++11", "-Wall", "-Wextra", "-Werror",
        "-I", str(firmware), str(source), "-o", str(executable),
    ], check=True, capture_output=True, text=True)
    subprocess.run([str(executable)], check=True, capture_output=True, text=True)
