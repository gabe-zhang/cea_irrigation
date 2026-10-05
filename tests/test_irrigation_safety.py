"""Exercise complete watering runs with deterministic time and no live hardware."""

from contextlib import ExitStack
import csv
import logging
from pathlib import Path
import subprocess
from unittest.mock import MagicMock, patch

import numpy as np
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
        for name in ("_init_serial", "_camera_loop", "_start_periodic_loggers", "_start_auto_ticker"):
            stack.enter_context(patch.object(gui.MainWindow, name))
        app = gui.MainWindow()
    app.withdraw()
    app.camera.is_available = False
    app.ser = MagicMock()
    app.ser.is_open = True
    clock = Clock()
    monkeypatch.setattr(gui, "AUTO_WATERING_DIR", tmp_path / "images")
    monkeypatch.setattr(gui, "TELEMETRY_DIR", tmp_path)
    monkeypatch.setattr(gui.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(app, "after", clock.after)
    monkeypatch.setattr(app, "after_cancel", clock.cancel)
    app.telemetry["moisture_pct"] = [35.8, 37.0, 100.0, 100.0]
    app._below_threshold_counts = [3, 3, 0, 0]
    yield app, clock
    app.destroy()


def mask(app):
    return app.ser.write.call_args.args[0]


def test_watering_logs_explain_run_and_safety_stop(watering, caplog):
    app, clock = watering
    caplog.set_level(logging.INFO, logger="cea_irrigation")
    app._start_auto_watering()
    clock.advance(gui.MAX_WATERING_SEC)
    events = {getattr(record, "event", None): record for record in caplog.records}
    started = events["watering.auto_started"].context
    stopped = events["watering.auto_stopped"].context
    assert started["channels"] == [1, 2]
    assert started["moisture_pct"][:2] == [35.8, 37.0]
    assert started["run_id"] == stopped["run_id"]
    assert stopped["reason"] == "safety_timeout"
    assert stopped["elapsed_sec"] == gui.MAX_WATERING_SEC
    assert events["watering.safety_stop"].levelno == logging.WARNING
    assert mask(app) == b"0000\n"


def test_pump_stop_write_failure_is_critical_with_traceback(watering, caplog):
    app, _ = watering
    app.ser.write.side_effect = OSError("device unplugged")
    app.send_bitmask("0000")
    failure = next(record for record in caplog.records if getattr(record, "event", None) == "serial.write_failed")
    assert failure.levelno == logging.CRITICAL
    assert failure.context["command"] == "0000"
    assert isinstance(failure.exc_info[1], OSError)


def test_image_false_return_is_logged_as_failure(watering, monkeypatch, tmp_path, caplog):
    app, _ = watering
    monkeypatch.setattr(gui.cv2, "imwrite", lambda *_: False)
    path = tmp_path / "snapshot.jpg"
    app._save_image(path, np.zeros((4, 4, 3), dtype=np.uint8))
    failure = next(record for record in caplog.records if getattr(record, "event", None) == "image.write_failed")
    assert failure.context["path"] == path
    assert not path.exists()


def test_recorded_sw_run_stops_channel_two_then_hard_stops_channel_one(watering):
    app, clock = watering
    app._start_auto_watering()
    assert mask(app) == b"1100\n"
    # Values from the September 30 10:01:12 telemetry record.
    clock.advance(59)
    receive(app, [37.2, 84.2, 100.0, 100.0])
    assert mask(app) == b"1000\n"
    assert app._auto_channels_active == [0]
    clock.advance(0.99)
    assert mask(app) == b"1000\n"
    clock.advance(0.01)
    assert mask(app) == b"0000\n"
    assert not app._auto_watering_active
    assert not clock.jobs


@pytest.mark.parametrize("duration", [1, 5, 60, 180, 900])
def test_time_mode_stops_at_duration_bounded_by_safety_limit(watering, duration):
    app, clock = watering
    app.off_method_var.set("Time")
    app.pump_duration_var.set(duration)
    app._start_auto_watering()
    receive(app, [100.0] * 4)
    deadline = min(duration, gui.MAX_WATERING_SEC)
    clock.advance(deadline - 0.01)
    assert mask(app) == b"1100\n"  # SW target does not stop a Time run.
    clock.advance(0.01)
    assert mask(app) == b"0000\n"
    assert not app._auto_watering_active
    assert not clock.jobs


def test_exact_sw_target_and_missing_sensor_stop_individually(watering):
    app, clock = watering
    app.soil_water_stop_setpoint.set(70)
    app._start_auto_watering()
    receive(app, [70.0, 69.9, 100.0, 100.0])
    assert mask(app) == b"0100\n"
    receive(app, [70.0, None, 100.0, 100.0])
    assert mask(app) == b"0000\n"
    assert not clock.jobs


def test_manual_switch_cancels_run_and_prevents_old_monitor_restarting(watering):
    app, clock = watering
    app._start_auto_watering()
    app.auto_var.set(0)
    app._on_auto_toggle()
    assert mask(app) == b"0000\n"
    assert not clock.jobs
    app._auto_watering_monitor()
    clock.advance(200)
    assert mask(app) == b"0000\n"
    app._start_auto_watering()
    assert not app._auto_watering_active


def test_no_channels_at_or_above_start_setpoint_are_watered(watering):
    app, clock = watering
    app.telemetry["moisture_pct"] = [40.0, 80.0, None, 100.0]
    app._start_auto_watering()
    app.ser.write.assert_not_called()
    assert not clock.jobs


def test_timers_exist_before_first_on_command(watering):
    app, clock = watering
    writes = []

    def record(payload):
        writes.append(payload)
        if payload == b"1100\n":
            assert app._auto_safety_job in clock.jobs
            assert app._auto_time_job in clock.jobs
            assert app._auto_monitor_job in clock.jobs

    app.ser.write.side_effect = record
    app.off_method_var.set("Time")
    app._start_auto_watering()
    assert writes == [b"1100\n"]


def test_timer_setup_failure_never_energizes_pumps(watering, monkeypatch):
    app, clock = watering
    original_after = app.after

    def fail_monitor(delay, callback):
        if callback == app._auto_watering_monitor:
            raise RuntimeError("Timer setup failure")
        return original_after(delay, callback)

    monkeypatch.setattr(app, "after", fail_monitor)
    with pytest.raises(RuntimeError, match="Timer setup"):
        app._start_auto_watering()
    assert mask(app) == b"0000\n"
    assert all(call.args[0] == b"0000\n" for call in app.ser.write.call_args_list)
    assert not app._auto_watering_active
    assert not clock.jobs


def test_control_error_stops_pumps_and_camera_error_cannot_disable_safety(watering, monkeypatch):
    app, clock = watering
    app.camera.is_available = True
    app.camera.capture_array.side_effect = RuntimeError("camera disconnected")
    app._start_auto_watering()
    receive(app, [35.8, 37.0, 100.0, 100.0])
    receive(app, [35.8, 37.0, 100.0, 100.0])
    assert app._auto_watering_active
    app.camera.capture_array.assert_called_once()
    clock.advance(gui.MAX_WATERING_SEC)
    assert mask(app) == b"0000\n"
    app._below_threshold_counts = [3, 3, 0, 0]
    app._start_auto_watering()
    monkeypatch.setattr(app.soil_water_stop_setpoint, "get", MagicMock(side_effect=ValueError("invalid target")))
    clock.advance(2)
    assert mask(app) == b"0000\n"
    assert not clock.jobs


def test_safety_callback_stops_without_monitor_and_wall_clock_is_irrelevant(watering):
    app, clock = watering
    app._start_auto_watering()
    clock.cancel(app._auto_monitor_job)
    with patch.object(gui.time, "time", return_value=-1000000):
        clock.advance(gui.MAX_WATERING_SEC)
    assert mask(app) == b"0000\n"
    assert not app._auto_watering_active


def receive(app, moist):
    data = {**app.telemetry, "relays": "".join(str(var.get()) for var in app.water_vars)}
    app._receive_telemetry(data, moist)


def start_ticker(app):
    app._below_threshold_counts = [0] * len(app.water_vars)
    app._start_auto_ticker()


def check(app, clock, moist):
    receive(app, moist)
    clock.advance(2)


def test_three_fresh_checks_start_per_channel_at_any_time(watering):
    app, clock = watering
    start_ticker(app)
    check(app, clock, [30, 40, None, 100])
    check(app, clock, [30, 39, None, 100])
    app.ser.write.assert_not_called()
    check(app, clock, [30, 39, None, 100])
    assert mask(app) == b"1000\n"
    assert clock.now == 6
    assert app._auto_channels_active == [0]


@pytest.mark.parametrize("interrupt", [40, 50, None])
def test_wet_or_missing_frame_between_checks_breaks_streak(watering, interrupt):
    app, clock = watering
    start_ticker(app)
    check(app, clock, [30, 100, 100, 100])
    check(app, clock, [30, 100, 100, 100])
    receive(app, [interrupt, 100, 100, 100])
    check(app, clock, [30, 100, 100, 100])
    check(app, clock, [30, 100, 100, 100])
    app.ser.write.assert_not_called()
    check(app, clock, [30, 100, 100, 100])
    assert mask(app) == b"1000\n"


def test_repeated_checks_do_not_count_cached_telemetry(watering):
    app, clock = watering
    start_ticker(app)
    check(app, clock, [30, 100, 100, 100])
    clock.advance(20)
    app.ser.write.assert_not_called()
    assert app._below_threshold_counts == [0] * 4
    check(app, clock, [30, 100, 100, 100])
    check(app, clock, [30, 100, 100, 100])
    app.ser.write.assert_not_called()
    check(app, clock, [30, 100, 100, 100])
    assert mask(app) == b"1000\n"


def test_manual_readings_do_not_arm_auto_start(watering):
    app, clock = watering
    start_ticker(app)
    app.auto_var.set(0)
    for _ in range(3):
        check(app, clock, [30, 100, 100, 100])
    app.ser.write.assert_not_called()
    app.auto_var.set(1)
    for _ in range(2):
        check(app, clock, [30, 100, 100, 100])
    app.ser.write.assert_not_called()
    check(app, clock, [30, 100, 100, 100])
    assert mask(app) == b"1000\n"


def test_new_run_requires_three_new_checks_without_cooldown(watering):
    app, clock = watering
    app._start_auto_watering()
    app._start_auto_ticker()
    receive(app, [80, 80, 100, 100])
    assert not app._auto_watering_active
    assert mask(app) == b"0000\n"
    for _ in range(2):
        check(app, clock, [30, 100, 100, 100])
        assert not app._auto_watering_active
    check(app, clock, [30, 100, 100, 100])
    assert mask(app) == b"1000\n"


def test_safety_stop_requires_new_readings_before_another_run(watering):
    app, clock = watering
    app._start_auto_watering()
    app._start_auto_ticker()
    clock.advance(gui.MAX_WATERING_SEC)
    assert mask(app) == b"0000\n"
    clock.advance(10)
    assert not app._auto_watering_active
    for _ in range(3):
        check(app, clock, [30, 100, 100, 100])
    assert mask(app) == b"1000\n"


def test_records_every_second_frame_and_stop_precedes_capture(watering):
    app, clock = watering
    app.camera.is_available = True
    app.camera.capture_array.return_value = np.zeros((8, 8, 3), dtype=np.uint8)
    app.data_record_var.set("1hr")
    app.image_record_var.set("1hr")
    app._start_auto_watering()
    csv_path = next(gui.TELEMETRY_DIR.glob("telemetry_*.csv"))

    def rows():
        with csv_path.open() as file:
            return list(csv.DictReader(file))

    assert rows()[0]["relays"] == "1100"  # Start boundary.
    for n in range(1, 5):
        receive(app, [35 + n, 37, 100, 100])
        assert len(rows()) == 1 + n // 2
        assert len(list(app._auto_run_dir.glob("IMG_*.jpg"))) == n // 2
    assert rows()[-1]["soil1_pct"] == "39.0"
    # Ordinary logging does not duplicate the watering records.
    app._periodic_data_logger()
    app._periodic_image_logger()
    assert len(rows()) == 3
    assert app.camera.capture_array.call_count == 2
    receive(app, [39, 37, 100, 100])  # Fifth frame: no recording.

    def capture_after_stop():
        assert mask(app) == b"0000\n"
        return np.zeros((8, 8, 3), dtype=np.uint8)

    app.camera.capture_array.side_effect = capture_after_stop
    receive(app, [80, 80, 100, 100])  # Sixth frame: immediate stop then recording.
    assert len(rows()) == 4
    assert rows()[-1]["relays"] == "0000"
    assert len(list(app._auto_run_dir.glob("IMG_*.jpg"))) == 3
    receive(app, [80, 80, 100, 100])
    assert len(rows()) == 4
    assert app.camera.capture_array.call_count == 3
    # Restore ordinary recording once watering has finished.
    app.camera.is_available = False
    app._periodic_data_logger()
    assert len(rows()) == 5


def test_serial_reader_queues_each_frame_for_tk_processing(watering, monkeypatch):
    app, clock = watering
    frames = (
        b"ACK: ready\n"
        b"soil,400,400,400,400,relays,0000\n"
        b"soil,432,408,427,424,relays,0000\n"
    )
    app.ser.in_waiting = len(frames)
    app.ser.read.return_value = frames
    queued = []

    def enqueue(delay, callback):
        assert delay == 0
        queued.append(callback)
        job = clock.after(delay, callback)
        if len(queued) == 2:
            app.stop_threads.set()
        return job

    monkeypatch.setattr(app, "after", enqueue)
    app._serial_reader()
    assert len(queued) == 2  # ACK does not count as a reading.
    assert app._telemetry_sequence == 0  # Reader does not mutate Tk-owned state.
    app.stop_threads.clear()
    clock.advance(0)
    assert app._telemetry_sequence == 2
    assert app.telemetry["soil"] == [432, 408, 427, 424]
    assert app.telemetry["moisture_pct"] == [0.0] * 4


def test_firmware_safety_runtime_repeated_commands_and_millis_rollover(tmp_path):
    """Compile and run the actual safety code with the same 32-bit AVR clock."""
    source = tmp_path / "pump_safety.cpp"
    source.write_text(r'''
#include <cassert>
#include "PumpSafety.h"
int main() {
    PumpSafety pumps[4];
    pumps[0].request(true, 1000);
    pumps[1].request(true, 21000);
    pumps[0].request(true, 60999); // Repeated ON must preserve original start.
    assert(pumps[0].on);
    pumps[0].check(61000);
    pumps[1].check(61000);
    assert(!pumps[0].on && pumps[0].timedOut);
    assert(pumps[1].on);
    pumps[0].request(true, 61001); // Cannot restart a timed-out pump.
    assert(!pumps[0].on);
    pumps[1].request(true, 81000); // Deadline coincides with an ON command.
    assert(!pumps[1].on && pumps[1].timedOut);
    assert(!pumps[2].on && !pumps[3].on);
    pumps[0].request(false, 90000); // Explicit OFF permits a new run.
    pumps[0].request(true, 90001);
    assert(pumps[0].on && !pumps[0].timedOut);
    pumps[0].check(150001);
    assert(!pumps[0].on);
    PumpSafety rollover;
    const uint32_t start = UINT32_MAX - 1000;
    rollover.request(true, start);
    rollover.check(uint32_t(start + 59999UL));
    assert(rollover.on);
    rollover.check(uint32_t(start + 60000UL));
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
