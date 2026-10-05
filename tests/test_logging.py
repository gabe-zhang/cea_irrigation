"""Verify the diagnostic contract, retention, and failure isolation."""

from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import threading
from unittest.mock import MagicMock

import pytest

from scripts import logging_config as config

logger = logging.getLogger("cea_irrigation.test")


@pytest.fixture(autouse=True)
def reset_logging():
    config.shutdown_logging()
    yield
    config.shutdown_logging()


def records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_structured_exception_context_and_idempotence(tmp_path, capsys):
    root = logging.getLogger()
    root_state = root.level, list(root.handlers)
    path = config.configure_logging("gui", log_dir=tmp_path)
    assert config.configure_logging("gui", log_dir=tmp_path) == path
    try:
        raise RuntimeError("sensor disconnected")
    except RuntimeError:
        logger.exception("Read failed", extra={"event": "sensor.failed", "context": {
            "channel": 2, "path": Path("sensor.csv"), "values": [float("nan"), float("inf")],
        }})
    entries = records(path)
    assert len(entries) == 2
    entry = entries[-1]
    assert entry["event"] == "sensor.failed"
    assert entry["level"] == "ERROR"
    assert entry["context"] == {"channel": 2, "path": "sensor.csv", "values": [None, None]}
    assert "RuntimeError: sensor disconnected" in entry["exception"]
    assert entry["timestamp"].endswith("+00:00")
    assert entry["session_id"] == entries[0]["session_id"]
    assert entry["application"] == "gui"
    assert entry["thread"] == "MainThread"
    assert entry["source"].startswith("test_logging.py:")
    assert (root.level, list(root.handlers)) == root_state
    assert capsys.readouterr().err.count("[sensor.failed]") == 1


def test_environment_level_and_location(tmp_path, monkeypatch):
    monkeypatch.setenv("CEA_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("CEA_LOG_LEVEL", "debug")
    path = config.configure_logging("controller")
    logger.debug("Debug telemetry", extra={"event": "telemetry.received"})
    assert path == tmp_path / "controller.jsonl"
    assert records(path)[-1]["event"] == "telemetry.received"


def test_default_info_excludes_high_frequency_debug(tmp_path):
    path = config.configure_logging("gui", log_dir=tmp_path, level="INFO")
    logger.debug("Frame")
    logger.info("Mode changed", extra={"event": "watering.mode_changed"})
    assert [r["event"] for r in records(path)] == ["application.started", "watering.mode_changed"]


def test_invalid_level_reports_fallback(tmp_path):
    path = config.configure_logging("gui", log_dir=tmp_path, level="INVALID")
    assert records(path)[0]["event"] == "logging.invalid_level"
    assert records(path)[-1]["context"]["level"] == "INFO"


def test_rotation_retains_bounded_backups_and_latest_event(tmp_path):
    path = config.configure_logging("gui", log_dir=tmp_path, max_bytes=1024, backup_count=2)
    for number in range(25):
        logger.info("Watering event %d", number, extra={"event": "watering.test", "context": {"number": number}})
    files = sorted(tmp_path.glob("gui.jsonl*"))
    assert len(files) == 3
    assert all(file.stat().st_size <= 1024 for file in files)
    assert records(path)[-1]["context"]["number"] == 24
    assert all(records(file) for file in files)


def test_unwritable_directory_keeps_console_logging(tmp_path, capsys):
    blocked = tmp_path / "not_a_directory"
    blocked.write_text("file")
    assert config.configure_logging("gui", log_dir=blocked) is None
    logger.error("Still visible", extra={"event": "test.visible"})
    stderr = capsys.readouterr().err
    assert "logging.file_unavailable" in stderr
    assert "test.visible" in stderr


def test_disk_full_keeps_console_and_does_not_raise(tmp_path, monkeypatch, capsys):
    config.configure_logging("gui", log_dir=tmp_path)
    handler = next(h for h in config._owned_handlers if isinstance(h, config.SafeRotatingFileHandler))

    def fail_write(_):
        raise OSError("No space left on device")

    monkeypatch.setattr(handler.stream, "write", fail_write)
    logger.critical("Pump stop failed", extra={"event": "serial.write_failed"})
    stderr = capsys.readouterr().err
    assert "Pump stop failed" in stderr
    assert "logging.file_write_failed" in stderr
    assert "No space left on device" in stderr


def test_recurring_errors_are_bounded_but_critical_events_are_kept(tmp_path, monkeypatch):
    path = config.configure_logging("gui", log_dir=tmp_path)
    now = [10.0]
    monkeypatch.setattr(config.time, "monotonic", lambda: now[0])
    for _ in range(4):
        logger.error("Camera error", extra={"event": "camera.capture_failed", "rate_limit": True})
    for _ in range(2):
        logger.critical("Pump stop failed", extra={"event": "serial.write_failed", "rate_limit": True})
    now[0] += 30
    logger.error("Camera error", extra={"event": "camera.capture_failed", "rate_limit": True})
    entries = records(path)
    assert sum(r["event"] == "camera.capture_failed" for r in entries) == 2
    assert sum(r["event"] == "serial.write_failed" for r in entries) == 2
    assert entries[-1]["suppressed_count"] == 3


def test_concurrent_workers_write_complete_json_lines(tmp_path):
    path = config.configure_logging("controller", log_dir=tmp_path)

    def emit(number):
        logger.info("Worker event", extra={"event": "worker.event", "context": {"number": number}})

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(emit, range(40)))
    entries = records(path)
    assert {r["context"]["number"] for r in entries[1:]} == set(range(40))


def test_exception_hooks_record_real_main_and_thread_failures(tmp_path):
    script = """
from scripts.logging_config import configure_logging, install_exception_hooks
import sys
import threading
configure_logging('controller', log_dir=sys.argv[1])
install_exception_hooks()
def fail():
    raise RuntimeError('worker died')
thread = threading.Thread(target=fail, name='test-worker')
thread.start()
thread.join()
raise RuntimeError('main died')
"""
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True,
                            cwd=Path(__file__).resolve().parent.parent)
    assert result.returncode == 1
    entries = records(tmp_path / "controller.jsonl")
    assert entries[-2]["event"] == "application.worker_failed"
    assert entries[-2]["context"]["worker"] == "test-worker"
    assert "RuntimeError: worker died" in entries[-2]["exception"]
    assert entries[-1]["event"] == "application.unhandled_exception"
    assert "RuntimeError: main died" in entries[-1]["exception"]


def test_shutdown_restores_hooks_and_preserves_host_handlers(tmp_path):
    app_logger = logging.getLogger(config.LOGGER_NAME)
    original_state = app_logger.level, app_logger.propagate
    hooks = sys.excepthook, threading.excepthook
    host_handler = logging.NullHandler()
    app_logger.addHandler(host_handler)
    try:
        config.configure_logging("gui", log_dir=tmp_path)
        config.install_exception_hooks()
        config.install_exception_hooks()
        config.shutdown_logging()
        assert (sys.excepthook, threading.excepthook) == hooks
        assert (app_logger.level, app_logger.propagate) == original_state
        assert host_handler in app_logger.handlers
    finally:
        app_logger.removeHandler(host_handler)


def test_import_has_no_file_or_root_configuration_side_effects(tmp_path):
    script = """
import logging
from pathlib import Path
root = logging.getLogger()
before = root.level, list(root.handlers)
import scripts.logging_config
assert (root.level, list(root.handlers)) == before
assert not list(Path('.').iterdir())
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("command, expected_level", [("0000", "CRITICAL"), ("1000", "ERROR")])
def test_cli_failed_commands_include_command_and_traceback(tmp_path, command, expected_level):
    from scripts.controller import send_command

    path = config.configure_logging("controller", log_dir=tmp_path)
    serial = MagicMock()
    serial.write.side_effect = OSError("device unplugged")
    assert send_command(serial, command) is False
    record = records(path)[-1]
    assert record["event"] == "serial.write_failed"
    assert record["level"] == expected_level
    assert record["context"]["command"] == command
    assert "OSError: device unplugged" in record["exception"]


@pytest.mark.parametrize("launch", [["scripts/controller.py"], ["-m", "scripts.controller"]])
def test_cli_launch_modes_configure_logging(tmp_path, launch):
    env = os.environ.copy()
    env["CEA_LOG_DIR"] = str(tmp_path)
    env["CEA_LOG_LEVEL"] = "INFO"
    result = subprocess.run([sys.executable, *launch, "--list"], env=env, capture_output=True, text=True,
                            cwd=Path(__file__).resolve().parent.parent)
    assert result.returncode == 0, result.stderr
    assert "Available Serial Ports" in result.stdout
    assert records(tmp_path / "controller.jsonl")[0]["event"] == "application.started"
