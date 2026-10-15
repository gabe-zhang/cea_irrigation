"""Shared hardware boundaries, GUI fixtures, and pytest options."""

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class Clock:
    """Advance monotonic time and execute due callbacks in registration order."""

    def __init__(self):
        self.now = 0.0
        self.jobs = {}
        self.sequence = 0

    def after(self, delay_ms, callback, *args):
        self.sequence += 1
        job = f"test-job-{self.sequence}"
        self.jobs[job] = (self.now + delay_ms / 1000, lambda: callback(*args))
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
def clock():
    return Clock()


@pytest.fixture
def serial_device():
    """Configure read chunks/errors and inspect writes without a real port."""
    device = MagicMock(name="serial_device")
    device.is_open = True
    device.in_waiting = 0
    device.read.return_value = b""
    device.write.side_effect = lambda payload: len(payload)
    device.close.side_effect = lambda: setattr(device, "is_open", False)
    return device


@pytest.fixture
def picamera_backend():
    """Inject this constructor at arducam.Picamera2 to exercise real Camera."""
    device = MagicMock(name="picamera")
    device.capture_array.return_value = np.zeros((48, 64, 3), dtype=np.uint8)
    return MagicMock(name="Picamera2", return_value=device)


@pytest.fixture
def litert_backend():
    """Inject this runtime at plant_ai.tflite to exercise real inference."""
    interpreter = MagicMock(name="interpreter")
    interpreter.get_input_details.return_value = [
        {"shape": [1, 200, 200, 3], "dtype": np.float32,
         "index": 0, "quantization": (0.0, 0)}
    ]
    interpreter.get_output_details.return_value = [
        {"index": 1, "quantization": (0.0, 0)}
    ]
    interpreter.get_tensor.return_value = np.zeros((1, 7, 5), dtype=np.float32)
    return SimpleNamespace(
        Interpreter=MagicMock(return_value=interpreter),
        load_delegate=MagicMock(return_value=object()),
    )


@pytest.fixture(scope="session")
def coral_delegate():
    """Reuse the real TPU delegate, releasing it before Python module teardown."""
    import ai_edge_litert.interpreter as tflite

    delegate = tflite.load_delegate("libedgetpu.so.1")
    yield delegate
    del delegate


@pytest.fixture(autouse=True)
def isolated_gui_environment(tmp_path, monkeypatch, request, serial_device):
    """Keep ordinary tests away from operator recordings and native hardware."""
    import gui.psc_irr_gui as gui
    from gui.modules import arducam, plant_ai

    directories = {
        "DATA_DIR": tmp_path / "Data",
        "TELEMETRY_DIR": tmp_path / "Data" / "telemetry",
        "IMAGES_DIR": tmp_path / "Data" / "images",
        "AUTO_WATERING_DIR": tmp_path / "Data" / "images" / "auto_watering",
    }
    for name, directory in directories.items():
        directory.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(gui, name, directory)
    if request.node.get_closest_marker("coral") is None:
        # Explicit per-test fixtures may replace these inactive boundaries.
        monkeypatch.setattr(arducam, "Picamera2", None)
        monkeypatch.setattr(plant_ai, "tflite", None)
        monkeypatch.setattr(gui.serial, "Serial", MagicMock(return_value=serial_device))
        monkeypatch.setattr(gui.serial.tools.list_ports, "comports", lambda: [])
    return directories


@pytest.fixture(autouse=True)
def isolated_safety_storage(tmp_path, monkeypatch):
    """Never read or reset the operator's persistent safety fault in tests."""
    from scripts import safety_state
    monkeypatch.setattr(safety_state, "DEFAULT_STATE_PATH", tmp_path / "safety_state.json")


@pytest.fixture
def controller_ready():
    """Give a hardware-free GUI an explicit supported-controller status."""
    def ready(app):
        data = {**app.telemetry, "safety_version": "1", "safety_locked": "0",
                "safety_generation": "0", "safety_source": "0", "safety_channels": "0"}
        app._receive_telemetry(data, app.telemetry.get("moisture_pct", []))
        app._telemetry_sequence = 0
        app._auto_checked_sequence = 0
        return app
    return ready


@pytest.fixture
def gui_app_factory(controller_ready, isolated_gui_environment):
    """Build real Tk controls; opt in to ready status and real AI rendering."""
    import gui.psc_irr_gui as gui
    from gui.modules.plant_ai import PlantAIDetector

    apps = []

    def create(*, ready=False, real_detector=False):
        camera = MagicMock(name="camera")
        camera.is_available = False
        camera.capture_array.return_value = None
        if real_detector:
            detector = PlantAIDetector()
        else:
            detector = MagicMock(name="plant_ai")
            detector.is_available = False
            detector.last_latency_ms = 0.0
            detector.detect_and_analyze.return_value = ([], 0.0)
            detector.draw_overlay.side_effect = lambda frame, *_args, **_kwargs: frame
            detector.draw_full_frame_heatmap.side_effect = lambda frame, *_args, **_kwargs: frame
        with ExitStack() as stack:
            stack.enter_context(patch.object(gui, "Camera", return_value=camera))
            stack.enter_context(patch.object(gui, "PlantAIDetector", return_value=detector))
            for name in ("_init_serial", "_camera_loop", "_start_periodic_loggers", "_start_auto_ticker"):
                stack.enter_context(patch.object(gui.MainWindow, name))
            app = gui.MainWindow()
        apps.append(app)
        app.withdraw()
        if ready:
            controller_ready(app)
        return app

    yield create
    for app in reversed(apps):
        app.stop_threads.set()
        # The root widget command disappears on destroy; winfo may remain as
        # a stub that raises TclError after the Tk application has closed.
        if app.tk.call("info", "commands", app._w):
            app.plotter.toggle(False)
            # Cancel real Tcl timers even when a control test replaced after().
            for job in app.tk.call("after", "info"):
                app.tk.call("after", "cancel", job)
            app.destroy()
        if app.ser and app.ser.is_open:
            app.ser.close()
        app.camera.stop()


@pytest.fixture
def headless_app(gui_app_factory):
    return gui_app_factory(ready=True)


@pytest.fixture
def headless_gui(gui_app_factory):
    return gui_app_factory(ready=True)


@pytest.fixture
def simulated_controller():
    """Model device-side safety commands; host state waits for explicit telemetry."""
    def attach(app):
        status = {tag: app.telemetry[tag] for tag in (
            "safety_version", "safety_locked", "safety_generation", "safety_source", "safety_channels"
        )}

        def write(payload):
            command = payload.decode().strip()
            if command.startswith("safety trip ") and status["safety_locked"] == "0":
                status.update(safety_locked="1", safety_generation=str(int(status["safety_generation"]) + 1),
                              safety_source="2", safety_channels=command.split()[-1])
            elif command == f"safety resolve {status['safety_generation']}" and status["safety_locked"] == "1":
                status.update(safety_locked="0", safety_generation=str(int(status["safety_generation"]) + 1),
                              safety_source="0", safety_channels="0")
            return len(payload)

        app.ser.write.side_effect = write
        return status

    return attach


@pytest.fixture
def watering(gui_app_factory, clock, serial_device, simulated_controller, monkeypatch):
    import gui.psc_irr_gui as gui

    app = gui_app_factory(ready=True)
    app.ser = serial_device
    monkeypatch.setattr(gui.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(app, "after", clock.after)
    monkeypatch.setattr(app, "after_cancel", clock.cancel)
    # Refresh the safety timestamp in the deterministic time domain.
    app.safety.last_status = clock.now
    app._test_controller_status = simulated_controller(app)
    app.telemetry["moisture_pct"] = [35.8, 37.0, 100.0, 100.0]
    app._below_threshold_counts = [3, 3, 0, 0]
    return app, clock


def pytest_addoption(parser):
    """Add custom command-line flags to pytest."""
    parser.addoption(
        "--coral",
        action="store_true",
        default=False,
        help="Run Google Coral Edge TPU tests (skipped by default).",
    )


def pytest_collection_modifyitems(config, items):
    """Skip Coral TPU tests unless --coral flag is specified."""
    if config.getoption("--coral"):
        return

    skip_coral = pytest.mark.skip(reason="Coral TPU test skipped (run with --coral to execute)")
    for item in items:
        if "coral" in item.keywords:
            item.add_marker(skip_coral)
