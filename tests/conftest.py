"""Pytest configuration and custom command-line options."""

import pytest


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
