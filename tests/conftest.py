"""Pytest configuration and custom command-line options."""

import pytest


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
