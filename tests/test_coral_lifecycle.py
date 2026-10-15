"""Check delegate ownership across pytest teardown without opening USB hardware."""

from pathlib import Path
import subprocess
import sys


def test_delegate_released_at_pytest_session_end():
    script = """
import gc
import weakref
import pytest
import ai_edge_litert.interpreter as tflite

delegates = []

class FakeDelegate:
    pass

def load_delegate(*args, **kwargs):
    delegate = FakeDelegate()
    delegates.append(weakref.ref(delegate))
    return delegate

tflite.load_delegate = load_delegate
assert pytest.main(['--coral', '-q',
                    'tests/test_coral_tpu.py::test_edgetpu_delegate_load']) == 0
gc.collect()
assert delegates and all(ref() is None for ref in delegates), 'Delegate survived pytest session cleanup'
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Exception ignored in:" not in result.stderr, result.stderr
