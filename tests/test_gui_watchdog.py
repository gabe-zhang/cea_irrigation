"""Exercise native freeze diagnostics in isolated processes."""

from pathlib import Path
import subprocess
import sys

import pytest


def run_watchdog(tmp_path, body):
    script = """
from pathlib import Path
import sys
import time
import tkinter as tk
from scripts.gui_watchdog import GuiWatchdog
from scripts.logging_config import configure_logging
directory = Path(sys.argv[1])
configure_logging('gui', log_dir=directory)
root = tk.Tk()
root.withdraw()
watchdog = GuiWatchdog(root, directory, timeout=0.2, interval_ms=20)
assert watchdog.start()
try:
""" + "\n".join("    " + line for line in body.splitlines()) + """
finally:
    watchdog.stop()
    root.destroy()
"""
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)],
                            cwd=Path(__file__).resolve().parent.parent,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    return next(tmp_path.glob("gui-stacks-*.log")).read_text(), (tmp_path / "gui.jsonl").read_text()


def test_healthy_event_loop_and_shutdown_do_not_dump(tmp_path):
    stacks, events = run_watchdog(tmp_path, """root.after(500, root.quit)
root.mainloop()
watchdog.stop()
time.sleep(0.3)""")
    assert "Timeout (" not in stacks
    assert "Thread 0x" not in stacks
    assert "gui.event_loop_recovered" not in events


def test_blocked_callback_dumps_once_then_records_recovery(tmp_path):
    stacks, events = run_watchdog(tmp_path, """def blocked_callback():
    time.sleep(0.8)
root.after(40, blocked_callback)
root.after(950, root.quit)
root.mainloop()""")
    assert stacks.count("Timeout (") == 1
    assert "blocked_callback" in stacks
    assert "mainloop" in stacks
    assert "Recovered " in stacks
    assert "gui.event_loop_recovered" in events


@pytest.mark.skipif(sys.platform != "linux", reason="Uses Linux libc and SIGUSR1")
def test_native_call_holding_gil_still_dumps(tmp_path):
    stacks, _ = run_watchdog(tmp_path, """import ctypes
def blocked_native_call():
    ctypes.PyDLL('libc.so.6').sleep(1)
blocked_native_call()""")
    assert stacks.count("Timeout (") == 1
    assert "blocked_native_call" in stacks


@pytest.mark.skipif(sys.platform != "linux", reason="Uses Linux SIGUSR1")
def test_manual_signal_dumps_without_terminating_gui(tmp_path):
    stacks, _ = run_watchdog(tmp_path, """import os
import signal
os.kill(os.getpid(), signal.SIGUSR1)
root.after(80, root.quit)
root.mainloop()""")
    assert "Current thread 0x" in stacks
    assert "Timeout (" not in stacks


def test_diagnostic_file_failure_does_not_stop_gui(tmp_path, caplog):
    from scripts.gui_watchdog import GuiWatchdog
    from unittest.mock import MagicMock

    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")
    root = MagicMock()
    watchdog = GuiWatchdog(root, blocked)
    assert watchdog.start() is False
    watchdog.stop()
    root.after.assert_not_called()
    assert "Cannot start GUI freeze diagnostics" in caplog.text
