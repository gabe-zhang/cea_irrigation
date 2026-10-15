"""Capture a stalled Tk event loop without depending on Python worker threads."""

from datetime import datetime
import faulthandler
import logging
import os
from pathlib import Path
import signal
import time

from scripts.logging_config import LOG_TIMEZONE

logger = logging.getLogger("cea_irrigation.gui")


class GuiWatchdog:
    """Rearm a native one-shot stack dump timer from a Tk heartbeat.

    Owns the process-wide faulthandler timer and SIGUSR1 handler; use only at
    the GUI entry point. A native timer can dump even when a C call holds the GIL.
    """

    def __init__(self, root, log_dir: Path, *, timeout: float = 30, interval_ms: int = 5000):
        if timeout <= 0 or not 0 < interval_ms < timeout * 1000:
            raise ValueError("heartbeat interval must be shorter than the timeout")
        self.root = root
        self.log_dir = Path(log_dir)
        self.timeout = timeout
        self.interval_ms = interval_ms
        self.path = None
        self._file = None
        self._job = None
        self._last_heartbeat = None
        self._signal_registered = False

    def start(self) -> bool:
        if self._file is not None:
            return True
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(LOG_TIMEZONE).strftime("%Y%m%d_%H%M%S_%f")
            self.path = self.log_dir / f"gui-stacks-{stamp}-{os.getpid()}.log"
            self._file = self.path.open("a", encoding="utf-8")
            self._file.write(
                f"Started {datetime.now(LOG_TIMEZONE).isoformat()} pid={os.getpid()} "
                f"timeout={self.timeout}s; Timeout means no GUI heartbeat.\n"
            )
            self._file.flush()
            # Keep the latest five sessions. Each freeze produces one dump.
            for old in sorted(self.log_dir.glob("gui-stacks-*.log"))[:-5]:
                old.unlink()
            if hasattr(signal, "SIGUSR1"):
                faulthandler.register(signal.SIGUSR1, file=self._file, all_threads=True)
                self._signal_registered = True
            self.heartbeat()
        except (OSError, RuntimeError):
            self.stop()
            logger.exception("Cannot start GUI freeze diagnostics", extra={"event": "gui.watchdog_unavailable"})
            return False
        logger.info("GUI freeze diagnostics enabled", extra={"event": "gui.watchdog_started", "context": {
            "stack_file": self.path, "timeout_sec": self.timeout, "manual_dump_signal": "SIGUSR1",
        }})
        return True

    def heartbeat(self) -> None:
        if self._file is None:
            return
        now = time.monotonic()
        if self._last_heartbeat is not None and now - self._last_heartbeat >= self.timeout:
            stalled_sec = round(now - self._last_heartbeat, 3)
            logger.warning("GUI event loop resumed after a stall", extra={"event": "gui.event_loop_recovered", "context": {
                "heartbeat_gap_sec": stalled_sec, "stack_file": self.path,
            }})
            try:
                self._file.write(f"Recovered {datetime.now(LOG_TIMEZONE).isoformat()} heartbeat_gap={stalled_sec}s\n")
                self._file.flush()
            except OSError:
                logger.exception("Cannot write GUI recovery marker", extra={"event": "gui.watchdog_write_failed", "rate_limit": True})
        self._last_heartbeat = now
        faulthandler.dump_traceback_later(self.timeout, file=self._file, repeat=False)
        self._job = self.root.after(self.interval_ms, self.heartbeat)

    def stop(self) -> None:
        if self._file is None:
            return
        # Cancel the native timer before closing its output descriptor.
        faulthandler.cancel_dump_traceback_later()
        if self._signal_registered:
            faulthandler.unregister(signal.SIGUSR1)
            self._signal_registered = False
        if self._job is not None:
            try:
                self.root.after_cancel(self._job)
            except Exception:
                pass  # The Tk root may already have been destroyed.
            self._job = None
        self._file.close()
        self._file = None
