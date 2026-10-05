"""Application-owned logging; imports never configure the process root logger."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from logging.handlers import RotatingFileHandler
import math
import os
from pathlib import Path
import sys
import threading
import time
import uuid

LOGGER_NAME = "cea_irrigation"
DEFAULT_LOG_DIR = Path(__file__).resolve().parent.parent / "Data" / "logs"
MAX_BYTES = 5 * 1024 * 1024
BACKUP_COUNT = 5
_logger = logging.getLogger(LOGGER_NAME)
_logger.addHandler(logging.NullHandler())
_lock = threading.RLock()
_configured = False
_log_path: Path | None = None
_original_hooks: tuple | None = None
_owned_handlers: list[logging.Handler] = []
_previous_state: tuple[int, bool] | None = None


def _json_safe(value):
    """Sensor NaN/Infinity values must not turn diagnostic events into invalid JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


class SafeRotatingFileHandler(RotatingFileHandler):
    """Report runtime storage failures without raising into irrigation control."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_error = float("-inf")

    def handleError(self, record: logging.LogRecord) -> None:
        now = time.monotonic()
        if now - self._last_error >= 30:
            self._last_error = now
            try:
                sys.stderr.write(f"ERROR logging.file_write_failed: cannot write {self.baseFilename}: {sys.exc_info()[1]}\n")
            except OSError:
                pass


class JsonFormatter(logging.Formatter):
    """One JSON object per line, including traceback and structured event context."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(record, "event", "message"),
            "message": record.getMessage(),
            "session_id": getattr(record, "session_id", None),
            "application": getattr(record, "application", None),
            "process_id": record.process,
            "thread": record.threadName,
            "source": f"{record.filename}:{record.lineno}",
            "context": getattr(record, "context", {}),
        }
        if getattr(record, "suppressed_count", 0):
            payload["suppressed_count"] = record.suppressed_count
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(_json_safe(payload), default=str, ensure_ascii=False, allow_nan=False)


class EventFilter(logging.Filter):
    """Attach session context and bound opt-in recurring errors to once per 30s.

    Each handler has its own filter so console filtering cannot hide file events.
    Critical records are never suppressed. The next emitted error reports the
    number suppressed since the previous record.
    """

    def __init__(self, application: str, session_id: str, interval: float = 30.0):
        super().__init__()
        self.application = application
        self.session_id = session_id
        self.interval = interval
        self._seen: dict[tuple[str, str], tuple[float, int]] = {}
        self._lock = threading.Lock()

    def filter(self, record: logging.LogRecord) -> bool:
        record.application = self.application
        record.session_id = self.session_id
        if not getattr(record, "rate_limit", False) or record.levelno >= logging.CRITICAL:
            return True
        key = (record.name, getattr(record, "event", record.msg))
        now = time.monotonic()
        with self._lock:
            last, count = self._seen.get(key, (float("-inf"), 0))
            if now - last < self.interval:
                self._seen[key] = (last, count + 1)
                return False
            record.suppressed_count = count
            self._seen[key] = (now, 0)
        return True


class ConsoleFormatter(logging.Formatter):
    converter = time.gmtime

    def __init__(self):
        super().__init__("%(asctime)s UTC %(levelname)s %(name)s [%(event)s] %(message)s", "%H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "event"):
            record.event = "message"
        result = super().format(record)
        if getattr(record, "context", None):
            result += " " + json.dumps(record.context, default=str, ensure_ascii=False)
        return result


def configure_logging(application: str, *, log_dir: Path | str | None = None,
                      level: str | None = None, max_bytes: int = MAX_BYTES,
                      backup_count: int = BACKUP_COUNT) -> Path | None:
    """Configure once at an entry point; retain host/third-party logging settings.

    File creation failures leave console logging active. Use separate directories
    for concurrent instances of the same application (stdlib rotation has one writer).
    """
    global _configured, _log_path, _previous_state
    with _lock:
        if _configured:
            return _log_path
        if application not in {"gui", "controller"}:
            raise ValueError("application must be 'gui' or 'controller'")
        if max_bytes <= 0 or backup_count < 1:
            raise ValueError("rotation requires positive max_bytes and backup_count")
        requested_level = (level or os.environ.get("CEA_LOG_LEVEL", "INFO")).upper()
        valid_level = requested_level in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        log_level = getattr(logging, requested_level) if valid_level else logging.INFO
        session_id = uuid.uuid4().hex
        console = logging.StreamHandler()
        console.setLevel(log_level)
        console.setFormatter(ConsoleFormatter())
        console.addFilter(EventFilter(application, session_id))
        _previous_state = (_logger.level, _logger.propagate)
        _logger.setLevel(log_level)
        _logger.propagate = False
        _logger.addHandler(console)
        _owned_handlers.append(console)
        _configured = True
        directory = Path(log_dir or os.environ.get("CEA_LOG_DIR", DEFAULT_LOG_DIR)).expanduser()
        try:
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{application}.jsonl"
            handler = SafeRotatingFileHandler(path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
            handler.setLevel(log_level)
            handler.setFormatter(JsonFormatter())
            handler.addFilter(EventFilter(application, session_id))
            _logger.addHandler(handler)
            _owned_handlers.append(handler)
            _log_path = path
        except OSError:
            _logger.exception("Cannot open log file; console logging remains active",
                              extra={"event": "logging.file_unavailable", "context": {"directory": directory}})
        if not valid_level:
            _logger.warning("Invalid log level; using INFO",
                            extra={"event": "logging.invalid_level", "context": {"requested_level": requested_level}})
        _logger.info("Application starting", extra={"event": "application.started", "context": {
            "log_file": _log_path, "python_version": sys.version.split()[0], "level": logging.getLevelName(log_level),
        }})
        return _log_path


def _process_exception(exc_type, value, traceback):
    if not issubclass(exc_type, KeyboardInterrupt):
        _logger.critical("Unhandled application exception", exc_info=(exc_type, value, traceback),
                         extra={"event": "application.unhandled_exception"})
    if _original_hooks:
        _original_hooks[0](exc_type, value, traceback)


def _thread_exception(args):
    if args.exc_type is not SystemExit:
        _logger.critical("Unhandled worker exception", exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
                         extra={"event": "application.worker_failed", "context": {"worker": args.thread.name if args.thread else None}})
    if _original_hooks:
        _original_hooks[1](args)


def install_exception_hooks() -> None:
    """Capture fatal main/worker exceptions while preserving default reporting."""
    global _original_hooks
    with _lock:
        if _original_hooks is None:
            _original_hooks = (sys.excepthook, threading.excepthook)
            sys.excepthook = _process_exception
            threading.excepthook = _thread_exception


def shutdown_logging() -> None:
    """Flush owned handlers and restore hooks; leave host logging untouched."""
    global _configured, _log_path, _original_hooks, _previous_state
    with _lock:
        if _original_hooks:
            if sys.excepthook is _process_exception:
                sys.excepthook = _original_hooks[0]
            if threading.excepthook is _thread_exception:
                threading.excepthook = _original_hooks[1]
            _original_hooks = None
        if _configured:
            for handler in _owned_handlers:
                _logger.removeHandler(handler)
                handler.close()
            _owned_handlers.clear()
            if _previous_state:
                _logger.setLevel(_previous_state[0])
                _logger.propagate = _previous_state[1]
                _previous_state = None
            _configured = False
            _log_path = None
