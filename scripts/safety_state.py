"""Shared persistent safety state for the GUI and serial CLI."""

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import tempfile
import threading
import time

logger = logging.getLogger("cea_irrigation.safety")
DEFAULT_STATE_PATH = Path(__file__).resolve().parent.parent / "Data" / "safety_state.json"
SAFETY_TAGS = ("safety_version", "safety_locked", "safety_generation", "safety_source", "safety_channels")


def safety_metadata(data):
    """Accept only complete, supported, well-formed controller status."""
    try:
        values = [int(data[tag]) for tag in SAFETY_TAGS]
        if any(str(data[tag]) != str(value) for tag, value in zip(SAFETY_TAGS, values)):
            return None
        version, locked, generation, source, channels = values
        if version != 1 or locked not in (0, 1) or not 0 <= generation <= 0xffffffff:
            return None
        if source not in (0, 1, 2, 3) or not 0 <= channels <= 15:
            return None
        return dict(zip(SAFETY_TAGS, values))
    except (KeyError, ValueError, TypeError, OverflowError):
        return None


def newer(a, b):
    return a != b and ((a - b) & 0xffffffff) < 0x80000000


class SafetyState:
    """A host fault cannot be cleared without a newer controller reset state.

    A pending host trip is forwarded until firmware confirms it. An acknowledged
    trip uses the firmware generation, allowing a CLI reset to be recognized
    after app reopening without resurrecting an already-resolved fault.
    """

    def __init__(self, path=None):
        self.path = Path(path) if path is not None else DEFAULT_STATE_PATH
        self.mutex = threading.RLock()
        self.ready = False
        self.last_status = None
        self.storage_error = None
        self.reset_generation = None
        self.state = {"version": 1, "locked": False, "pending_trip": False,
                      "generation": None, "context": {}}
        try:
            loaded = json.loads(self.path.read_text())
            generation = loaded.get("generation")
            if (type(loaded.get("version")) is not int or loaded.get("version") != 1 or type(loaded.get("locked")) is not bool
                    or type(loaded.get("pending_trip")) is not bool
                    or not isinstance(loaded.get("context"), dict)
                    or not (generation is None or type(generation) is int and 0 <= generation <= 0xffffffff)
                    or loaded["pending_trip"] and not loaded["locked"]):
                raise ValueError("Invalid saved safety state")
            mask = loaded["context"].get("channel_mask", 0)
            if type(mask) is not int or not 0 <= mask <= 15:
                raise ValueError("Invalid saved safety channel mask")
            for key in ("reason", "timestamp"):
                if key in loaded["context"] and not isinstance(loaded["context"][key], str):
                    raise ValueError("Invalid saved safety context")
            self.state = loaded
        except FileNotFoundError:
            pass
        except (OSError, ValueError, TypeError, AttributeError):
            logger.exception("Cannot restore safety state; pumping inhibited", extra={"event": "safety.restore_failed"})
            self.state.update(locked=True, pending_trip=True, context={"reason": "state_unavailable"})
            self.storage_error = "Cannot read the saved safety state"
        if self.locked:
            logger.warning("Restored pump safety lock", extra={"event": "safety.restored", "context": self.state})

    @property
    def locked(self):
        return self.state["locked"]

    @property
    def online(self):
        return (self.ready and self.last_status is not None
                and 0 <= time.monotonic() - self.last_status <= 4)

    @property
    def can_water(self):
        return self.online and not self.locked and not self.storage_error

    def _save(self):
        temporary = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", dir=self.path.parent, delete=False) as file:
                temporary = file.name
                json.dump(self.state, file, allow_nan=False)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, self.path)
            temporary = None
            if os.name == "posix":
                directory = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            self.storage_error = None
            return True
        except (OSError, ValueError, TypeError):
            self.storage_error = "Cannot save the safety state; watering remains blocked"
            logger.exception(self.storage_error, extra={"event": "safety.persist_failed"})
            return False
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass

    def trip(self, context, stop=None):
        with self.mutex:
            if self.locked:
                return
            self.state.update(locked=True, pending_trip=True, context={
                **context, "timestamp": datetime.now(timezone.utc).isoformat()})
            self.reset_generation = None
            try:
                if stop is not None:
                    stop()
            finally:
                self._save()
            logger.warning("All pumps locked after safety stop", extra={"event": "safety.locked", "context": self.state})

    def disconnect(self):
        self.ready = False
        self.reset_generation = None

    def reset_failed(self, message):
        with self.mutex:
            event = "safety.resolve_failed" if self.reset_generation is not None else "safety.controller_error"
            self.reset_generation = None
            logger.error("Controller safety operation failed", extra={
                "event": event, "context": {"message": message}})

    def observe(self, data):
        with self.mutex:
            status = safety_metadata(data)
            if status is None:
                self.disconnect()
                return "invalid"
            generation = status["safety_generation"]
            previous = self.state["generation"]
            # A journal recovery can lose the old counter. Its explicit locked
            # storage-fault status is safe to adopt and must remain resolvable.
            recovered = status["safety_locked"] and status["safety_source"] == 3
            if previous is not None and newer(previous, generation) and not recovered:
                self.disconnect()
                return "stale"
            self.ready = True
            self.last_status = time.monotonic()
            if status["safety_locked"]:
                changed = not self.locked or previous != generation or self.state["pending_trip"]
                context = self.state["context"] if self.locked else {
                    "reason": "firmware_timeout" if status["safety_source"] == 1 else "controller_lock",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                context = {**context, "source": status["safety_source"], "channel_mask": status["safety_channels"]}
                self.state.update(locked=True, pending_trip=False, generation=generation, context=context)
                if changed or self.storage_error:
                    self._save()
                if changed:
                    self.reset_generation = None
                    logger.warning("Controller confirms safety lock", extra={"event": "safety.controller_locked", "context": self.state})
                return "locked"
            if self.state["pending_trip"] or self.locked and (previous is None or not newer(generation, previous)):
                return "trip"
            if self.locked:
                # The controller increments only on trip/explicit reset. A newer
                # unlocked generation confirms resolution, including CLI resets.
                old_state = dict(self.state)
                self.state.update(locked=False, pending_trip=False, generation=generation)
                if not self._save():
                    self.state = old_state
                    return "storage_failed"
                self.reset_generation = None
                logger.info("Safety resolution confirmed; watering may resume", extra={"event": "safety.resolved", "context": self.state})
                return "resolved"
            if previous != generation or self.storage_error:
                self.state["generation"] = generation
                self._save()
            return "ready"

    def trip_command(self):
        context = self.state["context"]
        mask = context.get("channel_mask", 0)
        return f"safety trip {mask}"

    def resolve_command(self):
        with self.mutex:
            if (not self.locked or self.state["pending_trip"] or not self.online
                    or self.state["generation"] is None):
                return None
            self.reset_generation = self.state["generation"]
            logger.info("Operator requested safety resolution", extra={"event": "safety.resolve_requested", "context": self.state})
            return f"safety resolve {self.reset_generation}"
