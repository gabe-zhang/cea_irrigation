#pragma once

#include <stdint.h>

// Independent of the host and its wall clock. Repeated ON commands cannot
// extend a run or restart a pump after a timeout; an explicit OFF rearms it.
struct PumpSafety {
  static const uint32_t MAX_RUNTIME_MS = 60000UL;
  bool on = false;
  bool timedOut = false;
  uint32_t startedAt = 0;

  void check(uint32_t now) {
    if (on && uint32_t(now - startedAt) >= MAX_RUNTIME_MS) {
      on = false;
      timedOut = true;
    }
  }

  void request(bool requestedOn, uint32_t now) {
    check(now);
    if (!requestedOn) {
      on = false;
      timedOut = false;
    } else if (!on && !timedOut) {
      startedAt = now;
      on = true;
    }
  }
};
