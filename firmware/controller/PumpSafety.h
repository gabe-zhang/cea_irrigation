#pragma once

#include <stdint.h>

// Independent of the host and its wall clock. Repeated ON commands cannot
// extend a run or restart a pump after a timeout. Only an explicit safety
// reset rearms a timed-out pump; an ordinary OFF never clears the fault.
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
    } else if (!on && !timedOut) {
      startedAt = now;
      on = true;
    }
  }
};

struct SafetyRecord {
  uint32_t generation = 0;
  bool locked = false;
  uint8_t source = 0; // 1: firmware timeout, 2: host timeout, 3: storage recovery
  uint8_t channels = 0;
};

inline bool safetyNewer(uint32_t a, uint32_t b) {
  return a != b && uint32_t(a - b) < 0x80000000UL;
}

struct PumpGroupSafety {
  PumpSafety pumps[4];
  SafetyRecord record;

  void stopAll() {
    for (uint8_t i = 0; i < 4; ++i) pumps[i].on = false;
  }

  void trip(uint8_t source, uint8_t channels) {
    stopAll();
    if (record.locked) return;
    ++record.generation;
    record.locked = true;
    record.source = source;
    record.channels = channels;
  }

  void check(uint32_t now) {
    if (record.locked) { stopAll(); return; }
    uint8_t expired = 0;
    for (uint8_t i = 0; i < 4; ++i) {
      pumps[i].check(now);
      if (pumps[i].timedOut) expired |= uint8_t(1U << i);
    }
    if (expired) trip(1, expired);
  }

  void request(uint8_t channel, bool on, uint32_t now) {
    check(now); // Check ALL deadlines before accepting any ON command.
    if (channel < 4) pumps[channel].request(on && !record.locked, now);
  }

  bool resolve(uint32_t expectedGeneration) {
    if (!record.locked || expectedGeneration != record.generation) return false;
    stopAll();
    for (uint8_t i = 0; i < 4; ++i) pumps[i] = PumpSafety();
    ++record.generation;
    record.locked = false;
    record.source = 0;
    record.channels = 0;
    return true;
  }
};
