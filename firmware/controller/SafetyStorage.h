#pragma once

#include "PumpSafety.h"

// Two checksummed EEPROM slots plus a conservative lock marker. Write the
// marker before a trip record and clear it only after a verified reset record.
// Storage supplies read(address) and update(address, byte), as EEPROM does.
template <typename Storage> struct SafetyStorage {
  Storage& storage;
  static const uint8_t SLOT_SIZE = 12;
  explicit SafetyStorage(Storage& s) : storage(s) {}

  uint16_t crc(const uint8_t* data) {
    uint16_t value = 0xffff;
    for (uint8_t i = 1; i <= 8; ++i) {
      value ^= uint16_t(data[i]) << 8;
      for (uint8_t bit = 0; bit < 8; ++bit)
        value = (value & 0x8000) ? uint16_t((value << 1) ^ 0x1021) : uint16_t(value << 1);
    }
    return value;
  }

  bool readSlot(uint8_t slot, SafetyRecord& record) {
    uint8_t bytes[SLOT_SIZE];
    for (uint8_t i = 0; i < SLOT_SIZE; ++i) bytes[i] = storage.read(1 + slot * SLOT_SIZE + i);
    if (bytes[0] != 0xa5 || bytes[1] != 1 || bytes[6] > 1 || bytes[7] > 3 || bytes[8] > 15
        || crc(bytes) != uint16_t(bytes[9] | uint16_t(bytes[10]) << 8)) return false;
    record.generation = uint32_t(bytes[2]) | uint32_t(bytes[3]) << 8
        | uint32_t(bytes[4]) << 16 | uint32_t(bytes[5]) << 24;
    record.locked = bytes[6] != 0;
    record.source = bytes[7];
    record.channels = bytes[8];
    return true;
  }

  SafetyRecord load() {
    SafetyRecord a, b, result;
    bool validA = readSlot(0, a), validB = readSlot(1, b);
    if (validA) result = a;
    if (validB && (!validA || safetyNewer(b.generation, a.generation))) result = b;
    bool virgin = true;
    for (uint8_t i = 0; i < 1 + 2 * SLOT_SIZE; ++i)
      if (storage.read(i) != 0xff) virgin = false;
    if (virgin) return result;
    const uint8_t marker = storage.read(0);
    if ((!validA && !validB) || marker != 0x3c || result.locked) {
      if (!result.locked) { ++result.generation; result.source = 3; result.channels = 0; }
      result.locked = true;
    }
    return result;
  }

  bool save(const SafetyRecord& record) {
    SafetyRecord a, b;
    bool validA = readSlot(0, a), validB = readSlot(1, b);
    uint8_t target = !validA ? 0 : !validB ? 1 : safetyNewer(a.generation, b.generation) ? 1 : 0;
    if (record.locked) storage.update(0, 0xc3);
    uint8_t bytes[SLOT_SIZE] = {};
    bytes[0] = 0xa5; bytes[1] = 1;
    for (uint8_t i = 0; i < 4; ++i) bytes[2 + i] = uint8_t(record.generation >> (8 * i));
    bytes[6] = record.locked; bytes[7] = record.source; bytes[8] = record.channels;
    uint16_t checksum = crc(bytes);
    bytes[9] = uint8_t(checksum); bytes[10] = uint8_t(checksum >> 8);
    int address = 1 + target * SLOT_SIZE;
    storage.update(address, 0);
    for (uint8_t i = 1; i < SLOT_SIZE; ++i) storage.update(address + i, bytes[i]);
    storage.update(address, bytes[0]);
    SafetyRecord verified;
    if (!readSlot(target, verified) || verified.generation != record.generation
        || verified.locked != record.locked || verified.source != record.source
        || verified.channels != record.channels) return false;
    storage.update(0, record.locked ? 0xc3 : 0x3c);
    return storage.read(0) == (record.locked ? 0xc3 : 0x3c);
  }
};
