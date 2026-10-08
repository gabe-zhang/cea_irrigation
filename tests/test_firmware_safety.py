"""Compile the production lock and EEPROM journal with simulated power cuts."""

from pathlib import Path
import subprocess


def test_global_lock_and_persistence_through_power_cuts(tmp_path):
    source = tmp_path / "global_safety.cpp"
    source.write_text(r'''
#include <cassert>
#include <cstring>
#include <stdexcept>
#include "SafetyStorage.h"

struct FakeEEPROM {
    unsigned char bytes[25];
    int calls = 0;
    int failAfter = -1;
    FakeEEPROM() { memset(bytes, 0xff, sizeof(bytes)); }
    unsigned char read(int address) { assert(address >= 0 && address < 25); return bytes[address]; }
    void update(int address, unsigned char value) {
        if (calls++ == failAfter) throw std::runtime_error("power lost");
        bytes[address] = value;
    }
};

int main() {
    PumpGroupSafety group;
    group.request(0, true, 1000);
    group.request(1, true, 21000);
    group.request(0, true, 60999);
    assert(group.pumps[0].startedAt == 1000);
    group.request(3, true, 61000); // Check all deadlines before starting W4.
    assert(group.record.locked && group.record.channels == 1);
    assert(group.record.source == 1 && group.record.generation == 1);
    for (int i = 0; i < 4; ++i) assert(!group.pumps[i].on);
    group.request(0, false, 62000);
    group.request(1, true, 62001);
    assert(group.record.locked && !group.pumps[1].on);
    group.trip(2, 2); // Repeated trip preserves original fault and generation.
    assert(group.record.generation == 1 && group.record.channels == 1);
    assert(!group.resolve(0));
    assert(group.resolve(1));
    assert(!group.record.locked && group.record.generation == 2);
    for (int i = 0; i < 4; ++i) assert(!group.pumps[i].on);
    group.request(2, true, 70000);
    assert(group.pumps[2].on);

    PumpGroupSafety rollover;
    uint32_t start = UINT32_MAX - 1000;
    rollover.request(0, true, start);
    rollover.check(uint32_t(start + 59999UL));
    assert(!rollover.record.locked);
    rollover.check(uint32_t(start + 60000UL));
    assert(rollover.record.locked);
    rollover.record.generation = UINT32_MAX;
    assert(rollover.resolve(UINT32_MAX));
    assert(rollover.record.generation == 0);

    FakeEEPROM virgin;
    SafetyStorage<FakeEEPROM> storage(virgin);
    assert(!storage.load().locked);
    SafetyRecord unlocked;
    assert(storage.save(unlocked));
    FakeEEPROM baseline = virgin;
    SafetyRecord locked;
    locked.generation = 1; locked.locked = true; locked.source = 1; locked.channels = 3;
    virgin.calls = 0;
    assert(storage.save(locked));
    int tripWrites = virgin.calls;
    assert(storage.load().locked && storage.load().generation == 1);

    // Once the first persistent trip marker is written, every possible
    // interruption of the record write must restore a global lock.
    for (int cut = 1; cut < tripWrites; ++cut) {
        FakeEEPROM damaged = baseline;
        damaged.calls = 0; damaged.failAfter = cut;
        SafetyStorage<FakeEEPROM> interrupted(damaged);
        try { interrupted.save(locked); } catch (const std::runtime_error&) {}
        assert(interrupted.load().locked);
    }

    FakeEEPROM lockedBaseline = virgin;
    SafetyRecord resolved;
    resolved.generation = 2;
    virgin.calls = 0;
    assert(storage.save(resolved));
    int resetWrites = virgin.calls;
    assert(!storage.load().locked && storage.load().generation == 2);
    for (int cut = 0; cut < resetWrites; ++cut) {
        FakeEEPROM damaged = lockedBaseline;
        damaged.calls = 0; damaged.failAfter = cut;
        SafetyStorage<FakeEEPROM> interrupted(damaged);
        try { interrupted.save(resolved); } catch (const std::runtime_error&) {}
        assert(interrupted.load().locked);
    }
    // Corrupt both checksummed slots and an unknown marker: fail closed.
    virgin.bytes[1] = 0;
    virgin.bytes[13] = 0;
    assert(storage.load().locked && storage.load().source == 3);
    FakeEEPROM corrupt;
    corrupt.bytes[0] = 0;
    SafetyStorage<FakeEEPROM> unknown(corrupt);
    assert(unknown.load().locked);
}
''')
    firmware = Path(__file__).resolve().parents[1] / "firmware" / "controller"
    executable = tmp_path / "global_safety"
    subprocess.run(["g++", "-std=c++11", "-Wall", "-Wextra", "-Werror",
                    "-I", str(firmware), str(source), "-o", str(executable)], check=True)
    subprocess.run([str(executable)], check=True)
