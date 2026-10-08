# Persistent pump safety lock: design and implementation plan

Status: implemented in the app, CLI, and firmware. Software verification is recorded below; firmware upload and a physical bench check remain deployment steps.

## Confirmed requirements

- Create a lock only when an existing watering safety stop triggers. Do not add sensor-fault detection or turn other existing stop reasons into safety trips in this change.
- A safety trip stops all four pumps and blocks automatic watering, manual watering, and direct serial ON commands.
- Sensor monitoring, camera functions, and diagnostics remain available while locked.
- Keep the lock across app reopening, Arduino reset, and full power cycling.
- Show a visible warning/status and record the trip and reset in the existing event logs.
- An operator clears the lock with a Resolved button or an explicit command. This is the operator's confirmation that the physical issue has been addressed; no extra sensors or automatic repair verification are required.
- Ordinary OFF commands, mode changes, improved readings, and restarts never clear a safety lock.

## Recovery choices

- Confirmed: Resolved restores automatic watering after three fresh low checks when AUTO is selected. There is no separate Resume action. In MANUAL, resolving enables manual controls without starting a pump.
- Implemented: a reset requires an online controller to confirm that its persistent lock has been cleared. Do not queue an unattended reset while disconnected.
- Normal scheduled duration completion remains a normal stop. An actual GUI or Arduino safety-timeout event always locks, including during manual watering. Do not reinterpret a reported controller timeout as normal completion.
- A planned 60-second run coincides with the existing hard limit. Boundary behavior must be verified: the GUI should recognize an already-due normal completion before classifying its own safety timeout, but an Arduino timeout that actually fires must still lock. Do not increase the hard safety limit or conceal this race. Document the observed boundary behavior and revisit the timed-duration interface if normal 60-second completion cannot be reliable.

## Walkthrough

1. W1 starts watering; W2 may also be operating.
2. A safety timer expires. The component detecting it latches the global lock and requests all outputs OFF immediately, before logging, persistence, or UI work can delay shutdown.
3. Cancel watering callbacks and manual relay timers, clear dry-reading streaks, and reject subsequent ON requests. Preserve the original trip's context.
4. Persist the lock and surface one warning. Continue receiving live telemetry. Distinguish OFF requested from controller-reported outputs OFF; reported relay outputs do not prove water flow or mechanically verify the relay.
5. Reopening the app or resetting the controller restores the lock before watering can be enabled.
6. The operator repairs the issue and selects Resolved or issues the explicit reset command.
7. The controller clears its stored lock, keeps all outputs OFF, and reports the reset result. The app updates its saved record only after confirmation.
8. Clear reading streaks and cancel old callbacks again. In AUTO, require three new low checks before starting watering. No pump starts as a side effect of processing the reset command itself.
9. If the issue persists, a later safety stop creates a new lock. No automatic retry or bypass is introduced.

## Enforcement and persistence

### Arduino

- Replace per-pump OFF-rearms-timeout behavior with a global latched safety lock alongside the existing per-pump runtime tracking.
- Detect safety expiry across all channels before applying any ON bitmask, so command ordering cannot briefly energize another pump after a trip.
- Any channel's safety timeout de-energizes every relay. Once locked, allow OFF and non-pump commands while rejecting pump ON requests, including partial bitmasks.
- Add dedicated status, trip, and resolve operations. Proposed operator command: `safety resolve`; its parser must not conflict with existing servo commands.
- Both independently detected firmware trips and GUI-reported safety trips use the same global lock.
- Store lock state and a fault/reset generation in nonvolatile storage. Update only on state transitions, with a versioned record and corruption detection; do not write on each telemetry frame.
- Initialize relay outputs OFF before other startup work and restore persisted state before permitting ON commands. Test interrupted writes and invalid records; ambiguous state must not silently permit watering.
- Do not claim persistence of a transition that lost power before its first persistent write. The conservative marker protects interrupted journal writes after that initial marker is stored. To preserve the agreed scope, ordinary interrupted runs do not create an additional safety trip on boot.

### Shared Python state and protocol

- Use one shared safety-state helper for GUI and CLI rather than independent reset implementations.
- Store the app's active fault separately from rotating event logs, at a stable project-relative path such as `Data/safety_state.json`, independent of the launching working directory.
- Use atomic durable replacement for host state. Keep trip context: fault ID/generation, source, UTC time when available, triggering channels, run ID, elapsed time, and last moisture readings.
- Add protocol version/capability and explicit lock metadata to controller telemetry/status; update both existing telemetry parsers.
- Synchronize safety status before enabling watering on startup or reconnection. An unavailable controller or an older firmware without safety-lock support cannot be treated as a confirmed unlocked controller.
- Reconcile both directions: adopt a firmware-only trip, and forward a saved GUI trip that occurred while disconnected. Never clear a saved lock merely because the other side reports unlocked.
- Correlate explicit reset confirmation with the current fault/generation. A reset from the supported CLI must be recognized by the GUI without restoring a stale fault; a stale reset must not clear a newer trip.
- A failed trip persistence write leaves the in-memory lock active and reports the failure. A failed or unconfirmed reset leaves pumping inhibited. Storage problems are operational blockers, not additional sensor-triggered safety trips.

## Implementation sequence

1. **Firmware lock and persistence** — update `firmware/controller/PumpSafety.h` and `firmware/controller/controller.ino`; add host-compilable global-lock logic and a testable storage boundary. Preserve the 60-second maximum, repeated-ON protection, nonblocking command handling, and clock-rollover behavior.
2. **Protocol and shared host state** — introduce the shared safety helper; extend parsing, status synchronization, trip propagation, and reset confirmation for GUI and CLI.
3. **GUI integration** — load safety state before starting irrigation scheduling; route the existing safety callback through the lock; guard both watering entry points and the final command writer; cancel automatic/manual callbacks on trip.
4. **Warning and recovery UI** — display a persistent banner and a single warning window with Resolved and a way to dismiss the window. Closing the window preserves the lock and banner. Disable all pump controls while locked, and show pending/failed reset results without reporting success early.
5. **CLI recovery** — expose safety status and `safety resolve`, print the active warning, and log the result. Ordinary `0000` stays a stop command. The Arduino enforces the lock even against direct serial commands.
6. **Logs and documentation** — retain existing watering events and add structured events for lock creation/restoration, shutdown confirmation/failure, reset request/confirmation/failure, and resume if selected. Describe the firmware update requirement, persistence location, actual timeout boundary behavior, and operator recovery procedure in README.
7. **Verification** — run focused tests, the full default suite, and a documented hardware check before declaring the end-to-end behavior verified.

## Acceptance checks

- Fresh low readings never restart any pump after a safety stop, even after many checks.
- A trip on one channel stops other active channels and prevents inactive channels from starting.
- Manual controls, mode toggles, ordinary OFF, partial bitmasks, queued callbacks, and repeated ON cannot bypass the lock.
- A firmware-only timeout while the app is frozen or disconnected produces the same global lock and is visible on reconnection.
- App reopening and controller reboot restore a trip before ON is allowed. Exercise full power loss and interrupted writes through storage fakes and a hardware check.
- Missing or malformed status, incompatible firmware, corrupt persistence, and a failed reset do not silently enable pumps.
- Resolved and the CLI command clear the same fault, require confirmation under the proposed default, and leave relay outputs OFF. Verify interrupted reset recovery and stale fault/reset generations.
- A new trip supersedes an older reset request. Dismissing the warning never clears a fault.
- Ordinary moisture-target and planned-duration stops remain ordinary stops; existing sensor-missing and control-error stop behavior does not gain a lock in this scope.
- Cover durations immediately below, at, and beyond the hard limit, GUI callback ordering, serial delay, manual duration expiry, simultaneous channel deadlines, and 32-bit clock rollover.
- Replace `test_safety_stop_requires_new_readings_before_another_run`, which currently expects automatic restart, with persistent-lock coverage. Split existing tests that start a second run after a safety stop so they explicitly reset or use a fresh test instance.
- Use temporary safety storage and fake serial/controller state for tests; tests must never affect the operator's real lock or run physical pumps.
- Run `uv run --extra gui --extra dev pytest`; use `xvfb-run -a` if no display is available. Record hardware verification separately from simulated test results.

## Scope

The user authorized implementation after design review and selected automatic resumption after resolution. Firmware has been compiled, not flashed, and real pumps have not been operated. The existing root `plan.md` concerns a separate test refactor and is preserved.

## Hardware verification

1. With pump power isolated, upload the updated controller firmware using the normal Arduino workflow. Confirm telemetry includes supported safety status and starts with outputs OFF.
2. Use relay indicators or a bench load to exercise the 60-second runtime guard on one channel while another channel is active. Confirm all outputs switch OFF and all subsequent ON bitmasks are rejected.
3. Send ordinary OFF commands and switch modes. Confirm the lock remains. Close/reopen the app, reset the Arduino, and power-cycle the controller after the lock has been recorded; confirm the lock remains each time.
4. With the controller disconnected, attempt resolution and confirm the app preserves the lock. Reconnect, confirm the status, then resolve through the GUI. Confirm outputs stay OFF until three new low checks in AUTO.
5. Repeat recovery through the CLI. Confirm its reset is recognized when the GUI is reopened. Reject a reset command for an older generation after a new trip.
6. Measure an intentional 60-second timed run on the actual board/serial link. Record whether the normal stop reaches the controller before its hard limit. Every actual firmware timeout must lock, even if a normal GUI completion occurred first.
7. Reconnect real pump power only after these checks and after confirming the water supply and sensors are ready.

## Software verification

- Baseline: 136 passed, 7 skipped, 2 failed. The failures were calendar placement with a 1280-pixel virtual display and `test_analyze_crop_health_chlorosis` (pre-existing plant-color classification mismatch).
- A 1920 × 1080 virtual display resolves the calendar placement failure without changing calendar code or assertions.
- Final full suite: 168 passed, 7 skipped, 1 failed; the remaining failure is the same pre-existing chlorosis test. Command: `xvfb-run -a -s '-screen 0 1920x1080x24' .venv/bin/python -m pytest -q --tb=short`. The installed environment was used directly because sandbox access to uv's cache was unavailable.
- Final affected suite (GUI, irrigation safety, persistence, firmware, parsing, and logging): 128 passed.
- Production firmware compiles for `arduino:avr:uno` and `arduino:avr:mega`. The Uno build uses 18,282 bytes of flash and 1,093 bytes of static RAM (baseline: 14,300 and 1,135 bytes respectively). New diagnostic literals use flash storage to preserve RAM headroom.
- Host-compiled production lock/journal tests exercise global shutdown, stale reset rejection, clock/generation rollover, corrupt records, and simulated interruptions during trip/reset persistence. Trip interruption tests begin after the first persistent lock-marker write; physical output and power-cut behavior remain bench-verification items.
