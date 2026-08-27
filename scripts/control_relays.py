"""5-Pump Relay Manual Controller (5-Digit Bitmask Protocol).

Controls 5 relays individually or in batch using 5-digit bitmasks (e.g. '00100').

Usage:
    uv run python scripts/control_relays.py
"""

import sys
import threading
import time
import serial
import serial.tools.list_ports


def find_arduino_port() -> str:
    """Find Arduino COM port or fallback to COM3."""
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if p.vid in (0x2341, 0x3343, 0x1A86, 0x10C4) or "serial" in p.description.lower():
            return p.device
    return "COM3"


def listen_serial(ser: serial.Serial, stop_flag: list):
    """Print incoming messages from Arduino in the background."""
    while not stop_flag[0]:
        try:
            if ser.in_waiting:
                line = ser.readline().decode("utf-8", errors="replace").strip()
                if line:
                    print(f"\n[Arduino] {line}\n> ", end="", flush=True)
            else:
                time.sleep(0.05)
        except Exception:
            break


def send_bitmask(ser: serial.Serial, states: list[int]):
    """Send 5-digit bitmask over serial."""
    mask = "".join(str(s) for s in states)
    ser.write((mask + "\n").encode("utf-8"))
    ser.flush()
    return mask


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else find_arduino_port()
    baud = 9600

    print(f"Connecting to Arduino on {port} at {baud} baud...")
    print("Tip: Make sure the Arduino IDE Serial Monitor is CLOSED.\n")

    try:
        ser = serial.Serial(port=port, baudrate=baud, timeout=1.0)
        time.sleep(2.0)  # Wait for Arduino auto-reset
        ser.reset_input_buffer()
    except serial.SerialException as e:
        print(f"[Error] Could not open {port}: {e}")
        return

    # Track states of pumps 1 to 5: 0=OFF, 1=ON
    pump_states = [0, 0, 0, 0, 0]

    stop_flag = [False]
    listener = threading.Thread(target=listen_serial, args=(ser, stop_flag), daemon=True)
    listener.start()

    print("=" * 55)
    print("      5-PUMP RELAY CONTROLLER (BITMASK PROTOCOL)")
    print("=" * 55)
    print("  Direct Bitmask: Type 5 digits like '00100', '11000'")
    print("  Single Pump:    '<1-5> on' or '<1-5> off' (e.g. '3 on')")
    print("  All Pumps:      'all on' or 'all off'")
    print("  Query State:    'status'")
    print("  Exit:           'exit' (shuts off all pumps)")
    print("=" * 55)

    try:
        while True:
            cmd = input("> ").strip().lower()
            if not cmd:
                continue

            if cmd in ("exit", "quit", "q"):
                send_bitmask(ser, [0, 0, 0, 0, 0])
                time.sleep(0.2)
                break

            # Case 1: Direct 5-digit bitmask entered (e.g. '00100')
            if len(cmd) == 5 and all(c in "01" for c in cmd):
                pump_states = [int(c) for c in cmd]
                send_bitmask(ser, pump_states)
                continue

            # Case 2: 'all on' / 'all off'
            if cmd in ("all on", "all_on", "on"):
                pump_states = [1, 1, 1, 1, 1]
                send_bitmask(ser, pump_states)
                continue

            if cmd in ("all off", "all_off", "off", "stop"):
                pump_states = [0, 0, 0, 0, 0]
                send_bitmask(ser, pump_states)
                continue

            if cmd == "status":
                ser.write(b"STATUS\n")
                ser.flush()
                continue

            # Case 3: Single pump toggle: '<1-5> on' or '<1-5> off'
            parts = cmd.split()
            if len(parts) == 2 and parts[0].isdigit():
                pump_num = int(parts[0])
                action = parts[1]

                if 1 <= pump_num <= 5 and action in ("on", "1", "off", "0"):
                    state = 1 if action in ("on", "1") else 0
                    pump_states[pump_num - 1] = state
                    mask = send_bitmask(ser, pump_states)
                    print(f"Set Pump {pump_num} to {'ON' if state else 'OFF'} -> Sent bitmask: {mask}")
                    continue

            print("Invalid input. Examples: '3 on', '3 off', '00100', 'all off', 'status'")

    except (KeyboardInterrupt, EOFError):
        send_bitmask(ser, [0, 0, 0, 0, 0])
    finally:
        stop_flag[0] = True
        ser.close()
        print("\nDisconnected.")


if __name__ == "__main__":
    main()
