"""CEA Irrigation Controller Interactive CLI.

Connects over Serial (Windows COMx or Linux/Raspberry Pi /dev/tty*)
to the Arduino controller firmware to monitor sensors and control pump relays.

Features:
  - Dynamically detects number of active channels (1 to 5 pumps & soil sensors).
  - Explicit input buffer clearing to guarantee fresh telemetry readings.
  - Interactive shell for bitmask dispatch (e.g. "0001", "1111", "0000") and live telemetry.

Usage:
    uv run python scripts/controller.py
    uv run python scripts/controller.py --port COM3 --baud 115200
"""

import argparse
import json
import sys
import threading
import time
from datetime import datetime
import serial
import serial.tools.list_ports


def find_arduino_port() -> str | None:
    """Auto-detect Arduino or USB Serial adapter port."""
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        return None

    # Priority 1: Known Arduino/DFRobot VIDs
    for p in ports:
        if p.vid in (0x2341, 0x3343, 0x1A86, 0x10C4):
            return p.device

    # Priority 2: Keyword match in description or device name
    for p in ports:
        desc = (p.description or "").lower()
        dev = (p.device or "").lower()
        if any(k in desc or k in dev for k in ["arduino", "serial", "ch340", "cp210", "ftdi", "ttyacm", "ttyusb"]):
            return p.device

    # Priority 3: First available port
    return ports[0].device


def list_ports():
    """Display all connected serial ports."""
    ports = list(serial.tools.list_ports.comports())
    print("\n--- Available Serial Ports ---")
    if not ports:
        print("  No serial devices detected.")
    for p in ports:
        vid_pid = f"(VID: 0x{p.vid:04X}, PID: 0x{p.pid:04X})" if p.vid and p.pid else ""
        print(f"  * {p.device:12} - {p.description} {vid_pid}")
    print("------------------------------\n")


def parse_telemetry_line(raw_line: str) -> dict | None:
    """Attempt to parse a JSON telemetry string from Arduino."""
    line = raw_line.strip()
    if not (line.startswith("{") and line.endswith("}")):
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def read_fresh_telemetry(ser: serial.Serial, timeout_sec: float = 3.0) -> dict | None:
    """Flush input buffer to discard stale readings and fetch the latest fresh telemetry frame."""
    ser.reset_input_buffer()
    start_time = time.time()
    while (time.time() - start_time) < timeout_sec:
        line_bytes = ser.readline()
        if not line_bytes:
            continue
        line_str = line_bytes.decode("utf-8", errors="replace").strip()
        data = parse_telemetry_line(line_str)
        if data and "soil" in data:
            return data
    return None


def format_telemetry_compact(data: dict) -> str:
    """Format JSON telemetry into a single-line summary."""
    now = datetime.now().strftime("%H:%M:%S")
    soil_vals = data.get("soil", [])
    soil_str = "[" + ", ".join(f"{v:3d}" if isinstance(v, int) else str(v) for v in soil_vals) + "]"
    temp_val = data.get("temp")
    humi_val = data.get("humidity")
    temp_str = f"{temp_val:.1f}C" if isinstance(temp_val, (int, float)) else "N/A"
    humi_str = f"{humi_val:.1f}%" if isinstance(humi_val, (int, float)) else "N/A"
    light_val = data.get("light", "N/A")
    relays_str = data.get("relays", "")
    return f"[{now}] Light: {str(light_val):<4} | Temp: {temp_str:<6} | RH: {humi_str:<6} | Soil: {soil_str:<18} | Relays: {relays_str}"


def send_bitmask(ser: serial.Serial, mask: str, expected_len: int) -> bool:
    """Send a bitmask to Arduino and flush write buffer."""
    if len(mask) != expected_len or not all(c in "01" for c in mask):
        print(f"[Error] Invalid bitmask '{mask}'. Expected {expected_len} digits of 0 or 1.")
        return False

    ser.write((mask + "\n").encode("utf-8"))
    ser.flush()
    return True


def main():
    parser = argparse.ArgumentParser(description="CEA Irrigation Controller Interactive CLI")
    parser.add_argument("-p", "--port", type=str, default=None, help="Serial port (e.g. COM3, /dev/ttyACM0)")
    parser.add_argument("-b", "--baud", type=int, default=9600, help="Baud rate (default: 9600)")
    parser.add_argument("-l", "--list", action="store_true", help="List available serial ports and exit")
    parser.add_argument("-c", "--channels", type=int, default=None, choices=range(1, 6), help="Override channel count (1-5)")
    args = parser.parse_args()

    if args.list:
        list_ports()
        return

    port = args.port or find_arduino_port()
    if not port:
        print("[Error] No serial port detected. Please specify one with `--port`.")
        list_ports()
        sys.exit(1)

    print(f"Connecting to Arduino on {port} at {args.baud} baud...")
    try:
        ser = serial.Serial(port=port, baudrate=args.baud, timeout=1.0)
        time.sleep(2.0)  # Wait for Arduino DTR reset
        ser.reset_input_buffer()
    except serial.SerialException as e:
        print(f"[Error] Failed to open {port}: {e}")
        print("Tip: Make sure the Arduino IDE Serial Monitor or other serial terminals are closed.")
        return

    print("Fetching initial telemetry...")
    initial_data = read_fresh_telemetry(ser, timeout_sec=4.0)

    if initial_data and "soil" in initial_data:
        num_channels = args.channels or len(initial_data["soil"])
        print(f"--> Connected! Detected {num_channels} active channels.")
        print(f"--> Initial Reading: {format_telemetry_compact(initial_data)}")
    else:
        num_channels = args.channels or 4
        print(f"--> Connected! (Defaulting to {num_channels} channels)")

    zero_mask = "0" * num_channels

    print("=" * 63)
    print(f"     CEA IRRIGATION CONTROLLER INTERACTIVE SHELL ({num_channels} CHANNELS)")
    print("=" * 63)
    print(f"  * Send Bitmask:  Type {num_channels} digits (e.g. '{'0' * (num_channels-1) + '1'}', '{'1' * num_channels}', '{zero_mask}')")
    print(f"  * Exit:          Type 'exit' / 'q' (turns off all relays and exits)")
    print("=" * 63 + "\n")

    stop_event = threading.Event()

    def background_listener():
        while not stop_event.is_set():
            try:
                if ser.in_waiting:
                    line_bytes = ser.readline()
                    if line_bytes:
                        line_str = line_bytes.decode("utf-8", errors="replace").strip()
                        data = parse_telemetry_line(line_str)
                        if data and "soil" in data:
                            print(f"\n{format_telemetry_compact(data)}\n> ", end="", flush=True)
                        elif line_str:
                            print(f"\n[Arduino] {line_str}\n> ", end="", flush=True)
                else:
                    time.sleep(0.05)
            except Exception:
                break

    listener_thread = threading.Thread(target=background_listener, daemon=True)
    listener_thread.start()

    try:
        while True:
            cmd = input("> ").strip().lower()
            if not cmd:
                continue

            if cmd in ("exit", "quit", "q"):
                print("Turning off all relays before exit...")
                send_bitmask(ser, zero_mask, num_channels)
                time.sleep(0.3)
                break

            if len(cmd) == num_channels and all(c in "01" for c in cmd):
                send_bitmask(ser, cmd, num_channels)
                continue

            print(f"[Warning] Invalid input '{cmd}'. Please enter a {num_channels}-digit bitmask (e.g. '{zero_mask}') or 'exit'.")

    except (KeyboardInterrupt, EOFError):
        print("\nInterrupted. Shutting down all pumps...")
        try:
            send_bitmask(ser, zero_mask, num_channels)
            time.sleep(0.3)
        except Exception:
            pass
    finally:
        stop_event.set()
        ser.close()
        print("Serial connection closed. Exited cleanly.")


if __name__ == "__main__":
    main()
