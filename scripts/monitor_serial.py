"""Flexible Serial Monitor and Test Script.

Use this script to monitor and interact with any Arduino test sketch
running on Windows (COMx) or Raspberry Pi (/dev/tty*).

Usage:
    uv run python scripts/monitor_serial.py
    uv run python scripts/monitor_serial.py --port COM3 --baud 9600
    uv run python scripts/monitor_serial.py --list
"""

import argparse
import sys
import time
from datetime import datetime
import serial
import serial.tools.list_ports


def list_available_ports():
    """List all available serial ports with descriptions and hardware IDs."""
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("No serial ports found.")
        return

    print("\n--- Available Serial Ports ---")
    for p in ports:
        vid_pid = f"(VID: 0x{p.vid:04X}, PID: 0x{p.pid:04X})" if p.vid and p.pid else ""
        print(f"  * {p.device:12} - {p.description} {vid_pid}")
    print("------------------------------\n")


def auto_detect_port() -> str | None:
    """Auto-detect Arduino or USB Serial adapter port."""
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        return None

    # Priority 1: Known Arduino/DFRobot VIDs (0x2341, 0x3343, 0x1A86, 0x10C4)
    for p in ports:
        if p.vid in (0x2341, 0x3343, 0x1A86, 0x10C4):
            return p.device

    # Priority 2: Keyword match in description
    for p in ports:
        desc = p.description.lower()
        if any(k in desc for k in ["arduino", "serial", "ch340", "cp210", "ftdi", "ttyacm", "ttyusb"]):
            return p.device

    # Fallback to the first available port
    return ports[0].device


def main():
    parser = argparse.ArgumentParser(description="Flexible Arduino Serial Monitor for Windows & Raspberry Pi")
    parser.add_argument("-p", "--port", type=str, default="COM3", help="Serial port (e.g. COM3, /dev/ttyACM0)")
    parser.add_argument("-b", "--baud", type=int, default=9600, help="Baud rate (default: 9600)")
    parser.add_argument("-l", "--list", action="store_true", help="List all available serial ports and exit")
    parser.add_argument("--raw", action="store_true", help="Print raw lines without timestamp prefix")
    args = parser.parse_args()

    if args.list:
        list_available_ports()
        return

    port = args.port or auto_detect_port()
    if not port:
        print("[Error] No serial port detected. Please connect your board and try again.")
        list_available_ports()
        sys.exit(1)

    print(f"==================================================")
    print(f" Connecting to: {port} at {args.baud} baud")
    print(f" Tip: Close the Arduino IDE Serial Monitor if open.")
    print(f" Press Ctrl+C to exit.")
    print(f"==================================================\n")

    try:
        with serial.Serial(port=port, baudrate=args.baud, timeout=1.0) as ser:
            # Give Arduino time to auto-reset upon DTR assertion
            time.sleep(2.0)
            ser.reset_input_buffer()
            print(f"--> Connected to {port}! Waiting for data...\n")

            while True:
                line_bytes = ser.readline()
                if not line_bytes:
                    continue

                line_str = line_bytes.decode("utf-8", errors="replace").strip()
                if not line_str:
                    continue

                if args.raw:
                    print(line_str)
                else:
                    now = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    print(f"[{now}] {line_str}")

    except serial.SerialException as e:
        print(f"\n[Serial Error] Failed to communicate with {port}:")
        print(f"  -> {e}")
        print("\nSuggestions:")
        print("  1. Close the Arduino IDE Serial Monitor tab if it's currently open.")
        print("  2. Check if another script or terminal is using this COM port.")
        print("  3. Run `uv run python scripts/monitor_serial.py --list` to verify the port name.")
    except KeyboardInterrupt:
        print("\n\nMonitor stopped by user.")


if __name__ == "__main__":
    main()
