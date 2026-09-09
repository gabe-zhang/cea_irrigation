"""CEA Irrigation Controller Interactive CLI.

Connects over Serial (Windows COMx or Linux/Raspberry Pi /dev/tty*)
to the Arduino controller firmware to monitor sensors and control pump relays
and pan/tilt servos.

Features:
  - Robust CSV telemetry parser that gracefully handles missing sensors ('null' values).
  - Displays Soil Moisture (up to 4 channels), Soil Temp, Air Temp & Humidity, Relays, Pan & Tilt.
  - Interactive shell supporting:
      * Relay bitmasks (e.g. '0000', '1000', '01')
      * Pan angle commands (e.g. 'p 65', 'p 0', 'p 130')
      * Tilt angle commands (e.g. 't 30', 't 45', 't 60')
      * Servo home ('h', 'c')
      * Clean exit ('exit', 'q') with relay shutoff and servo homing.

Usage:
    uv run python scripts/controller.py
    uv run python scripts/controller.py --port COM3 --baud 9600
"""

import argparse
import sys
import threading
import time
from datetime import datetime
import serial
import serial.tools.list_ports

# Hardware safety bounds
PAN_MIN, PAN_MAX = 0, 130
TILT_MIN, TILT_MAX = 0, 60


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


def _safe_float(val: str) -> float | None:
    """Convert string to float, treating 'null', 'nan', or empty as None."""
    s = val.strip().lower()
    if not s or s in ("null", "none", "nan"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _safe_int(val: str) -> int | None:
    """Convert string to int, treating 'null', 'nan', or empty as None."""
    s = val.strip().lower()
    if not s or s in ("null", "none", "nan"):
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def parse_telemetry_line(raw_line: str) -> dict | None:
    """Parse CSV telemetry string into a dictionary.

    Supports:
      1. Tagged CSV (Self-describing):
         soil,s1,s2,s3,s4,soil_temp,val,temp,val,humi,val,light,val,relays,mask,pan,val,tilt,val
      2. Positional CSV (Fallback):
         s1,s2,s3,s4,soil_temp,temp,humidity,light,relays,pan,tilt

    Returns None if line is a header, ACK/ERR/FORMAT message, or corrupt.
    """
    line = raw_line.strip()
    if not line:
        return None

    # Ignore headers or system status lines
    line_lower = line.lower()
    if line_lower.startswith(("soil1,", "format:", "status:", "ack:", "err:")):
        return None

    tokens = [t.strip() for t in line.split(",")]

    # Case 1: Tagged CSV format
    if "soil" in [t.lower() for t in tokens]:
        data = {
            "soil": [None, None, None, None],
            "soil_temp": None,
            "temp": None,
            "humidity": None,
            "light": None,
            "relays": "0000",
            "pan": 65,
            "tilt": 60,
        }
        try:
            i = 0
            n = len(tokens)
            while i < n:
                tag = tokens[i].lower()
                if tag == "soil":
                    soil_vals = []
                    j = i + 1
                    while j < n and len(soil_vals) < 4:
                        val_str = tokens[j]
                        if val_str.lower() in ("soil_temp", "temp", "humi", "light", "relays", "pan", "tilt"):
                            break
                        soil_vals.append(_safe_int(val_str))
                        j += 1
                    while len(soil_vals) < 4:
                        soil_vals.append(None)
                    data["soil"] = soil_vals
                    i = j
                    continue
                elif tag == "soil_temp" and i + 1 < n:
                    data["soil_temp"] = _safe_float(tokens[i + 1])
                    i += 2
                elif tag in ("temp", "air_temp") and i + 1 < n:
                    data["temp"] = _safe_float(tokens[i + 1])
                    i += 2
                elif tag in ("humi", "humidity") and i + 1 < n:
                    data["humidity"] = _safe_float(tokens[i + 1])
                    i += 2
                elif tag == "light" and i + 1 < n:
                    data["light"] = _safe_int(tokens[i + 1])
                    i += 2
                elif tag == "relays" and i + 1 < n:
                    data["relays"] = tokens[i + 1].strip()
                    i += 2
                elif tag == "pan" and i + 1 < n:
                    data["pan"] = _safe_int(tokens[i + 1])
                    i += 2
                elif tag == "tilt" and i + 1 < n:
                    data["tilt"] = _safe_int(tokens[i + 1])
                    i += 2
                else:
                    i += 1
            return data
        except Exception:
            return None

    # Case 2: Positional CSV (11 columns fallback)
    if len(tokens) >= 11 and not any(t.lower() in ("relays", "soil1") for t in tokens):
        try:
            return {
                "soil": [_safe_int(t) for t in tokens[0:4]],
                "soil_temp": _safe_float(tokens[4]),
                "temp": _safe_float(tokens[5]),
                "humidity": _safe_float(tokens[6]),
                "light": _safe_int(tokens[7]),
                "relays": tokens[8].strip(),
                "pan": _safe_int(tokens[9]),
                "tilt": _safe_int(tokens[10]),
            }
        except Exception:
            return None

    return None


def format_telemetry_compact(data: dict) -> str:
    """Format parsed telemetry dictionary into a clean single-line summary."""
    now = datetime.now().strftime("%H:%M:%S")

    # Soil moisture formatting
    soil_list = data.get("soil", [])
    soil_strs = [f"{v:3d}" if v is not None else "---" for v in soil_list]
    soil_repr = "[" + ", ".join(soil_strs) + "]"

    # Soil temp
    st = data.get("soil_temp")
    soil_temp_str = f"{st:.1f}°C" if st is not None else "N/A"

    # Air Temp & Humidity
    t = data.get("temp")
    h = data.get("humidity")
    air_temp_str = f"{t:.1f}°C" if t is not None else "N/A"
    air_humi_str = f"{h:.1f}%" if h is not None else "N/A"

    # Light
    light_val = data.get("light")
    light_str = str(light_val) if light_val is not None else "N/A"

    # Relays
    relays_str = data.get("relays", "N/A")

    # Pan & Tilt
    pan_val = data.get("pan")
    tilt_val = data.get("tilt")
    pan_str = f"{pan_val}°" if pan_val is not None else "N/A"
    tilt_str = f"{tilt_val}°" if tilt_val is not None else "N/A"

    return (
        f"[{now}] Soil: {soil_repr:<19} | SoilTemp: {soil_temp_str:<6} | "
        f"Air: {air_temp_str:<6} {air_humi_str:<6} | Light: {light_str:<3} | "
        f"Relays: {relays_str:<4} | Pan: {pan_str:<4} | Tilt: {tilt_str:<3}"
    )


def read_fresh_telemetry(ser: serial.Serial, timeout_sec: float = 3.0) -> dict | None:
    """Flush input buffer and wait for the latest fresh CSV telemetry frame."""
    ser.reset_input_buffer()
    start_time = time.time()
    while (time.time() - start_time) < timeout_sec:
        line_bytes = ser.readline()
        if not line_bytes:
            continue
        line_str = line_bytes.decode("utf-8", errors="replace").strip()
        data = parse_telemetry_line(line_str)
        if data is not None:
            return data
    return None


def send_command(ser: serial.Serial, cmd: str) -> bool:
    """Send a command line to Arduino and flush write buffer."""
    try:
        ser.write((cmd.strip() + "\n").encode("utf-8"))
        ser.flush()
        return True
    except Exception as e:
        print(f"[Serial Write Error] {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="CEA Irrigation Controller Interactive CLI")
    parser.add_argument("-p", "--port", type=str, default=None, help="Serial port (e.g. COM3, /dev/ttyACM0)")
    parser.add_argument("-b", "--baud", type=int, default=9600, help="Baud rate (default: 9600)")
    parser.add_argument("-l", "--list", action="store_true", help="List available serial ports and exit")
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
        print("Tip: Make sure the Arduino IDE Serial Monitor or other terminals are closed.")
        return

    print("Fetching initial telemetry...")
    initial_data = read_fresh_telemetry(ser, timeout_sec=4.0)
    if initial_data:
        print(f"--> Connected! Initial Reading:\n    {format_telemetry_compact(initial_data)}")
    else:
        print("--> Connected! (Waiting for periodic telemetry broadcast...)")

    print("=" * 68)
    print("               CEA IRRIGATION CONTROLLER INTERACTIVE SHELL")
    print("=" * 68)
    print("  * Relays Bitmask: Type 1-4 digits (e.g. '0000', '1000', '1111')")
    print("  * Pan Control:    'p <0-130>'    (e.g. 'p 55', 'p 0', 'p 130')")
    print("  * Tilt Control:   't <0-60>'     (e.g. 't 30', 't 45', 't 60')")
    print("  * Home:           'h' / 'c'      (Resets Pan 55, Tilt 30)")
    print("  * Exit:           'exit' / 'q'   (Turns off relays, homes servos, exits)")
    print("=" * 68 + "\n")

    stop_event = threading.Event()

    def background_listener():
        while not stop_event.is_set():
            try:
                if ser.in_waiting:
                    line_bytes = ser.readline()
                    if line_bytes:
                        line_str = line_bytes.decode("utf-8", errors="replace").strip()
                        data = parse_telemetry_line(line_str)
                        if data is not None:
                            print(f"\n{format_telemetry_compact(data)}\n> ", end="", flush=True)
                        elif line_str:
                            # Print ACKs, headers, and status messages cleanly
                            print(f"\n[Arduino] {line_str}\n> ", end="", flush=True)
                else:
                    time.sleep(0.05)
            except Exception:
                break

    listener_thread = threading.Thread(target=background_listener, daemon=True)
    listener_thread.start()

    try:
        while True:
            cmd = input("> ").strip()
            if not cmd:
                continue

            cmd_lower = cmd.lower()

            # Exit command
            if cmd_lower in ("exit", "quit", "q"):
                print("Turning off relays and homing servos before exit...")
                send_command(ser, "0000")
                time.sleep(0.1)
                send_command(ser, "h")
                time.sleep(0.3)
                break

            # Home command ('h' or legacy 'c')
            if cmd_lower in ("h", "c"):
                send_command(ser, "h")
                continue

            # Pan command ('p <angle>' or 'P <angle>')
            if cmd_lower.startswith("p"):
                parts = cmd_lower.split()
                if len(parts) == 2 and parts[1].lstrip("-").isdigit():
                    angle = int(parts[1])
                    if angle < PAN_MIN or angle > PAN_MAX:
                        print(f"[Warning] Pan angle {angle}° clamped to safety bounds [{PAN_MIN}, {PAN_MAX}].")
                        angle = max(PAN_MIN, min(PAN_MAX, angle))
                    send_command(ser, f"p {angle}")
                    continue
                else:
                    print("[Error] Invalid pan command. Use 'p <0-130>' (e.g. 'p 65').")
                    continue

            # Tilt command ('t <angle>' or 'T <angle>')
            if cmd_lower.startswith("t"):
                parts = cmd_lower.split()
                if len(parts) == 2 and parts[1].lstrip("-").isdigit():
                    angle = int(parts[1])
                    if angle < TILT_MIN or angle > TILT_MAX:
                        print(f"[Warning] Tilt angle {angle}° clamped to safety bounds [{TILT_MIN}, {TILT_MAX}].")
                        angle = max(TILT_MIN, min(TILT_MAX, angle))
                    send_command(ser, f"t {angle}")
                    continue
                else:
                    print("[Error] Invalid tilt command. Use 't <0-60>' (e.g. 't 30').")
                    continue

            # Bitmask command: 1 to 4 digits of '0' and '1'
            if 1 <= len(cmd) <= 4 and all(c in "01" for c in cmd):
                send_command(ser, cmd)
                continue

            print(f"[Warning] Unknown input '{cmd}'.")
            print("  Allowed: 'p <0-130>', 't <0-60>', 'h', 'c', bitmask (e.g. '0000'), or 'exit'.")

    except (KeyboardInterrupt, EOFError):
        print("\nInterrupted. Turning off relays and homing servos...")
        try:
            send_command(ser, "0000")
            time.sleep(0.1)
            send_command(ser, "h")
            time.sleep(0.2)
        except Exception:
            pass
    finally:
        stop_event.set()
        try:
            ser.close()
        except Exception:
            pass
        print("Serial connection closed. Exited cleanly.")


if __name__ == "__main__":
    main()
