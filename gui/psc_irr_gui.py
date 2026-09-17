'''
/*****************************************************************************************************
*                                   NOTICE
*
* THIS COMPUTER SOFTWARE CONTAINS PROPRIETARY INFORMATION OF USDA-ARS-SPRU.
* NEITHER RECEIPT NOR POSSESSION THEREOF COVERS ANY RIGHT TO
* REPRODUCE OR USE, OR DISCLOSE IN WHOLE OR IN PART, ANY SUCH INFORMATION
* WITHOUT WRITTEN AUTHORIZATION FROM USDA-ARS/ALARC.  
* 
* ****************************************************************************************************
* Program: PSC_VG_GUI (GUI-based Pi Serial Communication for Vertual Garden)
* ****************************************************************************************************
* Purpose: Serial communication to Arduino from PC or RPi
* Author: James Kim (james.y.kim@usda.gov), Yuan Zhang
* Version:  V2.0
* Date:  2025-08-08  
* Revision: 2026-08-26
* ****************************************************************************************************/
'''

from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path
import signal
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog

import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.animation as animation
import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from PIL import Image, ImageTk
import serial
import serial.tools.list_ports

try:
    from gui.modules.arducam import Camera
    from gui.modules.plant_ai import PlantAIDetector
except (ImportError, ModuleNotFoundError):
    from modules.arducam import Camera
    from modules.plant_ai import PlantAIDetector

# Constants & Soil Calibration
BAUDRATE = 9600
SOIL_WATER_SETPOINT = 40.0  # Trigger pump below this moisture % in AUTO mode
SCHEDULED_TARGET_PCT = 80.0  # Scheduled irrigation shuts off when channel reaches 80%
SCHEDULED_MAX_WATERING_SEC = 180  # 3-minute hard safety timeout
PUMP_FLOW_RATE_LPH = 500.0  # Pump flow rate: 500 L/Hour
PUMP_FLOW_RATE_LPS = PUMP_FLOW_RATE_LPH / 3600.0  # ~0.13889 Liters/Second
DRY_BASELINES = [432.0, 408.0, 427.0, 424.0]
WET_BASELINES = [136.0, 92.0, 159.0, 160.0]

DATA_DIR = Path("Data")
TELEMETRY_DIR = DATA_DIR / "telemetry"
IMAGES_DIR = DATA_DIR / "images"
SCHEDULED_WATERING_DIR = IMAGES_DIR / "scheduled_watering"

for _p in (DATA_DIR, TELEMETRY_DIR, IMAGES_DIR, SCHEDULED_WATERING_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# Hardware Safety Bounds and Home
PAN_MIN, PAN_MAX = 0, 130
TILT_MIN, TILT_MAX = 0, 60
PAN_HOME, TILT_HOME = 55, 30
GIMBAL_STEP = 5  # degrees per nudge click

# Interval mapping
DATA_INTERVAL_MAP = {"10s": 10_000, "1min": 60_000, "1hr": 3_600_000}
IMAGE_INTERVAL_MAP = {"1sec": 1_000, "1min": 60_000, "1hr": 3_600_000, "1day": 86_400_000}


def parse_interval_to_ms(val: str, default_ms: int = 10_000) -> int:
    """Parse interval string into milliseconds.
    
    Supports '10s', '1min', '1hr', 'sec', 'min', 'hr', 'day', etc.
    """
    if not val:
        return default_ms
    s = str(val).strip().lower()
    mapping = {
        "1sec": 1_000,
        "sec": 1_000,
        "1s": 1_000,
        "10s": 10_000,
        "1min": 60_000,
        "min": 60_000,
        "1m": 60_000,
        "1hr": 3_600_000,
        "hr": 3_600_000,
        "1h": 3_600_000,
        "1day": 86_400_000,
        "day": 86_400_000,
        "24hr": 86_400_000,
    }
    return mapping.get(s, default_ms)


def _safe_float(val: str | None) -> float | None:
    """Convert string to float, treating 'null', 'none', 'nan', or empty as None."""
    if val is None:
        return None
    s = str(val).strip().lower()
    if not s or s in ("null", "none", "nan"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _safe_int(val: str | None) -> int | None:
    """Convert string to int, treating 'null', 'none', 'nan', or empty as None."""
    if val is None:
        return None
    s = str(val).strip().lower()
    if not s or s in ("null", "none", "nan"):
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def parse_telemetry_line(raw_line: str) -> dict | None:
    """Parse tagged CSV telemetry string into a dictionary."""
    line = raw_line.strip()
    if not line or line.lower().startswith(("soil1,", "format:", "status:", "ack:", "err:")):
        return None

    tokens = [t.strip() for t in line.split(",")]
    if "soil" not in [t.lower() for t in tokens]:
        return None

    data = {
        "soil": [None, None, None, None],
        "soil_temp": None,
        "temp": None,
        "humidity": None,
        "light": None,
        "relays": "0000",
        "pan": PAN_HOME,
        "tilt": TILT_HOME,
    }
    try:
        i, n = 0, len(tokens)
        while i < n:
            tag = tokens[i].lower()
            if tag == "soil":
                soil_vals, j = [], i + 1
                while j < n and len(soil_vals) < 4:
                    if tokens[j].lower() in ("soil_temp", "temp", "air_temp", "humi", "humidity", "light", "relays", "pan", "tilt"):
                        break
                    soil_vals.append(_safe_int(tokens[j]))
                    j += 1
                data["soil"] = soil_vals + [None] * (4 - len(soil_vals))
                i = j
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


def raw_to_moisture(raw: float | int | None, ch: int) -> float | None:
    """Convert raw soil ADC to 0-100% moisture percentage using two-point dry/wet calibration."""
    if raw is None:
        return None
    dry = DRY_BASELINES[ch] if ch < len(DRY_BASELINES) else (sum(DRY_BASELINES) / len(DRY_BASELINES))
    wet = WET_BASELINES[ch] if ch < len(WET_BASELINES) else (sum(WET_BASELINES) / len(WET_BASELINES))
    if dry == wet:
        return 0.0
    pct = ((dry - float(raw)) / (dry - wet)) * 100.0
    return max(0.0, min(100.0, pct))


def find_arduino_port() -> str | None:
    """Auto-detect connected Arduino or serial adapter port."""
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if getattr(p, "vid", None) in (0x2341, 0x3343, 0x1A86, 0x10C4):
            return p.device
    for p in ports:
        dev = f"{p.description or ''} {p.device or ''}".lower()
        if any(k in dev for k in ("arduino", "serial", "ch340", "cp210", "ftdi", "ttyacm", "ttyusb")):
            return p.device
    return ports[0].device if ports else None


def read_historical_telemetry(telemetry_dir: Path, range_mode: str, now: datetime | None = None) -> dict:
    """Read telemetry CSV records from telemetry_dir matching range_mode ('day', 'week', 'month').
    
    Returns:
        dict with keys: 'timestamps', 'moisture', 'soil_temp', 'air_temp', 'humidity', 'pump_volumes', 'pump_events'
    """
    if now is None:
        now = datetime.now()

    mode = range_mode.strip().lower()
    if mode == "day":
        cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif mode == "week":
        cutoff = now - timedelta(days=7)
    elif mode == "month":
        cutoff = now - timedelta(days=30)
    else:
        cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)

    timestamps: list[datetime] = []
    moistures: list[list[float | None]] = [[], [], [], []]
    soil_temps: list[float | None] = []
    air_temps: list[float | None] = []
    humidities: list[float | None] = []
    pump_volumes: list[list[float]] = [[], [], [], []]
    pump_events: list[dict] = []

    if not telemetry_dir.exists():
        return {
            "timestamps": timestamps,
            "moisture": moistures,
            "soil_temp": soil_temps,
            "air_temp": air_temps,
            "humidity": humidities,
            "pump_volumes": pump_volumes,
            "pump_events": pump_events,
        }

    csv_files = sorted(telemetry_dir.glob("telemetry_*.csv"))
    rows: list[tuple[datetime, list[float | None], float | None, float | None, float | None, str]] = []

    for fpath in csv_files:
        try:
            with fpath.open("r", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if not header:
                    continue
                for row in reader:
                    if not row or len(row) < 12:
                        continue
                    ts_str = row[0].strip()
                    dt = None
                    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
                        try:
                            dt = datetime.strptime(ts_str, fmt)
                            break
                        except ValueError:
                            pass
                    if dt is None or dt < cutoff:
                        continue
                    
                    m1 = _safe_float(row[5])
                    m2 = _safe_float(row[6])
                    m3 = _safe_float(row[7])
                    m4 = _safe_float(row[8])
                    st = _safe_float(row[9])
                    at = _safe_float(row[10])
                    rh = _safe_float(row[11])
                    relays_str = row[13].strip() if len(row) > 13 else "0000"
                    rows.append((dt, [m1, m2, m3, m4], st, at, rh, relays_str))
        except Exception as e:
            print(f"[Telemetry Reader] Error reading {fpath.name}: {e}")

    rows.sort(key=lambda x: x[0])
    active_runs: dict[int, dict] = {}

    for idx, (dt, m_list, st, at, rh, r_str) in enumerate(rows):
        timestamps.append(dt)
        for i in range(4):
            moistures[i].append(m_list[i])
        soil_temps.append(st)
        air_temps.append(at)
        humidities.append(rh)

        if idx > 0:
            dt_step = (dt - rows[idx - 1][0]).total_seconds()
            if dt_step < 0 or dt_step > 300.0:
                dt_step = 0.0
        else:
            dt_step = 0.0

        for i in range(4):
            is_on = (i < len(r_str) and r_str[i] == "1")
            step_vol = (dt_step * PUMP_FLOW_RATE_LPS) if is_on else 0.0
            pump_volumes[i].append(round(step_vol, 3))

            if is_on:
                if i not in active_runs:
                    active_runs[i] = {"start": dt, "last": dt}
                else:
                    active_runs[i]["last"] = dt
            else:
                if i in active_runs:
                    run = active_runs.pop(i)
                    dur = max(1.0, min((dt - run["start"]).total_seconds(), float(SCHEDULED_MAX_WATERING_SEC)))
                    vol = round(dur * PUMP_FLOW_RATE_LPS, 2)
                    pump_events.append({
                        "timestamp": run["start"],
                        "pump": i,
                        "volume": vol,
                        "duration": dur,
                    })

    for i, run in list(active_runs.items()):
        dur = max(1.0, min((run["last"] - run["start"]).total_seconds(), float(SCHEDULED_MAX_WATERING_SEC)))
        if dur <= 1.0 and rows:
            dur = 10.0
        vol = round(dur * PUMP_FLOW_RATE_LPS, 2)
        pump_events.append({
            "timestamp": run["start"],
            "pump": i,
            "volume": vol,
            "duration": dur,
        })

    return {
        "timestamps": timestamps,
        "moisture": moistures,
        "soil_temp": soil_temps,
        "air_temp": air_temps,
        "humidity": humidities,
        "pump_volumes": pump_volumes,
        "pump_events": pump_events,
    }


class PlotWindow:
    """Dynamic Matplotlib window displaying 60s live or historical telemetry."""

    def __init__(
        self,
        master: tk.Tk,
        get_telemetry_fn,
        range_var: tk.StringVar | None = None,
        setpoint_var: tk.DoubleVar | None = None,
        stop_setpoint_var: tk.DoubleVar | None = None,
    ) -> None:
        self.master = master
        self.get_telemetry = get_telemetry_fn
        self.range_var = range_var or tk.StringVar(value="min")
        self.setpoint_var = setpoint_var or getattr(master, "soil_water_setpoint", None)
        self.stop_setpoint_var = stop_setpoint_var or getattr(master, "soil_water_stop_setpoint", None)
        self.window: tk.Toplevel | None = None
        self.canvas_widget: FigureCanvasTkAgg | None = None
        self.ani: animation.FuncAnimation | None = None
        self.max_ch = 4
        self.colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]
        self.xdata: list[float] = []
        self.ydata_moist: list[list[float]] = [[] for _ in range(self.max_ch)]
        self.ydata_soil_temp: list[float] = []
        self.ydata_temp: list[float] = []
        self.ydata_rh: list[float] = []

        self.fig = Figure(figsize=(15.0, 10.5), dpi=100)
        self.lines_moist = []
        self.line_setpoint = None
        self.line_stop = None
        self.line_temp = None
        self.line_soil_temp = None
        self.line_rh = None
        self.ax_moist = None
        self.ax_vol = None
        self.ax_temp = None
        self.ax_rh = None

        self._init_live_figure()
        self._trace_id = self.range_var.trace_add("write", self._on_range_changed)
        self._sp_trace_id = None
        self._stop_sp_trace_id = None
        if self.setpoint_var:
            self._sp_trace_id = self.setpoint_var.trace_add("write", self._on_setpoint_changed)
        if self.stop_setpoint_var:
            self._stop_sp_trace_id = self.stop_setpoint_var.trace_add("write", self._on_stop_setpoint_changed)

    def _on_setpoint_changed(self, *args) -> None:
        if self.window and tk.Toplevel.winfo_exists(self.window):
            mode = self.range_var.get().strip().lower()
            if mode == "min":
                try:
                    sp = float(self.setpoint_var.get())
                    if hasattr(self, "line_setpoint") and self.line_setpoint:
                        self.line_setpoint.set_ydata([sp, sp])
                    if self.canvas_widget:
                        self.canvas_widget.draw_idle()
                except Exception:
                    pass
            else:
                self._render_current_mode()
            self.window.lift()

    def _on_stop_setpoint_changed(self, *args) -> None:
        if self.window and tk.Toplevel.winfo_exists(self.window):
            mode = self.range_var.get().strip().lower()
            if mode == "min":
                try:
                    ssp = float(self.stop_setpoint_var.get())
                    if hasattr(self, "line_stop") and self.line_stop:
                        self.line_stop.set_ydata([ssp, ssp])
                    if self.canvas_widget:
                        self.canvas_widget.draw_idle()
                except Exception:
                    pass
            else:
                self._render_current_mode()
            self.window.lift()

    def _init_live_figure(self) -> None:
        self.fig.clf()
        self.fig.suptitle("Environmental telemetry and soil moisture (Last 60 sec)", fontsize=24, fontweight="bold")

        # Top Plot: Air temp & Soil temp (0-50°C, Left Y-axis) & Relative Humidity (0-100%, Right Y-axis)
        self.ax_temp = self.fig.add_subplot(2, 1, 1)
        self.ax_rh = self.ax_temp.twinx()

        (self.line_temp,) = self.ax_temp.plot([], [], color="#d62728", linewidth=3.6, label="Air temp, °C")
        (self.line_soil_temp,) = self.ax_temp.plot(
            [], [], color="#d95f02", linewidth=3.6, linestyle="--", label="Soil temp, °C"
        )
        (self.line_rh,) = self.ax_rh.plot([], [], color="#00838f", linewidth=3.6, label="RH, %")

        self.ax_temp.set_ylim(0, 50)
        self.ax_temp.set_xlim(0, 60)
        self.ax_temp.set_xticks([0, 10, 20, 30, 40, 50, 60])
        self.ax_temp.set_ylabel("Air temp, °C", fontsize=20, fontweight="bold", color="#d62728")
        self.ax_temp.tick_params(axis="x", labelsize=18)
        self.ax_temp.tick_params(axis="y", labelcolor="#d62728", labelsize=18)
        self.ax_temp.grid(True, linestyle="--", alpha=0.6, linewidth=1.5)

        self.ax_rh.set_ylim(0, 100)
        self.ax_rh.set_ylabel("Relative humidity, %", fontsize=20, fontweight="bold", color="#00838f")
        self.ax_rh.tick_params(axis="y", labelcolor="#00838f", labelsize=18)

        self.ax_temp.legend(
            [self.line_temp, self.line_soil_temp, self.line_rh],
            ["Air temp, °C", "Soil temp, °C", "RH, %"],
            loc="lower right",
            fontsize=18,
            ncol=3,
            framealpha=0.92,
        )

        # Bottom Plot: Soil Moisture (0-100%)
        self.ax_moist = self.fig.add_subplot(2, 1, 2, sharex=self.ax_temp)
        self.lines_moist = []
        for i, c in enumerate(self.colors):
            (line,) = self.ax_moist.plot([], [], color=c, linewidth=3.6, label=f"S{i+1}")
            self.lines_moist.append(line)

        sp = float(self.setpoint_var.get()) if self.setpoint_var else SOIL_WATER_SETPOINT
        ssp = float(self.stop_setpoint_var.get()) if self.stop_setpoint_var else SCHEDULED_TARGET_PCT
        self.line_setpoint = self.ax_moist.axhline(
            sp, color="#e63946", linestyle="--", linewidth=2.5
        )
        self.line_stop = self.ax_moist.axhline(
            ssp, color="#2ecc71", linestyle="--", linewidth=2.5
        )

        self.ax_moist.set_ylim(0, 100)
        self.ax_moist.set_xlim(0, 60)
        self.ax_moist.set_xticks([0, 10, 20, 30, 40, 50, 60])
        self.ax_moist.set_xlabel("Time, sec", fontsize=20, fontweight="bold")
        self.ax_moist.set_ylabel("Soil moisture, %", fontsize=20, fontweight="bold")
        self.ax_moist.tick_params(axis="both", labelsize=18)
        self.ax_moist.grid(True, linestyle="--", alpha=0.6, linewidth=1.5)
        self.ax_moist.legend(loc="lower right", fontsize=18, ncol=4, framealpha=0.92)

        self.fig.tight_layout(rect=[0.02, 0.04, 0.98, 0.95])
        self.fig.subplots_adjust(hspace=0.35)

    def _render_historical_figure(self, mode: str) -> None:
        self.fig.clf()
        hist = read_historical_telemetry(TELEMETRY_DIR, mode)
        ts = hist["timestamps"]
        if not ts:
            ax = self.fig.add_subplot(1, 1, 1)
            ax.text(
                0.5, 0.5,
                f"No historical telemetry records found for range: '{mode}'.\nLogs are saved to {TELEMETRY_DIR}/",
                ha="center", va="center", fontsize=20, color="#6c757d", fontweight="bold"
            )
            ax.axis("off")
            today_str = datetime.now().strftime("%Y-%m-%d")
            if mode == "day":
                self.fig.suptitle(f"Environmental telemetry and soil moisture (Day \u2014 {today_str})", fontsize=24, fontweight="bold")
            else:
                self.fig.suptitle(f"Environmental telemetry and soil moisture ({mode.capitalize()})", fontsize=24, fontweight="bold")
            self.fig.tight_layout(rect=[0.02, 0.04, 0.98, 0.95])
            return

        today_str = datetime.now().strftime("%Y-%m-%d")
        if mode == "day":
            self.fig.suptitle(f"Environmental telemetry and soil moisture (Day \u2014 {today_str})", fontsize=24, fontweight="bold")
        else:
            self.fig.suptitle(f"Environmental telemetry and soil moisture ({mode.capitalize()})", fontsize=24, fontweight="bold")

        # Top Plot: Air temp & Soil temp (0-50°C, Left Y-axis) & Relative Humidity (0-100%, Right Y-axis)
        ax_temp = self.fig.add_subplot(2, 1, 1)
        ax_rh = ax_temp.twinx()

        y_temp = [v if v is not None else np.nan for v in hist["air_temp"]]
        y_soil = [v if v is not None else np.nan for v in hist["soil_temp"]]
        y_rh = [v if v is not None else np.nan for v in hist["humidity"]]

        (l_t,) = ax_temp.plot(ts, y_temp, color="#d62728", linewidth=2.8, label="Air temp, °C")
        (l_s,) = ax_temp.plot(ts, y_soil, color="#d95f02", linewidth=2.8, linestyle="--", label="Soil temp, °C")
        (l_rh,) = ax_rh.plot(ts, y_rh, color="#00838f", linewidth=2.8, label="RH, %")

        ax_temp.set_ylim(0, 50)
        ax_temp.set_ylabel("Air temp, °C", fontsize=20, fontweight="bold", color="#d62728")
        ax_temp.tick_params(axis="both", labelsize=16)
        ax_temp.tick_params(axis="y", labelcolor="#d62728")
        ax_temp.grid(True, linestyle="--", alpha=0.6, linewidth=1.5)

        ax_rh.set_ylim(0, 100)
        ax_rh.set_ylabel("Relative humidity, %", fontsize=20, fontweight="bold", color="#00838f")
        ax_rh.tick_params(axis="y", labelcolor="#00838f", labelsize=16)

        ax_temp.legend([l_t, l_s, l_rh], ["Air temp, °C", "Soil temp, °C", "RH, %"], loc="lower right", fontsize=16, ncol=3, framealpha=0.92)

        # Bottom Plot: Soil Moisture (0-100%) & Secondary Water Volume Axis (mL)
        ax_moist = self.fig.add_subplot(2, 1, 2, sharex=ax_temp)
        ax_vol = ax_moist.twinx()

        for i, c in enumerate(self.colors):
            y_vals = [v if v is not None else np.nan for v in hist["moisture"][i]]
            ax_moist.plot(ts, y_vals, color=c, linewidth=2.8, label=f"S{i+1}")

        sp = float(self.setpoint_var.get()) if self.setpoint_var else SOIL_WATER_SETPOINT
        ssp = float(self.stop_setpoint_var.get()) if self.stop_setpoint_var else SCHEDULED_TARGET_PCT
        ax_moist.axhline(sp, color="#e63946", linestyle="--", linewidth=2.2)
        ax_moist.axhline(ssp, color="#2ecc71", linestyle="--", linewidth=2.2)

        ax_moist.set_ylim(0, 100)
        ax_moist.set_ylabel("Soil moisture, %", fontsize=20, fontweight="bold")
        ax_moist.tick_params(axis="both", labelsize=16)
        ax_moist.grid(True, linestyle="--", alpha=0.6, linewidth=1.5)

        # Plot water volume bars on ax_vol for any pump events (converted to mL)
        pump_events = hist.get("pump_events", [])
        if mode == "day":
            base_w = timedelta(minutes=10)
            offsets = [timedelta(minutes=m) for m in (-15, -5, 5, 15)]
        elif mode == "week":
            base_w = timedelta(hours=1)
            offsets = [timedelta(hours=h) for h in (-1.5, -0.5, 0.5, 1.5)]
        else:  # month
            base_w = timedelta(hours=4)
            offsets = [timedelta(hours=h) for h in (-6, -2, 2, 6)]

        for ev in pump_events:
            p_idx = ev["pump"]
            vol_l = ev["volume"]
            vol_ml = vol_l * 1000.0
            ev_ts = ev["timestamp"]
            if vol_ml > 0:
                ax_vol.bar(
                    ev_ts + offsets[p_idx],
                    vol_ml,
                    width=base_w,
                    color=self.colors[p_idx],
                    alpha=0.45,
                    edgecolor=self.colors[p_idx],
                    linewidth=1.5,
                )

        ax_vol.set_ylabel("Water volume, mL", fontsize=20, fontweight="bold", color="#0288d1")
        ax_vol.tick_params(axis="y", labelcolor="#0288d1", labelsize=16)
        ax_vol.grid(False)
        ax_vol.set_ylim(0, 1000)

        if mode == "day":
            # Fixed 24-hour x-axis from midnight to midnight of current date
            today = datetime.now().date()
            ax_moist.set_xlim(
                datetime.combine(today, datetime.min.time()),
                datetime.combine(today, datetime.max.time().replace(microsecond=0))
            )
            ax_moist.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        elif mode == "week":
            ax_moist.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        else:  # month
            ax_moist.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        ax_moist.set_xlabel("Date / time", fontsize=18, fontweight="bold")

        ax_moist.legend(
            loc="lower right",
            fontsize=16,
            ncol=4,
            framealpha=0.92,
        )

        self.fig.tight_layout(rect=[0.02, 0.04, 0.98, 0.95])
        self.fig.subplots_adjust(hspace=0.35)

    def _render_current_mode(self) -> None:
        mode = self.range_var.get().strip().lower()
        if mode == "min":
            self.xdata.clear()
            for y_m in self.ydata_moist:
                y_m.clear()
            self.ydata_soil_temp.clear()
            self.ydata_temp.clear()
            self.ydata_rh.clear()

            self._init_live_figure()
            if self.canvas_widget:
                self.canvas_widget.draw_idle()
            if self.ani and self.ani.event_source:
                self.ani.event_source.stop()
            self.ani = animation.FuncAnimation(
                self.fig, self._update, self._gen, interval=500, cache_frame_data=False
            )
            if self.window:
                self.window.title("Environmental Telemetry & Soil Moisture (Last 60s)")
        else:
            if self.ani and self.ani.event_source:
                self.ani.event_source.stop()
            self.ani = None

            self._render_historical_figure(mode)
            if self.canvas_widget:
                self.canvas_widget.draw_idle()
            if self.window:
                self.window.title(f"Historical Telemetry ({mode.capitalize()})")

    def _on_range_changed(self, *args) -> None:
        if self.window and tk.Toplevel.winfo_exists(self.window):
            self._render_current_mode()
            self.window.lift()

    def _update(self, data: tuple):
        if len(data) >= 5:
            x, moistures, soil_temp, temp, rh = data[:5]
        elif len(data) == 4:
            x, moistures, temp, rh = data
            soil_temp = None
        else:
            x, moistures = data[0], data[1]
            soil_temp, temp, rh = None, None, None
        if not self.xdata or x < self.xdata[-1] or x == 0:
            self.xdata = [x]
            self.ydata_moist = [
                [m if m is not None else np.nan]
                for m in (moistures[:self.max_ch] + [None] * max(0, self.max_ch - len(moistures)))
            ]
            self.ydata_soil_temp = [soil_temp if soil_temp is not None else np.nan]
            self.ydata_temp = [temp if temp is not None else np.nan]
            self.ydata_rh = [rh if rh is not None else np.nan]
        else:
            self.xdata.append(x)
            for i in range(self.max_ch):
                m = moistures[i] if i < len(moistures) else None
                self.ydata_moist[i].append(m if m is not None else np.nan)
            self.ydata_soil_temp.append(soil_temp if soil_temp is not None else np.nan)
            self.ydata_temp.append(temp if temp is not None else np.nan)
            self.ydata_rh.append(rh if rh is not None else np.nan)

        for i in range(self.max_ch):
            if i < len(self.lines_moist):
                self.lines_moist[i].set_data(self.xdata, self.ydata_moist[i])

        if self.line_temp:
            self.line_temp.set_data(self.xdata, self.ydata_temp)
        if self.line_soil_temp:
            self.line_soil_temp.set_data(self.xdata, self.ydata_soil_temp)
        if self.line_rh:
            self.line_rh.set_data(self.xdata, self.ydata_rh)

        return tuple(self.lines_moist) + (self.line_temp, self.line_soil_temp, self.line_rh)

    def _gen(self):
        t0 = time.time()
        while True:
            elapsed = time.time() - t0
            if elapsed >= 60.0:
                t0, elapsed = time.time(), 0.0

            raw_data = self.get_telemetry()
            if isinstance(raw_data, dict):
                moistures = raw_data.get("moisture_pct", [])
                soil_temp = raw_data.get("soil_temp")
                temp = raw_data.get("temp")
                rh = raw_data.get("humidity")
            elif isinstance(raw_data, (list, tuple)):
                moistures = list(raw_data)
                soil_temp, temp, rh = None, None, None
            else:
                moistures, soil_temp, temp, rh = [], None, None, None

            yield round(elapsed, 1), moistures, soil_temp, temp, rh
            time.sleep(0.5)

    def toggle(self, show: bool, on_close=None) -> None:
        if show:
            if self.window is None or not tk.Toplevel.winfo_exists(self.window):
                self.window = tk.Toplevel(self.master)
                scr_w = self.master.winfo_screenwidth()
                scr_h = getattr(self.master, "scr_h", self.master.winfo_screenheight() - 75)
                margin_w = getattr(self.master, "margin_w", max(380, int(scr_w / 4.0)))

                # Expose right control panel (width margin_w) and keep plot window docked to the left
                win_w = max(600, scr_w - margin_w - 6)
                win_h = max(600, scr_h)
                pos_x = 0
                pos_y = 0

                self.window.geometry(f"{win_w}x{win_h}+{pos_x}+{pos_y}")
                self.window.attributes("-topmost", True)
                if self.master and getattr(self.master, "winfo_viewable", lambda: False)():
                    try:
                        self.window.transient(self.master)
                    except Exception:
                        pass
                self.window.protocol("WM_DELETE_WINDOW", lambda: (self.toggle(False), on_close and on_close()))
                btn_close = tk.Button(
                    self.window,
                    text="✕ Close",
                    font=("arial", 20, "bold"),
                    bg="#dc3545",
                    fg="white",
                    activebackground="#bb2d3b",
                    activeforeground="white",
                    bd=3,
                    command=lambda: (self.toggle(False), on_close and on_close()),
                )
                btn_close.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=6, ipady=8)
                self.canvas_widget = FigureCanvasTkAgg(self.fig, master=self.window)
                self.canvas_widget.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                self._render_current_mode()
            else:
                self.window.deiconify()
                self.window.attributes("-topmost", True)
                if self.master and getattr(self.master, "winfo_viewable", lambda: False)():
                    try:
                        self.window.transient(self.master)
                    except Exception:
                        pass
                self.window.lift()
        elif self.window and tk.Toplevel.winfo_exists(self.window):
            if self.ani and self.ani.event_source:
                self.ani.event_source.stop()
            self.window.destroy()
            self.window, self.ani, self.canvas_widget = None, None, None


class MainWindow(tk.Tk):
    """Main Application coordinating Camera, Serial Telemetry, and User Interface."""

    def __init__(self) -> None:
        super().__init__()
        self.title("CEA Irrigation Controller GUI")

        self.scr_w = self.winfo_screenwidth()
        self.scr_h = self.winfo_screenheight() - 75
        self.geometry(f"{self.scr_w}x{self.scr_h}+0+0")
        self.margin_w = max(380, int(self.scr_w / 4.0))
        margin_w = self.margin_w

        # Hardware & Telemetry State
        self.ser: serial.Serial | None = None
        self.serial_lock = threading.Lock()  # guards self.ser against reader/writer races
        self.stop_threads = threading.Event()
        self.telemetry = {
            "soil": [], "moisture_pct": [], "soil_temp": None,
            "temp": None, "humidity": None, "light": None,
            "relays": "0000", "pan": PAN_HOME, "tilt": TILT_HOME
        }
        self.current_pan, self.current_tilt = PAN_HOME, TILT_HOME
        self._repeat_job: str | None = None
        self._data_logger_job: str | None = None
        self._image_logger_job: str | None = None
        self._scheduled_ticker_job: str | None = None
        self._scheduled_monitor_job: str | None = None

        self.camera = Camera(width=self.scr_w - margin_w, height=self.scr_h - int(self.scr_h / 5))
        self.plant_ai = PlantAIDetector()
        self.flag_capture = False
        self.photo_ref: ImageTk.PhotoImage | None = None

        # State Models & Options
        self.auto_var = tk.IntVar(value=1)
        self.soil_water_setpoint = tk.DoubleVar(value=SOIL_WATER_SETPOINT)
        self.soil_water_stop_setpoint = tk.DoubleVar(value=SCHEDULED_TARGET_PCT)
        self.start_setpoint_display = tk.StringVar(value=f"{SOIL_WATER_SETPOINT:.0f}%")
        self.stop_setpoint_display = tk.StringVar(value=f"{SCHEDULED_TARGET_PCT:.0f}%")
        self.plot_var = tk.IntVar(value=0)
        self.plot_range_var = tk.StringVar(value="min")
        self.data_record_var = tk.StringVar(value="10s")
        self.image_record_var = tk.StringVar(value="1hr")
        self.live_var = tk.IntVar(value=1)
        self.plant_ai_var = tk.IntVar(value=0)
        self.heatmap_var = tk.IntVar(value=0)

        # Scheduled Watering State
        self._last_scheduled_date: str | None = None
        self._scheduled_watering_active: bool = False
        self._scheduled_run_dir: Path | None = None
        self._scheduled_start_time: float | None = None
        self._scheduled_channels_active: list[int] = []

        # Build UI Layout
        paned = tk.PanedWindow(self, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=4)
        paned.pack(fill=tk.BOTH, expand=True)
        left_frame = tk.Frame(paned, relief=tk.SUNKEN)
        right_frame = tk.Frame(paned, width=margin_w, bg="#f8f9fa")
        paned.add(left_frame, stretch="always")
        paned.add(right_frame, stretch="never", width=margin_w)

        self._build_bottom_bar(left_frame)

        # Left Camera Canvas (scrollbars removed, frame fits automatically)
        self.x_scrl = None
        self.y_scrl = None
        self._last_display_img: np.ndarray | Image.Image | None = None
        self.canvas = tk.Canvas(left_frame, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # Scrollable Sidebar Container
        self.sidebar_canvas = tk.Canvas(right_frame, bg="#f8f9fa", highlightthickness=0)
        self.sidebar_scrl = tk.Scrollbar(right_frame, orient=tk.VERTICAL, command=self.sidebar_canvas.yview)
        self.sidebar_content = tk.Frame(self.sidebar_canvas, bg="#f8f9fa")

        self.sidebar_win_id = self.sidebar_canvas.create_window((0, 0), window=self.sidebar_content, anchor="nw")
        self.sidebar_canvas.config(yscrollcommand=self.sidebar_scrl.set)

        self.sidebar_scrl.pack(side=tk.RIGHT, fill=tk.Y)
        self.sidebar_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.sidebar_content.bind("<Configure>", lambda e: self.sidebar_canvas.configure(scrollregion=self.sidebar_canvas.bbox("all")))
        self.sidebar_canvas.bind("<Configure>", lambda e: self.sidebar_canvas.itemconfig(self.sidebar_win_id, width=e.width))

        def _on_mousewheel(event):
            if event.num == 4:
                self.sidebar_canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                self.sidebar_canvas.yview_scroll(1, "units")
            elif getattr(event, "delta", 0):
                self.sidebar_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        self.sidebar_canvas.bind_all("<Button-4>", _on_mousewheel)
        self.sidebar_canvas.bind_all("<Button-5>", _on_mousewheel)

        # Build Sidebar Cards
        self._build_sidebar_cards()

        self.plotter = PlotWindow(self, lambda: self.telemetry, range_var=self.plot_range_var, setpoint_var=self.soil_water_setpoint)
        self._build_menu()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self._init_serial()
        self._camera_loop()
        self._auto_loop()
        self._start_periodic_loggers()
        self._start_scheduled_ticker()

    def _adjust_stop_setpoint(self, delta: float) -> None:
        try:
            cur = float(self.soil_water_stop_setpoint.get())
        except (ValueError, tk.TclError):
            cur = SCHEDULED_TARGET_PCT
        new_val = round(cur + delta, 1)
        new_val = max(10.0, min(100.0, new_val))
        self.soil_water_stop_setpoint.set(new_val)

    def _adjust_start_setpoint(self, delta: float) -> None:
        try:
            cur = float(self.soil_water_setpoint.get())
        except (ValueError, tk.TclError):
            cur = SOIL_WATER_SETPOINT
        new_val = round(cur + delta, 1)
        new_val = max(10.0, min(90.0, new_val))
        self.soil_water_setpoint.set(new_val)

    def _update_setpoint_displays(self, *_args) -> None:
        """Keep the large threshold readouts synchronized with their variables."""
        self.start_setpoint_display.set(f"{float(self.soil_water_setpoint.get()):.0f}%")
        self.stop_setpoint_display.set(f"{float(self.soil_water_stop_setpoint.get()):.0f}%")

    def _build_sidebar_cards(self) -> None:
        # Card 1: IRRIGATION
        card_irrigation = tk.LabelFrame(
            self.sidebar_content, text=" IRRIGATION ", font=("arial", 17, "bold"),
            bd=2, relief=tk.GROOVE, bg="#ffffff", fg="#212529"
        )
        card_irrigation.pack(fill=tk.X, padx=10, pady=(6, 5))
        card_irrigation.grid_columnconfigure(0, weight=1)
        card_irrigation.grid_columnconfigure(1, weight=1)

        self.ckb_auto = tk.Checkbutton(
            card_irrigation, text="AUTO", font=("arial", 23, "bold"), bg="white",
            selectcolor="#2ecc71", bd=3, indicatoron=False, variable=self.auto_var,
            command=self._on_auto_toggle
        )
        self.ckb_auto.grid(row=0, column=0, sticky="nsew", padx=(6, 3), pady=4, ipady=10)

        set_box = tk.Frame(card_irrigation, bg="white")
        set_box.grid(row=0, column=1, sticky="nsew", padx=(3, 6), pady=4)
        # Top row: Turn off at (stop threshold)
        self.off_row = tk.Frame(set_box, bg="white")
        off_row = self.off_row
        off_row.pack(side=tk.TOP, fill=tk.X, padx=2, pady=(2, 1))
        tk.Label(off_row, text="Turn off at", font=("arial", 18, "bold"), bg="white").pack(side=tk.LEFT, padx=(2, 5))
        tk.Label(
            off_row, textvariable=self.stop_setpoint_display, font=("arial", 22, "bold"),
            bg="white", width=4, anchor="e"
        ).pack(side=tk.LEFT, padx=(1, 4))
        self.btn_stop_down = tk.Button(
            off_row, text="−", font=("arial", 19, "bold"), width=2, bd=2, bg="#f1f3f5",
            activebackground="#ced4da", repeatdelay=400, repeatinterval=150,
            command=lambda: self._adjust_stop_setpoint(-5.0)
        )
        self.btn_stop_down.pack(side=tk.LEFT, padx=(1, 2))
        self.btn_stop_up = tk.Button(
            off_row, text="+", font=("arial", 19, "bold"), width=2, bd=2, bg="#f1f3f5",
            activebackground="#ced4da", repeatdelay=400, repeatinterval=150,
            command=lambda: self._adjust_stop_setpoint(5.0)
        )
        self.btn_stop_up.pack(side=tk.LEFT, padx=(2, 2))

        # Bottom row: Turn on at (start threshold)
        self.on_row = tk.Frame(set_box, bg="white")
        on_row = self.on_row
        on_row.pack(side=tk.TOP, fill=tk.X, padx=2, pady=(1, 2))
        tk.Label(on_row, text="Turn on at", font=("arial", 18, "bold"), bg="white").pack(side=tk.LEFT, padx=(2, 5))
        tk.Label(
            on_row, textvariable=self.start_setpoint_display, font=("arial", 22, "bold"),
            bg="white", width=4, anchor="e"
        ).pack(side=tk.LEFT, padx=(1, 4))
        self.btn_start_down = tk.Button(
            on_row, text="−", font=("arial", 19, "bold"), width=2, bd=2, bg="#f1f3f5",
            activebackground="#ced4da", repeatdelay=400, repeatinterval=150,
            command=lambda: self._adjust_start_setpoint(-5.0)
        )
        self.btn_start_down.pack(side=tk.LEFT, padx=(1, 2))
        self.btn_start_up = tk.Button(
            on_row, text="+", font=("arial", 19, "bold"), width=2, bd=2, bg="#f1f3f5",
            activebackground="#ced4da", repeatdelay=400, repeatinterval=150,
            command=lambda: self._adjust_start_setpoint(5.0)
        )
        self.btn_start_up.pack(side=tk.LEFT, padx=(2, 2))
        self.btn_setpoint_down = self.btn_start_down
        self.btn_setpoint_up = self.btn_start_up
        self.btn_manual_mode = tk.Button(
            card_irrigation, text="Manual", font=("arial", 23, "bold"), bg="#6c757d", fg="white",
            activebackground="#5c636a", activeforeground="white", bd=3,
            command=lambda: (self.auto_var.set(1), self._on_auto_toggle())
        )
        self.btn_manual_mode.grid(row=0, column=0, sticky="nsew", padx=(6, 3), pady=4, ipady=10)
        self.soil_water_setpoint.trace_add("write", self._update_setpoint_displays)
        self.soil_water_stop_setpoint.trace_add("write", self._update_setpoint_displays)

        self.water_frame = tk.Frame(card_irrigation, bg="white")
        self.water_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=6, pady=(2, 8))
        self.water_vars: list[tk.IntVar] = []
        self.water_btns: list[tk.Checkbutton] = []
        self.rebuild_relays(4)
        self._on_auto_toggle()

        # Card 2: PLOT (Untitled card with groove border)
        card_plot = tk.Frame(self.sidebar_content, relief=tk.GROOVE, bd=2, bg="#ffffff")
        card_plot.pack(fill=tk.X, padx=10, pady=5)
        card_plot.grid_columnconfigure(0, weight=1)
        card_plot.grid_columnconfigure(1, weight=1)

        self.ckb_plot = tk.Checkbutton(
            card_plot, text="PLOT", font=("arial", 23, "bold"), bg="white",
            selectcolor="#b0bec5", bd=3, indicatoron=False, variable=self.plot_var,
            command=lambda: self.plotter.toggle(bool(self.plot_var.get()), lambda: self.plot_var.set(0))
        )
        self.ckb_plot.grid(row=0, column=0, sticky="nsew", padx=(6, 3), pady=4, ipady=10)

        range_box = tk.Frame(card_plot, bg="white")
        range_box.grid(row=0, column=1, sticky="nsew", padx=(3, 6), pady=4)
        tk.Label(range_box, text="Range:", font=("arial", 18, "bold"), bg="white").pack(side=tk.LEFT, padx=(4, 2))
        self.plot_range_dropdown = tk.OptionMenu(range_box, self.plot_range_var, "min", "day", "week", "month")
        self.plot_range_dropdown.config(font=("arial", 17, "bold"), bg="#f8f9fa", width=5, pady=3)
        try:
            self.plot_range_dropdown["menu"].config(font=("arial", 15, "bold"))
        except Exception:
            pass
        self.plot_range_dropdown.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Card 3: RECORD
        card_record = tk.LabelFrame(
            self.sidebar_content, text=" RECORD ", font=("arial", 17, "bold"),
            bd=2, relief=tk.GROOVE, bg="#ffffff", fg="#212529"
        )
        card_record.pack(fill=tk.X, padx=10, pady=5)
        card_record.grid_columnconfigure(0, weight=1)
        card_record.grid_columnconfigure(1, weight=2)

        tk.Label(card_record, text="Data  :", font=("arial", 18, "bold"), bg="white", anchor="w").grid(
            row=0, column=0, sticky="w", padx=(10, 4), pady=4
        )
        self.data_record_dropdown = tk.OptionMenu(card_record, self.data_record_var, "10s", "1min", "1hr")
        self.data_record_dropdown.config(font=("arial", 17, "bold"), bg="#f8f9fa", width=5, pady=3)
        try:
            self.data_record_dropdown["menu"].config(font=("arial", 15, "bold"))
        except Exception:
            pass
        self.data_record_dropdown.grid(row=0, column=1, sticky="ew", padx=(4, 10), pady=4)

        tk.Label(card_record, text="Image :", font=("arial", 18, "bold"), bg="white", anchor="w").grid(
            row=1, column=0, sticky="w", padx=(10, 4), pady=4
        )
        self.image_record_dropdown = tk.OptionMenu(card_record, self.image_record_var, "1sec", "1min", "1hr", "1day")
        self.image_record_dropdown.config(font=("arial", 17, "bold"), bg="#f8f9fa", width=5, pady=3)
        try:
            self.image_record_dropdown["menu"].config(font=("arial", 15, "bold"))
        except Exception:
            pass
        self.image_record_dropdown.grid(row=1, column=1, sticky="ew", padx=(4, 10), pady=4)

        # Card 4: IMAGE
        card_image = tk.LabelFrame(
            self.sidebar_content, text=" IMAGE ", font=("arial", 17, "bold"),
            bd=2, relief=tk.GROOVE, bg="#ffffff", fg="#212529"
        )
        card_image.pack(fill=tk.X, padx=10, pady=5)
        card_image.grid_columnconfigure(0, weight=1)
        card_image.grid_columnconfigure(1, weight=1)

        self.ckb_live = tk.Checkbutton(
            card_image, text="LIVE", font=("arial", 23, "bold"), bg="white",
            selectcolor="#f1c40f", bd=3, indicatoron=False, variable=self.live_var,
            command=self._on_live_toggle
        )
        self.ckb_live.grid(row=0, column=0, sticky="nsew", padx=(6, 3), pady=4, ipady=10)

        self.btn_capture = tk.Button(
            card_image, text="Capture", font=("arial", 23, "bold"), bg="white", fg="black",
            activebackground="white", activeforeground="black", bd=3, command=self._on_capture
        )
        self.btn_capture.grid(row=0, column=1, sticky="nsew", padx=(3, 6), pady=4, ipady=10)
        self._bind_capture_click_hold()

        # Card 5: PLANT
        card_plant = tk.LabelFrame(
            self.sidebar_content, text=" PLANT ", font=("arial", 17, "bold"),
            bd=2, relief=tk.GROOVE, bg="#ffffff", fg="#212529"
        )
        card_plant.pack(fill=tk.X, padx=10, pady=5)
        card_plant.grid_columnconfigure(0, weight=1)
        card_plant.grid_columnconfigure(1, weight=1)

        self.ckb_plant_ai = tk.Checkbutton(
            card_plant, text="AI", font=("arial", 25, "bold"), bg="white",
            selectcolor="#ffb703", bd=3, indicatoron=False, variable=self.plant_ai_var,
            command=self._on_plant_ai_toggle
        )
        self.ckb_plant_ai.grid(row=0, column=0, sticky="nsew", padx=(6, 3), pady=4, ipady=10)

        self.ckb_heatmap = tk.Checkbutton(
            card_plant, text="Heatmap", font=("arial", 25, "bold"), bg="white",
            selectcolor="#e63946", bd=3, indicatoron=False, variable=self.heatmap_var,
            command=self._on_heatmap_toggle
        )
        self.ckb_heatmap.grid(row=0, column=1, sticky="nsew", padx=(3, 6), pady=4, ipady=10)

        # Card 6: Camera Gimbal (Untitled card with groove border)
        self.gimbal_frame = tk.Frame(self.sidebar_content, bg="#f8f9fa", bd=2, relief=tk.GROOVE)
        self.gimbal_frame.pack(fill=tk.X, padx=10, pady=(6, 4))
        for i in range(3):
            self.gimbal_frame.grid_columnconfigure(i, weight=1, uniform="g_col")
            self.gimbal_frame.grid_rowconfigure(i, weight=1, uniform="g_row")

        self.lbl_gimbal = tk.Label(
            self.gimbal_frame, text="Camera\nGimbal", font=("arial", 22, "bold"),
            fg="#0d6efd", bg="#f8f9fa"
        )
        self.lbl_gimbal.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        gcfg = {
            "font": ("arial", 30, "bold"), "bd": 3, "bg": "#495057", "fg": "white",
            "activebackground": "#6c757d", "activeforeground": "white"
        }
        dpad = [
            ("btn_tilt_up", "▲", 0, 1, lambda: self.nudge_tilt(-GIMBAL_STEP)),
            ("btn_pan_left", "◀", 1, 0, lambda: self.nudge_pan(GIMBAL_STEP)),
            ("btn_pan_right", "▶", 1, 2, lambda: self.nudge_pan(-GIMBAL_STEP)),
            ("btn_tilt_down", "▼", 2, 1, lambda: self.nudge_tilt(GIMBAL_STEP)),
        ]
        for attr, text, r, c, action in dpad:
            btn = tk.Button(self.gimbal_frame, text=text, **gcfg)
            btn.grid(row=r, column=c, sticky="nsew", padx=4, pady=4, ipady=8)
            btn.bind("<ButtonPress-1>", lambda e, a=action: self._start_repeat(a))
            btn.bind("<ButtonRelease-1>", self._stop_repeat)
            btn.bind("<Leave>", self._stop_repeat)
            setattr(self, attr, btn)

        self.btn_home = tk.Button(
            self.gimbal_frame, text="Home", font=("arial", 25, "bold"), bd=3,
            bg="#0d6efd", fg="white", activebackground="#0b5ed7", activeforeground="white",
            command=self.home_gimbal
        )
        self.btn_home.grid(row=1, column=1, sticky="nsew", padx=4, pady=4, ipady=8)

    def _build_bottom_bar(self, parent: tk.Frame) -> None:
        bar = tk.Frame(parent, bg="#1a1d20", height=130, relief=tk.GROOVE, bd=3)
        bar.pack(side=tk.BOTTOM, fill=tk.X, padx=2, pady=2)
        bar.pack_propagate(False)

        self.btn_exit = tk.Button(
            bar,
            text="✕ Exit",
            font=("arial", 18, "bold"),
            bg="#dc3545",
            fg="white",
            activebackground="#bb2d3b",
            activeforeground="white",
            bd=3,
            padx=16,
            pady=8,
            command=self.on_closing,
        )
        self.btn_exit.pack(side=tk.RIGHT, padx=14, pady=10)

        row1 = tk.Frame(bar, bg="#1a1d20")
        row1.pack(side=tk.TOP, fill=tk.X, padx=12, pady=(6, 0))
        tk.Label(row1, text="ENV:", font=("arial", 16, "bold"), fg="#90e0ef", bg="#1a1d20").pack(side=tk.LEFT, padx=(0, 12))

        self.lbl_soil_temp = tk.Label(row1, text="Soil: --.-°C", font=("arial", 16, "bold"), fg="#06d6a0", bg="#1a1d20")
        self.lbl_air_temp = tk.Label(row1, text="Air: --.-°C", font=("arial", 16, "bold"), fg="#ffd166", bg="#1a1d20")
        self.lbl_air_humi = tk.Label(row1, text="RH: --.-%", font=("arial", 16, "bold"), fg="#4cc9f0", bg="#1a1d20")
        self.lbl_light = tk.Label(row1, text="Light: -- lux", font=("arial", 16, "bold"), fg="#f72585", bg="#1a1d20")
        
        tpu_init_text = "AI: Ready" if getattr(self, "plant_ai", None) and self.plant_ai.is_available else "TPU: Disconnected"
        tpu_init_color = "#06d6a0" if getattr(self, "plant_ai", None) and self.plant_ai.is_available else "#e63946"
        self.lbl_tpu_status = tk.Label(row1, text=tpu_init_text, font=("arial", 16, "bold"), fg=tpu_init_color, bg="#1a1d20")

        for lbl in (self.lbl_soil_temp, self.lbl_air_temp, self.lbl_air_humi, self.lbl_light, self.lbl_tpu_status):
            lbl.pack(side=tk.LEFT, padx=10)

        row2 = tk.Frame(bar, bg="#1a1d20")
        row2.pack(side=tk.TOP, fill=tk.X, padx=12, pady=(2, 6))
        tk.Label(row2, text="MOIST:", font=("arial", 16, "bold"), fg="#90e0ef", bg="#1a1d20").pack(side=tk.LEFT, padx=(0, 12))
        self.lbl_soil_moist = tk.Label(row2, text="S1: --  |  S2: --  |  S3: --  |  S4: --", font=("arial", 16, "bold"), fg="#ffffff", bg="#1a1d20")
        self.lbl_soil_moist.pack(side=tk.LEFT, padx=10)

    def _start_periodic_loggers(self) -> None:
        data_ms = parse_interval_to_ms(self.data_record_var.get(), default_ms=10_000)
        self._data_logger_job = self.after(data_ms, self._periodic_data_logger)

        img_ms = parse_interval_to_ms(self.image_record_var.get(), default_ms=3_600_000)
        self._image_logger_job = self.after(img_ms, self._periodic_image_logger)

    def _periodic_data_logger(self) -> None:
        try:
            moist = self.telemetry.get("moisture_pct", [])
            self._log_telemetry_csv(self.telemetry, moist)
        except Exception as e:
            print(f"[Periodic Data Logger] Error: {e}")
        finally:
            interval_ms = parse_interval_to_ms(self.data_record_var.get(), default_ms=10_000)
            self._data_logger_job = self.after(interval_ms, self._periodic_data_logger)

    def _periodic_image_logger(self) -> None:
        try:
            if self.camera and self.camera.is_available:
                clean_frame = self.camera.capture_array()
                if clean_frame is not None:
                    img_path = IMAGES_DIR / f"IMG_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(str(img_path), clean_frame)
                    print(f"[Periodic Image Logger] Saved clean frame to {img_path}")
        except Exception as e:
            print(f"[Periodic Image Logger] Error: {e}")
        finally:
            interval_ms = parse_interval_to_ms(self.image_record_var.get(), default_ms=3_600_000)
            self._image_logger_job = self.after(interval_ms, self._periodic_image_logger)

    def _start_scheduled_ticker(self) -> None:
        self._scheduled_ticker_job = self.after(60_000, self._scheduled_check_ticker)

    def _scheduled_check_ticker(self) -> None:
        try:
            now = datetime.now()
            today_str = now.strftime("%Y-%m-%d")
            if now.hour == 10 and now.minute == 0:
                if self._last_scheduled_date != today_str:
                    if self.auto_var.get():
                        print(f"[Scheduled] 10:00 AM CST check triggered on {today_str}.")
                        self._start_scheduled_watering(today_str)
                    else:
                        print(f"[Scheduled] 10:00 AM CST check skipped: Mode is MANUAL.")
                        self._last_scheduled_date = today_str
        except Exception as e:
            print(f"[Scheduled Ticker] Error: {e}")
        finally:
            self._scheduled_ticker_job = self.after(60_000, self._scheduled_check_ticker)

    def _start_scheduled_watering(self, date_str: str | None = None) -> None:
        if self._scheduled_watering_active:
            return
        self._last_scheduled_date = date_str or datetime.now().strftime("%Y-%m-%d")
        moistures = self.telemetry.get("moisture_pct", [])
        setpoint = float(self.soil_water_setpoint.get())
        active_channels = []
        for ch in range(min(len(moistures), len(self.water_vars))):
            m = moistures[ch]
            if m is not None and m < setpoint:
                active_channels.append(ch)

        if not active_channels:
            print(f"[Scheduled] All soil moisture channels >= setpoint ({setpoint}%). No watering needed.")
            return

        self._scheduled_watering_active = True
        self._scheduled_start_time = time.time()
        self._scheduled_channels_active = list(active_channels)

        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._scheduled_run_dir = SCHEDULED_WATERING_DIR / run_stamp
        self._scheduled_run_dir.mkdir(parents=True, exist_ok=True)

        mask = ["1" if i in active_channels else "0" for i in range(len(self.water_vars))]
        for i in range(len(self.water_vars)):
            self.water_vars[i].set(1 if i in active_channels else 0)
        self.send_bitmask("".join(mask))
        stop_target = float(self.soil_water_stop_setpoint.get()) if hasattr(self, "soil_water_stop_setpoint") else SCHEDULED_TARGET_PCT
        print(f"[Scheduled] Watering started for channels {[c+1 for c in active_channels]} with target {stop_target}%. Dir: {self._scheduled_run_dir}")

        self._scheduled_monitor_job = self.after(2000, self._scheduled_watering_monitor)

    def _scheduled_watering_monitor(self) -> None:
        if not self._scheduled_watering_active:
            return

        elapsed = time.time() - (self._scheduled_start_time or time.time())

        # 1. Independent clean raw image capture
        if self.camera and self.camera.is_available and self._scheduled_run_dir:
            try:
                clean_frame = self.camera.capture_array()
                if clean_frame is not None:
                    img_name = f"IMG_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(str(self._scheduled_run_dir / img_name), clean_frame)
            except Exception as e:
                print(f"[Scheduled Monitor] Image capture error: {e}")

        # 2. Check per-channel target cutoff
        stop_target = float(self.soil_water_stop_setpoint.get()) if hasattr(self, "soil_water_stop_setpoint") else SCHEDULED_TARGET_PCT
        moistures = self.telemetry.get("moisture_pct", [])
        still_active = []
        for ch in self._scheduled_channels_active:
            m = moistures[ch] if ch < len(moistures) else None
            if m is not None and m >= stop_target:
                print(f"[Scheduled] Channel S{ch+1} reached {m:.1f}% >= target ({stop_target}%). Turning OFF pump.")
            else:
                still_active.append(ch)

        self._scheduled_channels_active = still_active
        timed_out = elapsed >= SCHEDULED_MAX_WATERING_SEC

        if timed_out and still_active:
            print(f"[Scheduled WARNING] Safety timeout ({SCHEDULED_MAX_WATERING_SEC}s) reached! Stopping all pumps.")
            summary_moist = [f"S{c+1}: {moistures[c]:.1f}%" if c < len(moistures) and moistures[c] is not None else f"S{c+1}: N/A" for c in still_active]
            print(f"[Scheduled WARNING] Active channels remaining: {', '.join(summary_moist)}")

        if not still_active or timed_out:
            for i in range(len(self.water_vars)):
                self.water_vars[i].set(0)
            self.send_bitmask("0" * len(self.water_vars))
            self._scheduled_watering_active = False
            self._scheduled_channels_active = []
            print(f"[Scheduled] Watering run completed in {elapsed:.1f}s. Timeout={timed_out}.")
            return

        # Update relays for remaining active channels
        mask = ["1" if i in self._scheduled_channels_active else "0" for i in range(len(self.water_vars))]
        for i in range(len(self.water_vars)):
            self.water_vars[i].set(1 if i in self._scheduled_channels_active else 0)
        self.send_bitmask("".join(mask))

        self._scheduled_monitor_job = self.after(2000, self._scheduled_watering_monitor)

    def _bind_capture_click_hold(self) -> None:
        def on_press(event=None):
            if self.live_var.get():
                self.btn_capture.config(bg="purple", fg="white", activebackground="purple", activeforeground="white")

        def on_release(event=None):
            if self.live_var.get():
                self.btn_capture.config(bg="white", fg="black", activebackground="white", activeforeground="black")

        self.btn_capture.bind("<ButtonPress-1>", on_press)
        self.btn_capture.bind("<ButtonRelease-1>", on_release)
        self.btn_capture.bind("<Leave>", on_release)

    def rebuild_relays(self, count: int) -> None:
        """Dynamically create W1..WN relay buttons in 1 row with solid black text."""
        for b in self.water_btns:
            b.destroy()
        self.water_btns.clear()
        self.water_vars.clear()
        relays_str = self.telemetry.get("relays", "")
        state = tk.DISABLED if self.auto_var.get() else tk.NORMAL

        for i in range(max(1, min(5, count))):
            var = tk.IntVar(value=1 if (i < len(relays_str) and relays_str[i] == '1') else 0)
            self.water_vars.append(var)
            btn = tk.Checkbutton(
                self.water_frame, text=f"W{i+1}", font=("arial", 22, "bold"), bg="white",
                fg="black", activeforeground="black", disabledforeground="#9e9e9e",
                selectcolor="#1e88e5", indicatoron=False, variable=var, bd=3, state=state,
                command=self._send_manual_relays
            )
            btn.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2, ipady=14, pady=2)
            self.water_btns.append(btn)

    def _on_auto_toggle(self) -> None:
        is_auto = bool(self.auto_var.get())
        self.ckb_auto.config(text="AUTO" if is_auto else "MANUAL")
        if is_auto:
            self.ckb_auto.grid()
            self.btn_manual_mode.grid_remove()
            self.btn_manual_mode.config(state=tk.DISABLED)
            self.off_row.pack(side=tk.TOP, fill=tk.X, padx=2, pady=(2, 1))
            self.on_row.pack(side=tk.TOP, fill=tk.X, padx=2, pady=(1, 2))
        else:
            self.ckb_auto.grid_remove()
            self.off_row.pack_forget()
            self.on_row.pack_forget()
            self.btn_manual_mode.config(state=tk.NORMAL)
            self.btn_manual_mode.grid()
        state = tk.DISABLED if is_auto else tk.NORMAL
        for b in self.water_btns:
            b.config(state=state, disabledforeground="#9e9e9e")
        if not is_auto:
            self._send_manual_relays()

    def _on_live_toggle(self) -> None:
        live = bool(self.live_var.get())
        self.btn_capture.config(
            state=tk.NORMAL if live else tk.DISABLED,
            bg="white" if live else "light grey",
            fg="black" if live else "dark grey",
            activebackground="white" if live else "light grey",
            activeforeground="black" if live else "dark grey"
        )

    def _update_tpu_status_label(self) -> None:
        ai_active = bool(self.plant_ai_var.get() or self.heatmap_var.get())
        if ai_active:
            if self.plant_ai.is_available:
                self.lbl_tpu_status.config(text=f"TPU: Active ({self.plant_ai.last_latency_ms:.1f} ms)", fg="#06d6a0")
            else:
                self.lbl_tpu_status.config(text="TPU: Disconnected", fg="#e63946")
        else:
            if self.plant_ai.is_available:
                self.lbl_tpu_status.config(text="TPU: Ready", fg="#06d6a0")
            else:
                self.lbl_tpu_status.config(text="TPU: Disconnected", fg="#e63946")

    def _on_plant_ai_toggle(self) -> None:
        self._update_tpu_status_label()

    def _on_heatmap_toggle(self) -> None:
        self._update_tpu_status_label()

    def _on_capture(self) -> None:
        if self.live_var.get() and self.camera.is_available:
            self.flag_capture = True
            self.btn_capture.config(bg="purple", fg="white", activebackground="purple", activeforeground="white")
            self.after(200, lambda: self.live_var.get() and self.btn_capture.config(
                bg="white", fg="black", activebackground="white", activeforeground="black"
            ))

    def _send_manual_relays(self) -> None:
        if not self.auto_var.get():
            self.send_bitmask("".join(str(v.get()) for v in self.water_vars))

    def send_bitmask(self, bitmask: str) -> None:
        self.send_command(bitmask)

    def send_command(self, cmd: str) -> None:
        if self.ser and self.ser.is_open:
            try:
                with self.serial_lock:
                    self.ser.write((cmd.strip() + "\n").encode("utf-8"))
                    self.ser.flush()
            except Exception as e:
                print(f"[Serial] Command error: {e}")

    def _start_repeat(self, action_fn) -> None:
        """Execute action immediately, then schedule repeated execution while held."""
        self._stop_repeat()
        action_fn()

        def _repeat_step():
            action_fn()
            self._repeat_job = self.after(100, _repeat_step)

        self._repeat_job = self.after(200, _repeat_step)

    def _stop_repeat(self, event=None) -> None:
        """Cancel active press-and-hold repeat timer."""
        if self._repeat_job is not None:
            try:
                self.after_cancel(self._repeat_job)
            except Exception:
                pass
            self._repeat_job = None

    def nudge_pan(self, delta: int) -> None:
        new_pan = max(PAN_MIN, min(PAN_MAX, self.current_pan + delta))
        if new_pan == self.current_pan:
            return
        self.current_pan = new_pan
        self.send_command(f"p {new_pan}")

    def nudge_tilt(self, delta: int) -> None:
        new_tilt = max(TILT_MIN, min(TILT_MAX, self.current_tilt + delta))
        if new_tilt == self.current_tilt:
            return
        self.current_tilt = new_tilt
        self.send_command(f"t {new_tilt}")

    def home_gimbal(self) -> None:
        self._stop_repeat()
        self.current_pan, self.current_tilt = PAN_HOME, TILT_HOME
        self.send_command("h")

    def _init_serial(self) -> None:
        port = find_arduino_port()
        if not port:
            print("[Serial] No Arduino detected.")
            return
        try:
            self.ser = serial.Serial(port=port, baudrate=BAUDRATE, timeout=1.0)
            time.sleep(2.0)
            self.ser.reset_input_buffer()
            print(f"[Serial] Connected on {port}!")
            threading.Thread(target=self._serial_reader, daemon=True).start()
        except Exception as e:
            print(f"[Serial] Error opening {port}: {e}")

    def _update_telemetry_ui(self, data: dict, moist: list[float | None]) -> None:
        st, at, ah, lv = data.get("soil_temp"), data.get("temp"), data.get("humidity"), data.get("light")
        self.lbl_soil_temp.config(text=f"Soil: {st:.1f}°C" if st is not None else "Soil: N/A")
        self.lbl_air_temp.config(text=f"Air: {at:.1f}°C" if at is not None else "Air: N/A")
        self.lbl_air_humi.config(text=f"RH: {ah:.1f}%" if ah is not None else "RH: N/A")
        self.lbl_light.config(text=f"Light: {lv} lux" if lv is not None else "Light: N/A")

        moist_strs = [f"S{i+1}: {m:.1f}%" if m is not None else f"S{i+1}: --" for i, m in enumerate(moist)]
        self.lbl_soil_moist.config(text="  |  ".join(moist_strs) if moist_strs else "No sensors")

        relays_str = data.get("relays", "")
        for i in range(min(len(relays_str), len(self.water_vars))):
            expected = 1 if relays_str[i] == "1" else 0
            if self.water_vars[i].get() != expected:
                self.water_vars[i].set(expected)

    def _log_telemetry_csv(self, data: dict, moist: list[float | None]) -> None:
        """Append a telemetry record to daily CSV file in TELEMETRY_DIR with a single header row."""
        try:
            csv_path = TELEMETRY_DIR / f"telemetry_{datetime.now().strftime('%Y%m%d')}.csv"
            write_header = not csv_path.exists() or csv_path.stat().st_size == 0
            with csv_path.open("a", encoding="utf-8") as f:
                if write_header:
                    f.write(
                        "timestamp,soil1_raw,soil2_raw,soil3_raw,soil4_raw,"
                        "soil1_pct,soil2_pct,soil3_pct,soil4_pct,"
                        "soil_temp_c,air_temp_c,humidity_pct,light,relays,pan_deg,tilt_deg\n"
                    )
                s_raw = [(str(v) if v is not None else "null") for v in (data.get("soil", []) + [None] * 4)[:4]]
                s_pct = [(f"{m:.1f}" if m is not None else "null") for m in (moist + [None] * 4)[:4]]
                st = f"{data['soil_temp']:.1f}" if data.get("soil_temp") is not None else "null"
                at = f"{data['temp']:.1f}" if data.get("temp") is not None else "null"
                ah = f"{data['humidity']:.1f}" if data.get("humidity") is not None else "null"
                lv = str(data["light"]) if data.get("light") is not None else "null"
                row = [datetime.now().strftime("%Y-%m-%d %H:%M:%S")] + s_raw + s_pct + [
                    st, at, ah, lv, data.get("relays", "0000"), str(data.get("pan", PAN_HOME)), str(data.get("tilt", TILT_HOME))
                ]
                f.write(",".join(row) + "\n")
        except Exception:
            pass

    def _serial_reader(self) -> None:
        rx_buf = ""
        while not self.stop_threads.is_set():
            if self.ser and self.ser.is_open:
                try:
                    n = self.ser.in_waiting
                    if n:
                        rx_buf += self.ser.read(n).decode("utf-8", errors="replace")
                        while "\n" in rx_buf:
                            line, rx_buf = rx_buf.split("\n", 1)
                            line = line.strip()
                            if not line:
                                continue
                            data = parse_telemetry_line(line)
                            if data:
                                soil = data.get("soil", [])
                                moist = [raw_to_moisture(v, i) for i, v in enumerate(soil)]
                                self.telemetry = {**data, "moisture_pct": moist}
                                if soil and len(soil) != len(self.water_vars):
                                    self.after(0, lambda n=len(soil): self.rebuild_relays(n))
                                self.after(0, lambda d=data, m=moist: self._update_telemetry_ui(d, m))
                    else:
                        time.sleep(0.05)
                except Exception:
                    time.sleep(0.1)

    def _apply_plant_ai_overlays(self, frame: np.ndarray) -> tuple[np.ndarray, float | None]:
        pai = bool(self.plant_ai_var.get())
        hmap = bool(self.heatmap_var.get())
        if not (pai or hmap):
            return frame, None

        if pai and not hmap:
            # AI only: run TPU inference, draw bbox + health labels
            results, latency = self.plant_ai.detect_and_analyze(frame)
            out = self.plant_ai.draw_overlay(frame, results, latency, show_bbox=True)
        elif not pai and hmap:
            # Heatmap only: full-frame HSV colormap, no TPU needed
            out = self.plant_ai.draw_full_frame_heatmap(frame)
            latency = 0.0
        else:
            # Both ON: no interaction. AI shows bboxes, heatmap overlays entire frame with dimmed background
            results, latency = self.plant_ai.detect_and_analyze(frame)
            hmap_frame = self.plant_ai.draw_full_frame_heatmap(frame, latency_ms=latency, show_hud=False)
            out = self.plant_ai.draw_overlay(hmap_frame, results, latency, show_bbox=True)

        return out, latency

    def _camera_loop(self) -> None:
        if self.live_var.get() and self.camera.is_available:
            frame = self.camera.capture_array()
            if frame is not None:
                display_frame, latency = self._apply_plant_ai_overlays(frame)
                if latency is not None:
                    if self.plant_ai.is_available:
                        self.lbl_tpu_status.config(text=f"TPU: Active ({latency:.1f} ms)", fg="#06d6a0")
                    else:
                        self.lbl_tpu_status.config(text="TPU: Disconnected", fg="#e63946")
                self.display_image(display_frame)
                if self.flag_capture:
                    # Guarantees saving clean raw camera frame without overlays
                    out = IMAGES_DIR / f"IMG_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(str(out), frame)
                    print(f"[Capture] Saved clean snapshot to {out}")
                    self.flag_capture = False
        self.after(30, self._camera_loop)

    def _auto_loop(self) -> None:
        if self.auto_var.get():
            moistures = self.telemetry.get("moisture_pct", [])
            setpoint = float(self.soil_water_setpoint.get())
            mask, changed = [], False
            for i in range(min(len(moistures), len(self.water_vars))):
                m = moistures[i]
                des = 1 if (m is not None and m < setpoint) else 0
                if self.water_vars[i].get() != des:
                    self.water_vars[i].set(des)
                    changed = True
                mask.append(str(des))
            if changed and mask:
                self.send_bitmask("".join(mask))
        self.after(500, self._auto_loop)

    def display_image(self, img: np.ndarray | Image.Image) -> None:
        self._last_display_img = img
        c_w = self.canvas.winfo_width()
        c_h = self.canvas.winfo_height()

        if isinstance(img, np.ndarray):
            img_h, img_w = img.shape[:2]
        else:
            img_w, img_h = img.size

        if c_w > 10 and c_h > 10 and img_w > 0 and img_h > 0:
            scale = min(c_w / img_w, c_h / img_h)
            new_w = max(1, int(img_w * scale))
            new_h = max(1, int(img_h * scale))
            if isinstance(img, np.ndarray):
                resized = cv2.resize(
                    img, (new_w, new_h),
                    interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
                )
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) if resized.ndim == 3 and resized.shape[2] == 3 else resized
                pil_img = Image.fromarray(rgb)
            else:
                pil_img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
            pos_x = c_w // 2
            pos_y = c_h // 2
            anchor = "center"
        else:
            if isinstance(img, np.ndarray):
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 and img.shape[2] == 3 else img
                pil_img = Image.fromarray(rgb)
            else:
                pil_img = img
            pos_x = 0
            pos_y = 0
            anchor = "nw"

        self.photo_ref = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(pos_x, pos_y, anchor=anchor, image=self.photo_ref)

    def _on_canvas_resize(self, event=None) -> None:
        if getattr(self, "_last_display_img", None) is not None and not getattr(self, "_is_closing", False):
            self.display_image(self._last_display_img)

    def _build_menu(self) -> None:
        menubar = tk.Menu(self)
        self.config(menu=menubar)
        fmenu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=fmenu)
        fmenu.add_command(label="Open Image", command=self._open_image)
        fmenu.add_command(label="Exit", command=self.on_closing)
        menubar.add_cascade(label="View", menu=tk.Menu(menubar, tearoff=0))

    def _open_image(self) -> None:
        p = filedialog.askopenfilename(title="Open Image", filetypes=[("Images", "*.jpg *.png *.tif")])
        if p:
            img = cv2.imread(p)
            if img is not None:
                display_frame, latency = self._apply_plant_ai_overlays(img)
                if latency is not None:
                    if self.plant_ai.is_available:
                        self.lbl_tpu_status.config(text=f"TPU: Active ({latency:.1f} ms)", fg="#06d6a0")
                    else:
                        self.lbl_tpu_status.config(text="TPU: Disconnected", fg="#e63946")
                self.display_image(display_frame)
            else:
                self.display_image(Image.open(p))

    def report_callback_exception(self, exc, val, tb) -> None:
        if issubclass(exc, KeyboardInterrupt):
            self.on_closing()
        else:
            super().report_callback_exception(exc, val, tb)

    def on_closing(self) -> None:
        if getattr(self, "_is_closing", False):
            return
        self._is_closing = True
        self._stop_repeat()

        # Cancel all pending after() jobs
        for job in (
            getattr(self, "_data_logger_job", None),
            getattr(self, "_image_logger_job", None),
            getattr(self, "_scheduled_ticker_job", None),
            getattr(self, "_scheduled_monitor_job", None),
            getattr(self, "_repeat_job", None),
        ):
            if job is not None:
                try:
                    self.after_cancel(job)
                except Exception:
                    pass

        print("\nShutting down GUI and all pumps...")
        self.stop_threads.set()
        try:
            self.send_bitmask("0" * max(1, len(self.water_vars)))
            self.send_command("h")  # direct home (reader thread already stopped)
            time.sleep(0.3)
        except Exception as e:
            print(f"[Shutdown] Command error: {e}")
        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass
        try:
            self.camera.stop()
        except Exception:
            pass
        try:
            self.destroy()
        except Exception:
            pass
        sys.exit(0)


if __name__ == "__main__":
    app = MainWindow()
    signal.signal(signal.SIGINT, lambda sig, frame: app.on_closing())
    signal.signal(signal.SIGTERM, lambda sig, frame: app.on_closing())
    try:
        app.mainloop()
    except KeyboardInterrupt:
        app.on_closing()