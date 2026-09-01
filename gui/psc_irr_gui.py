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

from datetime import datetime
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
DRY_BASELINES = [432.0, 408.0, 427.0, 424.0]
WET_BASELINES = [136.0, 92.0, 159.0, 160.0]
DATA_DIR = Path("Data")
DATA_DIR.mkdir(exist_ok=True)

# Hardware Safety Bounds and Centers
PAN_MIN, PAN_MAX = 0, 130
TILT_MIN, TILT_MAX = 0, 90
PAN_CENTER, TILT_CENTER = 65, 60
GIMBAL_STEP = 5  # degrees per nudge click


def _safe_float(val: str) -> float | None:
    """Convert string to float, treating 'null', 'none', 'nan', or empty as None."""
    s = val.strip().lower()
    if not s or s in ("null", "none", "nan"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _safe_int(val: str) -> int | None:
    """Convert string to int, treating 'null', 'none', 'nan', or empty as None."""
    s = val.strip().lower()
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
        "pan": PAN_CENTER,
        "tilt": TILT_CENTER,
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


class PlotWindow:
    """Dynamic Matplotlib window displaying 60s soil moisture, temperatures, and humidity telemetry."""

    def __init__(self, master: tk.Tk, get_telemetry_fn) -> None:
        self.master = master
        self.get_telemetry = get_telemetry_fn
        self.window: tk.Toplevel | None = None
        self.ani: animation.FuncAnimation | None = None
        self.max_ch = 4
        self.colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]
        self.xdata: list[float] = []
        self.ydata_moist: list[list[float]] = [[] for _ in range(self.max_ch)]
        self.ydata_soil_temp: list[float] = []
        self.ydata_temp: list[float] = []
        self.ydata_rh: list[float] = []

        self.fig = Figure(figsize=(15.0, 10.5), dpi=100)
        self.fig.suptitle("Soil Moisture & Environmental Telemetry (Last 60s)", fontsize=24, fontweight="bold")

        # Top Plot: Soil Moisture (0-100%)
        self.ax_moist = self.fig.add_subplot(2, 1, 1)
        self.lines_moist = []
        for i, c in enumerate(self.colors):
            (line,) = self.ax_moist.plot([], [], color=c, linewidth=3.6, label=f"S{i+1}")
            self.lines_moist.append(line)

        self.ax_moist.set_ylim(0, 100)
        self.ax_moist.set_xlim(0, 60)
        self.ax_moist.set_xticks([0, 10, 20, 30, 40, 50, 60])
        self.ax_moist.set_ylabel("Soil Moisture (%)", fontsize=20, fontweight="bold")
        self.ax_moist.tick_params(axis="both", labelsize=18)
        self.ax_moist.grid(True, linestyle="--", alpha=0.6, linewidth=1.5)
        self.ax_moist.legend(loc="upper left", fontsize=18, ncol=4, framealpha=0.92)

        # Bottom Plot: Temperatures (0-50°C, Left Y-axis) & Relative Humidity (0-100%, Right Y-axis)
        self.ax_temp = self.fig.add_subplot(2, 1, 2, sharex=self.ax_moist)
        self.ax_rh = self.ax_temp.twinx()

        (self.line_temp,) = self.ax_temp.plot([], [], color="#d62728", linewidth=3.6, label="Air Temp (°C)")
        (self.line_soil_temp,) = self.ax_temp.plot(
            [], [], color="#d95f02", linewidth=3.6, linestyle="--", label="Soil Temp (°C)"
        )
        (self.line_rh,) = self.ax_rh.plot([], [], color="#00838f", linewidth=3.6, label="RH (%)")

        self.ax_temp.set_ylim(0, 50)
        self.ax_temp.set_xlim(0, 60)
        self.ax_temp.set_xticks([0, 10, 20, 30, 40, 50, 60])
        self.ax_temp.set_xlabel("Time (s)", fontsize=20, fontweight="bold")
        self.ax_temp.set_ylabel("Temperature (°C)", fontsize=20, fontweight="bold", color="#d62728")
        self.ax_temp.tick_params(axis="x", labelsize=18)
        self.ax_temp.tick_params(axis="y", labelcolor="#d62728", labelsize=18)
        self.ax_temp.grid(True, linestyle="--", alpha=0.6, linewidth=1.5)

        self.ax_rh.set_ylim(0, 100)
        self.ax_rh.set_ylabel("Relative Humidity (%)", fontsize=20, fontweight="bold", color="#00838f")
        self.ax_rh.tick_params(axis="y", labelcolor="#00838f", labelsize=18)

        # Combined Legend for Temperatures and RH
        self.ax_temp.legend(
            [self.line_temp, self.line_soil_temp, self.line_rh],
            ["Air Temp (°C)", "Soil Temp (°C)", "RH (%)"],
            loc="upper left",
            fontsize=18,
            ncol=3,
            framealpha=0.92,
        )

        self.fig.tight_layout(rect=[0.02, 0.04, 0.98, 0.95])
        self.fig.subplots_adjust(hspace=0.35)

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
            self.lines_moist[i].set_data(self.xdata, self.ydata_moist[i])

        self.line_temp.set_data(self.xdata, self.ydata_temp)
        self.line_soil_temp.set_data(self.xdata, self.ydata_soil_temp)
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
                self.window.title("Soil Moisture & Environmental Telemetry (Last 60s)")
                scr_w = self.master.winfo_screenwidth()
                scr_h = self.master.winfo_screenheight()
                win_w = max(1100, int(scr_w * 0.94))
                win_h = max(800, int(scr_h * 0.93))
                pos_x = max(0, int((scr_w - win_w) / 2))
                pos_y = max(0, int((scr_h - win_h) / 2))
                self.window.geometry(f"{win_w}x{win_h}+{pos_x}+{pos_y}")
                self.window.protocol("WM_DELETE_WINDOW", lambda: (self.toggle(False), on_close and on_close()))
                FigureCanvasTkAgg(self.fig, master=self.window).get_tk_widget().pack(fill=tk.BOTH, expand=True)
                self.ani = animation.FuncAnimation(
                    self.fig, self._update, self._gen, interval=500, cache_frame_data=False
                )
            else:
                self.window.deiconify()
                self.window.lift()
        elif self.window and tk.Toplevel.winfo_exists(self.window):
            if self.ani and self.ani.event_source:
                self.ani.event_source.stop()
            self.window.destroy()
            self.window, self.ani = None, None


class MainWindow(tk.Tk):
    """Main Application coordinating Camera, Serial Telemetry, and User Interface."""

    def __init__(self) -> None:
        super().__init__()
        self.title("CEA Irrigation Controller GUI")

        self.scr_w = self.winfo_screenwidth()
        self.scr_h = self.winfo_screenheight() - 75
        self.geometry(f"{self.scr_w}x{self.scr_h}+0+0")
        margin_w, btn_h = int(self.scr_w / 5), int(self.scr_h / 9.5)
        gap_y = int(btn_h / 10)
        y_pos = lambda slot: btn_h * slot + gap_y * (slot + 1)

        # Hardware & Telemetry State
        self.ser: serial.Serial | None = None
        self.serial_lock = threading.Lock()  # guards self.ser against reader/writer races
        self.stop_threads = threading.Event()
        self.telemetry = {
            "soil": [], "moisture_pct": [], "soil_temp": None,
            "temp": None, "humidity": None, "light": None,
            "relays": "0000", "pan": PAN_CENTER, "tilt": TILT_CENTER
        }
        self.current_pan, self.current_tilt = PAN_CENTER, TILT_CENTER
        self._repeat_job: str | None = None
        self.camera = Camera(width=self.scr_w - margin_w, height=self.scr_h - int(self.scr_h / 5))
        self.plant_ai = PlantAIDetector()
        self.flag_capture = False
        self.photo_ref: ImageTk.PhotoImage | None = None

        # Build UI Layout
        paned = tk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        left_frame = tk.Frame(paned, relief=tk.SUNKEN)
        right_frame = tk.Frame(paned, width=margin_w)
        paned.add(left_frame, width=self.scr_w - margin_w, height=self.scr_h)
        paned.add(right_frame)

        self._build_bottom_bar(left_frame)

        # Canvas with Scrollbars
        self.y_scrl = tk.Scrollbar(left_frame, orient=tk.VERTICAL)
        self.y_scrl.pack(fill=tk.Y, side=tk.RIGHT)
        self.x_scrl = tk.Scrollbar(left_frame, orient=tk.HORIZONTAL)
        self.x_scrl.pack(fill=tk.X, side=tk.BOTTOM)
        self.canvas = tk.Canvas(left_frame, yscrollcommand=self.y_scrl.set, xscrollcommand=self.x_scrl.set, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.y_scrl.config(command=self.canvas.yview)
        self.x_scrl.config(command=self.canvas.xview)

        # Sidebar Buttons (right_frame)
        self.auto_var = tk.IntVar(value=1)
        self.ckb_auto = tk.Checkbutton(
            right_frame, text="AUTO", font=("arial", 32, "bold"), bg="white",
            selectcolor="light green", bd=4, indicatoron=False, variable=self.auto_var,
            command=self._on_auto_toggle
        )
        self.ckb_auto.place(x=0, y=y_pos(0), width=margin_w, height=btn_h)

        self.plot_var = tk.IntVar(value=0)
        self.ckb_plot = tk.Checkbutton(
            right_frame, text="Plot", font=("arial", 32, "bold"), bg="white",
            selectcolor="light grey", bd=4, indicatoron=False, variable=self.plot_var,
            command=lambda: self.plotter.toggle(bool(self.plot_var.get()), lambda: self.plot_var.set(0))
        )
        self.ckb_plot.place(x=0, y=y_pos(1), width=margin_w, height=btn_h)

        self.water_frame = tk.Frame(right_frame, bg="light grey")
        self.water_frame.place(x=0, y=y_pos(2), width=margin_w, height=btn_h)
        self.water_vars: list[tk.IntVar] = []
        self.water_btns: list[tk.Checkbutton] = []
        self.rebuild_relays(4)

        self.live_var = tk.IntVar(value=1)
        self.ckb_live = tk.Checkbutton(
            right_frame, text="Live", font=("arial", 32, "bold"), bg="white",
            selectcolor="yellow", bd=4, indicatoron=False, variable=self.live_var,
            command=self._on_live_toggle
        )
        self.ckb_live.place(x=0, y=y_pos(3), width=margin_w, height=btn_h)

        self.btn_capture = tk.Button(
            right_frame, text="Capture", font=("arial", 32, "bold"), bg="white", fg="black",
            activebackground="white", activeforeground="black", bd=4, command=self._on_capture
        )
        self.btn_capture.place(x=0, y=y_pos(4), width=margin_w, height=btn_h)
        self._bind_capture_click_hold()

        self.plant_ai_var = tk.IntVar(value=0)
        self.ckb_plant_ai = tk.Checkbutton(
            right_frame, text="Plant AI", font=("arial", 30, "bold"), bg="white",
            selectcolor="#ffb703", bd=4, indicatoron=False, variable=self.plant_ai_var,
            command=self._on_plant_ai_toggle
        )
        self.ckb_plant_ai.place(x=0, y=y_pos(5), width=margin_w, height=btn_h)

        self._build_gimbal_panel(right_frame, margin_w, y_pos(6), self.scr_h - y_pos(6) - gap_y)

        self.plotter = PlotWindow(self, lambda: self.telemetry)
        self._build_menu()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self._init_serial()
        self._camera_loop()
        self._auto_loop()

    def _build_bottom_bar(self, parent: tk.Frame) -> None:
        bar = tk.Frame(parent, bg="#1a1d20", height=170, relief=tk.GROOVE, bd=3)
        bar.pack(side=tk.BOTTOM, fill=tk.X, padx=2, pady=2)
        bar.pack_propagate(False)

        row1 = tk.Frame(bar, bg="#1a1d20")
        row1.pack(side=tk.TOP, fill=tk.X, padx=12, pady=(8, 0))
        tk.Label(row1, text="ENV:", font=("arial", 22, "bold"), fg="#90e0ef", bg="#1a1d20").pack(side=tk.LEFT, padx=(0, 16))

        self.lbl_soil_temp = tk.Label(row1, text="Soil: --.-°C", font=("arial", 22, "bold"), fg="#06d6a0", bg="#1a1d20")
        self.lbl_air_temp = tk.Label(row1, text="Air: --.-°C", font=("arial", 22, "bold"), fg="#ffd166", bg="#1a1d20")
        self.lbl_air_humi = tk.Label(row1, text="RH: --.-%", font=("arial", 22, "bold"), fg="#4cc9f0", bg="#1a1d20")
        self.lbl_light = tk.Label(row1, text="Light: --", font=("arial", 22, "bold"), fg="#f72585", bg="#1a1d20")
        
        tpu_init_text = "TPU: Ready" if getattr(self, "plant_ai", None) and self.plant_ai.is_available else "TPU: Disconnected"
        tpu_init_color = "#06d6a0" if getattr(self, "plant_ai", None) and self.plant_ai.is_available else "#e63946"
        self.lbl_tpu_status = tk.Label(row1, text=tpu_init_text, font=("arial", 22, "bold"), fg=tpu_init_color, bg="#1a1d20")

        for lbl in (self.lbl_soil_temp, self.lbl_air_temp, self.lbl_air_humi, self.lbl_light, self.lbl_tpu_status):
            lbl.pack(side=tk.LEFT, padx=14)

        row2 = tk.Frame(bar, bg="#1a1d20")
        row2.pack(side=tk.TOP, fill=tk.X, padx=12, pady=(0, 8))
        tk.Label(row2, text="MOIST:", font=("arial", 22, "bold"), fg="#90e0ef", bg="#1a1d20").pack(side=tk.LEFT, padx=(0, 16))
        self.lbl_soil_moist = tk.Label(row2, text="S1: --  |  S2: --  |  S3: --  |  S4: --", font=("arial", 22, "bold"), fg="#ffffff", bg="#1a1d20")
        self.lbl_soil_moist.pack(side=tk.LEFT, padx=14)

    def _build_gimbal_panel(self, parent: tk.Frame, width: int, y: int, height: int) -> None:
        self.gimbal_frame = tk.LabelFrame(
            parent, text=" GIMBAL ", font=("arial", 20, "bold"),
            fg="#0d6efd", bg="#f8f9fa", bd=3, relief=tk.GROOVE
        )
        self.gimbal_frame.place(x=0, y=y, width=width, height=height)
        for i in range(3):
            self.gimbal_frame.grid_columnconfigure(i, weight=1, uniform="g_col")
            self.gimbal_frame.grid_rowconfigure(i, weight=1, uniform="g_row")

        gcfg = {
            "font": ("arial", 28, "bold"), "bd": 4, "bg": "#495057", "fg": "white",
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
            btn.grid(row=r, column=c, sticky="nsew", padx=4, pady=2)
            btn.bind("<ButtonPress-1>", lambda e, a=action: self._start_repeat(a))
            btn.bind("<ButtonRelease-1>", self._stop_repeat)
            btn.bind("<Leave>", self._stop_repeat)
            setattr(self, attr, btn)

        self.btn_center = tk.Button(
            self.gimbal_frame, text="⌖ Center", font=("arial", 18, "bold"), bd=4,
            bg="#0d6efd", fg="white", activebackground="#0b5ed7", activeforeground="white",
            command=self.recenter_gimbal
        )
        self.btn_center.grid(row=1, column=1, sticky="nsew", padx=4, pady=2)

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
                self.water_frame, text=f"W{i+1}", font=("arial", 26, "bold"), bg="white",
                fg="black", activeforeground="black", disabledforeground="black",
                selectcolor="#1e88e5", indicatoron=False, variable=var, bd=3, state=state,
                command=self._send_manual_relays
            )
            btn.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=1)
            self.water_btns.append(btn)

    def _on_auto_toggle(self) -> None:
        is_auto = bool(self.auto_var.get())
        self.ckb_auto.config(text="AUTO" if is_auto else "MANUAL")
        state = tk.DISABLED if is_auto else tk.NORMAL
        for b in self.water_btns:
            b.config(state=state, disabledforeground="black")
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

    def _on_plant_ai_toggle(self) -> None:
        if self.plant_ai_var.get():
            if self.plant_ai.is_available:
                self.lbl_tpu_status.config(text=f"TPU: Active ({self.plant_ai.last_latency_ms:.1f} ms)", fg="#06d6a0")
            else:
                self.lbl_tpu_status.config(text="TPU: Disconnected", fg="#e63946")
        else:
            if self.plant_ai.is_available:
                self.lbl_tpu_status.config(text="TPU: Ready", fg="#06d6a0")
            else:
                self.lbl_tpu_status.config(text="TPU: Disconnected", fg="#e63946")

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

    def recenter_gimbal(self) -> None:
        self._stop_repeat()
        self.current_pan, self.current_tilt = PAN_CENTER, TILT_CENTER
        self.send_command("c")

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
        self.lbl_light.config(text=f"Light: {lv}" if lv is not None else "Light: N/A")

        moist_strs = [f"S{i+1}: {m:.1f}%" if m is not None else f"S{i+1}: --" for i, m in enumerate(moist)]
        self.lbl_soil_moist.config(text="  |  ".join(moist_strs) if moist_strs else "No sensors")

        relays_str = data.get("relays", "")
        for i in range(min(len(relays_str), len(self.water_vars))):
            expected = 1 if relays_str[i] == "1" else 0
            if self.water_vars[i].get() != expected:
                self.water_vars[i].set(expected)

    def _log_telemetry_csv(self, data: dict, moist: list[float | None]) -> None:
        """Append a telemetry record to daily CSV file in DATA_DIR with a single header row."""
        try:
            csv_path = DATA_DIR / f"telemetry_{datetime.now().strftime('%Y%m%d')}.csv"
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
                    st, at, ah, lv, data.get("relays", "0000"), str(data.get("pan", PAN_CENTER)), str(data.get("tilt", TILT_CENTER))
                ]
                f.write(",".join(row) + "\n")
        except Exception:
            pass

    def _serial_reader(self) -> None:
        last_log = 0.0
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
                                if time.time() - last_log >= 10.0:
                                    self._log_telemetry_csv(data, moist)
                                    last_log = time.time()
                    else:
                        time.sleep(0.05)
                except Exception:
                    time.sleep(0.1)

    def _camera_loop(self) -> None:
        if self.live_var.get() and self.camera.is_available:
            frame = self.camera.capture_array()
            if frame is not None:
                if self.plant_ai_var.get() == 1:
                    results, latency = self.plant_ai.detect_and_analyze(frame)
                    display_frame = self.plant_ai.draw_overlay(frame, results, latency)
                    if self.plant_ai.is_available:
                        self.lbl_tpu_status.config(text=f"TPU: Active ({latency:.1f} ms)", fg="#06d6a0")
                    else:
                        self.lbl_tpu_status.config(text="TPU: Disconnected", fg="#e63946")
                else:
                    display_frame = frame
                self.display_image(display_frame)
                if self.flag_capture:
                    out = DATA_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:19]}.jpg"
                    cv2.imwrite(str(out), frame)
                    print(f"[Capture] Saved snapshot to {out}")
                    self.flag_capture = False
        self.after(30, self._camera_loop)

    def _auto_loop(self) -> None:
        if self.auto_var.get():
            moistures = self.telemetry.get("moisture_pct", [])
            mask, changed = [], False
            for i in range(min(len(moistures), len(self.water_vars))):
                m = moistures[i]
                des = 1 if (m is not None and m < SOIL_WATER_SETPOINT) else 0
                if self.water_vars[i].get() != des:
                    self.water_vars[i].set(des)
                    changed = True
                mask.append(str(des))
            if changed and mask:
                self.send_bitmask("".join(mask))
        self.after(500, self._auto_loop)

    def display_image(self, img: np.ndarray | Image.Image) -> None:
        if isinstance(img, np.ndarray):
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 and img.shape[2] == 3 else img
            pil_img = Image.fromarray(rgb)
        else:
            pil_img = img
        self.photo_ref = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo_ref)
        self.canvas.config(scrollregion=(0, 0, pil_img.width, pil_img.height))

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
                if self.plant_ai_var.get() == 1:
                    results, latency = self.plant_ai.detect_and_analyze(img)
                    img = self.plant_ai.draw_overlay(img, results, latency)
                self.display_image(img)
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
        print("\nShutting down GUI and all pumps...")
        self.stop_threads.set()
        try:
            self.send_bitmask("0" * max(1, len(self.water_vars)))
            self.send_command("c")  # direct center (reader thread already stopped)
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