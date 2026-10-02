"""Regression checks for sensor data supplied to live and historical plots."""

from datetime import datetime
import math
import tkinter as tk

import pytest

from gui.psc_irr_gui import PlotWindow, read_historical_telemetry
from tests.mock_telemetry import HEADER


def test_historical_light_sorting_nulls_and_zero(tmp_path):
    rows = []
    for hour, light in [(14, "1200"), (9, "0"), (10, "null"), (11, "bad")]:
        rows.append(
            f"2026-10-02 {hour:02d}:00:00,400,400,400,400,"
            f"40,41,42,43,22,24,58,{light},0000,55,30\n"
        )
    # Older/truncated rows without the light column still supply other sensors.
    rows.append("2026-10-02 12:00:00,400,400,400,400,40,41,42,43,22,24,58\n")
    (tmp_path / "telemetry_20261002.csv").write_text(HEADER + "".join(rows))
    hist = read_historical_telemetry(tmp_path, "day", now=datetime(2026, 10, 2, 15))
    assert [ts.hour for ts in hist["timestamps"]] == [9, 10, 11, 12, 14]
    assert hist["light"] == [0.0, None, None, None, 1200.0]
    for key in ("soil_temp", "air_temp", "humidity", "light"):
        assert len(hist[key]) == len(hist["timestamps"])
    assert hist["pump_events"] == []
    assert read_historical_telemetry(tmp_path / "missing", "day")["light"] == []


@pytest.fixture
def plotter():
    root = tk.Tk()
    root.withdraw()
    plot = PlotWindow(root, lambda: {
        "moisture_pct": [40, None, 50, 60], "soil_temp": 22,
        "temp": 24, "humidity": 58, "light": 1200,
    })
    yield plot
    plot.toggle(False)
    root.destroy()


def test_live_light_legacy_tuples_and_sweep_reset(plotter):
    assert next(plotter._gen())[1:] == ([40, None, 50, 60], 22, 24, 58, 1200)
    plotter._update((1, [40], 22, 24, 58, 0))
    plotter._update((2, [41], 22, 24, 58, None))
    plotter._update((3, [42], 22, 24, 58))
    plotter._update((4, [43], 24, 58))
    plotter._update((5, [44]))
    assert plotter.ydata_light[0] == 0
    assert all(math.isnan(value) for value in plotter.ydata_light[1:])
    assert len(plotter.xdata) == len(plotter.ydata_light) == 5
    plotter._update((0, [50], 23, 25, 60, 1500))
    assert plotter.ydata_light == [1500]
    assert plotter.xdata == [0]
