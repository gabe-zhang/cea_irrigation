"""Regression checks for sensor data supplied to live and historical plots."""

from datetime import date, datetime
import math
import tkinter as tk

import pytest

from gui.psc_irr_gui import PlotWindow, read_historical_telemetry, telemetry_range_bounds, range_date_label
import gui.psc_irr_gui as gui
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


def assert_sensor_axes(plotter):
    assert plotter.ax_temp.get_ylim() == (0, 100)
    assert plotter.ax_temp.get_ylabel() == "Air T/RH, °C/%"
    assert plotter.ax_light.get_ylim() == (0, 4000)
    assert plotter.ax_light.get_ylabel() == "Light, Lux"
    assert plotter.line_rh.axes is plotter.ax_temp
    assert plotter.line_light.axes is plotter.ax_light
    assert plotter.line_light.get_color() == gui.LIGHT_COLOR
    for artist in (*plotter.lines_moist, plotter.line_temp,
                   plotter.line_soil_temp, plotter.line_rh, plotter.line_light):
        assert artist.get_linestyle() == "None"
        assert artist.get_marker() == "o"
    assert plotter.ax_light.yaxis.label.get_color() == gui.LIGHT_COLOR
    assert all(t.get_color() == gui.LIGHT_COLOR for t in plotter.ax_light.get_yticklabels())
    for axis, sensors in ((plotter.ax_temp, (plotter.line_temp, plotter.line_soil_temp,
                                           plotter.line_rh, plotter.line_light)),
                          (plotter.ax_moist, plotter.lines_moist)):
        legend = axis.get_legend()
        assert legend.handlelength == 2
        assert all(t.get_fontsize() == 18 for t in legend.get_texts())
        for handle, sensor in zip(legend.legend_handles, sensors):
            assert handle is not sensor
            assert handle.get_color() == sensor.get_color()
            assert handle.get_linestyle() == "-"
            assert handle.get_marker() == "None"
            assert handle.get_linewidth() == 3
    assert plotter.line_setpoint.get_linestyle() == "--"
    assert plotter.line_stop.get_linestyle() == "--"


def test_live_dots_leave_missing_samples_empty(plotter):
    plotter._update((0, [40], 22, 24, 58, 0))
    plotter._update((10, [None], None, None, None, None))
    artists = plotter._update((50, [45], 23, 25, 60, 1200))
    assert len(artists) == 8
    assert_sensor_axes(plotter)
    assert math.isnan(plotter.line_light.get_ydata()[1])
    assert math.isnan(plotter.lines_moist[0].get_ydata()[1])


def test_historical_axes_and_return_to_live(plotter, tmp_path, monkeypatch):
    monkeypatch.setattr(gui, "TELEMETRY_DIR", tmp_path)
    day = date.today().isoformat()
    (tmp_path / "telemetry_today.csv").write_text(
        HEADER + f"{day} 00:00:00,400,400,400,400,40,41,42,43,22,24,58,0,0000,55,30\n"
        + f"{day} 00:10:00,400,400,400,400,null,41,42,43,22,24,58,null,0000,55,30\n"
    )
    sizes = {"min": plotter.line_temp.get_markersize()}
    for mode in ("day", "week", "month", "period"):
        plotter._render_historical_figure(mode)
        assert_sensor_axes(plotter)
        assert math.isnan(plotter.line_light.get_ydata()[1])
        sizes[mode] = plotter.line_temp.get_markersize()
        assert all(artist.get_markersize() == sizes[mode] for artist in (
            *plotter.lines_moist, plotter.line_soil_temp, plotter.line_rh, plotter.line_light,
        ))
    assert sizes["min"] == 4
    assert 0 < sizes["month"] < sizes["week"] < sizes["day"] < sizes["min"]
    assert sizes["period"] == sizes["day"]
    plotter.start_date_var = lambda: date.today().replace(year=date.today().year - 1)
    plotter.end_date_var = date.today
    plotter._render_historical_figure("period")
    assert 0 < plotter.line_temp.get_markersize() < sizes["month"]
    plotter.toggle(True)
    plotter._update((1, [40], 22, 24, 58, 1200))
    plotter.range_var.set("day")
    plotter.range_var.set("min")
    assert plotter.ydata_light == []
    assert_sensor_axes(plotter)
    assert plotter.line_temp.get_markersize() == sizes["min"]


def test_range_bounds_and_labels_share_rolling_policy(tmp_path):
    now = datetime(2026, 10, 2, 12)
    assert telemetry_range_bounds("week", now) == (datetime(2026, 9, 25, 12), now)
    assert telemetry_range_bounds("month", now) == (datetime(2026, 9, 2, 12), now)
    assert range_date_label("week", now) == "09/25/2026 – 10/02/2026"
    assert range_date_label("month", now) == "09/02/2026 – 10/02/2026"
    (tmp_path / "telemetry_bounds.csv").write_text(
        HEADER + "2026-09-25 11:59:59,400,400,400,400,40,41,42,43,22,24,58,0,0000,55,30\n"
        + "2026-09-25 12:00:00,400,400,400,400,40,41,42,43,22,24,58,0,0000,55,30\n"
        + "2026-10-02 12:00:01,400,400,400,400,40,41,42,43,22,24,58,0,0000,55,30\n"
    )
    hist = read_historical_telemetry(tmp_path, "week", now=now)
    assert hist["timestamps"] == [datetime(2026, 9, 25, 12)]


@pytest.mark.parametrize("mode", ["week", "month", "period"])
@pytest.mark.parametrize("now", [datetime(2026, 10, 5, 16, 50, 37), datetime(2028, 3, 1)])
def test_historical_date_ticks_start_at_range_boundary(plotter, tmp_path, monkeypatch, mode, now):
    class FrozenDatetime(datetime):
        @classmethod
        def now(cls):
            return now

    monkeypatch.setattr(gui, "datetime", FrozenDatetime)
    monkeypatch.setattr(gui, "TELEMETRY_DIR", tmp_path)
    first, last = telemetry_range_bounds("month" if mode == "period" else mode, now)
    plotter.start_date_var = lambda: first.date()
    plotter.end_date_var = lambda: last.date()
    first, last = telemetry_range_bounds(mode, now, first.date(), last.date())
    (tmp_path / "telemetry_ticks.csv").write_text(
        HEADER + f"{first:%Y-%m-%d %H:%M:%S},400,400,400,400,40,41,42,43,22,24,58,0,0000,55,30\n"
    )

    plotter._render_historical_figure(mode)

    for axis in (plotter.ax_temp, plotter.ax_moist):
        assert axis.get_xlim() == pytest.approx(gui.mdates.date2num([first, last]), rel=0, abs=1e-9)
        ticks = axis.get_xticks()
        assert ticks[0] == pytest.approx(gui.mdates.date2num(first), rel=0, abs=1e-9)
        assert all(axis.get_xlim()[0] <= tick <= axis.get_xlim()[1] for tick in ticks)
        assert axis.xaxis.get_major_formatter()(ticks[0]) == gui.visible_date(first)
    assert gui.visible_date(first) in plotter.fig._suptitle.get_text()
