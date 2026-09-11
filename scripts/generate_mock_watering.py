#!/usr/bin/env python3
"""Generate synthetic telemetry CSV dataset with a 10:00 AM watering scenario.

This script creates realistic 24-hour telemetry logs matching the schema used by
the CEA Irrigation Controller. At 10:00 AM, soil moisture drops below the 40%
setpoint, triggering pumps 1 and 2 (500 L/h flow rate) for 90 seconds (12.5 L each)
until soil moisture reaches the 80% target cutoff.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import math
from pathlib import Path
import random


HEADER = (
    "timestamp,soil1_raw,soil2_raw,soil3_raw,soil4_raw,"
    "soil1_pct,soil2_pct,soil3_pct,soil4_pct,"
    "soil_temp_c,air_temp_c,humidity_pct,light,relays,pan_deg,tilt_deg\n"
)


def generate_mock_data(target_date: datetime.date, output_dir: Path) -> Path:
    """Generate mock 24-hour telemetry CSV with 10:00 AM watering event."""
    output_dir.mkdir(parents=True, exist_ok=True)
    date_str = target_date.strftime("%Y%m%d")
    out_file = output_dir / f"telemetry_{date_str}.csv"

    rows: list[str] = []
    base_dt = datetime(target_date.year, target_date.month, target_date.day, 0, 0, 0)

    # 1. Pre-watering baseline (00:00 to 09:55) every 5 minutes
    # Moisture starts at 58% and slowly dries to ~38% by 09:55
    t = base_dt
    t_end_morning = base_dt.replace(hour=9, minute=55)
    total_morning_steps = int((t_end_morning - t).total_seconds() / 300)

    step = 0
    while t < t_end_morning:
        frac = step / max(1, total_morning_steps)
        # Drying curve: 58% down to ~38%
        m_base = 58.0 - (20.0 * frac)
        m1 = round(m_base + random.uniform(-0.6, 0.6), 1)
        m2 = round(m_base - 0.5 + random.uniform(-0.6, 0.6), 1)
        m3 = round(m_base + 3.0 + random.uniform(-0.5, 0.5), 1)
        m4 = round(m_base + 4.0 + random.uniform(-0.5, 0.5), 1)

        # Environmental diurnal curves
        hour_rad = (t.hour + t.minute / 60.0) / 24.0 * 2.0 * math.pi
        at = round(21.0 + 3.5 * math.sin(hour_rad - 1.5) + random.uniform(-0.2, 0.2), 1)
        st = round(20.0 + 2.0 * math.sin(hour_rad - 2.0) + random.uniform(-0.2, 0.2), 1)
        rh = round(65.0 - 15.0 * math.sin(hour_rad - 1.5) + random.uniform(-0.8, 0.8), 1)
        light = int(max(0, 1500 * math.sin(hour_rad - 1.2)))

        row = (
            f"{t.strftime('%Y-%m-%d %H:%M:%S')},"
            f"350,360,370,380,"
            f"{m1:.1f},{m2:.1f},{m3:.1f},{m4:.1f},"
            f"{st:.1f},{at:.1f},{rh:.1f},{light},0000,55,30\n"
        )
        rows.append(row)
        t += timedelta(minutes=5)
        step += 1

    # 2. Watering Event at 10:00 AM (10:00:00 to 10:01:30) logged every 15 seconds
    # Pumps 1 and 2 active ('1100'), duration 90s -> 12.5 L each at 500 L/h
    t_event_start = base_dt.replace(hour=10, minute=0, second=0)
    t = t_event_start
    event_ticks = [0, 15, 30, 45, 60, 75, 90]

    for sec in event_ticks:
        t_tick = t_event_start + timedelta(seconds=sec)
        # Soil moisture rising during watering from ~37% to 80%
        prog = sec / 90.0
        m1 = round(37.5 + (80.0 - 37.5) * prog, 1)
        m2 = round(37.0 + (80.0 - 37.0) * prog, 1)
        m3 = round(41.0 - 0.5 * (1 - prog), 1)  # channel 3 stayed above setpoint
        m4 = round(42.0 - 0.5 * (1 - prog), 1)  # channel 4 stayed above setpoint

        relays = "1100" if sec < 90 else "0000"
        row = (
            f"{t_tick.strftime('%Y-%m-%d %H:%M:%S')},"
            f"380,390,370,380,"
            f"{m1:.1f},{m2:.1f},{m3:.1f},{m4:.1f},"
            f"22.5,24.8,55.0,1200,{relays},55,30\n"
        )
        rows.append(row)

    # 3. Post-watering drainage (10:05 to 23:55) every 5 minutes
    t = base_dt.replace(hour=10, minute=5, second=0)
    t_end_day = base_dt.replace(hour=23, minute=55, second=0)
    total_post_steps = int((t_end_day - t).total_seconds() / 300)

    step = 0
    while t <= t_end_day:
        frac = step / max(1, total_post_steps)
        # Slow drainage: 80% down to ~64%
        m_base = 80.0 - (16.0 * frac)
        m1 = round(m_base + random.uniform(-0.5, 0.5), 1)
        m2 = round(m_base - 0.4 + random.uniform(-0.5, 0.5), 1)
        m3 = round(m_base - 1.5 + random.uniform(-0.5, 0.5), 1)
        m4 = round(m_base - 1.0 + random.uniform(-0.5, 0.5), 1)

        hour_rad = (t.hour + t.minute / 60.0) / 24.0 * 2.0 * math.pi
        at = round(21.0 + 4.0 * math.sin(hour_rad - 1.5) + random.uniform(-0.2, 0.2), 1)
        st = round(20.0 + 2.5 * math.sin(hour_rad - 2.0) + random.uniform(-0.2, 0.2), 1)
        rh = round(65.0 - 18.0 * math.sin(hour_rad - 1.5) + random.uniform(-0.8, 0.8), 1)
        light = int(max(0, 1800 * math.sin(hour_rad - 1.2)))

        row = (
            f"{t.strftime('%Y-%m-%d %H:%M:%S')},"
            f"160,165,170,175,"
            f"{m1:.1f},{m2:.1f},{m3:.1f},{m4:.1f},"
            f"{st:.1f},{at:.1f},{rh:.1f},{light},0000,55,30\n"
        )
        rows.append(row)
        t += timedelta(minutes=5)
        step += 1

    with out_file.open("w", encoding="utf-8") as f:
        f.write(HEADER)
        f.writelines(rows)

    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate mock CEA telemetry dataset with 10:00 AM watering event."
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Target date in YYYY-MM-DD format (default: today)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="Data/telemetry",
        help="Target directory for telemetry CSV files (default: Data/telemetry)",
    )
    args = parser.parse_args()

    if args.date:
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    else:
        target_date = datetime.now().date()

    out_path = generate_mock_data(target_date, Path(args.outdir))
    print(f"[Mock Generator] Successfully generated mock telemetry dataset:")
    print(f"  File: {out_path}")
    print(f"  Date: {target_date.isoformat()}")
    print(f"  Event: 10:00 AM watering event")
    print(f"         Pumps 1 & 2 active for 90s (~12.5 L each at 500 L/h)")
    print(f"         Moisture: drops <40% (37.0%) -> rises to 80.0% target cutoff")


if __name__ == "__main__":
    main()
