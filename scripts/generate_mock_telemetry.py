"""Generate realistic CEA environmental and soil moisture telemetry mock data for plotting."""

from __future__ import annotations

from datetime import datetime, timedelta
import math
from pathlib import Path
import random
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from gui.psc_irr_gui import TELEMETRY_DIR, DRY_BASELINES, WET_BASELINES

HEADER = (
    "timestamp,soil1_raw,soil2_raw,soil3_raw,soil4_raw,"
    "soil1_pct,soil2_pct,soil3_pct,soil4_pct,"
    "soil_temp_c,air_temp_c,humidity_pct,light,relays,pan_deg,tilt_deg\n"
)


def pct_to_raw(pct: float, ch: int) -> int:
    """Convert moisture % back to raw ADC count based on dry/wet baseline."""
    dry = DRY_BASELINES[ch]
    wet = WET_BASELINES[ch]
    raw = dry - (pct / 100.0) * (dry - wet)
    return max(50, min(600, int(round(raw))))


def generate_mock_telemetry(
    output_dir: Path = TELEMETRY_DIR,
    days: int = 30,
    step_minutes: int = 15,
    end_time: datetime | None = None,
) -> list[Path]:
    """Generate daily telemetry CSV files covering the past `days` days up to `end_time`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if end_time is None:
        end_time = datetime.now()

    start_time = end_time - timedelta(days=days)
    current_time = start_time.replace(second=0, microsecond=0)

    # Initial moisture levels for 4 channels
    moistures = [65.0, 72.0, 58.0, 68.0]
    dry_rates = [0.12, 0.15, 0.11, 0.14]  # moisture loss per hour

    records_by_day: dict[str, list[str]] = {}
    created_files: list[Path] = []

    while current_time <= end_time:
        day_key = current_time.strftime("%Y%m%d")
        if day_key not in records_by_day:
            records_by_day[day_key] = []

        # Time of day factors (0.0 to 24.0)
        hour = current_time.hour + current_time.minute / 60.0

        # Sunlight curve (lights ON 06:00 to 22:00, 16h photoperiod)
        if 6.0 <= hour <= 22.0:
            sun_phase = math.sin(math.pi * (hour - 6.0) / 16.0)
            light = int(800 + 700 * sun_phase + random.uniform(-40, 40))
            air_temp = 23.0 + 4.5 * sun_phase + random.uniform(-0.3, 0.3)
            soil_temp = 21.0 + 2.5 * sun_phase + random.uniform(-0.2, 0.2)
            rh = 68.0 - 18.0 * sun_phase + random.uniform(-1.5, 1.5)
            hourly_loss_mult = 1.4
        else:
            light = int(random.uniform(5, 30))
            air_temp = 19.5 + random.uniform(-0.4, 0.4)
            soil_temp = 20.0 + random.uniform(-0.2, 0.2)
            rh = 72.0 + random.uniform(-1.5, 1.5)
            hourly_loss_mult = 0.6

        # Moisture decay & scheduled / auto watering simulation
        relays = ["0", "0", "0", "0"]
        step_hours = step_minutes / 60.0

        for ch in range(4):
            # Moisture loss
            moistures[ch] -= dry_rates[ch] * step_hours * hourly_loss_mult

            # If moisture drops below setpoint (~38%), simulate irrigation cycle
            if moistures[ch] < 36.0:
                relays[ch] = "1"
                moistures[ch] = min(82.0, moistures[ch] + random.uniform(35.0, 45.0))

            moistures[ch] = max(10.0, min(95.0, moistures[ch]))

        relays_str = "".join(relays)
        raw_vals = [pct_to_raw(moistures[c], c) for c in range(4)]

        row = [
            current_time.strftime("%Y-%m-%d %H:%M:%S"),
            str(raw_vals[0]), str(raw_vals[1]), str(raw_vals[2]), str(raw_vals[3]),
            f"{moistures[0]:.1f}", f"{moistures[1]:.1f}", f"{moistures[2]:.1f}", f"{moistures[3]:.1f}",
            f"{soil_temp:.1f}", f"{air_temp:.1f}", f"{rh:.1f}", str(light),
            relays_str, "55", "30",
        ]
        records_by_day[day_key].append(",".join(row) + "\n")

        current_time += timedelta(minutes=step_minutes)

    # Write daily CSV files
    for day_key, lines in records_by_day.items():
        csv_file = output_dir / f"telemetry_{day_key}.csv"
        with csv_file.open("w", encoding="utf-8") as f:
            f.write(HEADER)
            f.writelines(lines)
        created_files.append(csv_file)

    print(f"[Mock Telemetry] Successfully generated {len(created_files)} daily files in {output_dir}")
    return created_files


if __name__ == "__main__":
    generate_mock_telemetry()
