## Installation
Arduino environment (eg, RPi)
```bash
mkdir -p ~/Arduino/libraries
cp -r firmware/libraries/* ~/Arduino/libraries/
```

## Project Structure
```
cea_irrigation/
├── firmware/
│   ├── controller/              # Arduino production firmware
│   ├── hardware_tests/          # Arduino hardware test sketches (moved from tests/Sketches)
│   │   ├── light/
│   │   ├── pan_tilt/
│   │   ├── pump_relay/
│   │   ├── soil_moisture/
│   │   ├── soil_temp/
│   │   └── temp_humi/
│   └── libraries/               # Arduino C++ libraries (DHT, OneWire, DFRobot_VEML7700)
├── gui/                         # Desktop UI application & vision modules
├── scripts/                     # CLI controllers & serial tools
└── tests/                       # Pure Python pytest suite
├── __init__.py
└── test_controller_parsing.py
```