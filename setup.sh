#!/usr/bin/env bash
# =============================================================================
# CEA Irrigation System — Setup Script
# Tested on: Raspberry Pi 5, Raspberry Pi OS Bookworm (64-bit)
# =============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RED='\033[0;31m'
GRN='\033[0;32m'
YLW='\033[1;33m'
BLU='\033[1;34m'
NC='\033[0m'

info()    { echo -e "${BLU}[INFO]${NC}  $*"; }
success() { echo -e "${GRN}[OK]${NC}    $*"; }
warn()    { echo -e "${YLW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

echo ""
echo -e "${BLU}╔══════════════════════════════════════════════╗${NC}"
echo -e "${BLU}║   CEA Irrigation System — Setup Script       ║${NC}"
echo -e "${BLU}╚══════════════════════════════════════════════╝${NC}"
echo ""

# ── 1. System prerequisites ───────────────────────────────────────────────────
info "Step 1/6 — Installing system prerequisites..."
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    git curl python3 python3-dev python3-venv \
    libgpiod2 libcamera-dev python3-picamera2 \
    arduino-cli i2c-tools usbutils
success "System prerequisites installed."

# ── 2. uv + Python venv ───────────────────────────────────────────────────────
info "Step 2/6 — Installing uv and setting up Python environment..."
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck source=/dev/null
    source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
fi
success "uv is available: $(uv --version)"

cd "$REPO_DIR"
uv venv --python 3.13
uv sync --extra gui
success "Python environment ready at $REPO_DIR/.venv"

# ── 3. Google Coral USB TPU driver ────────────────────────────────────────────
info "Step 3/6 — Installing Google Coral USB TPU driver (libedgetpu)..."
if ! dpkg -l | grep -q libedgetpu1; then
    echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
        | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | sudo gpg --dearmor -o /etc/apt/trusted.gpg.d/coral-edgetpu.gpg
    sudo apt-get update -qq
    sudo apt-get install -y libedgetpu1-std
    success "Coral Edge TPU driver installed."
else
    success "Coral Edge TPU driver already installed."
fi

# Add user to 'plugdev' group for USB device access
if ! groups "$USER" | grep -q plugdev; then
    sudo usermod -aG plugdev "$USER"
    warn "Added '$USER' to plugdev group. Log out and back in for USB TPU access."
fi

# ── 4. USB3 max power for Coral TPU ──────────────────────────────────────────
info "Step 4/6 — Configuring USB3 max current for Coral TPU..."
CONFIG_FILE="/boot/firmware/config.txt"
if grep -q "usb_max_current_enable" "$CONFIG_FILE" 2>/dev/null; then
    success "USB max current already configured in $CONFIG_FILE."
else
    echo "" | sudo tee -a "$CONFIG_FILE" > /dev/null
    echo "# Enable USB3 max current for Google Coral USB TPU" | sudo tee -a "$CONFIG_FILE" > /dev/null
    echo "usb_max_current_enable=1" | sudo tee -a "$CONFIG_FILE" > /dev/null
    success "Added usb_max_current_enable=1 to $CONFIG_FILE."
    warn "A reboot is required for USB max current to take effect (see step 7)."
fi

# ── 5. Tailscale (SSH + VPN) ─────────────────────────────────────────────────
info "Step 5/6 — Installing Tailscale..."
if ! command -v tailscale &>/dev/null; then
    curl -fsSL https://tailscale.com/install.sh | sh
    success "Tailscale installed."
else
    success "Tailscale already installed: $(tailscale version | head -1)"
fi

echo ""
echo -e "${YLW}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${YLW}║  Tailscale Authentication Required                         ║${NC}"
echo -e "${YLW}║  You will be shown a URL — visit it to link this device.   ║${NC}"
echo -e "${YLW}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
sudo tailscale up
success "Tailscale connected."

# ── 6. Arduino libraries ──────────────────────────────────────────────────────
info "Step 6/6 — Installing Arduino libraries..."
ARDUINO_LIB_DIR="$HOME/Arduino/libraries"
mkdir -p "$ARDUINO_LIB_DIR"
cp -r "$REPO_DIR/firmware/libraries/"* "$ARDUINO_LIB_DIR/"
success "Arduino libraries installed to $ARDUINO_LIB_DIR"

# ── Final checklist ───────────────────────────────────────────────────────────
echo ""
echo -e "${GRN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GRN}║  Setup Complete — Final Checklist                            ║${NC}"
echo -e "${GRN}╠══════════════════════════════════════════════════════════════╣${NC}"
echo -e "${GRN}║  ✅  System packages installed                               ║${NC}"
echo -e "${GRN}║  ✅  Python 3.13 venv ready (.venv)                          ║${NC}"
echo -e "${GRN}║  ✅  Google Coral USB TPU driver (libedgetpu1-std)            ║${NC}"
echo -e "${GRN}║  ✅  USB3 max current configured in /boot/firmware/config.txt ║${NC}"
echo -e "${GRN}║  ✅  Tailscale installed and authenticated                    ║${NC}"
echo -e "${GRN}║  ✅  Arduino libraries installed                              ║${NC}"
echo -e "${GRN}╠══════════════════════════════════════════════════════════════╣${NC}"
echo -e "${YLW}║  ⚠️   REBOOT REQUIRED for USB3 max current to take effect     ║${NC}"
echo -e "${YLW}║       Run: sudo reboot                                        ║${NC}"
echo -e "${GRN}╠══════════════════════════════════════════════════════════════╣${NC}"
echo -e "${BLU}║  Next steps:                                                 ║${NC}"
echo -e "${BLU}║    1. Plug in Arduino via USB (check /dev/ttyACM0)           ║${NC}"
echo -e "${BLU}║    2. Flash firmware: firmware/controller/                   ║${NC}"
echo -e "${BLU}║    3. Run the GUI:                                            ║${NC}"
echo -e "${BLU}║       uv run --extra gui python -m gui.psc_irr_gui           ║${NC}"
echo -e "${GRN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
