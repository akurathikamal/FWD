🚀 Serial to UDP MAVLink Forwarding (WSL + Windows + Pixhawk)
📌 Overview

This guide explains how to:

Connect Pixhawk (USB) to WSL using usbipd

Forward MAVLink from Serial → Multiple UDP Ports

Run MAVProxy automatically for one or multiple devices

Monitor heartbeat and connection status

This setup supports:

Single Pixhawk

Multiple Pixhawks

Multiple UDP outputs (e.g., 14550, 14560, etc.)

🖥 Step 1: Share USB Device from Windows to WSL

Open Administrator PowerShell in Windows.

🔎 List USB Devices
usbipd list
Example Output
BUSID  VID:PID    DEVICE                                STATE
1-1    1209:5741  USB Serial Device (COM9)              Not shared
2-3    0403:6001  FTDI USB Serial                       Not shared
🔍 How to Identify Your Pixhawk

Look at:

DEVICE name → USB Serial Device (COM9)

VID:PID → 1209:5741 (ArduPilot Vendor ID)

COM Port → Visible in Device Manager

Example:

1-1    1209:5741  USB Serial Device (COM9)

Here:

BUSID = 1-1
🔗 Attach Device to WSL

Run:

usbipd attach --wsl --busid 1-1

Replace 1-1 with your actual BUSID.

🐧 Step 2: Create Forwarder Script in WSL

Open WSL terminal.

Create script:

nano forward.sh

Paste the full script (provided below).

Save:

CTRL + O
CTRL + X
🔐 Make Script Executable
chmod +x forward.sh

If file is green in ls, it's executable.

▶ Run Script
./forward.sh

Leave it running.

📡 Step 3: Connect MAVProxy to UDP

Open another WSL terminal.

Run:

mavproxy.py --master=127.0.0.1:14550

You can change port if needed.

📜 Complete Forwarder Script
#!/bin/bash

# ============================================================
#  MAVProxy Serial → UDP Forwarder
#  Scans /dev/serial/by-id/ → fans out to multiple UDP ports
# ============================================================

SERIAL_BASE="/dev/serial/by-id"
UDP_HOST="127.0.0.1"
UDP_OUT_PORTS=(14550 14560)
HEARTBEAT_WAIT=60

# ── Dependency check ─────────────────────────────────────────
check_deps() {
    for cmd in mavproxy.py setsid; do
        command -v "$cmd" &>/dev/null || {
            echo "[ERROR] Missing $cmd"
            exit 1
        }
    done
}

# ── Discover serial ports ─────────────────────────────────────
discover_ports() {
    if [[ ! -d "$SERIAL_BASE" ]]; then
        echo "[ERROR] $SERIAL_BASE not found."
        exit 1
    fi

    mapfile -t PORT_NAMES < <(ls "$SERIAL_BASE")

    if [[ ${#PORT_NAMES[@]} -eq 0 ]]; then
        echo "[ERROR] No serial devices found."
        exit 1
    fi
}

# ── Cleanup ───────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "Stopping MAVProxy..."
    for pid in "${MAVPIDS[@]}"; do
        kill "$pid" 2>/dev/null
    done
    exit 0
}
trap cleanup SIGINT SIGTERM

# ── Main ──────────────────────────────────────────────────────
check_deps
discover_ports

declare -a MAVPIDS

for port_name in "${PORT_NAMES[@]}"; do
    port_path="$SERIAL_BASE/$port_name"

    out_args=()
    for udp_port in "${UDP_OUT_PORTS[@]}"; do
        out_args+=(--out "udp:${UDP_HOST}:${udp_port}")
    done

    echo "[INFO] Starting MAVProxy for $port_name"

    setsid --fork mavproxy.py \
        --master="$port_path" \
        "${out_args[@]}" &

    pid=$!
    MAVPIDS+=($pid)

    echo "[INFO] PID $pid started"
done

echo ""
echo "[INFO] MAVProxy running. Press Ctrl+C to stop."

while true; do sleep 5; done
🧠 How It Works
🔄 Flow
Pixhawk → Serial (/dev/serial/by-id)
→ MAVProxy
→ UDP 127.0.0.1:14550
→ MAVProxy / Python / Ground Station
🎯 Features

Auto-detects serial devices

Forwards to multiple UDP ports

Background process using setsid

Clean shutdown on Ctrl+C

Works in WSL & native Linux

🔥 Multi-UDP Example
UDP_OUT_PORTS=(14550 14560 14570)

Now one Pixhawk streams to 3 different clients.

📊 Architecture Diagram
Pixhawk
   ↓ Serial
MAVProxy
   ↓
UDP:14550 → Ground Station
UDP:14560 → Python Script
UDP:14570 → Logger
⚠ Common Issues
Device Not Found

Run:

usbipd attach --wsl --busid <BUSID>
Permission Denied

Run:

sudo usermod -aG dialout $USER

Then restart WSL.

🚀 Professional Use Cases

Multi-drone lab setup

SITL + Real Hardware hybrid testing

Forwarding MAVLink to cloud

Telemetry monitoring system

Swarm drone architecture

📌 Summary
Component	Purpose
usbipd	Share USB to WSL
MAVProxy	Serial → MAVLink bridge
UDP	Network telemetry
setsid	Background process
Multiple ports	Multi-client support
🏁 Final Result

After setup:

Pixhawk → WSL → MAVProxy → UDP → Ground Control / Python