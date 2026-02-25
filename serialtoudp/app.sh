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