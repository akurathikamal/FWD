🚁 Drone Communication System – Complete Understanding Guide
📌 Overview

This document explains how the following components work together in a drone system:

Pixhawk

MAVLink

pymavlink

MAVProxy

It also explains how commands like SERVO9 movement travel from your computer to the drone.

🧠 1. Pixhawk – The Drone Brain

Pixhawk is a flight controller hardware board.

It is the main control unit inside the drone.

Responsibilities

Controls motors

Stabilizes flight

Reads sensors (IMU, GPS, Compass, Barometer)

Controls servos (e.g., camera on SERVO9)

Executes commands received via MAVLink

Firmware

Pixhawk runs firmware such as:

ArduPilot

PX4

🗣 2. MAVLink – The Communication Language

MAVLink is a communication protocol used in drones.

It defines:

Message format

Command structure

Telemetry structure

Error checking

Example MAVLink Messages

Arm the drone

Change flight mode

Move SERVO9 to 1500 PWM

Send GPS position

Send battery status

⚠️ Important:
Pixhawk only understands MAVLink.

🐍 3. pymavlink – Python Translator

pymavlink is a Python library that allows Python scripts to communicate using MAVLink.

It:

Converts Python commands into MAVLink packets

Sends messages to Pixhawk

Receives telemetry from Pixhawk

Example
from pymavlink import mavutil

master = mavutil.mavlink_connection('/dev/ttyACM0', baud=57600)
master.wait_heartbeat()
Communication Flow
Python Script
→ pymavlink
→ MAVLink message
→ Pixhawk
🎮 4. MAVProxy – Command Line Ground Station

MAVProxy is a command-line ground control software.

You can type commands like:

arm throttle
mode guided

MAVProxy:

Converts commands into MAVLink

Sends them to Pixhawk

Communication Flow
User
→ MAVProxy
→ MAVLink
→ Pixhawk

⚠️ Important:
MAVProxy internally uses pymavlink.

🚁 Two Ways to Control Pixhawk

There are two main methods.

✅ Method 1 – Using MAVProxy
User → MAVProxy → MAVLink → Pixhawk

No custom Python script required

Simple command-line control

✅ Method 2 – Using Your Own Python Script
User → Python Script → pymavlink → MAVLink → Pixhawk

Full programmatic control

Used for automation

Used for AI systems

Used for advanced drone systems

⚠️ Common Confusion

Incorrect idea:

MAVProxy → Python Script → MAVLink → Pixhawk

This is NOT the normal setup.

You usually use:

MAVProxy
OR

Your Python script

Both send MAVLink directly to Pixhawk.

🎥 Example: Moving SERVO9

If you want to rotate a camera using SERVO9:

Python Code
master.mav.command_long_send(
    master.target_system,
    master.target_component,
    mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
    0,
    9,       # Servo channel
    1500,    # PWM value
    0, 0, 0, 0, 0
)
What Happens Internally

Python sends command

pymavlink converts it into MAVLink packet

Packet goes through USB (/dev/ttyACM0)

Pixhawk receives packet

Pixhawk outputs PWM signal

SERVO9 moves

Camera rotates

🔌 Communication Paths
Physical Connection
Computer ↔ USB ↔ Pixhawk
Logical Communication
Software
→ MAVLink
→ Pixhawk Firmware
→ PWM Output
→ Servo
🌐 Networking Extension (Optional)

MAVLink can also travel over:

UDP

TCP

4G/5G

WiFi

Radio telemetry

Example UDP:

GroundStationIP:14550
📊 Component Summary
Component	Type	Role
Pixhawk	Hardware	Drone brain
MAVLink	Protocol	Communication language
pymavlink	Library	Python MAVLink interface
MAVProxy	Software	Command-line ground station
🎯 Final Understanding

Pixhawk only understands MAVLink.

So any software that wants to control Pixhawk must speak MAVLink.

That software can be:

MAVProxy

Mission Planner

QGroundControl

Your own Python script (using pymavlink)

🚀 Practical Advice

If your goal is:

Control SERVO9 using Python

You only need:

Python Script → pymavlink → MAVLink → Pixhawk

MAVProxy is optional.

🧠 Architecture Summary Diagram
Option A:
You → MAVProxy → MAVLink → Pixhawk

Option B:
You → Python Script → pymavlink → MAVLink → Pixhawk
🏁 Conclusion

Pixhawk = Brain

MAVLink = Language

pymavlink = Python translator

MAVProxy = Command console

All drone communication is based on MAVLink.