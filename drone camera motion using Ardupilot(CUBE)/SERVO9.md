🚁 Servo9 Angle Control Using Pixhawk (WSL + USBIPD + pymavlink)
📌 Overview

This guide explains how to:

Attach Pixhawk USB to WSL

Connect using pymavlink

Convert angle (0–180°) → PWM

Control SERVO9 from laptop

Run safely and correctly

🖥 Step 1: Attach Pixhawk to WSL

Open Administrator PowerShell.

🔎 1. List USB Devices
usbipd list

Example output:

BUSID  VID:PID    DEVICE                                STATE
1-1    1209:5741  USB Serial Device (COM9)              Not shared
Identify Your Device Using:

DEVICE name → USB Serial Device (COMx)

VID:PID → 1209:5741 (ArduPilot)

BUSID → First column (example: 1-1)

🔗 2. Attach Device to WSL
usbipd attach --wsl --busid 1-1

Replace 1-1 with your BUSID.

🐧 Step 2: Verify in WSL

Open WSL terminal:

ls /dev/ttyACM*

You should see:

/dev/ttyACM0

Now Pixhawk is connected to Linux.

📝 Step 3: Create Python File
nano servo9.py

Paste the code below.

Save:

CTRL + O
CTRL + X
🔐 Step 4: Make File Executable (Important)

Before running:

chmod +x servo9.py

Why?

This allows you to run:

./servo9.py

Instead of:

python3 servo9.py

⚠ Important:
Add this at top of file if using ./servo9.py:

#!/usr/bin/env python3
🐍 Complete Servo9 Control Code
#!/usr/bin/env python3

from pymavlink import mavutil
import time

print("Connecting to Pixhawk...")

master = mavutil.mavlink_connection('/dev/ttyACM0', baud=115200)
master.wait_heartbeat()

print("Connected!")

def move_servo(channel, pwm):
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
        0,
        channel,
        pwm,
        0, 0, 0, 0, 0
    )

while True:
    print("\nEnter angle (0 to 180) or -1 to exit:")
    
    try:
        angle = float(input())
    except:
        print("Invalid input!")
        continue

    if angle == -1:
        print("Exiting...")
        break

    if 0 <= angle <= 180:
        pwm = 1000 + (angle / 180.0) * 1000
        move_servo(9, pwm)
        print(f"Moved Servo 9 to {angle}° (PWM {int(pwm)})")
    else:
        print("Invalid angle!")
🎯 How Angle Manager Works

You enter:

Angle: 90°

The system converts:

PWM = 1000 + (angle / 180) * 1000
Mapping:
Angle	PWM
0°	1000
90°	1500
180°	2000

This converts degrees → PWM signal.

Pixhawk sends PWM to SERVO9.

Servo rotates accordingly.

🔄 Full Control Flow
Laptop
   ↓
Python Script
   ↓
pymavlink
   ↓
MAVLink
   ↓
USB (/dev/ttyACM0)
   ↓
Pixhawk
   ↓
PWM Output
   ↓
SERVO9
   ↓
Camera moves
🚀 How to Run

If executable:

./servo9.py

If not:

python3 servo9.py
⚠ Important Notes
1️⃣ Pixhawk Must Be:

Powered

Properly configured

Servo9 output enabled

In correct mode (GUIDED recommended)

2️⃣ If Servo Doesn’t Move

Check:

SERVO9_FUNCTION parameter

PWM range in ArduPilot

Safety switch

Arm state

3️⃣ If Connection Fails

Check:

ls /dev/ttyACM*

If not found → Re-run:

usbipd attach --wsl --busid <BUSID>
🧠 Professional Improvement Ideas

You can improve this by:

Adding smooth servo sweep

Adding auto-centering

Adding joystick control

Adding GUI

Adding multithreading

Checking COMMAND_ACK

📊 System Summary
Component	Role
usbipd	Share USB to WSL
/dev/ttyACM0	Serial port
pymavlink	Python MAVLink interface
MAV_CMD_DO_SET_SERVO	Servo command
PWM	Physical motor signal
🏁 Final Result

After setup:

You type angle → Servo moves instantly

Laptop becomes live servo controller.