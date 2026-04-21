# logger.py
# -----------------------------------------------------------
# Final Corrected Behavioral Biometrics Logger
# Real keyboard + mouse telemetry
# Rolling 10-second windows
# Metrics scaled correctly for model inference
#
# Run:
# python logger.py
#
# Output:
# live_metrics.csv
# -----------------------------------------------------------

import time
import math
import threading
import statistics
from datetime import datetime

import pandas as pd
from pynput import keyboard, mouse

# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------

WINDOW_SECONDS = 3600
CSV_FILE = "live_metrics.csv"

# -----------------------------------------------------------
# GLOBAL STATE
# -----------------------------------------------------------

lock = threading.Lock()

# keyboard
total_keys = 0
backspace_count = 0
key_timestamps = []

# mouse
mouse_distance = 0.0
click_count = 0
click_latencies = []

# previous mouse position
prev_mouse_x = None
prev_mouse_y = None

# activity tracking
last_activity_time = time.time()
last_mouse_move_time = None

# -----------------------------------------------------------
# KEYBOARD LISTENER
# -----------------------------------------------------------

def on_key_press(key):
    global total_keys, backspace_count
    global key_timestamps, last_activity_time

    now = time.time()

    with lock:
        total_keys += 1
        key_timestamps.append(now)
        last_activity_time = now

        if key == keyboard.Key.backspace:
            backspace_count += 1


# -----------------------------------------------------------
# MOUSE MOVE LISTENER
# -----------------------------------------------------------

def on_move(x, y):
    global prev_mouse_x, prev_mouse_y
    global mouse_distance
    global last_activity_time, last_mouse_move_time

    now = time.time()

    with lock:
        if prev_mouse_x is not None and prev_mouse_y is not None:
            dx = x - prev_mouse_x
            dy = y - prev_mouse_y
            dist = math.sqrt(dx * dx + dy * dy)
            mouse_distance += dist

        prev_mouse_x = x
        prev_mouse_y = y

        last_activity_time = now
        last_mouse_move_time = now


# -----------------------------------------------------------
# MOUSE CLICK LISTENER
# -----------------------------------------------------------

def on_click(x, y, button, pressed):
    global click_count, click_latencies
    global last_activity_time

    if not pressed:
        return

    now = time.time()

    with lock:
        click_count += 1
        last_activity_time = now

        if last_mouse_move_time is not None:
            latency_ms = (now - last_mouse_move_time) * 1000

            if latency_ms >= 0:
                click_latencies.append(latency_ms)


# -----------------------------------------------------------
# COMPUTE METRICS
# -----------------------------------------------------------

def compute_metrics():
    global total_keys, backspace_count
    global key_timestamps
    global mouse_distance, click_count, click_latencies
    global last_activity_time

    with lock:

        now = time.time()

        # ---------------------------------------------------
        # Keyboard timing intervals
        # ---------------------------------------------------

        key_intervals_ms = []

        if len(key_timestamps) >= 2:
            for i in range(1, len(key_timestamps)):
                diff = (key_timestamps[i] - key_timestamps[i - 1]) * 1000
                key_intervals_ms.append(diff)

        mean_key_interval = (
            statistics.mean(key_intervals_ms)
            if key_intervals_ms else 0.0
        )

        std_key_interval = (
            statistics.pstdev(key_intervals_ms)
            if len(key_intervals_ms) >= 2 else 0.0
        )

        # ---------------------------------------------------
        # Scale metrics to per-minute units
        # ---------------------------------------------------

        scale_factor = 60 / WINDOW_SECONDS

        typing_speed_kpm = total_keys * scale_factor

        click_rate = click_count * scale_factor

        backspace_rate = (
            backspace_count / total_keys
            if total_keys > 0 else 0.0
        )

        mouse_speed = mouse_distance / WINDOW_SECONDS

        mean_click_latency = (
            statistics.mean(click_latencies)
            if click_latencies else 0.0
        )

        idle_time = now - last_activity_time

        # ---------------------------------------------------
        # Output Row
        # ---------------------------------------------------

        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "window_seconds": WINDOW_SECONDS,
            "total_keys": total_keys,
            "typing_speed_kpm": round(typing_speed_kpm, 2),
            "backspace_count": backspace_count,
            "backspace_rate": round(backspace_rate, 4),
            "mean_key_interval_ms": round(mean_key_interval, 2),
            "std_key_interval_ms": round(std_key_interval, 2),
            "mouse_distance_px": round(mouse_distance, 2),
            "mouse_speed_px_per_sec": round(mouse_speed, 2),
            "click_count": click_count,
            "click_rate_per_min": round(click_rate, 2),
            "mean_click_latency_ms": round(mean_click_latency, 2),
            "idle_time_sec": round(idle_time, 2),
        }

        reset_window_state()

        return row


# -----------------------------------------------------------
# RESET STATE
# -----------------------------------------------------------

def reset_window_state():
    global total_keys, backspace_count
    global key_timestamps
    global mouse_distance, click_count
    global click_latencies
    global prev_mouse_x, prev_mouse_y

    total_keys = 0
    backspace_count = 0
    key_timestamps = []

    mouse_distance = 0.0
    click_count = 0
    click_latencies = []

    prev_mouse_x = None
    prev_mouse_y = None


# -----------------------------------------------------------
# SAVE CSV
# -----------------------------------------------------------

def append_to_csv(row):
    try:
        df = pd.DataFrame([row])

        with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
            header_needed = f.tell() == 0
            df.to_csv(f, header=header_needed, index=False)

    except Exception as e:
        print("CSV write error:", e)


# -----------------------------------------------------------
# LOOP
# -----------------------------------------------------------

def logger_loop():
    print(f"Logging every {WINDOW_SECONDS} seconds...")
    print("Press CTRL + C to stop.\n")

    while True:
        time.sleep(WINDOW_SECONDS)

        row = compute_metrics()
        append_to_csv(row)

        print("Logged:", row)


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------

def main():

    kb_listener = keyboard.Listener(
        on_press=on_key_press
    )

    ms_listener = mouse.Listener(
        on_move=on_move,
        on_click=on_click
    )

    kb_listener.start()
    ms_listener.start()

    logger_loop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped logger.")