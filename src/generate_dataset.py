import os
import random
import pandas as pd

random.seed(42)

rows = []

# -------------------------------------------------
# Helper
# -------------------------------------------------

def clamp(x, low, high):
    return max(low, min(x, high))

# -------------------------------------------------
# ALERT CLASS (0)
# -------------------------------------------------

for _ in range(1700):
    row = {
        "typing_speed_kpm": random.uniform(180, 320),
        "backspace_rate": random.uniform(0.01, 0.05),
        "mean_key_interval_ms": random.uniform(80, 180),
        "std_key_interval_ms": random.uniform(10, 40),
        "mouse_distance_px": random.uniform(4000, 12000),
        "mouse_speed_px_per_sec": random.uniform(70, 220),
        "click_rate_per_min": random.uniform(8, 25),
        "mean_click_latency_ms": random.uniform(50, 180),
        "idle_time_sec": random.uniform(0, 4),
        "session_minutes": random.uniform(0, 60),
        "fatigue_label": 0
    }
    rows.append(row)

# -------------------------------------------------
# MILD FATIGUE (1)
# -------------------------------------------------

for _ in range(1700):
    row = {
        "typing_speed_kpm": random.uniform(120, 220),
        "backspace_rate": random.uniform(0.04, 0.10),
        "mean_key_interval_ms": random.uniform(140, 260),
        "std_key_interval_ms": random.uniform(35, 90),
        "mouse_distance_px": random.uniform(2500, 9000),
        "mouse_speed_px_per_sec": random.uniform(40, 140),
        "click_rate_per_min": random.uniform(5, 18),
        "mean_click_latency_ms": random.uniform(150, 350),
        "idle_time_sec": random.uniform(3, 8),
        "session_minutes": random.uniform(30, 180),
        "fatigue_label": 1
    }
    rows.append(row)

# -------------------------------------------------
# HIGH FATIGUE (2)
# -------------------------------------------------

for _ in range(1600):
    row = {
        "typing_speed_kpm": random.uniform(60, 150),
        "backspace_rate": random.uniform(0.08, 0.22),
        "mean_key_interval_ms": random.uniform(220, 600),
        "std_key_interval_ms": random.uniform(70, 180),
        "mouse_distance_px": random.uniform(800, 5000),
        "mouse_speed_px_per_sec": random.uniform(10, 80),
        "click_rate_per_min": random.uniform(1, 12),
        "mean_click_latency_ms": random.uniform(250, 800),
        "idle_time_sec": random.uniform(6, 20),
        "session_minutes": random.uniform(90, 420),
        "fatigue_label": 2
    }
    rows.append(row)

# -------------------------------------------------
# Add slight noise
# -------------------------------------------------

for row in rows:
    for key in row:
        if key != "fatigue_label":
            noise = random.uniform(-0.03, 0.03)
            row[key] = clamp(row[key] * (1 + noise), 0, 999999)

# -------------------------------------------------
# Shuffle
# -------------------------------------------------

random.shuffle(rows)

# -------------------------------------------------
# Save
# -------------------------------------------------

df = pd.DataFrame(rows)

os.makedirs("data", exist_ok=True)
df.to_csv("data/fatigue_dataset.csv", index=False)

print("Dataset created successfully.")
print("Rows:", len(df))
print(df.head())