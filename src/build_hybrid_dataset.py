import os
import random
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

# ----------------------------------------------------
# PATHS
# ----------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "data", "monkey_type_dataset.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "behavior_final_dataset.csv")

# ----------------------------------------------------
# LOAD DATA
# ----------------------------------------------------

df = pd.read_csv(INPUT_PATH)

numeric_cols = [
    "wpm",
    "acc",
    "rawWpm",
    "consistency",
    "restartCount",
    "testDuration",
    "afkDuration",
    "incompleteTestSeconds"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=numeric_cols).reset_index(drop=True)

print("Loaded rows:", len(df))

# ----------------------------------------------------
# REALISTIC LABEL ENGINE
# ----------------------------------------------------

def assign_label(row):
    risk = 0

    if row["acc"] < 94:
        risk += 1

    if row["consistency"] < 70:
        risk += 1

    if row["restartCount"] >= 2:
        risk += 1

    if row["afkDuration"] > 4:
        risk += 1

    if row["wpm"] < 42:
        risk += 1

    if row["testDuration"] > 60:
        risk += 1

    if risk <= 1:
        return 0
    elif risk <= 3:
        return 1
    else:
        return 2

df["fatigue_label"] = df.apply(assign_label, axis=1)

# ----------------------------------------------------
# FEATURE BUILDER
# ----------------------------------------------------

rows = []

for _, row in df.iterrows():

    label = row["fatigue_label"]

    # overlap added intentionally
    if label == 0:
        mouse_speed = random.uniform(65, 220)
        click_rate = random.uniform(7, 24)
        click_latency = random.uniform(70, 240)
        idle = random.uniform(0, 6)
        session = random.uniform(0, 90)

    elif label == 1:
        mouse_speed = random.uniform(35, 170)
        click_rate = random.uniform(4, 18)
        click_latency = random.uniform(120, 420)
        idle = random.uniform(2, 10)
        session = random.uniform(20, 220)

    else:
        mouse_speed = random.uniform(12, 110)
        click_rate = random.uniform(1, 14)
        click_latency = random.uniform(220, 780)
        idle = random.uniform(5, 22)
        session = random.uniform(60, 420)

    row_out = {
        "typing_speed_kpm": row["wpm"] * random.uniform(0.88, 1.12),
        "backspace_rate": max(
            0.01,
            ((100 - row["acc"]) / 100) * random.uniform(0.7, 1.6)
        ),
        "mean_key_interval_ms": (60000 / max(row["wpm"], 1)) * random.uniform(0.9, 1.15),
        "std_key_interval_ms": max(8, (100 - row["consistency"])) * random.uniform(0.8, 1.5),
        "mouse_distance_px": mouse_speed * 60 * random.uniform(0.6, 1.6),
        "mouse_speed_px_per_sec": mouse_speed,
        "click_rate_per_min": click_rate,
        "mean_click_latency_ms": click_latency,
        "idle_time_sec": idle,
        "session_minutes": session,
        "fatigue_label": label
    }

    rows.append(row_out)

# ----------------------------------------------------
# AUGMENT ALERT + MILD FROM REAL DATA
# ----------------------------------------------------

augmented = []

for row in rows:

    augmented.append(row)

    label = row["fatigue_label"]

    copies = 4 if label == 0 else 3 if label == 1 else 2

    for _ in range(copies):
        clone = row.copy()

        for k in clone:
            if k != "fatigue_label":
                noise = random.uniform(-0.15, 0.15)
                clone[k] = max(0, clone[k] * (1 + noise))

        # slight ambiguity
        if random.random() < 0.04:
            clone["fatigue_label"] = random.choice([0,1,2])

        augmented.append(clone)

# ----------------------------------------------------
# EXTRA HIGH FATIGUE SYNTHETIC BOOST
# ----------------------------------------------------

for _ in range(1300):

    high = {
        "typing_speed_kpm": random.uniform(18, 95),
        "backspace_rate": random.uniform(0.09, 0.26),
        "mean_key_interval_ms": random.uniform(300, 950),
        "std_key_interval_ms": random.uniform(45, 220),
        "mouse_distance_px": random.uniform(300, 4200),
        "mouse_speed_px_per_sec": random.uniform(5, 70),
        "click_rate_per_min": random.uniform(0.5, 10),
        "mean_click_latency_ms": random.uniform(280, 950),
        "idle_time_sec": random.uniform(8, 28),
        "session_minutes": random.uniform(120, 520),
        "fatigue_label": 2
    }

    # some contradictory edge cases
    if random.random() < 0.12:
        high["typing_speed_kpm"] = random.uniform(90, 140)

    augmented.append(high)

# ----------------------------------------------------
# SAVE
# ----------------------------------------------------

final_df = pd.DataFrame(augmented)
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

final_df.to_csv(OUTPUT_PATH, index=False)

print("Saved:", OUTPUT_PATH)
print("Rows:", len(final_df))
print(final_df["fatigue_label"].value_counts())