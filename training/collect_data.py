"""
collect_data.py — Swipe Gesture Training Data Collector
========================================================

Records hand-landmark position history from your webcam and saves each
sample to a CSV file that train_model.py can use.

HOW TO USE
----------
1.  Run this script:
        python collect_data.py

2.  A window will open showing your webcam feed with the index-finger tip
    marked in red.

3.  Press one of the number keys to start recording samples for that class:

        0  →  No Swipe   (keep your hand still)
        1  →  Swipe Left  (move hand quickly to the left)
        2  →  Swipe Right (move hand quickly to the right)
        3  →  Swipe Up   (move hand quickly upward)
        4  →  Swipe Down (move hand quickly downward)

4.  While a key is held the script saves one sample every two frames, so
    just keep performing the gesture naturally.

5.  Press R to pause recording (so you can rest between gestures).
    Press ESC or Q to quit.

TIPS FOR GOOD DATA
------------------
- Collect at least 300 samples per class (aim for 500+).
- Vary your speed, hand position on screen, and lighting.
- Keep "No Swipe" samples (class 0) at roughly the same count as the
  other classes combined.
- Perform clean, full swipes — do not start recording in the middle of
  a gesture; let the history window fill with the complete motion.

VELOCITY FEATURES
-----------------
When USE_VELOCITY_FEATURES = True (the default) each sample contains:
  - 16 relative positions  (dx0,dy0 … dx15,dy15)  — 32 floats
  - 15 frame-to-frame velocities (vx0,vy0 … vx14,vy14) — 30 floats
  Total: 62 floats per sample.

This gives the model explicit information about HOW FAST and in WHAT
DIRECTION the finger is moving, which is the key signal that separates
swipe gestures from a still "No Swipe" hand.

Set USE_VELOCITY_FEATURES = False to revert to the original 32-float
format (useful if you have an existing CSV you want to keep using).

The data is appended to `training_data.csv` in the same folder.
Run train_model.py when you are satisfied with the data.
"""

import copy
import csv
import os
from collections import deque

import cv2 as cv
import mediapipe as mp
import numpy as np

# ── Configuration ────────────────────────────────────────────────────────────
HISTORY_LENGTH = 16          # Frames of position history per sample
DATA_FILE = "training_data.csv"
IMAGE_WIDTH = 960
IMAGE_HEIGHT = 540

# When True, each sample includes HISTORY_LENGTH-1 frame-to-frame velocity
# vectors appended after the relative-position features.  This gives the
# model an explicit signal about motion speed and direction, which strongly
# improves accuracy on swipe gestures.
# Must match the USE_VELOCITY_FEATURES setting in train_model.py.
USE_VELOCITY_FEATURES = True

GESTURE_CLASSES = {
    0: "No Swipe",
    1: "Swipe Left",
    2: "Swipe Right",
    3: "Swipe Up",
    4: "Swipe Down",
}
# ─────────────────────────────────────────────────────────────────────────────

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def pre_process_point_history(history: list, use_velocity: bool = USE_VELOCITY_FEATURES) -> list:
    """
    Convert a list of (x, y) positions into a normalised 1-D feature vector.

    Steps
    -----
    1. Make coordinates relative to the first point in the history window.
    2. Flatten to a single list  [dx0, dy0, dx1, dy1, …].
    3. Normalise by the maximum absolute value so all values are in [-1, 1].
    4. (Optional, recommended) Append frame-to-frame velocity vectors
       [vx0, vy0, vx1, vy1, …] normalised independently.  This provides
       an explicit motion-direction signal that significantly helps the
       model distinguish swipe directions from a stationary hand.

    Returns
    -------
    list of floats:
      - 32 values when use_velocity=False  (HISTORY_LENGTH * 2)
      - 62 values when use_velocity=True   (HISTORY_LENGTH * 2 + (HISTORY_LENGTH-1) * 2)
    """
    temp = copy.deepcopy(history)
    if not temp:
        return []

    base_x, base_y = temp[0]
    flattened: list[float] = []
    for x, y in temp:
        flattened.append(x - base_x)
        flattened.append(y - base_y)

    max_val = max((abs(v) for v in flattened), default=1.0)
    if max_val > 0:
        flattened = [v / max_val for v in flattened]

    if not use_velocity:
        return flattened

    # --- Velocity features ---------------------------------------------------
    # Compute frame-to-frame differences from the original (un-normalised)
    # relative coordinates before dividing by max_val, then normalise
    # the velocity vector independently.
    velocities: list[float] = []
    for i in range(1, len(temp)):
        vx = (temp[i][0] - base_x) - (temp[i - 1][0] - base_x)
        vy = (temp[i][1] - base_y) - (temp[i - 1][1] - base_y)
        velocities.append(vx)
        velocities.append(vy)

    vel_max = max((abs(v) for v in velocities), default=1.0)
    if vel_max > 0:
        velocities = [v / vel_max for v in velocities]

    return flattened + velocities


def save_to_csv(label: int, data: list) -> None:
    with open(DATA_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([label] + data)
    print(f"  Saved sample  class={label}  ({GESTURE_CLASSES[label]})")


def count_existing_samples() -> dict:
    counts = {k: 0 for k in GESTURE_CLASSES}
    if not os.path.exists(DATA_FILE):
        return counts
    with open(DATA_FILE, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                try:
                    label = int(row[0])
                    if label in counts:
                        counts[label] += 1
                except ValueError:
                    pass
    return counts


def main() -> None:
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)

    point_history: deque = deque(maxlen=HISTORY_LENGTH)
    recording_class: int = -1
    frame_count: int = 0

    print("=== Swipe Gesture Data Collector ===")
    print(f"Data will be saved to: {DATA_FILE}")
    num_features = HISTORY_LENGTH * 2 + (HISTORY_LENGTH - 1) * 2 if USE_VELOCITY_FEATURES else HISTORY_LENGTH * 2
    print(f"Feature mode: {'positions + velocities' if USE_VELOCITY_FEATURES else 'positions only'} ({num_features} floats)")
    existing = count_existing_samples()
    print("Existing samples per class:")
    for k, v in existing.items():
        print(f"  {k}: {GESTURE_CLASSES[k]}  →  {v} samples")
    print()

    with mp_hands.Hands(
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    ) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Cannot read from webcam. Exiting.")
                break

            frame = cv.flip(frame, 1)           # Mirror so left/right are natural
            h, w = frame.shape[:2]

            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            hand_detected = False
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    hand_detected = True

                    # Track index finger tip (landmark 8)
                    tip = hand_landmarks.landmark[8]
                    cx = int(tip.x * w)
                    cy = int(tip.y * h)
                    point_history.append([tip.x, tip.y])

                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

                    # Record one sample every two frames when a class key is held
                    if (
                        recording_class >= 0
                        and len(point_history) == HISTORY_LENGTH
                        and frame_count % 2 == 0
                    ):
                        data = pre_process_point_history(list(point_history))
                        expected = num_features
                        if len(data) == expected:
                            save_to_csv(recording_class, data)

                    # Draw tracking dot
                    color = (0, 255, 0) if recording_class >= 0 else (0, 0, 255)
                    cv.circle(frame, (cx, cy), 10, color, -1)
                    break  # Only track the first hand

            if not hand_detected:
                point_history.clear()

            frame_count += 1

            # ── Key handling ─────────────────────────────────────────────────
            key = cv.waitKey(1) & 0xFF
            if key in (27, ord("q")):          # ESC or Q → quit
                break
            elif key in (ord("0"), ord("1"), ord("2"), ord("3"), ord("4")):
                recording_class = key - ord("0")
                print(f"Recording class {recording_class}: {GESTURE_CLASSES[recording_class]}")
            elif key == ord("r"):
                recording_class = -1
                point_history.clear()
                print("Recording paused.")

            # ── HUD ──────────────────────────────────────────────────────────
            if recording_class >= 0:
                status = (
                    f"RECORDING  class {recording_class}: "
                    f"{GESTURE_CLASSES[recording_class]}  |  "
                    f"history {len(point_history)}/{HISTORY_LENGTH}"
                )
                hud_color = (0, 255, 0)
            else:
                status = "PAUSED  (press 0-4 to start recording)"
                hud_color = (0, 200, 200)

            cv.putText(frame, status, (10, 35), cv.FONT_HERSHEY_SIMPLEX, 0.8, hud_color, 2)
            cv.putText(
                frame,
                "0=No Swipe  1=Left  2=Right  3=Up  4=Down  R=Pause  ESC/Q=Quit",
                (10, h - 15),
                cv.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                1,
            )

            cv.imshow("Swipe Gesture Data Collection", frame)

    cap.release()
    cv.destroyAllWindows()

    final_counts = count_existing_samples()
    print("\n=== Collection complete ===")
    for k, v in final_counts.items():
        print(f"  {k}: {GESTURE_CLASSES[k]}  →  {v} samples")
    print(f"\nRun  python train_model.py  to train the model.")


if __name__ == "__main__":
    main()
