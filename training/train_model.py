"""
train_model.py — Train a Swipe Gesture Neural Network
======================================================

Reads the CSV file produced by collect_data.py, trains a small dense neural
network with Keras, and saves the best model as `swipe_gesture_model.hdf5`.

Run this script after collecting enough training data:

    python train_model.py

After training, run convert_model.py to export the model for the browser.

MODEL INPUT / OUTPUT
--------------------
  Input  : 32 floats  (HISTORY_LENGTH=16 frames × 2 coordinates per frame)
  Output : 5 classes  (No Swipe / Swipe Left / Right / Up / Down)

GUIDELINES
----------
- Aim for at least 200 samples per class before training.
- If test accuracy is low, collect more data (especially for the confused
  classes) and re-run this script.
"""

import os

# Suppress TensorFlow/CUDA log noise on CPU-only machines.  These must be set
# before tensorflow is imported (directly or transitively).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # suppress C++ log spam
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # disable GPU look-up

import matplotlib
matplotlib.use("Agg")           # Headless back-end — saves PNG without GUI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

# ── Configuration ────────────────────────────────────────────────────────────
DATA_FILE = "training_data.csv"
MODEL_SAVE_PATH = "swipe_gesture_model.hdf5"
HISTORY_PLOT_PATH = "training_history.png"

HISTORY_LENGTH = 16           # Must match collect_data.py and useSwipeDetector.ts
NUM_FEATURES = HISTORY_LENGTH * 2   # 32 inputs

GESTURE_CLASSES = {
    0: "No Swipe",
    1: "Swipe Left",
    2: "Swipe Right",
    3: "Swipe Up",
    4: "Swipe Down",
}
NUM_CLASSES = len(GESTURE_CLASSES)
# ─────────────────────────────────────────────────────────────────────────────


def load_data() -> tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"'{DATA_FILE}' not found. "
            "Run collect_data.py first to create training data."
        )

    df = pd.read_csv(DATA_FILE, header=None)
    X = df.iloc[:, 1:].values.astype(np.float32)
    y = df.iloc[:, 0].values.astype(np.int32)

    if X.shape[1] != NUM_FEATURES:
        raise ValueError(
            f"Expected {NUM_FEATURES} features per sample, "
            f"but got {X.shape[1]}.  "
            "Make sure HISTORY_LENGTH matches the value used in collect_data.py."
        )

    print(f"Loaded {len(X)} samples with {NUM_FEATURES} features each.")
    print("Samples per class:")
    for class_id, class_name in GESTURE_CLASSES.items():
        count = int(np.sum(y == class_id))
        bar = "█" * (count // 10)
        print(f"  [{class_id}] {class_name:15s}  {count:4d}  {bar}")

    return X, y


def build_model() -> Sequential:
    """
    Small fully-connected network.

    Architecture  :  32 → 24 (ReLU) → Dropout → 10 (ReLU) → 5 (Softmax)
    Parameter count: ~1 K  (tiny; runs fast in the browser via TensorFlow.js)
    """
    model = Sequential(
        [
            Dense(24, activation="relu", input_shape=(NUM_FEATURES,)),
            Dropout(0.3),
            Dense(10, activation="relu"),
            Dense(NUM_CLASSES, activation="softmax"),
        ],
        name="swipe_gesture_classifier",
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    return model


def plot_history(history, path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(path)
    print(f"Training plot saved to: {path}")


def main() -> None:
    print("=== Swipe Gesture Model Training ===\n")

    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(
        f"\nSplit: {len(X_train)} train / {len(X_test)} test\n"
    )

    model = build_model()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    print("\nTraining…")
    history = model.fit(
        X_train,
        y_train,
        epochs=200,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n── Test results ──────────────────────")
    print(f"  Loss    : {test_loss:.4f}")
    print(f"  Accuracy: {test_accuracy:.4f}  ({test_accuracy * 100:.1f} %)")

    if test_accuracy < 0.80:
        print(
            "\n⚠  Accuracy is below 80 %.  Consider collecting more data "
            "(especially for classes that are frequently confused) and "
            "re-running this script."
        )

    plot_history(history, HISTORY_PLOT_PATH)

    print(f"\n✓  Model saved to: {MODEL_SAVE_PATH}")
    print("\nNext step: run  python convert_model.py  to export for the browser.")


if __name__ == "__main__":
    main()
