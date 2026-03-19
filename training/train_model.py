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
  When USE_VELOCITY_FEATURES = True  (recommended, default):
    Input  : 62 floats  (HISTORY_LENGTH=16 positions × 2   = 32 floats
                       + (HISTORY_LENGTH-1)=15 velocities × 2 = 30 floats)
  When USE_VELOCITY_FEATURES = False:
    Input  : 32 floats  (HISTORY_LENGTH=16 frames × 2 coordinates per frame)
  Output : 5 classes  (No Swipe / Swipe Left / Right / Up / Down)

  Must match the USE_VELOCITY_FEATURES setting in collect_data.py.

GUIDELINES
----------
- Aim for at least 300 samples per class before training (500+ recommended).
- If test accuracy is low, collect more data (especially for the confused
  classes) and re-run this script.
- Velocity features (USE_VELOCITY_FEATURES=True) substantially improve
  accuracy because they give the model explicit motion-direction information.
"""

import os

import matplotlib
matplotlib.use("Agg")           # Headless back-end — saves PNG without GUI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# ── Configuration ────────────────────────────────────────────────────────────
DATA_FILE = "training_data.csv"
MODEL_SAVE_PATH = "swipe_gesture_model.hdf5"
HISTORY_PLOT_PATH = "training_history.png"

HISTORY_LENGTH = 16           # Must match collect_data.py and useSwipeDetector.ts

# When True, each sample is expected to contain velocity features appended
# after the position features.  Must match USE_VELOCITY_FEATURES in
# collect_data.py.  Velocity features strongly improve accuracy because they
# give the model explicit information about motion direction and speed.
USE_VELOCITY_FEATURES = True

# Feature counts
NUM_POSITION_FEATURES = HISTORY_LENGTH * 2                          # 32
NUM_VELOCITY_FEATURES = (HISTORY_LENGTH - 1) * 2 if USE_VELOCITY_FEATURES else 0  # 30 or 0
NUM_FEATURES = NUM_POSITION_FEATURES + NUM_VELOCITY_FEATURES        # 62 or 32

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
    Fully-connected network with BatchNormalization for stable training.

    Architecture  :  N → 64 (BN, ReLU) → Dropout(0.3) → 32 (BN, ReLU)
                       → Dropout(0.2) → NUM_CLASSES (Softmax)

    Where N = 62 (with velocity features) or 32 (position-only).

    BatchNormalization normalises activations between layers, which:
      - Allows a higher learning rate without diverging
      - Acts as a regulariser, reducing over-fitting
      - Makes training less sensitive to weight initialisation
    """
    model = Sequential(
        [
            Dense(64, input_shape=(NUM_FEATURES,)),
            BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            Dropout(0.3),
            Dense(32),
            BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            Dropout(0.2),
            Dense(NUM_CLASSES, activation="softmax"),
        ],
        name="swipe_gesture_classifier",
    )
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    return model


def augment_data(
    X: np.ndarray,
    y: np.ndarray,
    noise_std: float = 0.02,
    n_augments: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Expand the training set by adding small amounts of Gaussian noise.

    Each sample is duplicated `n_augments` times with independent noise
    drawn from N(0, noise_std).  The augmented values are clipped to [-1, 1]
    to stay in the same range as the original normalised features.

    This technique simulates natural variation in gesture execution (slightly
    different speed, position, angle) without requiring extra recording time
    and is especially helpful when the dataset is small.

    Parameters
    ----------
    X : (n_samples, n_features) float array
    y : (n_samples,) int array of class labels
    noise_std : standard deviation of the additive Gaussian noise
    n_augments : number of noisy copies to create per original sample

    Returns
    -------
    X_aug, y_aug : augmented arrays (original + n_augments × original size)
    """
    rng = np.random.default_rng(seed=42)
    augmented_X = [X]
    augmented_y = [y]
    for _ in range(n_augments):
        noise = rng.normal(0.0, noise_std, X.shape).astype(np.float32)
        noisy = np.clip(X + noise, -1.0, 1.0)
        augmented_X.append(noisy)
        augmented_y.append(y)
    return np.vstack(augmented_X), np.concatenate(augmented_y)


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
    print(f"Feature mode: {'positions + velocities' if USE_VELOCITY_FEATURES else 'positions only'} ({NUM_FEATURES} floats)\n")

    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(
        f"\nSplit: {len(X_train)} train / {len(X_test)} test\n"
    )

    # Augment training data with small Gaussian noise to simulate natural
    # variation in gesture speed and position.  The test set is kept clean.
    X_train_aug, y_train_aug = augment_data(X_train, y_train, noise_std=0.02, n_augments=3)
    print(f"After augmentation: {len(X_train_aug)} training samples\n")

    model = build_model()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        # Halve the learning rate when validation loss plateaus for 10 epochs.
        # This lets the optimiser escape shallow local minima and fine-tune
        # the weights more precisely without overshooting.
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print("\nTraining…")
    history = model.fit(
        X_train_aug,
        y_train_aug,
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
            "\n⚠  Accuracy is below 80 %.  Consider:\n"
            "  1. Collecting more samples (aim for 500+ per class).\n"
            "  2. Balancing classes — check counts printed above.\n"
            "  3. Enabling velocity features (USE_VELOCITY_FEATURES = True)\n"
            "     in both collect_data.py and train_model.py.\n"
            "  4. Improving recording quality: perform clean, full swipes;\n"
            "     vary speed, position on screen, and lighting."
        )

    plot_history(history, HISTORY_PLOT_PATH)

    print(f"\n✓  Model saved to: {MODEL_SAVE_PATH}")
    print("\nNext step: run  python convert_model.py  to export for the browser.")


if __name__ == "__main__":
    main()
