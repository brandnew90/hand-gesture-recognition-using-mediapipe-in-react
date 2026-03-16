"""
convert_model.py — Export a Keras Model to TensorFlow.js Graph Format
======================================================================

After training with train_model.py, run this script to convert
`swipe_gesture_model.hdf5` into the TensorFlow.js format expected by
the React app.

Usage:
    python convert_model.py

The converted files are written to:
    ../public/tf-models/point-history-classifier/
        model.json
        group1-shard1of1.bin

After conversion, the React app will automatically load the new model
(see usePointHistoryClassifier.ts).
"""

import os
import subprocess
import sys

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_HDF5 = "swipe_gesture_model.hdf5"

# Path is relative to this script's location (training/).
# The output goes into the public assets folder of the React app.
OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "public",
    "tf-models",
    "point-history-classifier",
)
# ─────────────────────────────────────────────────────────────────────────────


def check_dependencies() -> bool:
    try:
        result = subprocess.run(
            ["tensorflowjs_converter", "--version"],
            capture_output=True,
            text=True,
        )
        print(f"tensorflowjs_converter version: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        print(
            "Error: tensorflowjs_converter not found.\n"
            "Install it with:\n"
            "    pip install tensorflowjs"
        )
        return False


def main() -> None:
    print("=== Keras → TensorFlow.js Model Converter ===\n")

    if not os.path.exists(MODEL_HDF5):
        print(
            f"Error: '{MODEL_HDF5}' not found in the current directory.\n"
            "Run  python train_model.py  first."
        )
        sys.exit(1)

    if not check_dependencies():
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cmd = [
        "tensorflowjs_converter",
        "--input_format", "keras",
        "--output_format", "tfjs_graph_model",
        MODEL_HDF5,
        OUTPUT_DIR,
    ]

    print(f"Input  : {MODEL_HDF5}")
    print(f"Output : {OUTPUT_DIR}")
    print(f"\nRunning: {' '.join(cmd)}\n")

    try:
        subprocess.run(cmd, check=True)
        print("\n✓  Conversion successful!")
        print(f"\nModel files written to:\n  {os.path.abspath(OUTPUT_DIR)}")

        print(
            "\n── What to do next ──────────────────────────────────────────\n"
            "1. The React app now has the new model at:\n"
            "       public/tf-models/point-history-classifier/model.json\n\n"
            "2. Open  src/components/hands-capture/hooks/useSwipeDetector.ts\n"
            "   The app already uses rule-based swipe detection by default.\n"
            "   If you want to switch to the trained ML model, see\n"
            "   the usePointHistoryClassifier hook (create it following\n"
            "   useKeyPointClassifier.ts as a template, pointing to the\n"
            "   new model path).\n\n"
            "3. Update  constants.ts  if you changed the gesture class labels.\n\n"
            "4. Run  yarn dev  and test your gestures in the browser."
        )
    except subprocess.CalledProcessError as exc:
        print(f"\nConversion failed (exit code {exc.returncode}).")
        print("Check the error messages above and make sure the model file is valid.")
        sys.exit(1)


if __name__ == "__main__":
    main()
