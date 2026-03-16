# Training a Swipe Gesture Model — Step-by-Step Guide

This guide explains **exactly** how to teach the app to recognise hand-swipe
gestures (left, right, up, down).  No prior machine-learning experience is
required.

---

## Overview

The app already includes **rule-based swipe detection** that works out of the
box without any training — just run `yarn dev` and swipe your hand in front of
the camera.  
This guide explains how to **train an ML model** if you want more accuracy or
want to add custom gesture classes.

### Two-step process

```
Camera → collect_data.py → training_data.csv → train_model.py
      → swipe_gesture_model.hdf5 → convert_model.py
      → public/tf-models/point-history-classifier/ → React app
```

---

## Prerequisites

### 1. Python 3.9 or later

Download from https://www.python.org/downloads/  
Verify:
```bash
python --version   # should print Python 3.9.x or higher
```

### 2. A webcam

Any USB or built-in webcam works.

### 3. Install Python dependencies

Open a terminal in the `training/` folder and run:

```bash
pip install -r requirements.txt
```

> **Tip (Windows):** If `pip` is not found, try `python -m pip install -r requirements.txt`

This installs:
| Package | Purpose |
|---------|---------|
| `mediapipe` | Hand landmark detection |
| `opencv-python` | Webcam capture and display |
| `tensorflow` | Neural network framework |
| `tensorflowjs` | Convert the model for the browser |
| `scikit-learn` | Splitting data into train/test sets |
| `pandas` | Loading the CSV data file |
| `matplotlib` | Plotting training accuracy |

---

## Step 1 — Collect Training Data

> **Goal:** Record examples of each swipe gesture so the model has data to learn from.

Run the data collector:
```bash
python collect_data.py
```

A window will open showing your webcam feed.

### Keyboard controls

| Key | Action |
|-----|--------|
| `0` | Start recording **No Swipe** (hold hand still) |
| `1` | Start recording **Swipe Left** |
| `2` | Start recording **Swipe Right** |
| `3` | Start recording **Swipe Up** |
| `4` | Start recording **Swipe Down** |
| `R` | Pause recording |
| `ESC` or `Q` | Quit and save |

### How many samples to collect

| Class | Minimum | Recommended |
|-------|---------|-------------|
| 0 – No Swipe | 300 | 600 |
| 1 – Swipe Left | 150 | 300 |
| 2 – Swipe Right | 150 | 300 |
| 3 – Swipe Up | 150 | 300 |
| 4 – Swipe Down | 150 | 300 |

### Tips for good data

- **Vary speed**: some swipes slow, some fast.
- **Vary position**: move your hand at different heights and distances.
- **Vary lighting**: record in different rooms or at different times of day.
- **"No Swipe" matters**: record it with your hand in many positions so the
  model learns the difference between "nothing happening" and a real swipe.

When you quit, the data is saved to `training_data.csv`.  
You can run `collect_data.py` multiple times — new rows are **appended** to
the file.

---

## Step 2 — Train the Model

Once you have collected enough data, train the neural network:

```bash
python train_model.py
```

You will see progress like:

```
Epoch 1/200 – loss: 1.4532  accuracy: 0.3812
Epoch 2/200 – loss: 1.2011  accuracy: 0.5430
…
Epoch 47/200 – val_loss improved; saving model to swipe_gesture_model.hdf5
…
── Test results ──────────────────────
  Loss    : 0.1243
  Accuracy: 0.9540  (95.4 %)
```

When training finishes you will find:
- `swipe_gesture_model.hdf5` — the trained model
- `training_history.png` — accuracy/loss charts

### What is a good accuracy?

| Accuracy | Interpretation |
|----------|---------------|
| < 70 %   | Poor — collect more data, especially for confused classes |
| 70–85 %  | Acceptable |
| 85–95 %  | Good |
| > 95 %   | Excellent |

If accuracy is low, the most common fixes are:
1. Collect more samples.
2. Make sure samples are balanced (similar counts per class).
3. Improve recording quality (better lighting, steadier camera).

---

## Step 3 — Convert the Model for the Browser

The React app uses **TensorFlow.js** which needs a different file format than
Keras.  Convert the model:

```bash
python convert_model.py
```

This writes two files to `../public/tf-models/point-history-classifier/`:
- `model.json` — model architecture
- `group1-shard1of1.bin` — model weights

---

## Step 4 — Use the New Model in the React App

The app includes a `useSwipeDetector` hook that already handles swipe
detection using **rule-based logic** (no ML required).  

If you want to **also** use your trained ML model for improved accuracy,
follow these steps:

### 4a. Create a new hook

Copy `src/components/hands-capture/hooks/useKeyPointClassifier.ts` to
`usePointHistoryClassifier.ts` and change the model path and input:

```typescript
// usePointHistoryClassifier.ts
model.current = await tf.loadGraphModel(
  `/tf-models/point-history-classifier/model.json`
);
```

The input to this model is a **flat array of 32 floats** — the same format
produced by the `preProcessPointHistory()` function in `collect_data.py`:
- 16 frames × 2 coordinates = 32 values
- Relative to the first frame position
- Normalised to [-1, 1]

### 4b. Update constants.ts

`CONFIGS.pointHistoryClassifierLabels` already contains the correct labels:
```
['No Swipe', 'Swipe Left', 'Swipe Right', 'Swipe Up', 'Swipe Down']
```

### 4c. Test it

```bash
cd ..       # go back to the project root
yarn dev    # start the development server
```

Open http://localhost:3000 and try swiping your hand.

---

## How the Model Works (for the curious)

### What data is recorded

The script tracks your **index finger tip** (MediaPipe landmark #8) position
over 16 consecutive frames.  That gives us 16 × 2 = **32 numbers** per sample.

```
Frame  1: (0.52, 0.43)
Frame  2: (0.50, 0.43)
…
Frame 16: (0.28, 0.44)   ← finger moved left → Swipe Left
```

### Model architecture

```
Input (32)
  ↓
Dense 24 neurons + ReLU
  ↓
Dropout 30 %
  ↓
Dense 10 neurons + ReLU
  ↓
Dense 5 neurons + Softmax
  ↓
Output (5 probabilities)
```

The model is intentionally small so it runs in real-time in the browser.

### Training process

1. **Split**: 80 % of your data is used for training, 20 % for testing.
2. **Early stopping**: training stops automatically when the validation loss
   stops improving (prevents over-fitting).
3. **Best model saved**: only the checkpoint with the highest validation
   accuracy is kept.

---

## File Reference

| File | Purpose |
|------|---------|
| `collect_data.py` | Record training data from your webcam |
| `train_model.py` | Train the neural network |
| `convert_model.py` | Export the model to TensorFlow.js format |
| `requirements.txt` | Python package dependencies |
| `training_data.csv` | Your collected samples (created at runtime) |
| `swipe_gesture_model.hdf5` | Trained Keras model (created at runtime) |
| `training_history.png` | Training accuracy chart (created at runtime) |

---

## Troubleshooting

**"No module named mediapipe"**  
→ Run `pip install -r requirements.txt` again.

**"Cannot read from webcam"**  
→ Make sure no other app is using the webcam.  Try changing
  `cv.VideoCapture(0)` to `cv.VideoCapture(1)` in `collect_data.py`.

**Accuracy stays low after many epochs**  
→ Check that your CSV has a balanced number of samples per class.
  Run `python -c "import pandas as pd; df=pd.read_csv('training_data.csv', header=None); print(df[0].value_counts())"`.

**"tensorflowjs_converter not found"**  
→ Run `pip install tensorflowjs`.

**The app does not detect swipes after conversion**  
→ Open the browser's developer console (F12) and check for 404 errors on the
  model files.  Make sure `model.json` is in
  `public/tf-models/point-history-classifier/`.
