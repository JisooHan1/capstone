# Hand Gesture Recognition System

A deep learning-based system for real-time hand gesture recognition using MediaPipe and PyTorch.

## Features

- Real-time hand gesture detection and recognition
- Custom gesture dataset collection
- Deep learning model using CNN-BiGRU architecture
- Support for multiple gestures including:
  - Turn on/off Light
  - Turn on/off Fan
  - Turn on/off Music
  - Open/close Curtain

## Project Structure

```
.
├── config.py           # Configuration and constants
├── data_collection.py  # Data collection script
├── dataset.py         # Custom dataset class
├── inference.py       # Real-time inference script
├── model.py           # Model architecture
├── train.py           # Training script
├── data/             # Collected gesture data
└── model/            # Saved model weights
```

## Requirements

- Python 3.7+
- PyTorch
- OpenCV
- MediaPipe
- NumPy
- pandas
- tqdm

Install dependencies:
```bash
pip install torch opencv-python mediapipe numpy pandas tqdm
```

## Usage

### 1. Data Collection

Collect gesture data for training:

```bash
python data_collection.py
```

- Position your hand in front of the camera
- Press 'q' to stop recording
- Data will be saved in the `data/` directory

### 2. Training

Train the model on collected data:

```bash
python train.py
```

Training parameters can be modified in `TrainingConfig` class:
- Window size: 30 frames
- Batch size: 64
- Learning rate: 0.0001
- Validation ratio: 0.2

### 3. Inference

Run real-time gesture recognition:

```bash
python inference.py
```

- Shows real-time predictions with confidence scores
- Press 'q' to quit

## Model Architecture

The system uses a CNN-BiGRU architecture:
- 1D Convolutional layer for feature extraction
- Batch normalization and dropout for regularization
- Bidirectional GRU for temporal sequence learning
- Softmax classification for gesture recognition

Input features:
- Hand landmark coordinates (21 points × 4 dimensions)
- Finger joint angles
- Total feature dimension: 99
