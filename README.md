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
├── test.py           # Testing script
├── train_data/       # Training dataset
├── test_data/        # Testing dataset
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

Collect gesture data for training or testing:

```bash
# For training data
python data_collection.py --data_type train --gesture <gesture_class> --fps <frames_per_second>

# For testing data
python data_collection.py --data_type test --gesture <gesture_class> --fps <frames_per_second>
```

Parameters:
- `--data_type`: Choose between 'train' or 'test'
- `--gesture`: Gesture class number (0-7)
- `--fps`: Frames per second to collect (default: 10)

Example:
```bash
# Collect training data for "Turn on Light" gesture (class 0) at 10 fps
python data_collection.py --data_type train --gesture 0 --fps 10
```

Features:
- Real-time hand landmark detection using MediaPipe
- Automatic calculation of finger joint angles
- Data saved in CSV format
- Supports appending to existing datasets
- Data saved in `train_data/` or `test_data/` directories

Usage instructions:
1. Position your hand in front of the camera
2. Perform the gesture you want to record
3. The script will collect data at the specified FPS
4. Press 'q' to stop recording
5. Data will be automatically saved to the appropriate directory

### 2. Training

Train the model on collected data:

```bash
python train.py
```

Training parameters can be modified in the training script:
- Window size: 30 frames
- Batch size: 64
- Learning rate: 0.0001
- Validation ratio: 0.2

### 3. Testing

Test the model performance:

```bash
python test.py
```

### 4. Inference

Run real-time gesture recognition:

```bash
python inference.py
```

- Shows real-time predictions with confidence scores
- Press 'q' to quit

## Model Architecture

The system uses a CNN-BiGRU architecture:
- 1D Convolutional layer for feature extraction
- Batch normalization and dropout (0.3) for regularization
- Bidirectional GRU for temporal sequence learning
- Linear layer with softmax for gesture classification

## Gesture Classes

The system recognizes 8 different gestures:
1. Turn on Light
2. Turn off Light
3. Turn on Fan
4. Turn off Fan
5. Turn on Music
6. Turn off Music
7. Curtain Open
8. Curtain Close
