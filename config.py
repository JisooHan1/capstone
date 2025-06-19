# config.py

import json
import os

# Landmark indices
WRIST = 0
THUMB_INDICES = [1, 2, 3, 4]
INDEX_INDICES = [5, 6, 7, 8]
MIDDLE_INDICES = [9, 10, 11, 12]
RING_INDICES = [13, 14, 15, 16]
PINKY_INDICES = [17, 18, 19, 20]

def load_gestures(path='gestures.json'):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Gesture configuration file not found at: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        gesture_map = json.load(f)
    return {int(k): v for k, v in gesture_map.items()}

GESTURE = load_gestures()