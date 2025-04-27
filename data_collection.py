# data_collection.py

import cv2
import mediapipe as mp
import numpy as np
import sys
import os

# Version check
print("Python version:", sys.version)
print("cv2 version:", cv2.__version__)
print("mediapipe version:", mp.__version__)
print("numpy version:", np.__version__)

# Define gesture classes
gesture_cls = 2
gesture = {
    0: 'Turn on Light',
    1: 'Turn off Light',
    2: 'Turn on Fan',
    3: 'Turn off Fan'
}

# Define hand landmark indices
WRIST = 0
THUMB_INDICES = [1, 2, 3, 4]
INDEX_INDICES = [5, 6, 7, 8]
MIDDLE_INDICES = [9, 10, 11, 12]
RING_INDICES = [13, 14, 15, 16]
PINKY_INDICES = [17, 18, 19, 20]

# Initialize MediaPipe hand tracking model
max_num_hands = 1
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize data array with dummy data (for shape matching)
default_array = np.array(range(100), dtype='float64')

# Initialize webcam
webcam = cv2.VideoCapture(0)

def calculate_finger_angles(joint, finger_indices):
    """Calculate angles for a single finger"""
    angles = []
    # Add wrist to the beginning of finger indices
    points = [WRIST] + finger_indices

    # Calculate two angles for each finger
    for i in range(len(points)-2):
        p1, p2, p3 = points[i:i+3]
        # Get vectors
        v1 = joint[p2] - joint[p1]
        v2 = joint[p3] - joint[p2]
        # Normalize vectors
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        # Calculate angle
        angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
        angles.append(angle)

    return angles

while webcam.isOpened():
    seq = []
    status, frame = webcam.read()
    if not status:
        continue

    # Preprocess frame
    frame = cv2.flip(frame, 1)  # Mirror image
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            # Extract joint coordinates
            joint = np.zeros((21, 4))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

            # Calculate angles for each finger
            all_angles = []
            for finger in [THUMB_INDICES, INDEX_INDICES, MIDDLE_INDICES, RING_INDICES, PINKY_INDICES]:
                angles = calculate_finger_angles(joint, finger)
                all_angles.extend(angles)

            # Convert angles to degrees and prepare for storage
            all_angles = np.degrees(all_angles)
            all_angles = np.array([all_angles], dtype=np.float32)

            # Append gesture class label to angles
            angle_label = np.append(all_angles, gesture_cls)

            # Combine joint positions, angles, and label
            joint_angle_label = np.concatenate([joint.flatten(), angle_label])
            seq.append(joint_angle_label)

            # Visualize hand landmarks
            mp_drawing.draw_landmarks(frame, res, mp_hands.HAND_CONNECTIONS)

        # Stack the new data
        data = np.array(seq)
        default_array = np.vstack((default_array, data))

    # Display the current frame
    cv2.imshow('Dataset', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Save data to CSV file
save_dir = './data'
os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist

filename = f'{save_dir}/gesture_{gesture[gesture_cls]}.csv'

# Load existing file or initialize new one
if os.path.exists(filename):
    print(f'Found existing file: {filename} -> Appending data')
    existing_data = np.loadtxt(filename, delimiter=',')
    # Remove the dummy first row if it exists
    if len(existing_data.shape) == 1:
        existing_data = existing_data.reshape(1, -1)
    default_array = np.vstack((existing_data, default_array[1:]))  # Skip the dummy first row
else:
    print(f'Creating new file: {filename}')
    # Keep only the collected data, removing the dummy first row
    default_array = default_array[1:]

# Save the data
if len(default_array) > 0:  # Only save if we have collected data
    np.savetxt(filename, default_array, delimiter=',')
    print(f'Saved {len(default_array)} samples to {filename}')
else:
    print('No data was collected')

# Cleanup
webcam.release()
cv2.destroyAllWindows()
