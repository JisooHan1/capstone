# data_collection.py

import cv2
import mediapipe as mp
import numpy as np
import os
from config import WRIST, THUMB_INDICES, INDEX_INDICES, MIDDLE_INDICES, RING_INDICES, PINKY_INDICES, GESTURE

def init_mediapipe():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return mp_hands, mp_drawing, hands

def calculate_finger_angles(joint, finger_indices):
    angles = []
    points = [WRIST] + finger_indices

    for i in range(len(points)-2):
        p1, p2, p3 = points[i:i+3]
        # Vectors
        v1 = joint[p2] - joint[p1]
        v2 = joint[p3] - joint[p2]
        # Normalize
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        # Angle
        angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
        angles.append(angle)

    return angles

def process_frame(frame, hands, mp_drawing, mp_hands):
    """Process a single frame and extract hand landmarks"""
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    seq = []
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

            all_angles = np.degrees(all_angles)
            angle_label = np.append(all_angles, current_gesture_cls)
            joint_angle_label = np.concatenate([joint.flatten(), angle_label])
            seq.append(joint_angle_label)

            mp_drawing.draw_landmarks(frame, res, mp_hands.HAND_CONNECTIONS)

    return frame, seq

def save_data(data, gesture_cls):
    save_dir = './data'
    os.makedirs(save_dir, exist_ok=True)

    filename = f'{save_dir}/gesture_{GESTURE[gesture_cls]}.csv'

    if os.path.exists(filename):
        print(f'Found existing file: {filename} -> Appending data')
        existing_data = np.loadtxt(filename, delimiter=',')
        if len(existing_data.shape) == 1:
            existing_data = existing_data.reshape(1, -1)
        data = np.vstack((existing_data, data[1:]))
    else:
        print(f'Creating new file: {filename}')
        data = data[1:]  # Remove dummy first row

    if len(data) > 0:
        np.savetxt(filename, data, delimiter=',')
        print(f'Saved {len(data)} samples to {filename}')
    else:
        print('No data was collected')

def main():
    mp_hands, mp_drawing, hands = init_mediapipe()
    webcam = cv2.VideoCapture(0)
    collected_data = []  # Initialize empty list for data collection

    try:
        while webcam.isOpened():
            status, frame = webcam.read()
            if not status:
                continue

            # Process frame and get hand data
            frame, seq = process_frame(frame, hands, mp_drawing, mp_hands)

            if seq:
                # Append the new data
                collected_data.extend(seq)

            # Display the current frame
            cv2.imshow('Dataset', frame)

            # Press 'q' to quit
            if cv2.waitKey(1) == ord('q'):
                break

    finally:
        # Convert collected data to numpy array before saving
        if collected_data:
            final_data = np.array(collected_data)
            save_data(final_data, current_gesture_cls)
        else:
            print('No data was collected')

        # Cleanup
        webcam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Define gesture class to collect
    current_gesture_cls = 7
    main()
