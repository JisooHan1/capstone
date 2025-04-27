import cv2
import mediapipe as mp
import numpy as np
import torch
from collections import deque
from config import WRIST, THUMB_INDICES, INDEX_INDICES, MIDDLE_INDICES, RING_INDICES, PINKY_INDICES, GESTURE

# Load model
model = torch.load('./model/model.pt', map_location=torch.device('cpu'))
model.eval()

# Define gesture classes
actions = list(GESTURE.values())
seq_length = 30

# Initialize MediaPipe hand tracking model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def calculate_finger_angles(joint, finger_indices):
    """Calculate angles for a single finger"""
    angles = []
    # Add wrist to the beginning of finger indices
    points = [WRIST] + finger_indices

    # Calculate angles between consecutive segments
    for i in range(len(points)-2):
        p1, p2, p3 = points[i:i+3]
        # Get vectors
        v1 = joint[p2, :3] - joint[p1, :3]  # Using only x,y,z (excluding visibility)
        v2 = joint[p3, :3] - joint[p2, :3]
        # Normalize vectors
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        # Calculate angle
        angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
        angles.append(angle)

    return angles

# Start camera
cap = cv2.VideoCapture(0)

# Initialize sequences
seq = []
pred_queue = deque(maxlen=5)  # Store the most recent 5 predictions

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    # Preprocess frame
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

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

            # Convert angles to degrees
            all_angles = np.degrees(all_angles)

            # Construct feature vector
            d = np.concatenate([joint.flatten(), all_angles])
            seq.append(d)

            # Visualize hand landmarks
            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            if len(seq) < seq_length:
                continue

            # Prepare input data for model
            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
            input_tensor = torch.FloatTensor(input_data)

            # Get model prediction
            y_pred = model(input_tensor)
            values, indices = torch.max(y_pred.data, dim=1, keepdim=True)
            conf = values.item()

            # Skip if confidence is low
            if conf < 0.8:
                continue

            # Add prediction to queue
            action = actions[indices.item()]
            pred_queue.append(action)

            # Select the most frequent prediction from recent predictions
            if len(pred_queue) == pred_queue.maxlen:
                pred_list = list(pred_queue)
                most_common = max(set(pred_list), key=pred_list.count)
                count = pred_list.count(most_common)

                # Only display when the same prediction appears a certain number of times
                if count >= 3:
                    print(f'Gesture recognized: {most_common} (confidence: {conf:.2f})')

                    # Display recognized action on screen
                    x_pos = int(res.landmark[0].x * img.shape[1])
                    y_pos = int(res.landmark[0].y * img.shape[0]) + 20
                    cv2.putText(img, f'{most_common.upper()} ({conf:.2f})',
                               org=(x_pos, y_pos),
                               fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                               fontScale=1,
                               color=(255, 255, 255),
                               thickness=2)

    # Display output
    cv2.imshow('Gesture Recognition', img)
    if cv2.waitKey(1) == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
