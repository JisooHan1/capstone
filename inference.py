# inference.py

import cv2
import mediapipe as mp
import numpy as np
import torch
from collections import deque
from config import WRIST, THUMB_INDICES, INDEX_INDICES, MIDDLE_INDICES, RING_INDICES, PINKY_INDICES, GESTURE

def load_model(model_path='./model/model.pt'):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    return model

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
        v1 = joint[p2, :3] - joint[p1, :3]
        v2 = joint[p3, :3] - joint[p2, :3]
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
        angles.append(angle)

    return angles

def process_frame(frame, hands, mp_drawing, mp_hands):
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

            # Convert angles to degrees
            all_angles = np.degrees(all_angles)

            # Construct feature vector
            d = np.concatenate([joint.flatten(), all_angles])
            seq.append(d)

            # Visualize hand landmarks
            mp_drawing.draw_landmarks(frame, res, mp_hands.HAND_CONNECTIONS)

    return frame, result.multi_hand_landmarks, seq

def predict_gesture(model, seq, seq_length=40):
    if len(seq) < seq_length:
        return None, None, None

    input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
    input_tensor = torch.FloatTensor(input_data)

    y_pred = model(input_tensor)
    values, indices = torch.max(y_pred.data, dim=1, keepdim=True)
    conf = values.item()

    return conf, indices.item(), actions[indices.item()]

def display_prediction(frame, landmarks, action, conf):
    if landmarks and action and conf >= 0.8:
        x_pos = int(landmarks[0].landmark[0].x * frame.shape[1])
        y_pos = int(landmarks[0].landmark[0].y * frame.shape[0]) + 20
        cv2.putText(frame, f'{action.upper()} ({conf:.2f})',
                    org=(x_pos, y_pos),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(255, 255, 255),
                    thickness=2)
    return frame

def main():
    # Initialize
    model = load_model()
    mp_hands, mp_drawing, hands = init_mediapipe()
    cap = cv2.VideoCapture(0)

    # Initialize sequences
    seq = []
    pred_queue = deque(maxlen=5)  # Store the most recent 5 predictions

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # Process frame
        frame, landmarks, current_seq = process_frame(frame, hands, mp_drawing, mp_hands)
        if current_seq:
            seq.extend(current_seq)

            # Get prediction
            conf, idx, action = predict_gesture(model, seq)

            if conf and conf >= 0.8:
                pred_queue.append(action)

                # Process prediction queue
                if len(pred_queue) == pred_queue.maxlen:
                    pred_list = list(pred_queue)
                    most_common = max(set(pred_list), key=pred_list.count)
                    count = pred_list.count(most_common)

                    if count >= 5:
                        print(f'Gesture recognized: {most_common} (confidence: {conf:.2f})')
                        frame = display_prediction(frame, landmarks, most_common, conf)

        # Display output
        cv2.imshow('Gesture Recognition', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Define gesture classes
    actions = list(GESTURE.values())
    main()
