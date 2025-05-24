# inference2_refactored.py

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
from config import WRIST, THUMB_INDICES, INDEX_INDICES, MIDDLE_INDICES, RING_INDICES, PINKY_INDICES, GESTURE
import time

# Constants
CONFIRMATION_THRESHOLD = 0.8
WINDOW_SIZE = 40
TRIGGER_GESTURE_NAME = 'Trigger'
CONFIRMATION_FRAMES = 15  # Voting for robust recognition
DELAY_AFTER_TRIGGER_SECONDS = 5

# States
STATE_WAITING_FOR_TRIGGER = 0
STATE_TRIGGER_DETECTED_DELAY = 1  # Trigger detected, waiting for delay
STATE_LISTENING_FOR_ACTION = 2

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(path='./model/model.pt'):
    model = torch.load(path, map_location=device)
    model.to(device)
    model.eval()
    return model

def init_mediapipe():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    return mp_hands, mp.solutions.drawing_utils, hands

def calculate_finger_angles(joint, finger_indices):
    angles = []
    indices = [WRIST] + finger_indices

    for i in range(len(indices) - 2):
        v1 = joint[indices[i+1], :3] - joint[indices[i], :3]
        v2 = joint[indices[i+2], :3] - joint[indices[i+1], :3]
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            angles.append(0.0)
            continue
        v1 /= norm1
        v2 /= norm2
        angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
        angles.append(angle)
    return angles

def process_frame(frame, hands, mp_drawing, mp_hands):
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    seq_data, landmarks = [], None

    if result.multi_hand_landmarks:
        for res in result.multi_hand_landmarks:
            landmarks = res
            joint = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in res.landmark])
            angles = []
            for indices in [THUMB_INDICES, INDEX_INDICES, MIDDLE_INDICES, RING_INDICES, PINKY_INDICES]:
                angles += calculate_finger_angles(joint, indices)
            features = np.concatenate([joint.flatten(), np.degrees(angles)])
            seq_data.append(features)
            mp_drawing.draw_landmarks(frame, res, mp_hands.HAND_CONNECTIONS)

    return frame, landmarks, seq_data

def predict_gesture(model, sequence, seq_length=WINDOW_SIZE):
    if len(sequence) < seq_length:
        return None, None, None

    data = np.expand_dims(np.array(sequence[-seq_length:], dtype=np.float32), axis=0)
    input_tensor = torch.FloatTensor(data).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        conf, idx = torch.max(probs, dim=1)

    return conf.item(), idx.item(), actions[idx.item()]


def display_text(frame, text, sub="", y_start=30):
    cv2.putText(frame, text, (10, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if sub:
        cv2.putText(frame, sub, (10, y_start + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    return frame

def display_action(frame, landmarks, action, conf):
    if landmarks and action and conf >= CONFIRMATION_THRESHOLD:
        try:
            wrist = landmarks.landmark[WRIST]
            x, y = int(wrist.x * frame.shape[1]), int(wrist.y * frame.shape[0]) + 80
            cv2.putText(frame, f'{action.upper()} ({conf:.2f})', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        except:
            pass
    return frame

def main():
    global actions
    model = load_model()
    mp_hands, mp_drawing, hands = init_mediapipe()
    cap = cv2.VideoCapture(0)

    sequence = deque(maxlen=WINDOW_SIZE)
    history = deque(maxlen=CONFIRMATION_FRAMES)

    state = STATE_WAITING_FOR_TRIGGER
    last_conf, current_action = 0.0, None
    status, substatus = "Waiting for Trigger Gesture...", ""
    trigger_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        frame, landmarks, features = process_frame(frame, hands, mp_drawing, mp_hands)

        if state == STATE_TRIGGER_DETECTED_DELAY:
            elapsed = time.time() - trigger_time
            remaining = DELAY_AFTER_TRIGGER_SECONDS - elapsed
            status, substatus = "Trigger Detected! Preparing...", f"Waiting for {remaining:.1f}s"
            if elapsed >= DELAY_AFTER_TRIGGER_SECONDS:
                status, substatus = "Listening for Action Gesture...", ""
                state, current_action, last_conf = STATE_LISTENING_FOR_ACTION, None, 0.0
                history.clear()

        elif features:
            sequence.append(features[0])
            if len(sequence) == WINDOW_SIZE:
                conf, idx, name = predict_gesture(model, list(sequence))
                if conf and conf >= CONFIRMATION_THRESHOLD:
                    if state != STATE_TRIGGER_DETECTED_DELAY:
                        history.append(name)
                        current_action, last_conf = name, conf
                    if state == STATE_WAITING_FOR_TRIGGER and len(history) >= CONFIRMATION_FRAMES:
                        if all(h == TRIGGER_GESTURE_NAME for h in list(history)[-CONFIRMATION_FRAMES:]):
                            print(f"Trigger '{TRIGGER_GESTURE_NAME}' detected.")
                            trigger_time, state = time.time(), STATE_TRIGGER_DETECTED_DELAY
                            history.clear()
                            current_action, last_conf = None, 0.0
                    elif state == STATE_LISTENING_FOR_ACTION and len(history) >= CONFIRMATION_FRAMES:
                        if all(h != TRIGGER_GESTURE_NAME and h == history[-1] for h in list(history)[-CONFIRMATION_FRAMES:]):
                            print(f"Gesture: {history[-1]} (Confidence: {conf:.2f})\n")
                            state, status, substatus = STATE_WAITING_FOR_TRIGGER, "Waiting for Trigger Gesture...", ""
                            history.clear()
                            current_action, last_conf = None, 0.0
                elif state != STATE_TRIGGER_DETECTED_DELAY:
                    history.clear()
                    current_action, last_conf = None, 0.0
        elif state != STATE_TRIGGER_DETECTED_DELAY:
            history.clear()
            current_action, last_conf = None, 0.0

        frame = display_text(frame, status, substatus)
        if current_action and landmarks and last_conf >= CONFIRMATION_THRESHOLD and state != STATE_TRIGGER_DETECTED_DELAY:
            frame = display_action(frame, landmarks, current_action, last_conf)

        cv2.imshow('Gesture Recognition', frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('r'):
            state, status, substatus = STATE_WAITING_FOR_TRIGGER, "Waiting for Trigger Gesture...", ""
            history.clear()
            current_action, trigger_time = None, 0

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    actions = list(GESTURE.values())
    main()
