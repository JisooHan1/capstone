# inference2.py

import cv2
import mediapipe as mp
import numpy as np
import torch
from collections import deque
from config import WRIST, THUMB_INDICES, INDEX_INDICES, MIDDLE_INDICES, RING_INDICES, PINKY_INDICES, GESTURE
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CONFIRMATION_THRESHOLD = 0.8
WINDOW_SIZE = 40

# --- State-related settings ---
STATE_WAITING_FOR_TRIGGER = 0
STATE_TRIGGER_DETECTED_DELAY = 1 # New state: delay after trigger detection
STATE_LISTENING_FOR_ACTION = 2   # State number changed

TRIGGER_GESTURE_NAME = 'Trigger'
FRAMES_TO_CONFIRM_TRIGGER = 5
FRAMES_TO_CONFIRM_ACTION = 10
DELAY_AFTER_TRIGGER_SECONDS = 3 # Delay time after trigger detection (seconds)
# --- End of state-related settings ---

# (load_model, init_mediapipe, calculate_finger_angles, process_frame, predict_gesture functions remain the same)
def load_model(model_path='./model/model.pt'):
    model = torch.load(model_path, map_location=device)
    model.to(device)
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
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            angles.append(0.0)
            continue
        v1 = v1 / norm_v1
        v2 = v2 / norm_v2
        dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)
        angle = np.arccos(dot_product)
        angles.append(angle)
    return angles

def process_frame(frame, hands, mp_drawing, mp_hands):
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    processed_seq_data = []
    landmarks_for_display = None
    if result.multi_hand_landmarks:
        for res in result.multi_hand_landmarks:
            landmarks_for_display = res
            joint = np.zeros((21, 4))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]
            all_angles = []
            for finger_indices_set in [THUMB_INDICES, INDEX_INDICES, MIDDLE_INDICES, RING_INDICES, PINKY_INDICES]:
                angles = calculate_finger_angles(joint, finger_indices_set)
                all_angles.extend(angles)
            all_angles = np.degrees(all_angles)
            d = np.concatenate([joint.flatten(), all_angles])
            processed_seq_data.append(d)
            mp_drawing.draw_landmarks(frame, res, mp_hands.HAND_CONNECTIONS)
    return frame, landmarks_for_display, processed_seq_data

def predict_gesture(model, seq_data_for_model, seq_length=WINDOW_SIZE):
    if len(seq_data_for_model) < seq_length:
        return None, None, None
    input_data_np = np.array(seq_data_for_model[-seq_length:], dtype=np.float32)
    input_data_np_expanded = np.expand_dims(input_data_np, axis=0)
    input_tensor = torch.FloatTensor(input_data_np_expanded).to(device)
    with torch.no_grad():
        y_pred = model(input_tensor)
    values, indices = torch.max(y_pred.data, dim=1, keepdim=True)
    conf = values.item()
    predicted_idx = indices.item()
    return conf, predicted_idx, actions[predicted_idx]


def display_status_on_frame(frame, status_text, sub_text=""):
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if sub_text:
        cv2.putText(frame, sub_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    return frame

def display_recognized_action_on_frame(frame, landmarks, action, conf):
    if landmarks and action and conf >= CONFIRMATION_THRESHOLD:
        try:
            wrist_landmark = landmarks.landmark[WRIST]
            x_pos = int(wrist_landmark.x * frame.shape[1])
            y_pos = int(wrist_landmark.y * frame.shape[0]) + 80 # Position adjustment
            cv2.putText(frame, f'Detected: {action.upper()} ({conf:.2f})',
                        org=(x_pos, y_pos), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.7, color=(255, 255, 0), thickness=2) # Color change
        except Exception as e:
            pass
    return frame

def main():
    model = load_model()
    mp_hands, mp_drawing, hands = init_mediapipe()
    cap = cv2.VideoCapture(0)

    accumulated_feature_sequence = []
    prediction_history = deque(maxlen=max(FRAMES_TO_CONFIRM_TRIGGER, FRAMES_TO_CONFIRM_ACTION))

    current_program_state = STATE_WAITING_FOR_TRIGGER
    last_valid_conf = 0.0
    current_display_action = None
    status_text_on_frame = "Waiting for Trigger Gesture..."
    sub_status_text = "" # For displaying delay time

    trigger_detection_time = 0 # Record trigger detection time

    print(status_text_on_frame)

    while cap.isOpened():
        ret, frame = cap.read()

        processed_frame, current_landmarks, hand_feature_vectors = process_frame(frame, hands, mp_drawing, mp_hands)

        # --- State-specific logic ---
        if current_program_state == STATE_TRIGGER_DETECTED_DELAY:
            # Delay state after trigger detection
            elapsed_time = time.time() - trigger_detection_time
            remaining_time = DELAY_AFTER_TRIGGER_SECONDS - elapsed_time
            status_text_on_frame = "Trigger Detected! Preparing..."
            sub_status_text = f"Waiting for {remaining_time:.1f}s"

            if elapsed_time >= DELAY_AFTER_TRIGGER_SECONDS:
                print("Input your action:")
                status_text_on_frame = "Listening for Action Gesture..."
                sub_status_text = ""
                current_program_state = STATE_LISTENING_FOR_ACTION
                prediction_history.clear() # Important: Initialize for next action recognition
                current_display_action = None
                last_valid_conf = 0.0
            # Gesture prediction logic can be skipped during delay (or ignored)
            # The gesture processing logic below should not be executed in this state

        elif hand_feature_vectors: # When hand is detected and not in delay state
            accumulated_feature_sequence.append(hand_feature_vectors[0])
            if len(accumulated_feature_sequence) > WINDOW_SIZE:
                accumulated_feature_sequence.pop(0)

            if len(accumulated_feature_sequence) == WINDOW_SIZE:
                conf, idx, recognized_action_name = predict_gesture(model, accumulated_feature_sequence, seq_length=WINDOW_SIZE)

                if conf and conf >= CONFIRMATION_THRESHOLD:
                    # Update prediction_history only when not in delay state
                    if current_program_state != STATE_TRIGGER_DETECTED_DELAY:
                        prediction_history.append(recognized_action_name)
                        current_display_action = recognized_action_name
                        last_valid_conf = conf

                    if current_program_state == STATE_WAITING_FOR_TRIGGER:
                        confirmation_target_frames = FRAMES_TO_CONFIRM_TRIGGER
                        if len(prediction_history) >= confirmation_target_frames:
                            recent_predictions = list(prediction_history)[-confirmation_target_frames:]
                            if len(set(recent_predictions)) == 1 and recent_predictions[0] == TRIGGER_GESTURE_NAME:
                                print(f"Trigger '{TRIGGER_GESTURE_NAME}' detected.")
                                current_program_state = STATE_TRIGGER_DETECTED_DELAY
                                trigger_detection_time = time.time() # 지연 시작 시간 기록
                                prediction_history.clear() # 지연 시작 시 history 초기화
                                current_display_action = None # 화면 정리
                                last_valid_conf = 0.0
                                # "동작을 입력하세요"는 지연 후 출력됨

                    elif current_program_state == STATE_LISTENING_FOR_ACTION:
                        confirmation_target_frames = FRAMES_TO_CONFIRM_ACTION
                        if len(prediction_history) >= confirmation_target_frames:
                            recent_predictions = list(prediction_history)[-confirmation_target_frames:]
                            if len(set(recent_predictions)) == 1 and recent_predictions[0] != TRIGGER_GESTURE_NAME:
                                confirmed_action = recent_predictions[0]
                                print(f"Gesture: {confirmed_action} (Confidence: {conf:.2f})\n")
                                status_text_on_frame = "Waiting for Trigger Gesture..."
                                sub_status_text = ""
                                current_program_state = STATE_WAITING_FOR_TRIGGER
                                prediction_history.clear()
                                current_display_action = None
                                last_valid_conf = 0.0
                elif current_program_state != STATE_TRIGGER_DETECTED_DELAY: # 확신도 낮고, 지연상태 아님
                    prediction_history.clear()
                    current_display_action = None
                    last_valid_conf = 0.0
        elif current_program_state != STATE_TRIGGER_DETECTED_DELAY: # 손 감지 안됨, 지연상태 아님
            prediction_history.clear()
            current_display_action = None
            last_valid_conf = 0.0

        # 화면 업데이트
        processed_frame_with_status = display_status_on_frame(processed_frame, status_text_on_frame, sub_status_text)
        if current_display_action and current_landmarks and last_valid_conf >= CONFIRMATION_THRESHOLD and \
           current_program_state != STATE_TRIGGER_DETECTED_DELAY : # 지연 중에는 현재 감지 제스처 표시 안함 (선택)
             processed_frame_with_status = display_recognized_action_on_frame(processed_frame_with_status, current_landmarks, current_display_action, last_valid_conf)

        cv2.imshow('Gesture Recognition', processed_frame_with_status)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('r'):
            current_program_state = STATE_WAITING_FOR_TRIGGER
            prediction_history.clear()
            status_text_on_frame = "Waiting for Trigger Gesture..."
            sub_status_text = ""
            current_display_action = None
            trigger_detection_time = 0

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    actions = list(GESTURE.values())
    main()