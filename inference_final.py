# inference_final.py

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F
import os
import time
from collections import deque
from config import WRIST, THUMB_INDICES, INDEX_INDICES, MIDDLE_INDICES, RING_INDICES, PINKY_INDICES, load_gestures
import multiprocessing
from train import main as train_main
# from firebase_utils import update_status

class Settings:
    MODEL_PATH = './model/model.pt'
    GESTURE_CONFIG_PATH = 'gestures.json'
    CONFIRMATION_THRESHOLD = 0.8
    WINDOW_SIZE = 40
    CONFIRMATION_FRAMES = 15
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    TRIGGER_GESTURE_NAME = 'Trigger_Action'
    TRIGGER_REGISTER_NAME = 'Trigger_Register'
    TRIGGER_YES = 'Yes'
    TRIGGER_NO = 'No'

    STATE_WAITING = 0
    STATE_CONFIRMING_TRIGGER = 1
    STATE_LISTENING = 3
    STATE_CONFIRMING_ACTION = 4
    STATE_AWAITING_TARGET_GESTURE = 5
    STATE_CONFIRMING_START_RECORDING = 6
    STATE_PREPARE_FOR_NEW_GESTURE = 7
    STATE_REGISTERING_NEW_GESTURE = 8
    STATE_CONFIRMING_SAVE = 9
    STATE_CONFIRMING_START_TRAINING = 10

    REGISTER_DURATION_SECONDS = 15
    DELAY_FOR_NEW_GESTURE_SECONDS = 5
    IGNORE_DURATION_SECONDS = 1.0
    REGISTER_OUTPUT_DIR = './train_data'


class GestureRecognizer:
    def __init__(self, settings):
        self.settings = settings
        self.gestures = load_gestures(self.settings.GESTURE_CONFIG_PATH)
        self.model = self._load_model()
        self.last_model_mtime = os.path.getmtime(self.settings.MODEL_PATH) if os.path.exists(self.settings.MODEL_PATH) else 0
        self.mp_hands, self.mp_drawing, self.hands = self._init_mediapipe()
        self.cap = cv2.VideoCapture(0)

        self.state = self.settings.STATE_WAITING
        self.status = "Waiting for Trigger..."
        self.substatus = ""

        self.sequence = deque(maxlen=self.settings.WINDOW_SIZE)
        self.history = deque(maxlen=self.settings.CONFIRMATION_FRAMES)

        self.current_action, self.last_conf = None, 0.0
        self.pending_action = None

        self.register_data = []
        self.register_start_time = None
        self.pending_mapping_id = None
        self.ignore_gestures_until = 0
        self.cooldown_timer_start = 0
        self.training_process = None

    def _load_model(self):
        print(f"Loading model from {self.settings.MODEL_PATH} to {self.settings.DEVICE}")
        model = torch.load(self.settings.MODEL_PATH, map_location=self.settings.DEVICE)
        model.to(self.settings.DEVICE)
        model.eval()
        return model

    def _init_mediapipe(self):
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        return mp_hands, mp.solutions.drawing_utils, hands

    def _extract_features(self, landmarks):
        joint = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks.landmark])
        angles = []
        for indices in [THUMB_INDICES, INDEX_INDICES, MIDDLE_INDICES, RING_INDICES, PINKY_INDICES]:
            angles.extend(self._calculate_finger_angles(joint, indices))
        return np.concatenate([joint.flatten(), np.degrees(angles)])

    def _calculate_finger_angles(self, joint, finger_indices):
        angles = []
        indices = [WRIST] + finger_indices
        for i in range(len(indices) - 2):
            p1, p2, p3 = joint[indices[i], :3], joint[indices[i+1], :3], joint[indices[i+2], :3]
            v1, v2 = p2 - p1, p3 - p2
            v1_norm, v2_norm = np.linalg.norm(v1), np.linalg.norm(v2)
            if v1_norm == 0 or v2_norm == 0: angles.append(0.0); continue
            v1_normalized, v2_normalized = v1 / v1_norm, v2 / v2_norm
            dot_product = np.dot(v1_normalized, v2_normalized)
            angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
            angles.append(angle)
        return angles

    def _process_hand_landmarks(self, frame):
        frame_flipped = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame_flipped, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb_frame)
        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        landmarks = result.multi_hand_landmarks[0] if result.multi_hand_landmarks else None
        if landmarks: self.mp_drawing.draw_landmarks(bgr_frame, landmarks, self.mp_hands.HAND_CONNECTIONS)
        return bgr_frame, landmarks

    def _predict(self):
        if len(self.sequence) < self.settings.WINDOW_SIZE: return None, None
        data = np.expand_dims(np.array(list(self.sequence), dtype=np.float32), axis=0)
        input_tensor = torch.FloatTensor(data).to(self.settings.DEVICE)
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = F.softmax(output, dim=1)
            conf, idx = torch.max(probs, dim=1)
        gesture_name = self.gestures.get(idx.item(), "Unknown")
        return conf.item(), gesture_name

    def _update_display(self, frame):
        cv2.putText(frame, self.status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
        if self.substatus: cv2.putText(frame, self.substatus, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if self.current_action and self.last_conf >= self.settings.CONFIRMATION_THRESHOLD:
            text = f'Action: {self.current_action.upper()}'
            cv2.putText(frame, text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        return frame

    def _reset_to_waiting_state(self, status="Waiting for Trigger...", substatus=""):
        self.state = self.settings.STATE_WAITING
        self.status, self.substatus = status, substatus
        self.history.clear(); self.sequence.clear()
        self.current_action, self.last_conf = None, 0.0
        self.pending_action = None
        self.register_data = []
        self.register_start_time = None
        self.pending_mapping_id = None

    def _is_gesture_confirmed(self, name):
        return len(self.history) >= self.settings.CONFIRMATION_FRAMES and all(h == name for h in list(self.history))

    def _start_cooldown(self):
        self.ignore_gestures_until = time.time() + self.settings.IGNORE_DURATION_SECONDS
        self.history.clear()

    def _handle_confirming_action_state(self, name):
        if not self._is_gesture_confirmed(name): return

        if name == self.settings.TRIGGER_YES:
            print(f"Action '{self.pending_action}' confirmed by user.")

            gesture_to_id = {v: k for k, v in self.gestures.items()}
            action_id = gesture_to_id.get(self.pending_action)

            if action_id is not None:
                print(f"Updating Firebase with action ID: {action_id}")
                # update_status(action_id)

            self._reset_to_waiting_state(status=f"Action '{self.pending_action}' done!")

        elif name == self.settings.TRIGGER_NO:
            self._reset_to_waiting_state(status=f"Action '{self.pending_action}' cancelled.")

    def _handle_waiting_state(self, name):
        if self._is_gesture_confirmed(name) and name in [self.settings.TRIGGER_GESTURE_NAME, self.settings.TRIGGER_REGISTER_NAME]:
            self.pending_action = name
            self.state = self.settings.STATE_CONFIRMING_TRIGGER
            self.status, self.substatus = f"Confirm: {name}?", "Show 'Yes' or 'No' gesture"
            self._start_cooldown()

    def _handle_confirming_trigger_state(self, name):
        if not self._is_gesture_confirmed(name): return
        if name == self.settings.TRIGGER_YES:
            if self.pending_action == self.settings.TRIGGER_GESTURE_NAME:
                self.state = self.settings.STATE_LISTENING
                self.status, self.substatus = "Listening for action...", ""
            elif self.pending_action == self.settings.TRIGGER_REGISTER_NAME:
                self.state = self.settings.STATE_AWAITING_TARGET_GESTURE
                self.status, self.substatus = "Map to which action?", "Show original gesture ('No' to cancel)"
            self._start_cooldown()
        elif name == self.settings.TRIGGER_NO:
            self._reset_to_waiting_state(status=f"'{self.pending_action}' cancelled")

    def _handle_listening_state(self, name):
        if self._is_gesture_confirmed(name):
            gesture_to_id = {v: k for k, v in self.gestures.items()}
            action_id = gesture_to_id.get(name)
            if action_id is not None and 0 <= action_id <= 7:
                self.pending_action = name
                self.state = self.settings.STATE_CONFIRMING_ACTION
                self.status, self.substatus = f"Confirm action: {name}?", "Show 'Yes' or 'No'"
                self._start_cooldown()

    def _handle_awaiting_target_gesture(self, name):
        if self._is_gesture_confirmed(name):
            gesture_to_id = {v: k for k, v in self.gestures.items()}
            action_id = gesture_to_id.get(name)
            if action_id is not None and 0 <= action_id <= 7:
                self.pending_mapping_id = action_id
                action_name = self.gestures[action_id]
                self.state = self.settings.STATE_CONFIRMING_START_RECORDING
                self.status, self.substatus = f"Target: '{action_name}'. Ready?", "Show 'Yes' or 'No'"
                self._start_cooldown()
            elif name == self.settings.TRIGGER_NO:
                self._reset_to_waiting_state(status="Registration cancelled.")

    def _handle_confirming_start_recording(self, name):
        if not self._is_gesture_confirmed(name): return
        if name == self.settings.TRIGGER_YES:
            self.state = self.settings.STATE_PREPARE_FOR_NEW_GESTURE
            self.status, self.substatus = "Get ready for NEW gesture!", ""
            self.cooldown_timer_start = time.time()
            self.history.clear()
        elif name == self.settings.TRIGGER_NO:
            self.state = self.settings.STATE_AWAITING_TARGET_GESTURE
            self.status, self.substatus = "Map to which action?", "Show original gesture ('No' to cancel)"
            self._start_cooldown()

    def _handle_prepare_for_new_gesture(self):
        elapsed = time.time() - self.cooldown_timer_start
        remaining = self.settings.DELAY_FOR_NEW_GESTURE_SECONDS - elapsed
        self.substatus = f"Recording starts in {remaining:.1f}s..."
        if elapsed >= self.settings.DELAY_FOR_NEW_GESTURE_SECONDS:
            self.state = self.settings.STATE_REGISTERING_NEW_GESTURE
            self.status, self.substatus = "Show the NEW gesture now!", ""
            self.register_start_time = time.time()
            self.register_data = []

    def _handle_registering_new_gesture(self, landmarks):
        elapsed = time.time() - self.register_start_time
        remaining = max(0, self.settings.REGISTER_DURATION_SECONDS - elapsed)
        action_name = self.gestures.get(self.pending_mapping_id, "Unknown")
        self.status, self.substatus = f"Recording for '{action_name}'...", f"Time left: {remaining:.1f}s"
        if landmarks:
            features = self._extract_features(landmarks)
            self.register_data.append(features)
        if elapsed >= self.settings.REGISTER_DURATION_SECONDS:
            self.state = self.settings.STATE_CONFIRMING_SAVE
            self.status, self.substatus = "Save this new gesture?", "Show 'Yes' or 'No'"
            self._start_cooldown()

    def _handle_confirming_save_state(self, name):
        if not self._is_gesture_confirmed(name): return
        if name == self.settings.TRIGGER_YES:
            action_id = self.pending_mapping_id
            features_to_save = [np.append(features, action_id) for features in self.register_data]
            action_name = self.gestures.get(action_id, "action")
            os.makedirs(self.settings.REGISTER_OUTPUT_DIR, exist_ok=True)
            filename = f"action_{action_id}_{action_name.replace(' ', '_')}_{int(time.time())}.csv"
            filepath = os.path.join(self.settings.REGISTER_OUTPUT_DIR, filename)
            np.savetxt(filepath, np.array(features_to_save), delimiter=',')
            print(f"Saved {len(features_to_save)} samples to {filepath}")
            self.state = self.settings.STATE_CONFIRMING_START_TRAINING
            self.status, self.substatus = "Start training now?", "Show 'Yes' or 'No'"
            self._start_cooldown()
        elif name == self.settings.TRIGGER_NO:
            self._reset_to_waiting_state(status="Save cancelled.")

    def _handle_confirming_start_training(self, name):
        if not self._is_gesture_confirmed(name): return
        if name == self.settings.TRIGGER_YES:
            if self.training_process is None or not self.training_process.is_alive():
                print("\nStarting background training process...")
                self.training_process = multiprocessing.Process(target=train_main, daemon=True)
                self.training_process.start()
                self._reset_to_waiting_state(status="Data saved!", substatus="Training started in background...")
            else:
                self._reset_to_waiting_state(status="Data saved!", substatus="Previous training still running...")
        elif name == self.settings.TRIGGER_NO:
            self._reset_to_waiting_state(status="Data saved!", substatus="Training deferred.")

    def main(self):
        state_handlers = {
            self.settings.STATE_WAITING: self._handle_waiting_state,
            self.settings.STATE_CONFIRMING_TRIGGER: self._handle_confirming_trigger_state,
            self.settings.STATE_LISTENING: self._handle_listening_state,
            self.settings.STATE_CONFIRMING_ACTION: self._handle_confirming_action_state,
            self.settings.STATE_AWAITING_TARGET_GESTURE: self._handle_awaiting_target_gesture,
            self.settings.STATE_CONFIRMING_START_RECORDING: self._handle_confirming_start_recording,
            self.settings.STATE_CONFIRMING_SAVE: self._handle_confirming_save_state,
            self.settings.STATE_CONFIRMING_START_TRAINING: self._handle_confirming_start_training,
        }

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret: break

            if os.path.exists(self.settings.MODEL_PATH) and os.path.getmtime(self.settings.MODEL_PATH) != self.last_model_mtime:
                print("New model version detected. Reloading...")
                try:
                    time.sleep(1)
                    self.model = self._load_model()
                    self.last_model_mtime = os.path.getmtime(self.settings.MODEL_PATH)
                    self._reset_to_waiting_state("Model Reloaded Successfully!")
                except Exception as e:
                    print(f"Failed to reload model: {e}")
                    self.last_model_mtime = os.path.getmtime(self.settings.MODEL_PATH)

            bgr_frame, landmarks = self._process_hand_landmarks(frame)

            if self.state == self.settings.STATE_PREPARE_FOR_NEW_GESTURE:
                self._handle_prepare_for_new_gesture()
            elif time.time() < self.ignore_gestures_until:
                self.history.clear()
            elif landmarks:
                if self.state == self.settings.STATE_REGISTERING_NEW_GESTURE:
                    self._handle_registering_new_gesture(landmarks)
                else:
                    features = self._extract_features(landmarks)
                    self.sequence.append(features)
                    conf, name = self._predict()
                    if conf and name and conf >= self.settings.CONFIRMATION_THRESHOLD:
                        self.history.append(name)
                        self.current_action, self.last_conf = name, conf
                        if self.state in state_handlers:
                            state_handlers[self.state](name)
                    else:
                        self.history.clear()
                        self.current_action, self.last_conf = None, 0.0
            else:
                self.history.clear()
                self.current_action, self.last_conf = None, 0.0

            final_frame = self._update_display(bgr_frame)
            cv2.imshow('Gesture Recognition', final_frame)
            key = cv2.waitKey(1)
            if key == ord('q'): break
            elif key == ord('r'): self._reset_to_waiting_state("State reset by user.")

        self.cleanup()

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    settings = Settings()
    recognizer = GestureRecognizer(settings)
    recognizer.main()