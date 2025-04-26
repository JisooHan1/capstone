import cv2
import mediapipe as mp
import numpy as np
import torch

# 모델 로드
model = torch.load('./model/model.pt', map_location=torch.device('cpu'))
model.eval()

gesture = {
    0 : 'Turn on Light',
    1 : 'Turn off Light',
    2 : 'Turn on Fan',
    3 : 'Turn off Fan'
}

actions = list(gesture.values())
seq_length = 30

# MediaPipe 설정
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 카메라 켜기
cap = cv2.VideoCapture(0)

seq = []
action_seq = []

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 4))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

            # 관절 벡터 계산
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3]
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3]
            v = v2 - v1
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # 관절 각도 계산
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:]))
            angle = np.degrees(angle)

            # 특징 벡터 구성
            d = np.concatenate([joint.flatten(), angle])
            seq.append(d)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            if len(seq) < seq_length:
                continue

            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
            input_tensor = torch.FloatTensor(input_data)

            y_pred = model(input_tensor)
            values, indices = torch.max(y_pred.data, dim=1, keepdim=True)
            conf = values.item()

            if conf < 0.9:
                continue

            action = actions[indices.item()]
            action_seq.append(action)

            if len(action_seq) < 3:
                continue

            # 최근 3개 예측이 같으면 확정
            this_action = '?'
            if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                this_action = action
                print(f'제스처 인식됨: {this_action}')

            x_pos = int(res.landmark[0].x * img.shape[1])
            y_pos = int(res.landmark[0].y * img.shape[0]) + 20
            cv2.putText(img, f'{this_action.upper()}', org=(x_pos, y_pos),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                        color=(255, 255, 255), thickness=2)

    # 화면 출력
    cv2.imshow('Gesture Recognition', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
