import cv2
import mediapipe as mp
import numpy as np
import sys
import os

# 버전 확인 (디버깅용)
print("Python version:", sys.version)
print("cv2 version:", cv2.__version__)
print("mediapipe version:", mp.__version__)
print("numpy version:", np.__version__)

# ======================= 클래스 번호 설정 =======================
# 수집하려는 제스처 번호 설정 (0~3)
class_num = 3  # << 제스처 바뀔 때마다 꼭 바꿔야 함!

gesture = {
    0: 'Turn on Light',
    1: 'Turn off Light',
    2: 'Turn on Fan',
    3: 'Turn off Fan'
}

# ======================= MediaPipe 모델 설정 =======================
max_num_hands = 1

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ======================= 데이터 초기화 =======================
# 첫 줄은 더미데이터 (모양 맞추기용)
default_array = np.array(range(100), dtype='float64')

# ======================= 웹캠 설정 =======================
webcam = cv2.VideoCapture(0)

while webcam.isOpened():
    seq = []
    status, frame = webcam.read()
    if not status:
        continue

    frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 4))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

            # 각 관절 간의 벡터 계산
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :]
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :]
            v = v2 - v1
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # 벡터 간 각도 계산
            angle = np.arccos(np.einsum('nt,nt->n',
                        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18], :],
                        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19], :]))
            angle = np.degrees(angle)
            angle = np.array([angle], dtype=np.float32)

            # 각도 + 라벨 붙이기
            angle_label = np.append(angle, class_num)

            # 관절 위치 + 각도 + 라벨 합치기
            joint_angle_label = np.concatenate([joint.flatten(), angle_label])
            seq.append(joint_angle_label)

            # 손 관절 시각화
            mp_drawing.draw_landmarks(frame, res, mp_hands.HAND_CONNECTIONS)

        data = np.array(seq)
        default_array = np.vstack((default_array, data))

    # 화면에 현재 프레임 출력
    cv2.imshow('Dataset', frame)

    # q 키 누르면 종료
    if cv2.waitKey(1) == ord('q'):
        break

# ======================= CSV로 저장 =======================

# 저장할 폴더 경로 지정
save_dir = './data'
os.makedirs(save_dir, exist_ok=True)  # 폴더 없으면 생성

filename = f'{save_dir}/gesture_{gesture[class_num]}.csv'

# 기존 파일이 있으면 이어쓰기, 없으면 초기화
if os.path.exists(filename):
    print(f'🔁 기존 파일 발견: {filename} → 이어서 데이터 추가')
    default_array = np.loadtxt(filename, delimiter=',')
else:
    print(f'🆕 새 파일 생성 예정: {filename}')
    default_array = np.array(range(100), dtype='float64')  # 첫 줄은 더미

# ======================= 종료 =======================
webcam.release()
cv2.destroyAllWindows()
