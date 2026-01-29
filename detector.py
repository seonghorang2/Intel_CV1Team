import time
from datetime import datetime
import hashlib

import numpy as np
import cv2
import onnxruntime as ort
import mediapipe as mp
import os


from db import init_db, insert_event

LIVE_FRAME_PATH = os.path.join("data", "live.jpg")
LIVE_SAVE_EVERY_SEC = 0.3   # 0.2~0.5 권장 (3~5fps 느낌)

_last_save_time = 0.0


# =========================
# 0) 설정 (필요시 수정)
# =========================
MODEL_PATH = "model/fall_lstm.onnx"  # ✅ 네 onnx 파일명으로 수정
INPUT_NAME = "input"                 # inspect 결과: input
SEQ_LEN = 30
FEAT_DIM = 135

# 낙상 임계값/중복 방지
CONF_TH = 0.70
COOLDOWN_SEC = 10      # 이벤트 중복 저장 방지
MIN_FALL_FRAMES = 3    # 연속으로 N번 fall로 판단되면 저장(오탐 완화)

# 카메라(노트북 웹캠) 고정 정보 (app.py와 동일해야 함)
TARGET_GU = "종로구"
LAPTOP_LAT = 37.583266
LAPTOP_LON = 126.966548
SOURCE_ID = "laptop_cam_01"
CAMERA_ADDRESS = "서울특별시 종로구 옥인동 47-264(노트북 웹캠)"

def make_camera_id(address: str, lat: float, lon: float) -> str:
    s = f"{address}|{lat:.6f}|{lon:.6f}".encode("utf-8")
    h = hashlib.sha1(s).hexdigest()[:10]
    return f"CAM_{h}"

CAMERA_ID = make_camera_id(CAMERA_ADDRESS, LAPTOP_LAT, LAPTOP_LON)

# MediaPipe landmark index
NOSE = 0
LEFT_HIP = 23
RIGHT_HIP = 24

# =========================
# 1) MediaPipe Pose 준비
# =========================
mp_pose = mp.solutions.pose

def build_pose():
    # model_complexity=1 정도가 웹캠에서 무난
    return mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

# =========================
# 2) (x,y) 33개 추출 -> 66
#    + velocity 66
#    + hip center 2
#    + height 1
# =========================
def landmarks_to_xy66(landmarks) -> np.ndarray:
    """
    MediaPipe Pose landmarks -> (66,) float32
    (x,y) 33개를 [x0,y0,x1,y1,...] 형태로 펼침
    """
    xy = np.zeros((33, 2), dtype=np.float32)
    for i, lm in enumerate(landmarks):
        xy[i, 0] = float(lm.x)  # normalized 0~1
        xy[i, 1] = float(lm.y)
    return xy.reshape(-1)  # (66,)

def make_feature135(xy66_curr: np.ndarray, xy66_prev: np.ndarray | None) -> np.ndarray:
    """
    135 = pose(66) + velocity(66) + center(2) + height(1)
    - velocity = curr - prev (prev 없으면 0)
    - center = midpoint of hips (left/right hip)
    - height = |nose_y - hip_center_y|  (거리 개념)
    """
    if xy66_prev is None:
        vel66 = np.zeros((66,), dtype=np.float32)
    else:
        vel66 = (xy66_curr - xy66_prev).astype(np.float32)

    # hip center (x,y)
    # xy66 is [x0,y0,x1,y1,...] so index conversion:
    # landmark i -> x at 2*i, y at 2*i+1
    lx, ly = xy66_curr[2 * LEFT_HIP], xy66_curr[2 * LEFT_HIP + 1]
    rx, ry = xy66_curr[2 * RIGHT_HIP], xy66_curr[2 * RIGHT_HIP + 1]
    center_x = (lx + rx) / 2.0
    center_y = (ly + ry) / 2.0
    center2 = np.array([center_x, center_y], dtype=np.float32)

    nose_y = xy66_curr[2 * NOSE + 1]
    height1 = np.array([abs(float(nose_y - center_y))], dtype=np.float32)

    feat = np.concatenate([xy66_curr.astype(np.float32), vel66, center2, height1], axis=0)
    assert feat.shape[0] == FEAT_DIM, f"feature dim mismatch: {feat.shape}"
    return feat

# =========================
# 3) ONNX 출력 해석 (1,2) = [normal_prob, fall_prob]
# =========================
def decode_output(outputs) -> tuple[bool, float, float]:
    out = np.array(outputs[0], dtype=np.float32)  # shape (1,2)
    if out.ndim != 2 or out.shape != (1, 2):
        return False, 0.0, 0.0

    p0, p1 = float(out[0, 0]), float(out[0, 1])

    # 출력이 이미 확률이면 합이 1 근처일 가능성이 큼.
    # 혹시 logits면 softmax로 변환
    s = p0 + p1
    if not (0.98 <= s <= 1.02) or (p0 < 0 or p1 < 0):
        logits = out[0]
        exp = np.exp(logits - np.max(logits))
        probs = exp / (np.sum(exp) + 1e-9)
        p0, p1 = float(probs[0]), float(probs[1])

    is_fall = (p1 >= CONF_TH)
    return is_fall, p0, p1

# =========================
# 4) 메인 루프 (웹캠 → 포즈 → feature → 시퀀스 → 추론 → DB)
# =========================
def main():
    init_db()

    sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    print("[OK] ONNX loaded:", MODEL_PATH)
    print("[INFO] input name/shape:", sess.get_inputs()[0].name, sess.get_inputs()[0].shape)

    pose = build_pose()

    cap = cv2.VideoCapture(0)

    _last_save_time = 0.0   # ✅ 반드시 while 이전

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        if now - _last_save_time >= LIVE_SAVE_EVERY_SEC:
            os.makedirs("data", exist_ok=True)
            preview = cv2.resize(frame, (960, 540))
            cv2.imwrite(LIVE_FRAME_PATH, preview)
            _last_save_time = now

    
    
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다. 다른 앱(Zoom/Teams 등)이 점유 중인지 확인하세요.")

    feat_buffer: list[np.ndarray] = []
    xy66_prev: np.ndarray | None = None

    last_saved = 0.0
    fall_streak = 0

    print("[RUN] Press 'q' to quit. Buffering 30 frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        # MediaPipe는 RGB 필요
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(frame_rgb)

        status = "NO_PERSON"
        conf_fall = 0.0
        conf_norm = 0.0
        is_fall = False

        if res.pose_landmarks is not None:
            # (x,y) 33개 -> 66
            xy66_curr = landmarks_to_xy66(res.pose_landmarks.landmark)
            feat135 = make_feature135(xy66_curr, xy66_prev)
            xy66_prev = xy66_curr

            feat_buffer.append(feat135)
            if len(feat_buffer) > SEQ_LEN:
                feat_buffer.pop(0)

            if len(feat_buffer) < SEQ_LEN:
                status = f"BUFFER {len(feat_buffer)}/{SEQ_LEN}"
            else:
                x = np.stack(feat_buffer, axis=0).astype(np.float32)  # (30,135)
                x = np.expand_dims(x, axis=0)                          # (1,30,135)

                outputs = sess.run(None, {INPUT_NAME: x})
                is_fall, conf_norm, conf_fall = decode_output(outputs)

                if is_fall:
                    fall_streak += 1
                else:
                    fall_streak = 0

                status = "FALL" if is_fall else "OK"

                # DB 저장(연속 fall + 쿨다운)
                now = time.time()
                if fall_streak >= MIN_FALL_FRAMES and (now - last_saved) >= COOLDOWN_SEC:
                    insert_event(
                        lat=LAPTOP_LAT,
                        lon=LAPTOP_LON,
                        dong=TARGET_GU,
                        cctv_id=CAMERA_ID,       # ✅ app.py에서 camera_id로 필터링
                        event_type="fall",
                        confidence=float(conf_fall),
                        source_id=SOURCE_ID,
                    )
                    last_saved = now
                    fall_streak = 0
                    print(f"[DB] Saved fall event {datetime.now().strftime('%H:%M:%S')} fall_prob={conf_fall:.3f}")

        # 화면 오버레이
        color = (0, 0, 255) if status == "FALL" else (0, 255, 0) if status == "OK" else (0, 255, 255)
        text = f"{status}  fall={conf_fall:.2f} normal={conf_norm:.2f}  streak={fall_streak}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)

        cv2.imshow("detector (q to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    pose.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
