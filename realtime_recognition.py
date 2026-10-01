import os
import cv2
import joblib
import numpy as np
import mediapipe as mp
from collections import deque, Counter

# =========================================
# CONFIG
# =========================================

MODEL_PATH = "models/sign_model.pkl"

CONFIDENCE_THRESHOLD = 0.80      # chỉ chấp nhận nếu confidence >= 80%
STABLE_FRAMES = 10               # số frame lưu lại để kiểm tra ổn định
MIN_STABLE_COUNT = 8             # phải xuất hiện ít nhất 8/10 frame
COOLDOWN_FRAMES = 20             # sau khi nhận 1 ký hiệu, chờ 20 frame mới nhận tiếp

# =========================================
# CHECK MODEL
# =========================================

if not os.path.exists(MODEL_PATH):
    print(f"Khong tim thay model: {MODEL_PATH}")
    print("Hay train model truoc bang:")
    print("python training/train_model.py")
    exit()

print("Dang tai model...")
model = joblib.load(MODEL_PATH)
print("Tai model thanh cong!")

# =========================================
# MEDIAPIPE
# =========================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =========================================
# CAMERA
# =========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Khong the mo camera")
    exit()

# =========================================
# VARIABLES
# =========================================

recent_predictions = deque(maxlen=STABLE_FRAMES)

sentence = []
last_accepted_sign = ""
cooldown_counter = 0

current_prediction = ""
current_confidence = 0.0

# map label -> text hien thi dep hon
DISPLAY_MAP = {
    "A": "A",
    "B": "B",
    "C": "C",
    "D": "D",
    "E": "E",
    "F": "F",
    "G": "G",
    "H": "H",
    "I": "I",
    "J": "J",
    "K": "K",
    "L": "L",
    "M": "M",
    "N": "N",
    "O": "O",
    "P": "P",
    "Q": "Q",
    "R": "R",
    "S": "S",
    "T": "T",
    "U": "U",
    "V": "V",
    "W": "W",
    "X": "X",
    "Y": "Y",
    "Z": "Z",
    "TOI": "Toi",
    "BAN": "Ban",
    "CO": "Co",
    "KHONG": "Khong",
    "GIUP_DO": "Giup do",
    "XIN_CHAO": "Xin chao",
    "CAM_ON": "Cam on",
    "XIN_LOI": "Xin loi",
    "AN": "An",
    "UONG": "Uong"
}

# =========================================
# MAIN LOOP
# =========================================

while True:
    ret, frame = cap.read()

    if not ret:
        print("Khong doc duoc frame")
        break

    frame = cv2.flip(frame, 1)
    display_frame = frame.copy()

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    current_prediction = ""
    current_confidence = 0.0

    if cooldown_counter > 0:
        cooldown_counter -= 1

    # =========================================
    # HAND DETECTION + FEATURE EXTRACTION
    # =========================================
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        mp_drawing.draw_landmarks(
            display_frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        landmarks = hand_landmarks.landmark

        # Lay wrist (landmark 0) lam goc
        base_x = landmarks[0].x
        base_y = landmarks[0].y
        base_z = landmarks[0].z

        features = []

        for landmark in landmarks:
            x = landmark.x - base_x
            y = landmark.y - base_y
            z = landmark.z - base_z

            features.extend([x, y, z])

        features = np.array(features).reshape(1, -1)

        # =========================================
        # PREDICT
        # =========================================
        predicted_label = model.predict(features)[0]

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            confidence = np.max(probabilities)
        else:
            confidence = 1.0

        current_prediction = predicted_label
        current_confidence = confidence

        # Luu cac du doan gan day de kiem tra on dinh
        if confidence >= CONFIDENCE_THRESHOLD:
            recent_predictions.append(predicted_label)

        # =========================================
        # STABILITY CHECK + ANTI DUPLICATE
        # =========================================
        if len(recent_predictions) == STABLE_FRAMES and cooldown_counter == 0:
            counter = Counter(recent_predictions)
            most_common_label, count = counter.most_common(1)[0]

            if count >= MIN_STABLE_COUNT:
                # chi them vao sentence neu khong bi lap lien tiep
                if most_common_label != last_accepted_sign:
                    sentence.append(most_common_label)
                    last_accepted_sign = most_common_label
                    cooldown_counter = COOLDOWN_FRAMES
                    recent_predictions.clear()
                else:
                    # neu giong ky hieu truoc do, van cooldown de tranh lap
                    cooldown_counter = COOLDOWN_FRAMES
                    recent_predictions.clear()

    else:
        # Neu khong thay tay, xoa du doan gan day de tranh nhiu
        recent_predictions.clear()

    # =========================================
    # DISPLAY TEXT
    # =========================================

    # Chuyen sentence sang dang chu dep hon
    display_sentence = " ".join(DISPLAY_MAP.get(word, word) for word in sentence)

    # Prediction
    cv2.putText(
        display_frame,
        f"Prediction: {DISPLAY_MAP.get(current_prediction, current_prediction)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # Confidence
    cv2.putText(
        display_frame,
        f"Confidence: {current_confidence * 100:.2f}%",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    # Cooldown
    cv2.putText(
        display_frame,
        f"Cooldown: {cooldown_counter}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 165, 255),
        2
    )

    # Huong dan phim
    cv2.putText(
        display_frame,
        "Press C: Clear | Press B: Backspace | Press Q: Quit",
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        2
    )

    # Sentence title
    cv2.putText(
        display_frame,
        "Sentence:",
        (20, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    # Sentence content
    cv2.putText(
        display_frame,
        display_sentence,
        (20, 230),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2
    )

    cv2.imshow("Realtime Sign Language Recognition", display_frame)

    # =========================================
    # KEYBOARD
    # =========================================
    key = cv2.waitKey(1) & 0xFF

    # q = quit
    if key == ord("q"):
        break

    # c = clear sentence
    elif key == ord("c"):
        sentence.clear()
        last_accepted_sign = ""
        recent_predictions.clear()
        cooldown_counter = 0
        print("Da xoa cau.")

    # b = xoa tu/cu ky hieu cuoi
    elif key == ord("b"):
        if len(sentence) > 0:
            sentence.pop()
            if len(sentence) > 0:
                last_accepted_sign = sentence[-1]
            else:
                last_accepted_sign = ""
            print("Da xoa ky hieu cuoi.")

# =========================================
# CLEANUP
# =========================================

cap.release()
hands.close()
cv2.destroyAllWindows()
print("Da dong chuong trinh.")