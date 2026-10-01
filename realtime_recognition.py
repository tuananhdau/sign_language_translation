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

CONFIDENCE_THRESHOLD = 0.80
STABLE_FRAMES = 10
MIN_STABLE_COUNT = 8

# =========================================
# CHECK MODEL
# =========================================

if not os.path.exists(MODEL_PATH):
    print("Khong tim thay model!")
    print("Hay train model truoc:")
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

text_result = ""

current_prediction = ""
current_confidence = 0.0

# Sau khi chap nhan 1 ky tu
# phai bo tay ra khoi camera moi duoc nhan ky tu tiep theo
waiting_for_hand_release = False

# =========================================
# MAIN LOOP
# =========================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Khong doc duoc camera")
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb_frame)

    current_prediction = ""
    current_confidence = 0.0

    # =====================================
    # HAND DETECTED
    # =====================================

    if results.multi_hand_landmarks:

        hand_landmarks = results.multi_hand_landmarks[0]

        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        # =================================
        # FEATURE EXTRACTION
        # =================================

        landmarks = hand_landmarks.landmark

        base_x = landmarks[0].x
        base_y = landmarks[0].y
        base_z = landmarks[0].z

        features = []

        for landmark in landmarks:

            x = landmark.x - base_x
            y = landmark.y - base_y
            z = landmark.z - base_z

            features.extend([
                x,
                y,
                z
            ])

        features = np.array(
            features
        ).reshape(1, -1)

        # =================================
        # PREDICT
        # =================================

        prediction = model.predict(features)[0]

        current_prediction = str(prediction)

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                features
            )[0]

            current_confidence = float(
                np.max(probabilities)
            )

        else:

            current_confidence = 1.0

        # =================================
        # STABILITY CHECK
        # =================================

        if (
            current_confidence >= CONFIDENCE_THRESHOLD
            and not waiting_for_hand_release
        ):

            recent_predictions.append(
                current_prediction
            )

            if len(recent_predictions) == STABLE_FRAMES:

                counter = Counter(
                    recent_predictions
                )

                label, count = counter.most_common(1)[0]

                if count >= MIN_STABLE_COUNT:

                    text_result += label

                    print(
                        f"Da them: {label}"
                    )

                    print(
                        f"Chuoi hien tai: {text_result}"
                    )

                    waiting_for_hand_release = True

                    recent_predictions.clear()

    # =====================================
    # NO HAND
    # =====================================

    else:

        recent_predictions.clear()

        # Neu da them ky tu truoc do
        # va nguoi dung bo tay ra
        # thi cho phep nhan ky tu tiep theo

        if waiting_for_hand_release:

            waiting_for_hand_release = False

            print(
                "Da reset. Co the nhap ky tu tiep theo."
            )

    # =====================================
    # DISPLAY
    # =====================================

    cv2.putText(
        frame,
        f"Prediction: {current_prediction}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Confidence: {current_confidence * 100:.2f}%",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Text:",
        (20, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        text_result,
        (20, 175),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 255),
        3
    )

    if waiting_for_hand_release:

        status = "REMOVE HAND TO CONTINUE"

        status_color = (0, 0, 255)

    else:

        status = "READY"

        status_color = (0, 255, 0)

    cv2.putText(
        frame,
        status,
        (20, 220),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        status_color,
        2
    )

    cv2.putText(
        frame,
        "C: Clear | B: Backspace | Q: Quit",
        (20, 260),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Sign Language Realtime Recognition",
        frame
    )

    # =====================================
    # KEYBOARD
    # =====================================

    key = cv2.waitKey(1) & 0xFF

    # Q = Quit
    if key == ord("q"):

        break

    # C = Clear
    elif key == ord("c"):

        text_result = ""

        recent_predictions.clear()

        waiting_for_hand_release = False

        print("Da xoa toan bo chuoi.")

    # B = Backspace
    elif key == ord("b"):

        if len(text_result) > 0:

            text_result = text_result[:-1]

            print(
                f"Chuoi sau khi xoa: {text_result}"
            )

# =========================================
# CLEANUP
# =========================================

cap.release()

hands.close()

cv2.destroyAllWindows()

print("Da dong chuong trinh.")