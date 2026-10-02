import os
import cv2
import json
import numpy as np
import mediapipe as mp

from collections import deque, Counter
from tensorflow.keras.models import load_model


# =========================================
# CONFIG
# =========================================

MODEL_PATH = "models/dynamic_sign_lstm.keras"
LABEL_PATH = "models/dynamic_labels.json"

SEQUENCE_LENGTH = 30
FEATURE_COUNT = 63

CONFIDENCE_THRESHOLD = 0.80

STABLE_RESULTS = 5
MIN_STABLE_COUNT = 4


# =========================================
# CHECK FILES
# =========================================

if not os.path.exists(MODEL_PATH):
    print("Khong tim thay model LSTM:")
    print(MODEL_PATH)
    exit()

if not os.path.exists(LABEL_PATH):
    print("Khong tim thay file labels:")
    print(LABEL_PATH)
    exit()


# =========================================
# LOAD MODEL
# =========================================

print("Dang tai LSTM model...")

model = load_model(MODEL_PATH)

print("Tai model thanh cong.")


# =========================================
# LOAD LABELS
# =========================================

with open(
    LABEL_PATH,
    "r",
    encoding="utf-8"
) as file:

    label_dict = json.load(file)


print("Labels:")

for index, label in label_dict.items():
    print(index, "->", label)


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

sequence = deque(
    maxlen=SEQUENCE_LENGTH
)

recent_results = deque(
    maxlen=STABLE_RESULTS
)

current_action = ""
current_confidence = 0.0

confirmed_action = ""

text_result = []


# =========================================
# MAIN LOOP
# =========================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb_frame)

    current_action = ""
    current_confidence = 0.0


    # =====================================
    # HAND DETECTED
    # =====================================

    if results.multi_hand_landmarks:

        hand_landmarks = \
            results.multi_hand_landmarks[0]

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

            features.extend([
                landmark.x - base_x,
                landmark.y - base_y,
                landmark.z - base_z
            ])


        if len(features) == FEATURE_COUNT:

            sequence.append(features)


        # =================================
        # LSTM PREDICTION
        # =================================

        if len(sequence) == SEQUENCE_LENGTH:

            input_data = np.array(
                sequence,
                dtype=np.float32
            )

            input_data = np.expand_dims(
                input_data,
                axis=0
            )

            probabilities = model.predict(
                input_data,
                verbose=0
            )[0]

            predicted_index = int(
                np.argmax(probabilities)
            )

            confidence = float(
                probabilities[predicted_index]
            )

            predicted_label = label_dict[
                str(predicted_index)
            ]

            current_action = predicted_label
            current_confidence = confidence


            # =================================
            # STABILITY CHECK
            # =================================

            if (
                confidence
                >= CONFIDENCE_THRESHOLD
            ):

                recent_results.append(
                    predicted_label
                )


                if len(recent_results) == STABLE_RESULTS:

                    counter = Counter(
                        recent_results
                    )

                    label, count = \
                        counter.most_common(1)[0]


                    if count >= MIN_STABLE_COUNT:

                        confirmed_action = label


    # =====================================
    # NO HAND
    # =====================================

    else:

        sequence.clear()
        recent_results.clear()


    # =====================================
    # DISPLAY
    # =====================================

    cv2.putText(
        frame,
        f"Action: {current_action}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Confidence: {current_confidence * 100:.2f}%",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Frames: {len(sequence)}/{SEQUENCE_LENGTH}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Confirmed: {confirmed_action}",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "A: Add action",
        (20, 210),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "C: Clear | Q: Quit",
        (20, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # =====================================
    # DISPLAY TEXT RESULT
    # =====================================

    sentence = " ".join(text_result)

    cv2.putText(
        frame,
        f"Text: {sentence}",
        (20, 290),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 255),
        2
    )


    cv2.imshow(
        "Dynamic Sign Recognition",
        frame
    )


    # =====================================
    # KEYBOARD
    # =====================================

    key = cv2.waitKey(1) & 0xFF


    # Q
    if key == ord("q"):

        break


    # A = add confirmed action
    elif key == ord("a"):

        if confirmed_action:

            text_result.append(
                confirmed_action
            )

            print(
                "Da them:",
                confirmed_action
            )

            confirmed_action = ""

            sequence.clear()

            recent_results.clear()


    # C = clear
    elif key == ord("c"):

        text_result.clear()

        confirmed_action = ""

        sequence.clear()

        recent_results.clear()

        print(
            "Da xoa ket qua."
        )


# =========================================
# CLEANUP
# =========================================

cap.release()

hands.close()

cv2.destroyAllWindows()

print("Da dong chuong trinh.")