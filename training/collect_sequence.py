import os
import cv2
import numpy as np
import mediapipe as mp
import sys

SEQUENCE_LENGTH = 30
NUM_SAMPLES = 100

if len(sys.argv) < 2:
    print("Vi du:")
    print("python training/collect_sequence.py XIN_CHAO")
    exit()

label = sys.argv[1].upper()

SAVE_DIR = f"sequence_dataset/{label}"

os.makedirs(SAVE_DIR, exist_ok=True)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

sample_index = 0
recording = False
sequence = []

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        mp_drawing.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        landmarks = hand.landmark

        base_x = landmarks[0].x
        base_y = landmarks[0].y
        base_z = landmarks[0].z

        features = []

        for lm in landmarks:

            features.extend([
                lm.x - base_x,
                lm.y - base_y,
                lm.z - base_z
            ])

        if recording:

            sequence.append(features)

    cv2.putText(
        frame,
        f"Label: {label}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Sample: {sample_index}/{NUM_SAMPLES}",
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
        "S: Record | Q: Quit",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Collect Dynamic Gesture",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s") and not recording:

        sequence = []

        recording = True

        print(
            f"Bat dau thu sample {sample_index + 1}"
        )

    elif key == ord("q"):

        break

    if recording and len(sequence) >= SEQUENCE_LENGTH:

        sequence = np.array(sequence[:SEQUENCE_LENGTH])

        path = os.path.join(
            SAVE_DIR,
            f"sample_{sample_index:03d}.npy"
        )

        np.save(
            path,
            sequence
        )

        print(
            f"Da luu: {path}"
        )

        sample_index += 1

        recording = False

        sequence = []

        if sample_index >= NUM_SAMPLES:

            print("Da thu du du lieu.")

            break

cap.release()

hands.close()

cv2.destroyAllWindows()