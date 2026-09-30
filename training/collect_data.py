import cv2
import mediapipe as mp
import csv
import os
import sys

# ==============================
# CONFIG
# ==============================

DATASET_PATH = "dataset/landmarks.csv"
MAX_SAMPLES = 300

# Lấy label từ command
if len(sys.argv) < 2:
    print("Vui long nhap ten ky hieu.")
    print("Vi du:")
    print("python training/collect_data.py XIN_CHAO")
    sys.exit()

label = sys.argv[1].upper()

print(f"Dang thu thap du lieu cho ky hieu: {label}")


# ==============================
# MEDIAPIPE
# ==============================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ==============================
# CAMERA
# ==============================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Khong the mo camera")
    sys.exit()


# ==============================
# CREATE DATASET FOLDER
# ==============================

os.makedirs("dataset", exist_ok=True)


# ==============================
# CSV HEADER
# ==============================

file_exists = os.path.exists(DATASET_PATH)

csv_file = open(
    DATASET_PATH,
    mode="a",
    newline="",
    encoding="utf-8"
)

writer = csv.writer(csv_file)

if not file_exists:

    header = ["label"]

    for i in range(21):

        header.extend([
            f"x{i}",
            f"y{i}",
            f"z{i}"
        ])

    writer.writerow(header)


# ==============================
# VARIABLES
# ==============================

sample_count = 0
collecting = False


# ==============================
# MAIN LOOP
# ==============================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb_frame)


    # ==============================
    # DETECT HAND
    # ==============================

    if results.multi_hand_landmarks:

        hand_landmarks = results.multi_hand_landmarks[0]

        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )


        # ==============================
        # COLLECT DATA
        # ==============================

        if collecting and sample_count < MAX_SAMPLES:

            landmarks = hand_landmarks.landmark

            # Landmark 0 = wrist
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

            row = [label] + features

            writer.writerow(row)

            sample_count += 1


    # ==============================
    # DISPLAY INFORMATION
    # ==============================

    cv2.putText(
        frame,
        f"Label: {label}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Samples: {sample_count}/{MAX_SAMPLES}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    if collecting:

        status = "COLLECTING"

    else:

        status = "PRESS S TO START"


    cv2.putText(
        frame,
        status,
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )


    cv2.imshow(
        "Collect Sign Language Dataset",
        frame
    )


    # ==============================
    # KEYBOARD
    # ==============================

    key = cv2.waitKey(1) & 0xFF

    # Start collecting
    if key == ord("s"):

        collecting = True

        print("Bat dau thu thap...")


    # Quit
    elif key == ord("q"):

        break


    # Auto stop
    if sample_count >= MAX_SAMPLES:

        print(
            f"Da thu thap du {MAX_SAMPLES} mau cho {label}"
        )

        break


# ==============================
# CLEANUP
# ==============================

csv_file.close()

cap.release()

hands.close()

cv2.destroyAllWindows()

print("Hoan thanh!")