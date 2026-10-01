import cv2
import mediapipe as mp
import csv
import os
import sys
import pandas as pd

# =========================================
# CONFIG
# =========================================

DATASET_PATH = "dataset/landmarks.csv"
MAX_SAMPLES = 300

# =========================================
# GET LABEL
# =========================================

if len(sys.argv) < 2:
    print("Vui long nhap ten ky hieu.")
    print("Vi du:")
    print("python training/collect_data.py A")
    sys.exit()

label = sys.argv[1].upper()

print(f"Label hien tai: {label}")

# =========================================
# CREATE DATASET FOLDER
# =========================================

os.makedirs("dataset", exist_ok=True)

# =========================================
# FUNCTION: DELETE ALL DATA OF LABEL
# =========================================

def delete_label_data(target_label):
    if not os.path.exists(DATASET_PATH):
        print("Dataset chua ton tai.")
        return

    df = pd.read_csv(DATASET_PATH)

    if df.empty:
        print("Dataset dang rong.")
        return

    before = len(df)

    df = df[df["label"] != target_label]

    after = len(df)

    deleted = before - after

    df.to_csv(
        DATASET_PATH,
        index=False
    )

    print(
        f"Da xoa {deleted} mau cua label {target_label}"
    )


# =========================================
# FUNCTION: DELETE LAST ROW
# =========================================

def delete_last_sample(target_label):
    if not os.path.exists(DATASET_PATH):
        return False

    df = pd.read_csv(DATASET_PATH)

    if df.empty:
        return False

    indexes = df.index[
        df["label"] == target_label
    ].tolist()

    if len(indexes) == 0:
        return False

    last_index = indexes[-1]

    df = df.drop(last_index)

    df.to_csv(
        DATASET_PATH,
        index=False
    )

    return True


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
    sys.exit()

# =========================================
# CREATE CSV HEADER
# =========================================

if not os.path.exists(DATASET_PATH):

    with open(
        DATASET_PATH,
        mode="w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.writer(csv_file)

        header = ["label"]

        for i in range(21):
            header.extend([
                f"x{i}",
                f"y{i}",
                f"z{i}"
            ])

        writer.writerow(header)

# =========================================
# VARIABLES
# =========================================

sample_count = 0
collecting = False

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

    # =====================================
    # HAND DETECTION
    # =====================================

    if results.multi_hand_landmarks:

        hand_landmarks = results.multi_hand_landmarks[0]

        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        # =================================
        # COLLECT
        # =================================

        if collecting and sample_count < MAX_SAMPLES:

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

            row = [label] + features

            # Append 1 sample
            with open(
                DATASET_PATH,
                mode="a",
                newline="",
                encoding="utf-8"
            ) as csv_file:

                writer = csv.writer(csv_file)

                writer.writerow(row)

            sample_count += 1

    # =====================================
    # DISPLAY
    # =====================================

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

    cv2.putText(
        frame,
        "S: Start/Pause",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "U: Undo last sample",
        (20, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "R: Delete ALL current label",
        (20, 220),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2
    )

    cv2.putText(
        frame,
        "Q: Quit",
        (20, 250),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Collect Sign Language Dataset",
        frame
    )

    # =====================================
    # KEYBOARD
    # =====================================

    key = cv2.waitKey(1) & 0xFF

    # S = Start / Pause
    if key == ord("s"):

        collecting = not collecting

        if collecting:
            print("Bat dau thu thap...")
        else:
            print("Tam dung thu thap.")

    # U = Undo
    elif key == ord("u"):

        collecting = False

        if sample_count > 0:

            success = delete_last_sample(label)

            if success:
                sample_count -= 1
                print(
                    f"Da xoa sample gan nhat. "
                    f"Con {sample_count} sample trong phien."
                )
        else:
            print("Khong co sample nao trong phien de xoa.")

    # R = Delete current label
    elif key == ord("r"):

        collecting = False

        print(
            f"Ban sap xoa TOAN BO du lieu cua label {label}"
        )

        confirm = input(
            "Nhap YES de xac nhan: "
        )

        if confirm.upper() == "YES":

            delete_label_data(label)

            sample_count = 0

            print(
                f"Da reset label {label}. "
                "Ban co the thu lai."
            )

        else:
            print("Da huy thao tac xoa.")

    # Q = Quit
    elif key == ord("q"):

        break

    # =====================================
    # AUTO STOP
    # =====================================

    if sample_count >= MAX_SAMPLES:

        collecting = False

        print(
            f"Da thu du {MAX_SAMPLES} mau cho {label}"
        )

# =========================================
# CLEANUP
# =========================================

cap.release()

hands.close()

cv2.destroyAllWindows()

print("Da dong chuong trinh.")