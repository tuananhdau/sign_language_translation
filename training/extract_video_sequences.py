import os
import cv2
import numpy as np
import mediapipe as mp


# =========================================
# CONFIG
# =========================================

VIDEO_DATASET_DIR = "video_dataset"

SEQUENCE_DATASET_DIR = (
    "sequence_dataset"
)

SEQUENCE_LENGTH = 30

FEATURE_COUNT = 63


# =========================================
# MEDIAPIPE
# =========================================

mp_hands = mp.solutions.hands


hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# =========================================
# FEATURE EXTRACTION
# =========================================

def extract_features(
    hand_landmarks
):

    landmarks = (
        hand_landmarks.landmark
    )


    if len(landmarks) != 21:

        return None


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


    return np.array(
        features,
        dtype=np.float32
    )


# =========================================
# READ VIDEO FRAMES
# =========================================

def read_video_frames(
    video_path
):

    cap = cv2.VideoCapture(
        video_path
    )


    frames = []


    while True:

        ret, frame = cap.read()


        if not ret:

            break


        frames.append(
            frame
        )


    cap.release()


    return frames


# =========================================
# SELECT 30 FRAMES
# =========================================

def select_frames(
    frames
):

    total_frames = len(
        frames
    )


    if (
        total_frames
        < SEQUENCE_LENGTH
    ):

        return None


    indices = np.linspace(
        0,
        total_frames - 1,
        SEQUENCE_LENGTH
    ).astype(
        int
    )


    selected = [
        frames[index]
        for index in indices
    ]


    return selected


# =========================================
# VIDEO -> SEQUENCE
# =========================================

def video_to_sequence(
    video_path
):

    frames = read_video_frames(
        video_path
    )


    if not frames:

        return None


    selected_frames = (
        select_frames(
            frames
        )
    )


    if selected_frames is None:

        print(
            "Video qua ngan:",
            video_path
        )

        return None


    sequence = []


    for frame in selected_frames:

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        results = hands.process(
            rgb_frame
        )


        # =================================
        # KHONG PHAT HIEN TAY
        # =================================

        if not (
            results.multi_hand_landmarks
        ):

            return None


        hand_landmarks = (
            results
            .multi_hand_landmarks[0]
        )


        features = extract_features(
            hand_landmarks
        )


        if (
            features is None
            or
            len(features)
            != FEATURE_COUNT
        ):

            return None


        sequence.append(
            features
        )


    sequence = np.array(
        sequence,
        dtype=np.float32
    )


    if sequence.shape != (
        SEQUENCE_LENGTH,
        FEATURE_COUNT
    ):

        return None


    return sequence


# =========================================
# PROCESS LABEL
# =========================================

def process_label(
    label
):

    input_dir = os.path.join(
        VIDEO_DATASET_DIR,
        label
    )


    output_dir = os.path.join(
        SEQUENCE_DATASET_DIR,
        label
    )


    os.makedirs(
        output_dir,
        exist_ok=True
    )


    video_files = [

        file

        for file in os.listdir(
            input_dir
        )

        if file.lower().endswith(
            (
                ".mp4",
                ".avi",
                ".mov",
                ".mkv"
            )
        )
    ]


    video_files.sort()


    success_count = 0

    failed_count = 0


    for index, filename in enumerate(
        video_files,
        start=1
    ):

        video_path = os.path.join(
            input_dir,
            filename
        )


        print()
        print(
            f"[{index}/{len(video_files)}]"
        )

        print(
            "Dang xu ly:",
            video_path
        )


        sequence = (
            video_to_sequence(
                video_path
            )
        )


        if sequence is None:

            print(
                "BO QUA:"
            )

            print(
                "Khong du landmark hoac video loi."
            )

            failed_count += 1

            continue


        success_count += 1


        output_name = (
            f"sample_"
            f"{success_count:04d}"
            f".npy"
        )


        output_path = os.path.join(
            output_dir,
            output_name
        )


        np.save(
            output_path,
            sequence
        )


        print(
            "Da luu:",
            output_path
        )

        print(
            "Shape:",
            sequence.shape
        )


    return (
        success_count,
        failed_count
    )


# =========================================
# MAIN
# =========================================

if __name__ == "__main__":

    if not os.path.exists(
        VIDEO_DATASET_DIR
    ):

        print(
            "Khong tim thay thu muc:"
        )

        print(
            VIDEO_DATASET_DIR
        )

        hands.close()

        exit()


    os.makedirs(
        SEQUENCE_DATASET_DIR,
        exist_ok=True
    )


    labels = [

        name

        for name in os.listdir(
            VIDEO_DATASET_DIR
        )

        if os.path.isdir(
            os.path.join(
                VIDEO_DATASET_DIR,
                name
            )
        )
    ]


    labels.sort()


    print()
    print(
        "=============================="
    )

    print(
        "EXTRACT VIDEO SEQUENCES"
    )

    print(
        "=============================="
    )

    print(
        "Labels:",
        labels
    )

    print()


    total_success = 0

    total_failed = 0


    for label in labels:

        print()
        print(
            "=============================="
        )

        print(
            "LABEL:",
            label
        )

        print(
            "=============================="
        )


        success, failed = (
            process_label(
                label
            )
        )


        total_success += success

        total_failed += failed


        print()
        print(
            "Thanh cong:",
            success
        )

        print(
            "Bo qua:",
            failed
        )


    hands.close()


    print()
    print(
        "=============================="
    )

    print(
        "HOAN TAT"
    )

    print(
        "=============================="
    )

    print(
        "Tong sample thanh cong:",
        total_success
    )

    print(
        "Tong sample bo qua:",
        total_failed
    )