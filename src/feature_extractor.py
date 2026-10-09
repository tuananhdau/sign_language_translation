import numpy as np


# =========================================================
# CONFIG
# =========================================================

LANDMARK_COUNT = 21

COORDINATE_COUNT = 3

HAND_FEATURE_COUNT = (
    LANDMARK_COUNT
    * COORDINATE_COUNT
)

TWO_HAND_FEATURE_COUNT = (
    HAND_FEATURE_COUNT * 2
)


# =========================================================
# EXTRACT ONE HAND
# =========================================================

def extract_features(
    hand_landmarks
):

    """
    Chuyển 21 landmark của một bàn tay
    thành 63 features.

    Mỗi landmark:
        x, y, z

    Chuẩn hóa theo landmark 0 (cổ tay).

    Return:
        numpy array shape (63,)
    """

    if hand_landmarks is None:

        return np.zeros(
            HAND_FEATURE_COUNT,
            dtype=np.float32
        )


    landmarks = (
        hand_landmarks.landmark
    )


    if (
        len(landmarks)
        != LANDMARK_COUNT
    ):

        return np.zeros(
            HAND_FEATURE_COUNT,
            dtype=np.float32
        )


    # Landmark 0 = wrist
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


# =========================================================
# EXTRACT TWO HANDS
# =========================================================

def extract_two_hand_features(
    hands_data
):

    """
    Ghép feature của tay trái và tay phải.

    Format:
        63 Left
        +
        63 Right
        =
        126 features

    Nếu thiếu một tay:
        features của tay đó = 0

    Return:
        numpy array shape (126,)
    """

    left_hand = hands_data.get(
        "Left"
    )

    right_hand = hands_data.get(
        "Right"
    )


    # LEFT
    if left_hand is not None:

        left_features = (
            extract_features(
                left_hand
            )
        )

    else:

        left_features = np.zeros(
            HAND_FEATURE_COUNT,
            dtype=np.float32
        )


    # RIGHT
    if right_hand is not None:

        right_features = (
            extract_features(
                right_hand
            )
        )

    else:

        right_features = np.zeros(
            HAND_FEATURE_COUNT,
            dtype=np.float32
        )


    # Left trước, Right sau
    features = np.concatenate([
        left_features,
        right_features
    ])


    return features.astype(
        np.float32
    )


# =========================================================
# VALIDATE
# =========================================================

def is_valid_single_hand(
    features
):

    return (
        features is not None
        and
        len(features)
        == HAND_FEATURE_COUNT
    )


def is_valid_two_hands(
    features
):

    return (
        features is not None
        and
        len(features)
        == TWO_HAND_FEATURE_COUNT
    )


    