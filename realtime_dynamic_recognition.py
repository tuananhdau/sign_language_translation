import os
import cv2
import json
import numpy as np
import mediapipe as mp

from collections import deque, Counter

from tensorflow.keras.models import load_model


# =========================================================
# CONFIG
# =========================================================

MODEL_PATH = "models/dynamic_sign_lstm.keras"

LABEL_PATH = "models/dynamic_labels.json"


# Model hiện tại train bằng 30 frame
SEQUENCE_LENGTH = 30

# 1 tay:
# 21 landmarks x 3 tọa độ = 63
FEATURE_COUNT = 63


# Ngưỡng tin cậy
CONFIDENCE_THRESHOLD = 0.80


# Số prediction gần nhất dùng để xác nhận
STABLE_RESULTS = 5

# Trong 5 prediction phải có ít nhất 4 giống nhau
MIN_STABLE_COUNT = 4


# =========================================================
# PERFORMANCE CONFIG
# =========================================================

# = 1:
# MediaPipe xử lý tất cả frame
AI_FRAME_SKIP = 1


# Khi sequence đã đủ 30 frame,
# chỉ chạy LSTM mỗi 3 frame
PREDICT_EVERY = 3


# Camera
CAMERA_WIDTH = 480

CAMERA_HEIGHT = 360

CAMERA_FPS = 30


# =========================================================
# CHECK FILE
# =========================================================

if not os.path.exists(MODEL_PATH):

    print("Khong tim thay model:")
    print(MODEL_PATH)

    exit()


if not os.path.exists(LABEL_PATH):

    print("Khong tim thay labels:")
    print(LABEL_PATH)

    exit()


# =========================================================
# LOAD LABELS
# =========================================================

with open(
    LABEL_PATH,
    "r",
    encoding="utf-8"
) as file:

    label_dict = json.load(file)


print()
print("==============================")
print("LABELS")
print("==============================")


for index, label in label_dict.items():

    print(
        index,
        "->",
        label
    )


# =========================================================
# LOAD MODEL
# =========================================================

print()
print("Dang tai LSTM model...")


dynamic_model = load_model(
    MODEL_PATH,
    compile=False
)


print("Tai model thanh cong.")


# =========================================================
# CHECK MODEL INPUT SHAPE
# =========================================================

print()
print(
    "Model input shape:",
    dynamic_model.input_shape
)


expected_shape = (
    None,
    SEQUENCE_LENGTH,
    FEATURE_COUNT
)


if dynamic_model.input_shape != expected_shape:

    print()
    print("CANH BAO:")
    print(
        "Model dang co input:",
        dynamic_model.input_shape
    )

    print(
        "Code mong doi:",
        expected_shape
    )


# =========================================================
# FORMAT LABEL
# =========================================================

def format_label(label):

    mapping = {

        "XIN_CHAO":
            "Xin chao",

        "XIN_LOI":
            "Xin loi",

        "CAM_ON":
            "Cam on",

        "GIUP_DO":
            "Giup do",

        "TOI":
            "Toi",

        "BAN":
            "Ban",

        "CO":
            "Co",

        "KHONG":
            "Khong",

        "AN":
            "An",

        "UONG":
            "Uong"
    }


    if not label:

        return ""


    return mapping.get(
        label,
        label.replace(
            "_",
            " "
        ).title()
    )


# =========================================================
# MEDIAPIPE
# =========================================================

mp_hands = mp.solutions.hands

mp_drawing = (
    mp.solutions.drawing_utils
)


hands = mp_hands.Hands(

    static_image_mode=False,

    max_num_hands=1,

    model_complexity=0,

    min_detection_confidence=0.5,

    min_tracking_confidence=0.5
)


# =========================================================
# CAMERA
# =========================================================

cap = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)


cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    CAMERA_WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    CAMERA_HEIGHT
)

cap.set(
    cv2.CAP_PROP_FPS,
    CAMERA_FPS
)

cap.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)


if not cap.isOpened():

    print(
        "Khong the mo camera."
    )

    hands.close()

    exit()


# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_features(
    hand_landmarks
):

    """
    21 landmark
    x,y,z
    => 63 features

    Chuẩn hóa theo landmark 0
    """

    if hand_landmarks is None:

        return None


    landmarks = (
        hand_landmarks.landmark
    )


    if len(landmarks) != 21:

        return None


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


    features = np.array(
        features,
        dtype=np.float32
    )


    if (
        features.shape[0]
        != FEATURE_COUNT
    ):

        return None


    return features


# =========================================================
# STATE
# =========================================================

# Chuỗi 30 frame
sequence = deque(
    maxlen=SEQUENCE_LENGTH
)


# Các kết quả dự đoán gần nhất
recent_results = deque(
    maxlen=STABLE_RESULTS
)


current_action = ""

current_confidence = 0.0

confirmed_action = ""


# Câu hiện tại
sentence = []


# Sau khi nhận 1 từ
# yêu cầu bỏ tay ra
waiting_for_release = False


# Đếm frame
frame_count = 0


# Kết quả MediaPipe gần nhất
last_results = None


# =========================================================
# RESET AFTER NO HAND
# =========================================================

def reset_after_release():

    global current_action
    global current_confidence
    global waiting_for_release


    sequence.clear()

    recent_results.clear()


    current_action = ""

    current_confidence = 0.0


    waiting_for_release = False


# =========================================================
# MAIN LOOP
# =========================================================

while True:

    success, frame = cap.read()


    if not success:

        continue


    # Lật camera
    frame = cv2.flip(
        frame,
        1
    )


    frame_count += 1


    # =====================================================
    # MEDIA PIPE EVERY FRAME
    # =====================================================

    run_ai = (

        frame_count
        % AI_FRAME_SKIP

        == 0
    )


    hand_detected = False


    # =====================================================
    # MEDIAPIPE
    # =====================================================

    if run_ai:

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        rgb_frame.flags.writeable = (
            False
        )


        results = hands.process(
            rgb_frame
        )


        rgb_frame.flags.writeable = (
            True
        )


        last_results = results


        # =================================================
        # HAND DETECTED
        # =================================================

        if (
            results.multi_hand_landmarks
        ):

            hand_detected = True


            hand_landmarks = (
                results
                .multi_hand_landmarks[0]
            )


            # =============================================
            # FEATURE
            # =============================================

            features = extract_features(
                hand_landmarks
            )


            if (
                features is not None
                and
                not waiting_for_release
            ):

                sequence.append(
                    features
                )


            # =============================================
            # CHỈ PREDICT KHI:
            #
            # 1. Đủ 30 frame
            # 2. Không trong trạng thái chờ bỏ tay
            # 3. Đúng chu kỳ PREDICT_EVERY
            # =============================================

            should_predict = (

                len(sequence)
                == SEQUENCE_LENGTH

                and

                not waiting_for_release

                and

                frame_count
                % PREDICT_EVERY
                == 0
            )


            if should_predict:

                # =========================================
                # PREPARE INPUT
                # =========================================

                input_data = np.array(
                    sequence,
                    dtype=np.float32
                )


                input_data = np.expand_dims(
                    input_data,
                    axis=0
                )


                # shape:
                # (1, 30, 63)


                # =========================================
                # LSTM PREDICT
                # =========================================

                probabilities = (
                    dynamic_model.predict(
                        input_data,
                        verbose=0
                    )[0]
                )


                predicted_index = int(
                    np.argmax(
                        probabilities
                    )
                )


                confidence = float(
                    probabilities[
                        predicted_index
                    ]
                )


                predicted_label = (
                    label_dict.get(

                        str(
                            predicted_index
                        ),

                        str(
                            predicted_index
                        )
                    )
                )


                current_action = (
                    predicted_label
                )


                current_confidence = (
                    confidence
                )


                # =========================================
                # CONFIDENCE CHECK
                # =========================================

                if (
                    confidence
                    >= CONFIDENCE_THRESHOLD
                ):

                    recent_results.append(
                        predicted_label
                    )

                else:

                    recent_results.clear()


                # =========================================
                # STABLE CHECK
                # =========================================

                if (
                    len(
                        recent_results
                    )
                    == STABLE_RESULTS
                ):

                    counter = Counter(
                        recent_results
                    )


                    stable_label, count = (
                        counter
                        .most_common(1)[0]
                    )


                    if (
                        count
                        >= MIN_STABLE_COUNT
                    ):

                        confirmed_action = (
                            stable_label
                        )


                        formatted_word = (
                            format_label(
                                stable_label
                            )
                        )


                        # =================================
                        # AUTO ADD TO SENTENCE
                        # =================================

                        sentence.append(
                            formatted_word
                        )


                        print()
                        print(
                            "Nhan dien:"
                        )

                        print(
                            stable_label
                        )

                        print(
                            "Confidence:",
                            round(
                                confidence
                                * 100,
                                2
                            ),
                            "%"
                        )

                        print(
                            "Tu dong them:",
                            formatted_word
                        )


                        # =================================
                        # ANTI REPEAT
                        # =================================

                        waiting_for_release = (
                            True
                        )


                        sequence.clear()

                        recent_results.clear()


        # =================================================
        # NO HAND
        # =================================================

        else:

            hand_detected = False


            reset_after_release()


    # =====================================================
    # DRAW LANDMARK
    # =====================================================

    if (
        last_results is not None
        and
        last_results.multi_hand_landmarks
    ):

        for hand_landmarks in (
            last_results.multi_hand_landmarks
        ):

            mp_drawing.draw_landmarks(

                frame,

                hand_landmarks,

                mp_hands.HAND_CONNECTIONS
            )


    # =====================================================
    # UI VALUES
    # =====================================================

    display_action = (
        format_label(
            current_action
        )
    )


    display_confirmed = (
        format_label(
            confirmed_action
        )
    )


    # =====================================================
    # TITLE
    # =====================================================

    cv2.putText(

        frame,

        "DYNAMIC SIGN - LSTM",

        (20, 30),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (255, 255, 255),

        2
    )


    # =====================================================
    # ACTION
    # =====================================================

    cv2.putText(

        frame,

        (
            f"Action: "
            f"{display_action}"
        ),

        (20, 65),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (0, 255, 0),

        2
    )


    # =====================================================
    # CONFIDENCE
    # =====================================================

    cv2.putText(

        frame,

        (
            f"Confidence: "
            f"{current_confidence * 100:.1f}%"
        ),

        (20, 100),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (255, 255, 0),

        2
    )


    # =====================================================
    # SEQUENCE
    # =====================================================

    cv2.putText(

        frame,

        (
            f"Sequence: "
            f"{len(sequence)}"
            f"/{SEQUENCE_LENGTH}"
        ),

        (20, 135),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (255, 255, 255),

        2
    )


    # =====================================================
    # CONFIRMED
    # =====================================================

    cv2.putText(

        frame,

        (
            f"Confirmed: "
            f"{display_confirmed}"
        ),

        (20, 170),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (0, 255, 255),

        2
    )


    # =====================================================
    # STATUS
    # =====================================================

    if waiting_for_release:

        status_text = (
            "Remove hand to continue"
        )

    elif len(sequence) < SEQUENCE_LENGTH:

        status_text = (
            "Collecting frames..."
        )

    else:

        status_text = (
            "Recognizing..."
        )


    cv2.putText(

        frame,

        status_text,

        (20, 205),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.5,

        (255, 255, 255),

        1
    )


    # =====================================================
    # TEXT RESULT
    # =====================================================

    text_result = " ".join(
        sentence
    )


    # Giới hạn chữ để không tràn camera
    display_text = text_result


    if len(display_text) > 45:

        display_text = (
            "..."
            + display_text[-42:]
        )


    cv2.putText(

        frame,

        (
            f"Text: "
            f"{display_text}"
        ),

        (20, 245),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (255, 0, 255),

        2
    )


    # =====================================================
    # PERFORMANCE INFO
    # =====================================================

    cv2.putText(

        frame,

        (
            f"Camera: {CAMERA_FPS} FPS"
            f" | Predict / {PREDICT_EVERY}"
        ),

        (20, 280),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.45,

        (255, 255, 255),

        1
    )


    # =====================================================
    # HELP
    # =====================================================

    cv2.putText(

        frame,

        "C: Clear | Q: Quit",

        (20, 315),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.45,

        (255, 255, 255),

        1
    )


    # =====================================================
    # SHOW
    # =====================================================

    cv2.imshow(

        "Dynamic Sign Recognition",

        frame
    )


    # =====================================================
    # KEY
    # =====================================================

    key = (
        cv2.waitKey(1)
        & 0xFF
    )


    # =====================================================
    # QUIT
    # =====================================================

    if key == ord("q"):

        break


    # =====================================================
    # CLEAR
    # =====================================================

    elif key == ord("c"):

        sentence.clear()

        sequence.clear()

        recent_results.clear()


        confirmed_action = ""

        current_action = ""

        current_confidence = 0.0


        waiting_for_release = (
            False
        )


        print()
        print(
            "Da xoa ket qua."
        )


# =========================================================
# CLEANUP
# =========================================================

cap.release()

hands.close()

cv2.destroyAllWindows()


print()
print(
    "Da dong chuong trinh."
)