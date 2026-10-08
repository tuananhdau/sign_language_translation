import os
import json
import threading

from collections import deque, Counter

import cv2
import joblib
import numpy as np

from flask import (
    Flask,
    render_template,
    Response,
    jsonify,
    request
)

from tensorflow.keras.models import (
    load_model,
    Sequential
)

from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dropout,
    Dense
)

from src.hand_detector import HandDetector

from src.feature_extractor import (
    extract_features
)


# =========================================================
# FLASK
# =========================================================

app = Flask(__name__)


# =========================================================
# CONFIG
# =========================================================

STATIC_MODEL_PATH = (
    "models/sign_model.pkl"
)

DYNAMIC_MODEL_PATH = (
    "models/dynamic_sign_lstm.keras"
)

DYNAMIC_LABEL_PATH = (
    "models/dynamic_labels.json"
)


# =========================================================
# STATIC CONFIG
# =========================================================

STATIC_CONFIDENCE = 0.80

STATIC_STABLE_FRAMES = 10

STATIC_MIN_COUNT = 8


# =========================================================
# DYNAMIC CONFIG
# =========================================================

SEQUENCE_LENGTH = 30

DYNAMIC_CONFIDENCE = 0.80

DYNAMIC_STABLE_RESULTS = 5

DYNAMIC_MIN_COUNT = 4


# =========================================================
# CAMERA CONFIG
# =========================================================

CAMERA_WIDTH = 480

CAMERA_HEIGHT = 360

CAMERA_FPS = 30

JPEG_QUALITY = 65


# Chạy MediaPipe + AI mỗi 2 frame
AI_FRAME_SKIP = 2


# =========================================================
# CHECK FILES
# =========================================================

if not os.path.exists(
    STATIC_MODEL_PATH
):

    raise FileNotFoundError(
        f"Khong tim thay: {STATIC_MODEL_PATH}"
    )


if not os.path.exists(
    DYNAMIC_MODEL_PATH
):

    raise FileNotFoundError(
        f"Khong tim thay: {DYNAMIC_MODEL_PATH}"
    )


if not os.path.exists(
    DYNAMIC_LABEL_PATH
):

    raise FileNotFoundError(
        f"Khong tim thay: {DYNAMIC_LABEL_PATH}"
    )


# =========================================================
# LOAD STATIC MODEL
# =========================================================

print(
    "Dang tai Random Forest..."
)

static_model = joblib.load(
    STATIC_MODEL_PATH
)

print(
    "Random Forest OK"
)


# =========================================================
# LOAD DYNAMIC LABELS
# =========================================================

with open(
    DYNAMIC_LABEL_PATH,
    "r",
    encoding="utf-8"
) as file:

    dynamic_labels = json.load(
        file
    )


print(
    "Dynamic labels:"
)

for index, label in (
    dynamic_labels.items()
):

    print(
        index,
        "->",
        label
    )


# =========================================================
# LOAD DYNAMIC MODEL
# =========================================================

def create_dynamic_model():

    return Sequential([

        Input(
            shape=(
                SEQUENCE_LENGTH,
                63
            )
        ),

        LSTM(
            128,
            return_sequences=True
        ),

        Dropout(
            0.3
        ),

        LSTM(
            64
        ),

        Dropout(
            0.3
        ),

        Dense(
            64,
            activation="relu"
        ),

        Dropout(
            0.2
        ),

        Dense(
            len(dynamic_labels),
            activation="softmax"
        )
    ])


print(
    "Dang tai LSTM..."
)


try:

    # Thử load model bình thường
    dynamic_model = load_model(
        DYNAMIC_MODEL_PATH,
        compile=False
    )

    print(
        "LSTM load_model OK"
    )


except Exception as error:

    print(
        "Khong load duoc config model cu."
    )

    print(
        "Chuyen sang load weights..."
    )

    print(
        "Chi tiet:",
        error
    )


    dynamic_model = (
        create_dynamic_model()
    )


    dynamic_model.load_weights(
        DYNAMIC_MODEL_PATH
    )


    print(
        "LSTM load_weights OK"
    )


# =========================================================
# HAND DETECTOR
# =========================================================

hand_detector = HandDetector(
    max_num_hands=2,
    detection_confidence=0.5,
    tracking_confidence=0.5
)


# =========================================================
# CAMERA
# =========================================================

camera = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)


camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    CAMERA_WIDTH
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    CAMERA_HEIGHT
)

camera.set(
    cv2.CAP_PROP_FPS,
    CAMERA_FPS
)

camera.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)


if not camera.isOpened():

    raise RuntimeError(
        "Khong the mo camera"
    )


# =========================================================
# GLOBAL STATE
# =========================================================

lock = threading.Lock()


# Chế độ mặc định
recognition_mode = (
    "static"
)


# Prediction hiện tại
current_prediction = ""

current_confidence = 0.0


# Kết quả vừa xác nhận
confirmed_result = ""


# Câu kết quả
text_result = ""


# Sau khi nhận thành công
# phải bỏ tay ra rồi mới nhận tiếp
waiting_for_release = False


# =========================================================
# STATIC STATE
# =========================================================

static_predictions = deque(
    maxlen=STATIC_STABLE_FRAMES
)


# =========================================================
# DYNAMIC STATE
# =========================================================

dynamic_sequence = deque(
    maxlen=SEQUENCE_LENGTH
)


dynamic_results = deque(
    maxlen=DYNAMIC_STABLE_RESULTS
)


# =========================================================
# FORMAT DYNAMIC LABEL
# =========================================================

def format_dynamic_label(label):

    mapping = {

        "XIN_CHAO":
            "Xin chào",

        "XIN_LOI":
            "Xin lỗi",

        "CAM_ON":
            "Cảm ơn",

        "GIUP_DO":
            "Giúp đỡ",

        "TOI":
            "Tôi",

        "BAN":
            "Bạn",

        "CO":
            "Có",

        "KHONG":
            "Không",

        "AN":
            "Ăn",

        "UONG":
            "Uống"
    }


    if not label:

        return ""


    return mapping.get(

        label,

        label
        .replace(
            "_",
            " "
        )
        .title()
    )


# =========================================================
# STATIC RECOGNITION
# =========================================================

def process_static(features):

    global current_prediction
    global current_confidence
    global confirmed_result
    global waiting_for_release
    global text_result


    # =====================================================
    # INPUT
    # =====================================================

    data = np.array(
        features,
        dtype=np.float32
    ).reshape(
        1,
        -1
    )


    # =====================================================
    # PREDICTION
    # =====================================================

    prediction = (
        static_model.predict(
            data
        )[0]
    )


    # =====================================================
    # CONFIDENCE
    # =====================================================

    if hasattr(
        static_model,
        "predict_proba"
    ):

        probabilities = (
            static_model
            .predict_proba(
                data
            )[0]
        )


        confidence = float(
            np.max(
                probabilities
            )
        )

    else:

        confidence = 1.0


    # =====================================================
    # CURRENT RESULT
    # =====================================================

    with lock:

        current_prediction = str(
            prediction
        )

        current_confidence = (
            confidence
        )


    # =====================================================
    # ĐÃ NHẬN RỒI -> CHỜ BỎ TAY
    # =====================================================

    if waiting_for_release:

        return


    # =====================================================
    # CONFIDENCE CHECK
    # =====================================================

    if (
        confidence
        >= STATIC_CONFIDENCE
    ):

        static_predictions.append(
            str(
                prediction
            )
        )

    else:

        static_predictions.clear()

        return


    # =====================================================
    # STABILITY
    # =====================================================

    if (
        len(
            static_predictions
        )
        == STATIC_STABLE_FRAMES
    ):

        counter = Counter(
            static_predictions
        )


        label, count = (
            counter
            .most_common(1)[0]
        )


        if (
            count
            >= STATIC_MIN_COUNT
        ):

            with lock:

                confirmed_result = (
                    label
                )


                # =========================================
                # TỰ ĐỘNG THÊM CHỮ
                # =========================================

                text_result += (
                    label
                )


            print(
                "Tu dong them chu:",
                label
            )


            # Chặn việc thêm AAAAA...
            waiting_for_release = (
                True
            )


            static_predictions.clear()


# =========================================================
# DYNAMIC RECOGNITION
# =========================================================

def process_dynamic(features):

    global current_prediction
    global current_confidence
    global confirmed_result
    global waiting_for_release
    global text_result


    # =====================================================
    # ĐÃ NHẬN -> CHỜ BỎ TAY
    # =====================================================

    if waiting_for_release:

        return


    # =====================================================
    # ADD FRAME
    # =====================================================

    dynamic_sequence.append(
        features
    )


    # Chưa đủ 30 frame
    if (
        len(
            dynamic_sequence
        )
        < SEQUENCE_LENGTH
    ):

        return


    # =====================================================
    # PREPARE LSTM INPUT
    # =====================================================

    input_data = np.array(
        dynamic_sequence,
        dtype=np.float32
    )


    input_data = np.expand_dims(
        input_data,
        axis=0
    )


    # =====================================================
    # PREDICTION
    # =====================================================

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


    label = dynamic_labels[
        str(
            predicted_index
        )
    ]


    # =====================================================
    # CURRENT RESULT
    # =====================================================

    with lock:

        current_prediction = (
            label
        )

        current_confidence = (
            confidence
        )


    # =====================================================
    # CONFIDENCE
    # =====================================================

    if (
        confidence
        >= DYNAMIC_CONFIDENCE
    ):

        dynamic_results.append(
            label
        )

    else:

        dynamic_results.clear()

        return


    # =====================================================
    # STABILITY
    # =====================================================

    if (
        len(
            dynamic_results
        )
        == DYNAMIC_STABLE_RESULTS
    ):

        counter = Counter(
            dynamic_results
        )


        stable_label, count = (
            counter
            .most_common(1)[0]
        )


        if (
            count
            >= DYNAMIC_MIN_COUNT
        ):

            word = (
                format_dynamic_label(
                    stable_label
                )
            )


            with lock:

                confirmed_result = (
                    stable_label
                )


                # =========================================
                # TỰ ĐỘNG THÊM TỪ
                # =========================================

                if text_result:

                    if not (
                        text_result
                        .endswith(
                            " "
                        )
                    ):

                        text_result += (
                            " "
                        )


                text_result += (
                    word
                )


            print(
                "Tu dong them tu:",
                word
            )


            waiting_for_release = (
                True
            )


            dynamic_sequence.clear()

            dynamic_results.clear()


# =========================================================
# CAMERA STREAM
# =========================================================

def generate_frames():

    global current_prediction
    global current_confidence
    global waiting_for_release


    # =====================================================
    # PERFORMANCE STATE
    # =====================================================

    frame_count = 0


    # Kết quả MediaPipe gần nhất
    last_results = None


    left_hand = None

    right_hand = None


    while True:

        # =================================================
        # READ CAMERA
        # =================================================

        success, frame = (
            camera.read()
        )


        if not success:

            continue


        frame = cv2.flip(
            frame,
            1
        )


        frame_count += 1


        # =================================================
        # CHỈ CHẠY AI MỖI 2 FRAME
        # =================================================

        run_ai = (

            frame_count
            % AI_FRAME_SKIP

            == 0
        )


        # =================================================
        # AI
        # =================================================

        if run_ai:

            # =============================================
            # MEDIAPIPE
            # =============================================

            last_results = (
                hand_detector.detect(
                    frame
                )
            )


            # =============================================
            # LEFT / RIGHT
            # =============================================

            hands_data = (
                hand_detector.get_hands(
                    last_results
                )
            )


            left_hand = (
                hands_data[
                    "Left"
                ]
            )


            right_hand = (
                hands_data[
                    "Right"
                ]
            )


            # =============================================
            # CHECK HAND
            # =============================================

            has_hand = (

                left_hand
                is not None

                or

                right_hand
                is not None
            )


            # =============================================
            # HAND DETECTED
            # =============================================

            if has_hand:

                # =================================================
                # MODEL HIỆN TẠI VẪN LÀ 1 TAY / 63 FEATURES
                # =================================================

                # Ưu tiên Right
                # Không có thì dùng Left

                hand_landmarks = (

                    right_hand

                    if (
                        right_hand
                        is not None
                    )

                    else

                    left_hand
                )


                features = (
                    extract_features(
                        hand_landmarks
                    )
                )


                if (
                    len(features)
                    == 63
                ):

                    # =========================================
                    # STATIC
                    # =========================================

                    if (
                        recognition_mode
                        == "static"
                    ):

                        process_static(
                            features
                        )


                    # =========================================
                    # DYNAMIC
                    # =========================================

                    elif (
                        recognition_mode
                        == "dynamic"
                    ):

                        process_dynamic(
                            features
                        )


            # =============================================
            # NO HAND
            # =============================================

            else:

                static_predictions.clear()

                dynamic_sequence.clear()

                dynamic_results.clear()


                with lock:

                    current_prediction = ""

                    current_confidence = (
                        0.0
                    )


                # =========================================
                # BỎ TAY RA -> CHO PHÉP NHẬN TIẾP
                # =========================================

                waiting_for_release = (
                    False
                )


        # =================================================
        # DRAW LAST LANDMARKS
        # =================================================

        if (
            last_results
            is not None
        ):

            hand_detector.draw_hands(
                frame,
                last_results
            )


        # =================================================
        # DISPLAY DATA
        # =================================================

        with lock:

            prediction = (
                current_prediction
            )

            confidence = (
                current_confidence
            )


        # =================================================
        # MODE
        # =================================================

        if (
            recognition_mode
            == "dynamic"
        ):

            display_prediction = (
                format_dynamic_label(
                    prediction
                )
            )


            mode_text = (
                "DYNAMIC - LSTM"
            )

        else:

            display_prediction = (
                prediction
            )


            mode_text = (
                "STATIC - RANDOM FOREST"
            )


        # =================================================
        # HAND COUNT
        # =================================================

        hand_count = 0


        if (
            left_hand
            is not None
        ):

            hand_count += 1


        if (
            right_hand
            is not None
        ):

            hand_count += 1


        # =================================================
        # DRAW TEXT
        # =================================================

        cv2.putText(
            frame,
            mode_text,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            (
                f"Prediction: "
                f"{display_prediction}"
            ),
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            (
                f"Confidence: "
                f"{confidence * 100:.1f}%"
            ),
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2
        )


        cv2.putText(
            frame,
            (
                f"Hands: "
                f"{hand_count}/2"
            ),
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


        # =================================================
        # DYNAMIC SEQUENCE
        # =================================================

        if (
            recognition_mode
            == "dynamic"
        ):

            cv2.putText(
                frame,
                (
                    f"Sequence: "
                    f"{len(dynamic_sequence)}"
                    f"/{SEQUENCE_LENGTH}"
                ),
                (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )


        # =================================================
        # JPEG
        # =================================================

        ret, buffer = (
            cv2.imencode(

                ".jpg",

                frame,

                [
                    int(
                        cv2.IMWRITE_JPEG_QUALITY
                    ),
                    JPEG_QUALITY
                ]
            )
        )


        if not ret:

            continue


        frame_bytes = (
            buffer.tobytes()
        )


        # =================================================
        # STREAM
        # =================================================

        yield (

            b"--frame\r\n"

            b"Content-Type: image/jpeg\r\n\r\n"

            + frame_bytes

            + b"\r\n"
        )


# =========================================================
# HOME
# =========================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# =========================================================
# VIDEO
# =========================================================

@app.route(
    "/video_feed"
)
def video_feed():

    return Response(

        generate_frames(),

        mimetype=(
            "multipart/x-mixed-replace;"
            " boundary=frame"
        )
    )


# =========================================================
# SET MODE
# =========================================================

@app.route(
    "/set_mode",
    methods=[
        "POST"
    ]
)
def set_mode():

    global recognition_mode
    global current_prediction
    global current_confidence
    global confirmed_result
    global waiting_for_release


    data = request.get_json(
        silent=True
    )


    if not data:

        return jsonify({

            "success":
                False
        })


    mode = data.get(
        "mode"
    )


    if mode not in [
        "static",
        "dynamic"
    ]:

        return jsonify({

            "success":
                False
        })


    with lock:

        recognition_mode = (
            mode
        )

        current_prediction = ""

        current_confidence = 0.0

        confirmed_result = ""


    static_predictions.clear()

    dynamic_sequence.clear()

    dynamic_results.clear()


    waiting_for_release = (
        False
    )


    return jsonify({

        "success":
            True,

        "mode":
            mode
    })


# =========================================================
# STATUS
# =========================================================

@app.route(
    "/status"
)
def status():

    with lock:

        prediction = (
            current_prediction
        )

        confirmed = (
            confirmed_result
        )


        # =========================================
        # DYNAMIC DISPLAY
        # =========================================

        if (
            recognition_mode
            == "dynamic"
        ):

            prediction = (
                format_dynamic_label(
                    prediction
                )
            )

            confirmed = (
                format_dynamic_label(
                    confirmed
                )
            )


        return jsonify({

            "mode":
                recognition_mode,

            "prediction":
                prediction,

            "confidence":
                round(
                    current_confidence
                    * 100,
                    2
                ),

            "confirmed":
                confirmed,

            "text":
                text_result,

            "sequence":
                len(
                    dynamic_sequence
                ),

            "waiting":
                waiting_for_release
        })


# =========================================================
# SPACE
# =========================================================

@app.route(
    "/add_space",
    methods=[
        "POST"
    ]
)
def add_space():

    global text_result


    with lock:

        if (
            text_result

            and

            not text_result.endswith(
                " "
            )
        ):

            text_result += (
                " "
            )


    return jsonify({

        "success":
            True
    })


# =========================================================
# DELETE LAST
# =========================================================

@app.route(
    "/delete_last",
    methods=[
        "POST"
    ]
)
def delete_last():

    global text_result


    with lock:

        if text_result:

            text_result = (
                text_result[:-1]
            )


    return jsonify({

        "success":
            True
    })


# =========================================================
# CLEAR
# =========================================================

@app.route(
    "/clear",
    methods=[
        "POST"
    ]
)
def clear():

    global text_result
    global confirmed_result
    global current_prediction
    global current_confidence
    global waiting_for_release


    with lock:

        text_result = ""

        confirmed_result = ""

        current_prediction = ""

        current_confidence = (
            0.0
        )


    static_predictions.clear()

    dynamic_sequence.clear()

    dynamic_results.clear()


    waiting_for_release = (
        False
    )


    return jsonify({

        "success":
            True
    })


# =========================================================
# CLEANUP
# =========================================================

def cleanup():

    try:

        camera.release()

    except Exception:

        pass


    try:

        hand_detector.close()

    except Exception:

        pass


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    try:

        app.run(

            host="127.0.0.1",

            port=5000,

            debug=False,

            threaded=True
        )

    finally:

        cleanup()