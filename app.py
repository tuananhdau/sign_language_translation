import os
import json
import threading
from collections import deque, Counter

import cv2
import joblib
import numpy as np
import mediapipe as mp

from flask import (
    Flask,
    render_template,
    Response,
    jsonify,
    request
)


from tensorflow.keras.models import load_model


# =========================================================
# FLASK
# =========================================================

app = Flask(__name__)


# =========================================================
# CONFIG
# =========================================================

STATIC_MODEL_PATH = "models/sign_model.pkl"

DYNAMIC_MODEL_PATH = (
    "models/dynamic_sign_lstm.keras"
)

DYNAMIC_LABEL_PATH = (
    "models/dynamic_labels.json"
)


# STATIC
STATIC_CONFIDENCE = 0.80
STATIC_STABLE_FRAMES = 10
STATIC_MIN_COUNT = 8


# DYNAMIC
SEQUENCE_LENGTH = 30

DYNAMIC_CONFIDENCE = 0.80

DYNAMIC_STABLE_RESULTS = 5
DYNAMIC_MIN_COUNT = 4


# =========================================================
# LOAD STATIC MODEL
# =========================================================

if not os.path.exists(
    STATIC_MODEL_PATH
):

    raise FileNotFoundError(
        STATIC_MODEL_PATH
    )


print("Dang tai Random Forest...")

static_model = joblib.load(
    STATIC_MODEL_PATH
)

print("Random Forest OK")


# =========================================================
# LOAD DYNAMIC MODEL
# =========================================================

if not os.path.exists(
    DYNAMIC_MODEL_PATH
):

    raise FileNotFoundError(
        DYNAMIC_MODEL_PATH
    )


print("Dang tai LSTM...")

dynamic_model = load_model(
    DYNAMIC_MODEL_PATH
)

print("LSTM OK")


# =========================================================
# LOAD DYNAMIC LABEL
# =========================================================

with open(
    DYNAMIC_LABEL_PATH,
    "r",
    encoding="utf-8"
) as file:

    dynamic_labels = json.load(
        file
    )


print("Dynamic labels:")

for index, label in (
    dynamic_labels.items()
):

    print(
        index,
        "->",
        label
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

camera = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)


camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)

camera.set(
    cv2.CAP_PROP_FPS,
    30
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
recognition_mode = "static"


# Kết quả hiện tại
current_prediction = ""

current_confidence = 0.0

confirmed_result = ""


# Văn bản
text_result = ""


# =========================================================
# STATIC STATE
# =========================================================

static_predictions = deque(
    maxlen=STATIC_STABLE_FRAMES
)

waiting_for_release = False


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

        "CAM_ON":
            "Cảm ơn",

        "XIN_LOI":
            "Xin lỗi",

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
        .replace("_", " ")
        .title()
    )


# =========================================================
# LANDMARK → 63 FEATURES
# =========================================================

def extract_features(
    hand_landmarks
):

    landmarks = (
        hand_landmarks.landmark
    )


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


    return features


# =========================================================
# STATIC RECOGNITION
# =========================================================

def process_static(features):

    global current_prediction
    global current_confidence
    global confirmed_result
    global waiting_for_release


    data = np.array(
        features,
        dtype=np.float32
    ).reshape(
        1,
        -1
    )


    prediction = (
        static_model.predict(
            data
        )[0]
    )


    if hasattr(
        static_model,
        "predict_proba"
    ):

        probabilities = (
            static_model.predict_proba(
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


    with lock:

        current_prediction = str(
            prediction
        )

        current_confidence = (
            confidence
        )


    # =====================================
    # STABILITY
    # =====================================

    if (
        confidence
        >= STATIC_CONFIDENCE
        and
        not waiting_for_release
    ):

        static_predictions.append(
            str(prediction)
        )


        if (
            len(static_predictions)
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


    dynamic_sequence.append(
        features
    )


    # Chưa đủ 30 frame
    if (
        len(dynamic_sequence)
        < SEQUENCE_LENGTH
    ):

        return


    input_data = np.array(
        dynamic_sequence,
        dtype=np.float32
    )


    input_data = np.expand_dims(
        input_data,
        axis=0
    )


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
        str(predicted_index)
    ]


    with lock:

        current_prediction = (
            label
        )

        current_confidence = (
            confidence
        )


    # =====================================
    # STABILITY
    # =====================================

    if (
        confidence
        >= DYNAMIC_CONFIDENCE
    ):

        dynamic_results.append(
            label
        )

    else:

        dynamic_results.clear()


    if (
        len(dynamic_results)
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

            with lock:

                confirmed_result = (
                    stable_label
                )


# =========================================================
# CAMERA STREAM
# =========================================================

def generate_frames():

    global current_prediction
    global current_confidence
    global waiting_for_release


    while True:

        success, frame = (
            camera.read()
        )


        if not success:

            continue


        frame = cv2.flip(
            frame,
            1
        )


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


        # =====================================
        # HAND
        # =====================================

        if results.multi_hand_landmarks:

            hand_landmarks = (
                results
                .multi_hand_landmarks[0]
            )


            mp_drawing.draw_landmarks(

                frame,

                hand_landmarks,

                mp_hands.HAND_CONNECTIONS

            )


            features = extract_features(
                hand_landmarks
            )


            if len(features) == 63:

                if (
                    recognition_mode
                    == "static"
                ):

                    process_static(
                        features
                    )


                elif (
                    recognition_mode
                    == "dynamic"
                ):

                    process_dynamic(
                        features
                    )


        # =====================================
        # NO HAND
        # =====================================

        else:

            static_predictions.clear()

            dynamic_sequence.clear()

            dynamic_results.clear()


            with lock:

                current_prediction = ""

                current_confidence = 0.0


            if recognition_mode == "static":

                waiting_for_release = (
                    False
                )


        # =====================================
        # CAMERA TEXT
        # =====================================

        with lock:

            prediction = (
                current_prediction
            )

            confidence = (
                current_confidence
            )


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


        cv2.putText(

            frame,

            mode_text,

            (20, 35),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            (255, 255, 255),

            2
        )


        cv2.putText(

            frame,

            (
                f"Prediction: "
                f"{display_prediction}"
            ),

            (20, 70),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.75,

            (0, 255, 0),

            2
        )


        cv2.putText(

            frame,

            (
                f"Confidence: "
                f"{confidence * 100:.1f}%"
            ),

            (20, 105),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (255, 255, 0),

            2
        )


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

                (20, 140),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.6,

                (255, 255, 255),

                2
            )


        # =====================================
        # JPEG
        # =====================================

        ret, buffer = cv2.imencode(

            ".jpg",

            frame,

            [
                int(
                    cv2.IMWRITE_JPEG_QUALITY
                ),
                75
            ]
        )


        if not ret:

            continue


        frame_bytes = (
            buffer.tobytes()
        )


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

@app.route("/video_feed")
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
    methods=["POST"]
)
def set_mode():

    global recognition_mode
    global current_prediction
    global current_confidence
    global confirmed_result
    global waiting_for_release


    data = request.get_json()


    mode = data.get(
        "mode"
    )


    if mode not in [
        "static",
        "dynamic"
    ]:

        return jsonify({
            "success": False
        })


    with lock:

        recognition_mode = mode

        current_prediction = ""

        current_confidence = 0.0

        confirmed_result = ""


    static_predictions.clear()

    dynamic_sequence.clear()

    dynamic_results.clear()

    waiting_for_release = False


    return jsonify({

        "success": True,

        "mode": mode
    })


# =========================================================
# STATUS
# =========================================================

@app.route("/status")
def status():

    with lock:

        prediction = (
            current_prediction
        )

        confirmed = (
            confirmed_result
        )


        if recognition_mode == "dynamic":

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
                )
        })


# =========================================================
# ADD RESULT
# =========================================================

@app.route(
    "/add_result",
    methods=["POST"]
)
def add_result():

    global text_result
    global confirmed_result


    with lock:

        if not confirmed_result:

            return jsonify({
                "success": False
            })


        # STATIC: ghép chữ
        if recognition_mode == "static":

            text_result += (
                confirmed_result
            )


        # DYNAMIC: ghép từ
        else:

            word = (
                format_dynamic_label(
                    confirmed_result
                )
            )


            if text_result:

                if not (
                    text_result.endswith(
                        " "
                    )
                ):

                    text_result += " "


            text_result += word


        confirmed_result = ""


    static_predictions.clear()

    dynamic_results.clear()

    dynamic_sequence.clear()


    return jsonify({
        "success": True
    })


# =========================================================
# SPACE
# =========================================================

@app.route(
    "/add_space",
    methods=["POST"]
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

            text_result += " "


    return jsonify({
        "success": True
    })


# =========================================================
# DELETE
# =========================================================

@app.route(
    "/delete_last",
    methods=["POST"]
)
def delete_last():

    global text_result


    with lock:

        if text_result:

            text_result = (
                text_result[:-1]
            )


    return jsonify({
        "success": True
    })


# =========================================================
# CLEAR
# =========================================================

@app.route(
    "/clear",
    methods=["POST"]
)
def clear():

    global text_result
    global confirmed_result


    with lock:

        text_result = ""

        confirmed_result = ""


    static_predictions.clear()

    dynamic_sequence.clear()

    dynamic_results.clear()


    return jsonify({
        "success": True
    })


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    app.run(

        host="127.0.0.1",

        port=5000,

        debug=False,

        threaded=True
    )
    