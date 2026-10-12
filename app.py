import cv2
import threading

from collections import deque, Counter

from flask import (
    Flask,
    render_template,
    Response,
    jsonify,
    request
)


from src.hand_detector import HandDetector

from src.feature_extractor import (
    extract_features
)

from src.predictor import Predictor

from src.sentence_builder import (
    SentenceBuilder
)


# =========================================================
# FLASK
# =========================================================

app = Flask(__name__)


# =========================================================
# MODEL PATH
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


# Random Forest không nhất thiết
# predict tất cả 30 frame / giây
STATIC_PREDICT_EVERY = 2


# =========================================================
# DYNAMIC CONFIG
# =========================================================

# Model train từ video:
# 30 frame x 63 features
SEQUENCE_LENGTH = 30

FEATURE_COUNT = 63


DYNAMIC_CONFIDENCE = 0.80

DYNAMIC_STABLE_RESULTS = 5

DYNAMIC_MIN_COUNT = 4


# =========================================================
# PERFORMANCE
# =========================================================

# MediaPipe chạy mỗi frame
# để gom 30 frame nhanh
AI_FRAME_SKIP = 1


# Sau khi đủ 30 frame:
# LSTM predict mỗi 3 frame
PREDICT_EVERY = 3


# =========================================================
# CAMERA
# =========================================================

CAMERA_WIDTH = 480

CAMERA_HEIGHT = 360

CAMERA_FPS = 30

JPEG_QUALITY = 65


# =========================================================
# LOAD MODELS
# =========================================================

print()
print(
    "=============================="
)

print(
    "LOAD AI MODELS"
)

print(
    "=============================="
)


predictor = Predictor(

    static_model_path=(
        STATIC_MODEL_PATH
    ),

    dynamic_model_path=(
        DYNAMIC_MODEL_PATH
    ),

    dynamic_label_path=(
        DYNAMIC_LABEL_PATH
    ),

    sequence_length=(
        SEQUENCE_LENGTH
    ),

    dynamic_feature_count=(
        FEATURE_COUNT
    )
)


print(
    "Da tai model."
)


# =========================================================
# SENTENCE BUILDER
# =========================================================

sentence_builder = (
    SentenceBuilder()
)


# =========================================================
# MEDIAPIPE HAND DETECTOR
# =========================================================

hand_detector = HandDetector(

    max_num_hands=2,

    detection_confidence=0.5,

    tracking_confidence=0.5,

    model_complexity=0
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
        "Khong the mo camera."
    )


# =========================================================
# THREAD LOCK
# =========================================================

state_lock = (
    threading.Lock()
)


# =========================================================
# GLOBAL STATE
# =========================================================

recognition_mode = (
    "static"
)


current_prediction = ""

current_confidence = 0.0

confirmed_result = ""


# Sau khi nhận thành công
# phải bỏ tay ra mới nhận tiếp
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

        label.replace(
            "_",
            " "
        ).title()
    )


# =========================================================
# RESET RECOGNITION
# =========================================================

def reset_recognition_state():

    global current_prediction
    global current_confidence
    global waiting_for_release


    static_predictions.clear()

    dynamic_sequence.clear()

    dynamic_results.clear()


    with state_lock:

        current_prediction = ""

        current_confidence = (
            0.0
        )


    waiting_for_release = (
        False
    )


# =========================================================
# STATIC PROCESS
# =========================================================

def process_static(features):

    global current_prediction
    global current_confidence
    global confirmed_result
    global waiting_for_release


    # =====================================================
    # ĐÃ NHẬN -> CHỜ BỎ TAY
    # =====================================================

    if waiting_for_release:

        return


    # =====================================================
    # RANDOM FOREST
    # =====================================================

    label, confidence = (
        predictor.predict_static(
            features
        )
    )


    with state_lock:

        current_prediction = label

        current_confidence = (
            confidence
        )


    # =====================================================
    # CONFIDENCE
    # =====================================================

    if (
        confidence
        < STATIC_CONFIDENCE
    ):

        static_predictions.clear()

        return


    static_predictions.append(
        label
    )


    # =====================================================
    # STABILITY
    # =====================================================

    if (
        len(static_predictions)
        < STATIC_STABLE_FRAMES
    ):

        return


    counter = Counter(
        static_predictions
    )


    stable_label, count = (
        counter
        .most_common(1)[0]
    )


    if (
        count
        < STATIC_MIN_COUNT
    ):

        return


    # =====================================================
    # CONFIRMED
    # =====================================================

    with state_lock:

        confirmed_result = (
            stable_label
        )


    # =====================================================
    # AUTO ADD CHARACTER
    # =====================================================

    sentence_builder.add_character(
        stable_label
    )


    print(
        "[STATIC] Them:",
        stable_label
    )


    # =====================================================
    # ANTI REPEAT
    # =====================================================

    waiting_for_release = (
        True
    )


    static_predictions.clear()


# =========================================================
# DYNAMIC PROCESS
# =========================================================

def process_dynamic(
    features,
    frame_count
):

    global current_prediction
    global current_confidence
    global confirmed_result
    global waiting_for_release


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


    # =====================================================
    # CHƯA ĐỦ 30 FRAME
    # =====================================================

    if (
        len(dynamic_sequence)
        < SEQUENCE_LENGTH
    ):

        return


    # =====================================================
    # CHỈ PREDICT MỖI 3 FRAME
    # =====================================================

    if (
        frame_count
        % PREDICT_EVERY
        != 0
    ):

        return


    # =====================================================
    # LSTM
    # =====================================================

    label, confidence = (
        predictor.predict_dynamic(
            dynamic_sequence
        )
    )


    if not label:

        return


    with state_lock:

        current_prediction = label

        current_confidence = (
            confidence
        )


    # =====================================================
    # CONFIDENCE CHECK
    # =====================================================

    if (
        confidence
        < DYNAMIC_CONFIDENCE
    ):

        dynamic_results.clear()

        return


    dynamic_results.append(
        label
    )


    # =====================================================
    # STABILITY
    # =====================================================

    if (
        len(dynamic_results)
        < DYNAMIC_STABLE_RESULTS
    ):

        return


    counter = Counter(
        dynamic_results
    )


    stable_label, count = (
        counter
        .most_common(1)[0]
    )


    if (
        count
        < DYNAMIC_MIN_COUNT
    ):

        return


    # =====================================================
    # CONFIRMED
    # =====================================================

    with state_lock:

        confirmed_result = (
            stable_label
        )


    word = format_dynamic_label(
        stable_label
    )


    # =====================================================
    # AUTO ADD WORD
    # =====================================================

    sentence_builder.add_word(
        word
    )


    print()
    print(
        "[DYNAMIC] Nhan dien:",
        stable_label
    )

    print(
        "[DYNAMIC] Them:",
        word
    )

    print(
        "[DYNAMIC] Confidence:",
        round(
            confidence * 100,
            2
        ),
        "%"
    )


    # =====================================================
    # ANTI REPEAT
    # =====================================================

    waiting_for_release = (
        True
    )


    dynamic_sequence.clear()

    dynamic_results.clear()


# =========================================================
# GENERATE CAMERA FRAMES
# =========================================================

def generate_frames():

    global current_prediction
    global current_confidence
    global waiting_for_release


    frame_count = 0

    last_results = None


    while True:

        # =================================================
        # CAMERA READ
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
        # AI FRAME
        # =================================================

        run_ai = (

            frame_count
            % AI_FRAME_SKIP

            == 0
        )


        hand_count = 0


        # =================================================
        # MEDIAPIPE
        # =================================================

        if run_ai:

            last_results = (
                hand_detector.detect(
                    frame
                )
            )


            hands_data = (
                hand_detector.get_hands(
                    last_results
                )
            )


            left_hand = (
                hands_data.get(
                    "Left"
                )
            )


            right_hand = (
                hands_data.get(
                    "Right"
                )
            )


            if left_hand is not None:

                hand_count += 1


            if right_hand is not None:

                hand_count += 1


            # =================================================
            # CÓ TAY
            # =================================================

            if hand_count > 0:

                # =================================================
                # MODEL HIỆN TẠI = 63 FEATURES
                #
                # Nên chỉ đưa một bàn tay vào model.
                # Ưu tiên Right.
                # =================================================

                if right_hand is not None:

                    selected_hand = (
                        right_hand
                    )

                else:

                    selected_hand = (
                        left_hand
                    )


                features = extract_features(
                    selected_hand
                )


                # =============================================
                # CHECK FEATURE
                # =============================================

                if (
                    features is not None

                    and

                    len(features)
                    == FEATURE_COUNT
                ):

                    # =========================================
                    # STATIC
                    # =========================================

                    if (
                        recognition_mode
                        == "static"
                    ):

                        if (
                            frame_count
                            % STATIC_PREDICT_EVERY
                            == 0
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

                            features,

                            frame_count
                        )


            # =================================================
            # NO HAND
            # =================================================

            else:

                static_predictions.clear()

                dynamic_sequence.clear()

                dynamic_results.clear()


                with state_lock:

                    current_prediction = ""

                    current_confidence = (
                        0.0
                    )


                # Sau khi bỏ tay ra
                # cho phép nhận ký hiệu tiếp
                waiting_for_release = (
                    False
                )


        # =================================================
        # DRAW HAND LANDMARKS
        # =================================================

        if last_results is not None:

            hand_detector.draw_hands(

                frame,

                last_results
            )


        # =================================================
        # GET CURRENT STATE
        # =================================================

        with state_lock:

            prediction = (
                current_prediction
            )

            confidence = (
                current_confidence
            )


        # =================================================
        # MODE DISPLAY
        # =================================================

        if (
            recognition_mode
            == "dynamic"
        ):

            mode_text = (
                "DYNAMIC - VIDEO LSTM"
            )


            display_prediction = (
                format_dynamic_label(
                    prediction
                )
            )

        else:

            mode_text = (
                "STATIC - RANDOM FOREST"
            )


            display_prediction = (
                prediction
            )


        # =================================================
        # TITLE
        # =================================================

        cv2.putText(

            frame,

            mode_text,

            (15, 25),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (255, 255, 255),

            2
        )


        # =================================================
        # PREDICTION
        # =================================================

        cv2.putText(

            frame,

            (
                f"Predict: "
                f"{display_prediction}"
            ),

            (15, 55),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.60,

            (0, 255, 0),

            2
        )


        # =================================================
        # CONFIDENCE
        # =================================================

        cv2.putText(

            frame,

            (
                f"Confidence: "
                f"{confidence * 100:.1f}%"
            ),

            (15, 85),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.50,

            (255, 255, 0),

            2
        )


        # =================================================
        # HAND COUNT
        # =================================================

        cv2.putText(

            frame,

            (
                f"Hands: "
                f"{hand_count}/2"
            ),

            (15, 115),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.50,

            (255, 255, 255),

            2
        )


        # =================================================
        # DYNAMIC INFO
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

                (15, 145),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.50,

                (255, 255, 255),

                2
            )


        # =================================================
        # WAITING STATUS
        # =================================================

        if waiting_for_release:

            cv2.putText(

                frame,

                "Remove hand to continue",

                (15, 175),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.50,

                (0, 255, 255),

                2
            )


        # =================================================
        # JPEG
        # =================================================

        success_encode, buffer = (
            cv2.imencode(

                ".jpg",

                frame,

                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    JPEG_QUALITY
                ]
            )
        )


        if not success_encode:

            continue


        frame_bytes = (
            buffer.tobytes()
        )


        # =================================================
        # MJPEG STREAM
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
# VIDEO STREAM
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
# STATUS API
# =========================================================

@app.route(
    "/status"
)
def status():

    with state_lock:

        prediction = (
            current_prediction
        )

        confidence = (
            current_confidence
        )

        confirmed = (
            confirmed_result
        )


    # =====================================================
    # FORMAT DYNAMIC
    # =====================================================

    if (
        recognition_mode
        == "dynamic"
    ):

        display_prediction = (
            format_dynamic_label(
                prediction
            )
        )


        display_confirmed = (
            format_dynamic_label(
                confirmed
            )
        )

    else:

        display_prediction = (
            prediction
        )

        display_confirmed = (
            confirmed
        )


    return jsonify({

        "mode":
            recognition_mode,

        "prediction":
            display_prediction,

        "confidence":
            round(
                confidence * 100,
                2
            ),

        "confirmed":
            display_confirmed,

        "text":
            sentence_builder.get_text(),

        "sequence":
            len(
                dynamic_sequence
            ),

        "sequence_length":
            SEQUENCE_LENGTH,

        "waiting":
            waiting_for_release
    })


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
                False,

            "message":
                "Khong co du lieu."
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
                False,

            "message":
                "Mode khong hop le."
        })


    with state_lock:

        recognition_mode = mode

        current_prediction = ""

        current_confidence = (
            0.0
        )

        confirmed_result = ""


    static_predictions.clear()

    dynamic_sequence.clear()

    dynamic_results.clear()


    waiting_for_release = (
        False
    )


    print()
    print(
        "Chuyen mode:",
        mode
    )


    return jsonify({

        "success":
            True,

        "mode":
            mode
    })


# =========================================================
# ADD SPACE
# =========================================================

@app.route(
    "/add_space",
    methods=[
        "POST"
    ]
)
def add_space():

    text = (
        sentence_builder.add_space()
    )


    return jsonify({

        "success":
            True,

        "text":
            text
    })


# =========================================================
# DELETE LAST CHARACTER
# =========================================================

@app.route(
    "/delete_last",
    methods=[
        "POST"
    ]
)
def delete_last():

    text = (
        sentence_builder.delete_last()
    )


    return jsonify({

        "success":
            True,

        "text":
            text
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

    global current_prediction
    global current_confidence
    global confirmed_result
    global waiting_for_release


    sentence_builder.clear()


    static_predictions.clear()

    dynamic_sequence.clear()

    dynamic_results.clear()


    with state_lock:

        current_prediction = ""

        current_confidence = (
            0.0
        )

        confirmed_result = ""


    waiting_for_release = (
        False
    )


    return jsonify({

        "success":
            True,

        "text":
            ""
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

        print()
        print(
            "=============================="
        )

        print(
            "SIGN LANGUAGE WEB"
        )

        print(
            "http://127.0.0.1:5000"
        )

        print(
            "=============================="
        )

        app.run(

            host="127.0.0.1",

            port=5000,

            debug=False,

            threaded=True
        )

    finally:

        cleanup()