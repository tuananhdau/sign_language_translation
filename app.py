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
from src.feature_extractor import extract_features
from src.predictor import Predictor
from src.sentence_builder import SentenceBuilder


# =========================================================
# FLASK
# =========================================================

app = Flask(__name__)


# =========================================================
# MODEL PATH
# =========================================================

STATIC_MODEL_PATH = "models/sign_model.pkl"

DYNAMIC_MODEL_PATH = "models/dynamic_sign_lstm.keras"

DYNAMIC_LABEL_PATH = "models/dynamic_labels.json"


# =========================================================
# STATIC CONFIG
# =========================================================

STATIC_CONFIDENCE = 0.80

STATIC_STABLE_FRAMES = 10

STATIC_MIN_COUNT = 8

STATIC_PREDICT_EVERY = 2


# =========================================================
# DYNAMIC CONFIG
# =========================================================

SEQUENCE_LENGTH = 30

FEATURE_COUNT = 63

DYNAMIC_CONFIDENCE = 0.80

DYNAMIC_STABLE_RESULTS = 5

DYNAMIC_MIN_COUNT = 4


# =========================================================
# PERFORMANCE
# =========================================================

# MediaPipe xử lý mọi frame
AI_FRAME_SKIP = 1

# Sau khi đủ 30 frame
# LSTM predict mỗi 3 frame
DYNAMIC_PREDICT_EVERY = 3


# =========================================================
# CAMERA
# =========================================================

CAMERA_WIDTH = 480

CAMERA_HEIGHT = 360

CAMERA_FPS = 30

JPEG_QUALITY = 65


# =========================================================
# LOAD MODEL
# =========================================================

print()
print("===================================")
print("LOAD MODEL")
print("===================================")


predictor = Predictor(

    static_model_path=STATIC_MODEL_PATH,

    dynamic_model_path=DYNAMIC_MODEL_PATH,

    dynamic_label_path=DYNAMIC_LABEL_PATH,

    sequence_length=SEQUENCE_LENGTH,

    dynamic_feature_count=FEATURE_COUNT
)


print("Load model thanh cong.")


# =========================================================
# SENTENCE BUILDER
# =========================================================

sentence_builder = SentenceBuilder()


# =========================================================
# HAND DETECTOR
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
# STATE LOCK
# =========================================================

state_lock = threading.Lock()


# =========================================================
# GLOBAL STATE
# =========================================================

# static:
# đánh vần chữ cái
#
# dynamic:
# từ giao tiếp

recognition_mode = "static"


current_prediction = ""

current_confidence = 0.0

confirmed_result = ""


# Sau khi nhận một ký hiệu
# cần bỏ tay ra để nhận tiếp

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
        .replace("_", " ")
        .title()
    )


# =========================================================
# RESET RECOGNITION
# =========================================================

def reset_recognition():

    global current_prediction
    global current_confidence
    global waiting_for_release


    static_predictions.clear()

    dynamic_sequence.clear()

    dynamic_results.clear()


    with state_lock:

        current_prediction = ""

        current_confidence = 0.0


    waiting_for_release = False


# =========================================================
# STATIC
#
# A B C D...
# dùng để đánh vần tên người
# hoặc tên địa điểm
# =========================================================

def process_static(features):

    global current_prediction
    global current_confidence
    global confirmed_result
    global waiting_for_release


    # Đã nhận chữ rồi
    # chờ người dùng bỏ tay

    if waiting_for_release:
        return


    # =====================================================
    # PREDICT RANDOM FOREST
    # =====================================================

    label, confidence = (
        predictor.predict_static(
            features
        )
    )


    if not label:
        return


    label = str(label).upper()


    with state_lock:

        current_prediction = label

        current_confidence = confidence


    # =====================================================
    # CONFIDENCE
    # =====================================================

    if confidence < STATIC_CONFIDENCE:

        static_predictions.clear()

        return


    static_predictions.append(
        label
    )


    # =====================================================
    # STABLE
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
        counter.most_common(1)[0]
    )


    if count < STATIC_MIN_COUNT:
        return


    # =====================================================
    # CONFIRMED
    # =====================================================

    with state_lock:

        confirmed_result = stable_label


    # =====================================================
    # TỰ ĐỘNG THÊM CHỮ
    #
    # H → I → E → U
    # => HIEU
    # =====================================================

    sentence_builder.add_letter(
        stable_label
    )


    print()
    print(
        "[STATIC] Chu cai:",
        stable_label
    )

    print(
        "[TEXT]",
        sentence_builder.get_text()
    )


    # =====================================================
    # ANTI REPEAT
    # =====================================================

    waiting_for_release = True

    static_predictions.clear()


# =========================================================
# DYNAMIC
#
# XIN_CHAO
# XIN_LOI
# ...
# =========================================================

def process_dynamic(
    features,
    frame_count
):

    global current_prediction
    global current_confidence
    global confirmed_result
    global waiting_for_release


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
    # PREDICT MỖI 3 FRAME
    # =====================================================

    if (
        frame_count
        % DYNAMIC_PREDICT_EVERY
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

        current_confidence = confidence


    # =====================================================
    # CONFIDENCE
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
    # STABLE
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
        counter.most_common(1)[0]
    )


    if (
        count
        < DYNAMIC_MIN_COUNT
    ):

        return


    # =====================================================
    # FORMAT WORD
    # =====================================================

    phrase = format_dynamic_label(
        stable_label
    )


    # =====================================================
    # CONFIRMED
    # =====================================================

    with state_lock:

        confirmed_result = stable_label


    # =====================================================
    # TỰ ĐỘNG THÊM TỪ
    #
    # XIN_CHAO
    # => Xin chào
    # =====================================================

    sentence_builder.add_phrase(
        phrase
    )


    print()
    print(
        "[DYNAMIC]",
        stable_label
    )

    print(
        "[DYNAMIC] Confidence:",
        round(
            confidence * 100,
            2
        ),
        "%"
    )

    print(
        "[TEXT]",
        sentence_builder.get_text()
    )


    # =====================================================
    # ANTI REPEAT
    # =====================================================

    waiting_for_release = True


    dynamic_sequence.clear()

    dynamic_results.clear()


# =========================================================
# CAMERA STREAM
# =========================================================

def generate_frames():

    global current_prediction
    global current_confidence
    global waiting_for_release


    frame_count = 0

    last_results = None

    hand_count = 0


    while True:

        # =================================================
        # READ CAMERA
        # =================================================

        success, frame = camera.read()


        if not success:
            continue


        frame = cv2.flip(
            frame,
            1
        )


        frame_count += 1


        # =================================================
        # RUN AI
        # =================================================

        run_ai = (

            frame_count
            % AI_FRAME_SKIP

            == 0
        )


        if run_ai:

            # =============================================
            # MEDIAPIPE
            # =============================================

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


            hand_count = 0


            if left_hand is not None:
                hand_count += 1


            if right_hand is not None:
                hand_count += 1


            # =============================================
            # CÓ TAY
            # =============================================

            if hand_count > 0:

                # Model hiện tại train 1 tay
                # => 63 features
                #
                # ưu tiên tay phải,
                # không có thì dùng tay trái

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


                if (
                    features is not None
                    and
                    len(features)
                    == FEATURE_COUNT
                ):

                    # =====================================
                    # STATIC MODE
                    # =====================================

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


                    # =====================================
                    # DYNAMIC MODE
                    # =====================================

                    elif (
                        recognition_mode
                        == "dynamic"
                    ):

                        process_dynamic(
                            features,
                            frame_count
                        )


            # =============================================
            # KHÔNG CÓ TAY
            # =============================================

            else:

                static_predictions.clear()

                dynamic_sequence.clear()

                dynamic_results.clear()


                with state_lock:

                    current_prediction = ""

                    current_confidence = 0.0


                # Đã bỏ tay
                # => cho phép nhận ký hiệu tiếp

                waiting_for_release = False


        # =================================================
        # DRAW LANDMARK
        # =================================================

        if last_results is not None:

            hand_detector.draw_hands(
                frame,
                last_results
            )


        # =================================================
        # CURRENT STATE
        # =================================================

        with state_lock:

            prediction = current_prediction

            confidence = current_confidence


        # =================================================
        # DISPLAY MODE
        # =================================================

        if (
            recognition_mode
            == "dynamic"
        ):

            mode_text = (
                "TU GIAO TIEP - LSTM"
            )


            display_prediction = (
                format_dynamic_label(
                    prediction
                )
            )

        else:

            mode_text = (
                "DANH VAN CHU CAI - RANDOM FOREST"
            )


            display_prediction = (
                prediction
            )


        # =================================================
        # DRAW INFO
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


        cv2.putText(

            frame,

            f"Predict: {display_prediction}",

            (15, 55),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.60,

            (0, 255, 0),

            2
        )


        cv2.putText(

            frame,

            (
                "Confidence: "
                f"{confidence * 100:.1f}%"
            ),

            (15, 85),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.50,

            (255, 255, 0),

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
                    "Sequence: "
                    f"{len(dynamic_sequence)}"
                    f"/{SEQUENCE_LENGTH}"
                ),

                (15, 115),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.50,

                (255, 255, 255),

                2
            )


        # =================================================
        # WAIT
        # =================================================

        if waiting_for_release:

            cv2.putText(

                frame,

                "Remove hand to continue",

                (15, 145),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.50,

                (0, 255, 255),

                2
            )


        # =================================================
        # JPEG
        # =================================================

        encode_success, buffer = (
            cv2.imencode(

                ".jpg",

                frame,

                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    JPEG_QUALITY
                ]
            )
        )


        if not encode_success:
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
# STATUS
# =========================================================

@app.route("/status")
def status():

    with state_lock:

        prediction = current_prediction

        confidence = current_confidence

        confirmed = confirmed_result


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
                confidence * 100,
                2
            ),

        "confirmed":
            confirmed,

        "text":
            sentence_builder
            .get_text(),

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
    methods=["POST"]
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

        current_confidence = 0.0

        confirmed_result = ""


    static_predictions.clear()

    dynamic_sequence.clear()

    dynamic_results.clear()


    waiting_for_release = False


    print()
    print(
        "Chuyen che do:",
        mode
    )


    return jsonify({

        "success":
            True,

        "mode":
            mode
    })


# =========================================================
# SPACE
# =========================================================

@app.route(
    "/add_space",
    methods=["POST"]
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
# DELETE
# =========================================================

@app.route(
    "/delete_last",
    methods=["POST"]
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
    methods=["POST"]
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

        current_confidence = 0.0

        confirmed_result = ""


    waiting_for_release = False


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
        print("===================================")
        print("SIGN LANGUAGE WEB")
        print("http://127.0.0.1:5000")
        print("===================================")


        app.run(

            host="127.0.0.1",

            port=5000,

            debug=False,

            threaded=True
        )


    finally:

        cleanup()