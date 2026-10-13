import cv2
import threading
import numpy as np

from collections import deque, Counter

from flask import (
    Flask,
    render_template,
    Response,
    jsonify
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
# MODEL
# =========================================================

STATIC_MODEL_PATH = "models/sign_model.pkl"

DYNAMIC_MODEL_PATH = "models/dynamic_sign_lstm.keras"

DYNAMIC_LABEL_PATH = "models/dynamic_labels.json"


# =========================================================
# CAMERA
# =========================================================

CAMERA_WIDTH = 480
CAMERA_HEIGHT = 360
CAMERA_FPS = 30

JPEG_QUALITY = 65


# =========================================================
# FEATURE
# =========================================================

FEATURE_COUNT = 63


# =========================================================
# STATIC CONFIG
# =========================================================

STATIC_CONFIDENCE = 0.80

STATIC_STABLE_FRAMES = 8

STATIC_MIN_COUNT = 6

STATIC_PREDICT_EVERY = 2


# =========================================================
# DYNAMIC CONFIG
# =========================================================

SEQUENCE_LENGTH = 30

DYNAMIC_CONFIDENCE = 0.80

DYNAMIC_STABLE_RESULTS = 4

DYNAMIC_MIN_COUNT = 3

DYNAMIC_PREDICT_EVERY = 3


# =========================================================
# AUTO DETECTION CONFIG
# =========================================================

# Số frame bỏ qua khi bàn tay vừa xuất hiện
HAND_WARMUP_FRAMES = 5


# Lịch sử chuyển động
MOTION_HISTORY_LENGTH = 5


# Nếu motion lớn hơn mức này
# nhiều frame liên tục -> Dynamic
DYNAMIC_MOTION_THRESHOLD = 0.008


# Tay ổn định dưới mức này -> Static
STATIC_MOTION_THRESHOLD = 0.0035


# Cần chuyển động liên tục bao nhiêu frame
# mới khóa sang Dynamic
DYNAMIC_TRIGGER_FRAMES = 3


# Cần đứng yên bao nhiêu frame
# trước khi nhận chữ cái
STATIC_STILL_FRAMES = 8


# Lưu vài frame trước khi phát hiện Dynamic
PRE_DYNAMIC_FRAMES = 8


# =========================================================
# LOAD MODEL
# =========================================================

print()
print("====================================")
print("LOAD AI MODELS")
print("====================================")


predictor = Predictor(

    static_model_path=STATIC_MODEL_PATH,

    dynamic_model_path=DYNAMIC_MODEL_PATH,

    dynamic_label_path=DYNAMIC_LABEL_PATH,

    sequence_length=SEQUENCE_LENGTH,

    dynamic_feature_count=FEATURE_COUNT
)


print("Models loaded.")


# =========================================================
# SENTENCE
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
# LOCK
# =========================================================

state_lock = threading.Lock()


# =========================================================
# GLOBAL STATE
# =========================================================

auto_mode = "waiting"

current_prediction = ""

current_confidence = 0.0

confirmed_result = ""

waiting_for_release = False


# Motion hiện tại
current_motion = 0.0


# =========================================================
# AUTO MODE STATE
# =========================================================

previous_motion_vector = None

motion_history = deque(
    maxlen=MOTION_HISTORY_LENGTH
)

still_frame_count = 0

dynamic_trigger_count = 0

hand_visible_frames = 0

dynamic_locked = False


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


# Lưu các frame trước khi chuyển sang Dynamic
pre_dynamic_buffer = deque(
    maxlen=PRE_DYNAMIC_FRAMES
)


# =========================================================
# FORMAT DYNAMIC
# =========================================================

def format_dynamic_label(label):

    mapping = {

        "XIN_CHAO": "Xin chào",

        "XIN_LOI": "Xin lỗi",

        "CAM_ON": "Cảm ơn",

        "GIUP_DO": "Giúp đỡ",

        "TOI": "Tôi",

        "BAN": "Bạn",

        "CO": "Có",

        "KHONG": "Không",

        "AN": "Ăn",

        "UONG": "Uống"
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
# RAW LANDMARK VECTOR
#
# Dùng để đo chuyển động.
#
# Không dùng feature đã chuẩn hóa theo wrist vì nếu
# toàn bộ bàn tay di chuyển trái/phải thì feature chuẩn hóa
# có thể gần như không đổi.
# =========================================================

def get_motion_vector(
    hand_landmarks
):

    if hand_landmarks is None:
        return None


    values = []


    for landmark in (
        hand_landmarks.landmark
    ):

        values.extend([

            landmark.x,

            landmark.y,

            landmark.z
        ])


    return np.array(
        values,
        dtype=np.float32
    )


# =========================================================
# CALCULATE MOTION
# =========================================================

def calculate_motion(
    previous,
    current
):

    if (
        previous is None
        or
        current is None
    ):
        return 0.0


    if (
        previous.shape
        != current.shape
    ):
        return 0.0


    difference = np.abs(
        current - previous
    )


    return float(
        np.mean(
            difference
        )
    )


# =========================================================
# RESET AUTO DETECTOR
# =========================================================

def reset_auto_detection():

    global auto_mode
    global previous_motion_vector
    global still_frame_count
    global dynamic_trigger_count
    global hand_visible_frames
    global dynamic_locked
    global current_motion


    auto_mode = "waiting"

    previous_motion_vector = None

    still_frame_count = 0

    dynamic_trigger_count = 0

    hand_visible_frames = 0

    dynamic_locked = False

    current_motion = 0.0


    motion_history.clear()

    pre_dynamic_buffer.clear()

    static_predictions.clear()

    dynamic_sequence.clear()

    dynamic_results.clear()


# =========================================================
# STATIC PROCESS
# =========================================================

def process_static(features):

    global current_prediction
    global current_confidence
    global confirmed_result
    global waiting_for_release


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


    if not label:
        return


    label = str(label).upper()


    with state_lock:

        current_prediction = label

        current_confidence = confidence


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
        counter.most_common(1)[0]
    )


    if count < STATIC_MIN_COUNT:
        return


    # =====================================================
    # CONFIRM
    # =====================================================

    sentence_builder.add_letter(
        stable_label
    )


    with state_lock:

        confirmed_result = (
            stable_label
        )


    print()
    print(
        "[STATIC]",
        stable_label
    )

    print(
        "[TEXT]",
        sentence_builder.get_text()
    )


    # Không cho AAAAAA
    waiting_for_release = True

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


    if waiting_for_release:
        return


    # =====================================================
    # ADD FRAME
    # =====================================================

    dynamic_sequence.append(
        features
    )


    # =====================================================
    # CHƯA ĐỦ 30
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

        current_prediction = (
            format_dynamic_label(
                label
            )
        )

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


    if count < DYNAMIC_MIN_COUNT:
        return


    phrase = format_dynamic_label(
        stable_label
    )


    # =====================================================
    # ADD PHRASE
    # =====================================================

    sentence_builder.add_phrase(
        phrase
    )


    with state_lock:

        confirmed_result = phrase


    print()
    print(
        "[DYNAMIC]",
        stable_label
    )

    print(
        "[CONFIDENCE]",
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
# AUTOMATIC ROUTING
# =========================================================

def process_auto(
    hand_landmarks,
    features,
    frame_count
):

    global auto_mode
    global previous_motion_vector
    global still_frame_count
    global dynamic_trigger_count
    global hand_visible_frames
    global dynamic_locked
    global current_motion


    # =====================================================
    # MOTION VECTOR
    # =====================================================

    motion_vector = (
        get_motion_vector(
            hand_landmarks
        )
    )


    motion = calculate_motion(

        previous_motion_vector,

        motion_vector
    )


    previous_motion_vector = (
        motion_vector.copy()
    )


    motion_history.append(
        motion
    )


    current_motion = (

        sum(motion_history)
        /
        len(motion_history)

        if motion_history

        else 0.0
    )


    hand_visible_frames += 1


    # Lưu feature liên tục
    # để khi phát hiện Dynamic có thể lấy lại
    # vài frame đầu của chuyển động

    pre_dynamic_buffer.append(
        features.copy()
    )


    # =====================================================
    # VỪA ĐƯA TAY VÀO CAMERA
    # =====================================================

    if (
        hand_visible_frames
        <= HAND_WARMUP_FRAMES
    ):

        auto_mode = "waiting"

        return


    # =====================================================
    # ĐÃ KHÓA DYNAMIC
    # =====================================================

    if dynamic_locked:

        auto_mode = "dynamic"


        process_dynamic(
            features,
            frame_count
        )


        return


    # =====================================================
    # PHÁT HIỆN CHUYỂN ĐỘNG
    # =====================================================

    if (
        current_motion
        >= DYNAMIC_MOTION_THRESHOLD
    ):

        dynamic_trigger_count += 1

        still_frame_count = 0

        static_predictions.clear()


        # =============================================
        # ĐỦ FRAME CHUYỂN ĐỘNG -> LOCK DYNAMIC
        # =============================================

        if (
            dynamic_trigger_count
            >= DYNAMIC_TRIGGER_FRAMES
        ):

            auto_mode = "dynamic"

            dynamic_locked = True


            # Xóa dữ liệu cũ
            dynamic_sequence.clear()

            dynamic_results.clear()


            # =========================================
            # ĐƯA VÀI FRAME TRƯỚC VÀO LSTM
            # để không mất đầu động tác
            # =========================================

            for buffered_feature in (
                pre_dynamic_buffer
            ):

                dynamic_sequence.append(
                    buffered_feature
                )


            process_dynamic(
                features,
                frame_count
            )


        else:

            auto_mode = "waiting"


        return


    # =====================================================
    # KHÔNG CÒN CHUYỂN ĐỘNG MẠNH
    # =====================================================

    dynamic_trigger_count = 0


    # =====================================================
    # TAY ỔN ĐỊNH
    # =====================================================

    if (
        current_motion
        <= STATIC_MOTION_THRESHOLD
    ):

        still_frame_count += 1


        if (
            still_frame_count
            >= STATIC_STILL_FRAMES
        ):

            auto_mode = "static"


            dynamic_sequence.clear()

            dynamic_results.clear()


            if (
                frame_count
                % STATIC_PREDICT_EVERY
                == 0
            ):

                process_static(
                    features
                )


        else:

            auto_mode = "waiting"


    # =====================================================
    # VÙNG TRUNG GIAN
    # =====================================================

    else:

        still_frame_count = 0

        auto_mode = "waiting"


# =========================================================
# VIDEO STREAM
# =========================================================

def generate_frames():

    global current_prediction
    global current_confidence
    global confirmed_result
    global waiting_for_release


    frame_count = 0


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


        frame_count += 1


        # =================================================
        # MEDIAPIPE
        # =================================================

        results = (
            hand_detector.detect(
                frame
            )
        )


        hands_data = (
            hand_detector.get_hands(
                results
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


        # =================================================
        # 1 CAMERA
        #
        # MODEL HIỆN TẠI VẪN LÀ 1 TAY = 63 FEATURES
        # Ưu tiên tay phải.
        # =================================================

        selected_hand = None


        if right_hand is not None:

            selected_hand = (
                right_hand
            )


        elif left_hand is not None:

            selected_hand = (
                left_hand
            )


        # =================================================
        # CÓ TAY
        # =================================================

        if selected_hand is not None:

            features = extract_features(
                selected_hand
            )


            if (
                features is not None
                and
                len(features)
                == FEATURE_COUNT
            ):

                process_auto(

                    selected_hand,

                    features,

                    frame_count
                )


        # =================================================
        # KHÔNG CÓ TAY
        # =================================================

        else:

            reset_auto_detection()


            with state_lock:

                current_prediction = ""

                current_confidence = 0.0


            waiting_for_release = False


        # =================================================
        # DRAW HAND
        # =================================================

        hand_detector.draw_hands(
            frame,
            results
        )


        # =================================================
        # GET STATE
        # =================================================

        with state_lock:

            prediction = (
                current_prediction
            )

            confidence = (
                current_confidence
            )


        # =================================================
        # MODE TEXT
        # =================================================

        if auto_mode == "static":

            mode_text = (
                "AUTO: LETTER"
            )


        elif auto_mode == "dynamic":

            mode_text = (
                "AUTO: DYNAMIC"
            )


        else:

            mode_text = (
                "AUTO: DETECTING"
            )


        # =================================================
        # OVERLAY
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

            (
                "Predict: "
                f"{prediction}"
            ),

            (15, 55),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

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


        cv2.putText(

            frame,

            (
                "Motion: "
                f"{current_motion:.4f}"
            ),

            (15, 115),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.45,

            (255, 255, 255),

            1
        )


        # =================================================
        # DYNAMIC SEQUENCE
        # =================================================

        if auto_mode == "dynamic":

            cv2.putText(

                frame,

                (
                    "Sequence: "
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
        # WAIT RELEASE
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


        yield (

            b"--frame\r\n"

            b"Content-Type: image/jpeg\r\n\r\n"

            + buffer.tobytes()

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

        prediction = (
            current_prediction
        )

        confidence = (
            current_confidence
        )

        confirmed = (
            confirmed_result
        )


    return jsonify({

        "mode":
            auto_mode,

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
            sentence_builder.get_text(),

        "sequence":
            len(
                dynamic_sequence
            ),

        "sequence_length":
            SEQUENCE_LENGTH,

        "motion":
            round(
                current_motion,
                5
            ),

        "waiting":
            waiting_for_release
    })


# =========================================================
# SPACE
# =========================================================

@app.route(
    "/add_space",
    methods=["POST"]
)
def add_space():

    return jsonify({

        "success":
            True,

        "text":
            sentence_builder.add_space()
    })


# =========================================================
# DELETE
# =========================================================

@app.route(
    "/delete_last",
    methods=["POST"]
)
def delete_last():

    return jsonify({

        "success":
            True,

        "text":
            sentence_builder.delete_last()
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

    reset_auto_detection()


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
        print("====================================")
        print("AUTO SIGN LANGUAGE SYSTEM")
        print("http://127.0.0.1:5000")
        print("====================================")


        app.run(

            host="127.0.0.1",

            port=5000,

            debug=False,

            threaded=True
        )


    finally:

        cleanup()