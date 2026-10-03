import os
import threading
from collections import deque, Counter

import cv2
import joblib
import numpy as np
import mediapipe as mp

from flask import Flask, render_template, Response, jsonify


app = Flask(__name__)


# =========================================
# CONFIG
# =========================================

MODEL_PATH = "models/sign_model.pkl"

CONFIDENCE_THRESHOLD = 0.80

STABLE_FRAMES = 10
MIN_STABLE_COUNT = 8


# =========================================
# LOAD MODEL
# =========================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Khong tim thay model: {MODEL_PATH}"
    )

print("Dang tai Random Forest model...")

model = joblib.load(MODEL_PATH)

print("Tai model thanh cong!")


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

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError(
        "Khong the mo camera"
    )


# =========================================
# VARIABLES
# =========================================

recent_predictions = deque(
    maxlen=STABLE_FRAMES
)

current_prediction = ""
current_confidence = 0.0

confirmed_character = ""

text_result = ""

waiting_for_hand_release = False

lock = threading.Lock()


# =========================================
# EXTRACT FEATURES
# =========================================

def extract_features(hand_landmarks):

    landmarks = hand_landmarks.landmark

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


# =========================================
# GENERATE CAMERA
# =========================================

def generate_frames():

    global current_prediction
    global current_confidence
    global confirmed_character
    global waiting_for_hand_release

    while True:

        success, frame = camera.read()

        if not success:
            break

        frame = cv2.flip(
            frame,
            1
        )

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = hands.process(
            rgb_frame
        )


        # =====================================
        # HAND DETECTED
        # =====================================

        if results.multi_hand_landmarks:

            hand_landmarks = (
                results.multi_hand_landmarks[0]
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

                features_array = np.array(
                    features
                ).reshape(1, -1)


                prediction = model.predict(
                    features_array
                )[0]


                if hasattr(
                    model,
                    "predict_proba"
                ):

                    probabilities = (
                        model.predict_proba(
                            features_array
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
                    >= CONFIDENCE_THRESHOLD
                    and
                    not waiting_for_hand_release
                ):

                    recent_predictions.append(
                        str(prediction)
                    )


                    if (
                        len(recent_predictions)
                        == STABLE_FRAMES
                    ):

                        counter = Counter(
                            recent_predictions
                        )

                        label, count = (
                            counter.most_common(1)[0]
                        )


                        if (
                            count
                            >= MIN_STABLE_COUNT
                        ):

                            with lock:

                                confirmed_character = (
                                    label
                                )

                                waiting_for_hand_release = (
                                    True
                                )


                            recent_predictions.clear()


        # =====================================
        # NO HAND
        # =====================================

        else:

            recent_predictions.clear()

            with lock:

                current_prediction = ""

                current_confidence = 0.0


                if waiting_for_hand_release:

                    waiting_for_hand_release = (
                        False
                    )


        # =====================================
        # CAMERA TEXT
        # =====================================

        with lock:

            display_prediction = (
                current_prediction
            )

            display_confidence = (
                current_confidence
            )


        cv2.putText(
            frame,
            f"Prediction: {display_prediction}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            (
                f"Confidence: "
                f"{display_confidence * 100:.1f}%"
            ),
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )


        # =====================================
        # STREAM
        # =====================================

        ret, buffer = cv2.imencode(
            ".jpg",
            frame
        )

        if not ret:
            continue


        frame_bytes = buffer.tobytes()


        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


# =========================================
# HOME
# =========================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# =========================================
# CAMERA STREAM
# =========================================

@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype=(
            "multipart/x-mixed-replace;"
            " boundary=frame"
        )
    )


# =========================================
# STATUS
# =========================================

@app.route("/status")
def status():

    with lock:

        return jsonify({

            "prediction":
                current_prediction,

            "confidence":
                round(
                    current_confidence * 100,
                    2
                ),

            "confirmed":
                confirmed_character,

            "text":
                text_result
        })


# =========================================
# ADD CHARACTER
# =========================================

@app.route(
    "/add_character",
    methods=["POST"]
)
def add_character():

    global text_result
    global confirmed_character

    with lock:

        if confirmed_character:

            text_result += (
                confirmed_character
            )

            added = confirmed_character

            confirmed_character = ""

            recent_predictions.clear()

            return jsonify({
                "success": True,
                "added": added
            })


    return jsonify({
        "success": False
    })


# =========================================
# SPACE
# =========================================

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
            not text_result.endswith(" ")
        ):

            text_result += " "


    return jsonify({
        "success": True
    })


# =========================================
# DELETE LAST
# =========================================

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


# =========================================
# CLEAR
# =========================================

@app.route(
    "/clear",
    methods=["POST"]
)
def clear():

    global text_result
    global confirmed_character
    global current_prediction
    global current_confidence
    global waiting_for_hand_release

    with lock:

        text_result = ""

        confirmed_character = ""

        current_prediction = ""

        current_confidence = 0.0

        waiting_for_hand_release = False

        recent_predictions.clear()


    return jsonify({
        "success": True
    })


# =========================================
# RUN
# =========================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True
    )