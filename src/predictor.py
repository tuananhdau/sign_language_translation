import os
import json

import joblib
import numpy as np

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


class Predictor:

    def __init__(
        self,
        static_model_path=None,
        dynamic_model_path=None,
        dynamic_label_path=None,
        sequence_length=30,
        dynamic_feature_count=63
    ):

        self.static_model = None

        self.dynamic_model = None

        self.dynamic_labels = {}

        self.sequence_length = (
            sequence_length
        )

        self.dynamic_feature_count = (
            dynamic_feature_count
        )


        # =========================================
        # STATIC
        # =========================================

        if static_model_path:

            self.load_static_model(
                static_model_path
            )


        # =========================================
        # LABEL
        # =========================================

        if dynamic_label_path:

            self.load_dynamic_labels(
                dynamic_label_path
            )


        # =========================================
        # DYNAMIC
        # =========================================

        if dynamic_model_path:

            self.load_dynamic_model(
                dynamic_model_path
            )


    # =====================================================
    # STATIC MODEL
    # =====================================================

    def load_static_model(
        self,
        model_path
    ):

        if not os.path.exists(
            model_path
        ):

            raise FileNotFoundError(
                model_path
            )


        print(
            "Dang tai Random Forest..."
        )


        self.static_model = (
            joblib.load(
                model_path
            )
        )


        print(
            "Random Forest OK"
        )


    # =====================================================
    # DYNAMIC LABELS
    # =====================================================

    def load_dynamic_labels(
        self,
        label_path
    ):

        if not os.path.exists(
            label_path
        ):

            raise FileNotFoundError(
                label_path
            )


        with open(
            label_path,
            "r",
            encoding="utf-8"
        ) as file:

            self.dynamic_labels = (
                json.load(
                    file
                )
            )


    # =====================================================
    # CREATE LSTM
    # =====================================================

    def _create_dynamic_model(
        self
    ):

        class_count = len(
            self.dynamic_labels
        )


        if class_count == 0:

            raise ValueError(
                "Dynamic labels rong."
            )


        model = Sequential([

            Input(
                shape=(
                    self.sequence_length,
                    self.dynamic_feature_count
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
                class_count,
                activation="softmax"
            )
        ])


        return model


    # =====================================================
    # LOAD DYNAMIC
    # =====================================================

    def load_dynamic_model(
        self,
        model_path
    ):

        if not os.path.exists(
            model_path
        ):

            raise FileNotFoundError(
                model_path
            )


        print(
            "Dang tai LSTM..."
        )


        try:

            self.dynamic_model = (
                load_model(
                    model_path,
                    compile=False
                )
            )


            print(
                "LSTM load_model OK"
            )


        except Exception as error:

            print(
                "Load model truc tiep that bai."
            )

            print(
                "Thu load weights..."
            )

            print(
                error
            )


            self.dynamic_model = (
                self._create_dynamic_model()
            )


            self.dynamic_model.load_weights(
                model_path
            )


            print(
                "LSTM load_weights OK"
            )


    # =====================================================
    # STATIC PREDICTION
    # =====================================================

    def predict_static(
        self,
        features
    ):

        if self.static_model is None:

            return "", 0.0


        data = np.array(
            features,
            dtype=np.float32
        ).reshape(
            1,
            -1
        )


        prediction = (
            self.static_model.predict(
                data
            )[0]
        )


        confidence = 1.0


        if hasattr(
            self.static_model,
            "predict_proba"
        ):

            probabilities = (
                self.static_model
                .predict_proba(
                    data
                )[0]
            )


            confidence = float(
                np.max(
                    probabilities
                )
            )


        return (
            str(prediction),
            confidence
        )


    # =====================================================
    # DYNAMIC PREDICTION
    # =====================================================

    def predict_dynamic(
        self,
        sequence
    ):

        if self.dynamic_model is None:

            return "", 0.0


        if (
            len(sequence)
            != self.sequence_length
        ):

            return "", 0.0


        data = np.array(
            sequence,
            dtype=np.float32
        )


        expected_shape = (

            self.sequence_length,

            self.dynamic_feature_count
        )


        if (
            data.shape
            != expected_shape
        ):

            print(
                "Sai shape LSTM:"
            )

            print(
                "Nhan:",
                data.shape
            )

            print(
                "Can:",
                expected_shape
            )


            return "", 0.0


        data = np.expand_dims(
            data,
            axis=0
        )


        probabilities = (
            self.dynamic_model.predict(
                data,
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


        label = (
            self.dynamic_labels.get(
                str(
                    predicted_index
                ),
                str(
                    predicted_index
                )
            )
        )


        return (
            label,
            confidence
        )