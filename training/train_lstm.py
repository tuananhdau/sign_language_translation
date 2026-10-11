import os
import json
import numpy as np

from sklearn.model_selection import (
    train_test_split
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

from tensorflow.keras.models import (
    Sequential
)

from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout
)

from tensorflow.keras.utils import (
    to_categorical
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)


# =========================================================
# CONFIG
# =========================================================

DATASET_DIR = (
    "sequence_dataset"
)

MODEL_DIR = (
    "models"
)

MODEL_PATH = (
    "models/dynamic_sign_lstm.keras"
)

BEST_MODEL_PATH = (
    "models/best_dynamic_sign_lstm.keras"
)

LABEL_PATH = (
    "models/dynamic_labels.json"
)


SEQUENCE_LENGTH = 30

FEATURE_COUNT = 63


TEST_SIZE = 0.20

VALIDATION_SIZE = 0.20


EPOCHS = 100

BATCH_SIZE = 16

RANDOM_STATE = 42


# =========================================================
# CHECK DATASET
# =========================================================

if not os.path.exists(
    DATASET_DIR
):

    print(
        "Khong tim thay dataset:"
    )

    print(
        DATASET_DIR
    )

    exit()


os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# =========================================================
# GET LABELS
# =========================================================

labels = [

    name

    for name in os.listdir(
        DATASET_DIR
    )

    if os.path.isdir(
        os.path.join(
            DATASET_DIR,
            name
        )
    )
]


labels.sort()


if len(labels) < 2:

    print(
        "Can it nhat 2 label de train."
    )

    exit()


print()
print(
    "=============================="
)

print(
    "LABELS"
)

print(
    "=============================="
)


for index, label in enumerate(
    labels
):

    print(
        index,
        "->",
        label
    )


# =========================================================
# LABEL DICTIONARY
# =========================================================

label_dict = {

    str(index):
        label

    for index, label
    in enumerate(labels)
}


with open(
    LABEL_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        label_dict,
        file,
        ensure_ascii=False,
        indent=4
    )


print()
print(
    "Da luu labels:"
)

print(
    LABEL_PATH
)


# =========================================================
# LOAD DATASET
# =========================================================

X = []

y = []


print()
print(
    "=============================="
)

print(
    "LOAD DATASET"
)

print(
    "=============================="
)


for label_index, label in enumerate(
    labels
):

    label_dir = os.path.join(
        DATASET_DIR,
        label
    )


    sample_files = [

        file

        for file in os.listdir(
            label_dir
        )

        if file.lower().endswith(
            ".npy"
        )
    ]


    sample_files.sort()


    print()
    print(
        "Label:",
        label
    )

    print(
        "Samples:",
        len(sample_files)
    )


    valid_count = 0

    invalid_count = 0


    for filename in sample_files:

        file_path = os.path.join(
            label_dir,
            filename
        )


        try:

            sequence = np.load(
                file_path
            )


        except Exception as error:

            print(
                "Khong doc duoc:",
                file_path
            )

            print(
                error
            )

            invalid_count += 1

            continue


        # =====================================
        # CHECK SHAPE
        # =====================================

        expected_shape = (
            SEQUENCE_LENGTH,
            FEATURE_COUNT
        )


        if sequence.shape != (
            expected_shape
        ):

            print(
                "Sai shape:",
                file_path
            )

            print(
                "Nhan:",
                sequence.shape
            )

            print(
                "Can:",
                expected_shape
            )

            invalid_count += 1

            continue


        X.append(
            sequence
        )

        y.append(
            label_index
        )

        valid_count += 1


    print(
        "Hop le:",
        valid_count
    )

    print(
        "Bo qua:",
        invalid_count
    )


# =========================================================
# CONVERT NUMPY
# =========================================================

X = np.array(
    X,
    dtype=np.float32
)

y = np.array(
    y,
    dtype=np.int32
)


print()
print(
    "=============================="
)

print(
    "DATASET INFO"
)

print(
    "=============================="
)


print(
    "X shape:",
    X.shape
)

print(
    "y shape:",
    y.shape
)


if len(X) == 0:

    print(
        "Khong co du lieu hop le."
    )

    exit()


# =========================================================
# CHECK CLASS COUNT
# =========================================================

print()

for index, label in enumerate(
    labels
):

    count = int(
        np.sum(
            y == index
        )
    )

    print(
        label,
        ":",
        count
    )


# =========================================================
# ONE HOT
# =========================================================

num_classes = len(
    labels
)


y_onehot = to_categorical(
    y,
    num_classes=num_classes
)


# =========================================================
# TRAIN / TEST
# =========================================================

X_train_val, X_test, y_train_val, y_test = (
    train_test_split(

        X,

        y_onehot,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE,

        stratify=y
    )
)


# =========================================================
# TRAIN / VALIDATION
# =========================================================

train_val_labels = np.argmax(
    y_train_val,
    axis=1
)


X_train, X_val, y_train, y_val = (
    train_test_split(

        X_train_val,

        y_train_val,

        test_size=VALIDATION_SIZE,

        random_state=RANDOM_STATE,

        stratify=train_val_labels
    )
)


print()
print(
    "=============================="
)

print(
    "SPLIT"
)

print(
    "=============================="
)


print(
    "Train:",
    X_train.shape
)

print(
    "Validation:",
    X_val.shape
)

print(
    "Test:",
    X_test.shape
)


# =========================================================
# BUILD MODEL
# =========================================================

model = Sequential([

    Input(
        shape=(
            SEQUENCE_LENGTH,
            FEATURE_COUNT
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
        num_classes,
        activation="softmax"
    )
])


# =========================================================
# COMPILE
# =========================================================

model.compile(

    optimizer="adam",

    loss="categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


print()
print(
    "=============================="
)

print(
    "MODEL"
)

print(
    "=============================="
)


model.summary()


# =========================================================
# CALLBACKS
# =========================================================

callbacks = [

    EarlyStopping(

        monitor="val_loss",

        patience=12,

        restore_best_weights=True,

        verbose=1
    ),


    ModelCheckpoint(

        BEST_MODEL_PATH,

        monitor="val_accuracy",

        save_best_only=True,

        verbose=1
    ),


    ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=5,

        min_lr=0.00001,

        verbose=1
    )
]


# =========================================================
# TRAIN
# =========================================================

print()
print(
    "=============================="
)

print(
    "TRAINING"
)

print(
    "=============================="
)


history = model.fit(

    X_train,

    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=callbacks,

    verbose=1
)


# =========================================================
# TEST
# =========================================================

print()
print(
    "=============================="
)

print(
    "TEST MODEL"
)

print(
    "=============================="
)


test_loss, test_accuracy = (
    model.evaluate(

        X_test,

        y_test,

        verbose=0
    )
)


print(
    "Test Loss:",
    round(
        float(test_loss),
        4
    )
)


print(
    "Test Accuracy:",
    round(
        float(test_accuracy)
        * 100,
        2
    ),
    "%"
)


# =========================================================
# PREDICT TEST
# =========================================================

probabilities = model.predict(
    X_test,
    verbose=0
)


y_pred = np.argmax(
    probabilities,
    axis=1
)


y_true = np.argmax(
    y_test,
    axis=1
)


accuracy = accuracy_score(
    y_true,
    y_pred
)


print()
print(
    "Accuracy:",
    round(
        accuracy * 100,
        2
    ),
    "%"
)


# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print()
print(
    "=============================="
)

print(
    "CLASSIFICATION REPORT"
)

print(
    "=============================="
)


print(
    classification_report(

        y_true,

        y_pred,

        target_names=labels,

        zero_division=0
    )
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

print(
    "=============================="
)

print(
    "CONFUSION MATRIX"
)

print(
    "=============================="
)


matrix = confusion_matrix(
    y_true,
    y_pred
)


print(
    matrix
)


# =========================================================
# SAVE FINAL MODEL
# =========================================================

model.save(
    MODEL_PATH
)


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
    "Final model:"
)

print(
    MODEL_PATH
)


print(
    "Best model:"
)

print(
    BEST_MODEL_PATH
)


print(
    "Labels:"
)

print(
    LABEL_PATH
)