import os
import json
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)


# =========================================
# CONFIG
# =========================================

DATASET_DIR = "sequence_dataset"

MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "dynamic_sign_lstm.keras"
)

LABEL_PATH = os.path.join(
    MODEL_DIR,
    "dynamic_labels.json"
)

SEQUENCE_LENGTH = 30

FEATURE_COUNT = 63

TEST_SIZE = 0.2

RANDOM_STATE = 42

EPOCHS = 100

BATCH_SIZE = 16


# =========================================
# CHECK DATASET
# =========================================

if not os.path.exists(DATASET_DIR):

    print(
        f"Khong tim thay thu muc: {DATASET_DIR}"
    )

    print(
        "Hay thu du lieu truoc bang collect_sequence.py"
    )

    exit()


# =========================================
# LOAD DATASET
# =========================================

X = []
y = []

print("Dang doc sequence dataset...")

labels_found = []


for label in sorted(
    os.listdir(DATASET_DIR)
):

    label_path = os.path.join(
        DATASET_DIR,
        label
    )

    if not os.path.isdir(label_path):
        continue

    sample_count = 0

    for filename in os.listdir(label_path):

        if not filename.endswith(".npy"):
            continue

        file_path = os.path.join(
            label_path,
            filename
        )

        try:

            sequence = np.load(
                file_path
            )

        except Exception as e:

            print(
                f"Loi doc {file_path}: {e}"
            )

            continue


        # =================================
        # CHECK SHAPE
        # =================================

        if sequence.shape != (
            SEQUENCE_LENGTH,
            FEATURE_COUNT
        ):

            print(
                f"Bo qua {filename}: "
                f"shape={sequence.shape}"
            )

            continue


        X.append(sequence)

        y.append(label)

        sample_count += 1


    if sample_count > 0:

        labels_found.append(label)

        print(
            f"{label}: {sample_count} samples"
        )


# =========================================
# CONVERT NUMPY
# =========================================

X = np.array(
    X,
    dtype=np.float32
)

y = np.array(y)


print("\n==============================")
print("THONG TIN DATASET")
print("==============================")

print(
    "X shape:",
    X.shape
)

print(
    "y shape:",
    y.shape
)

print(
    "So classes:",
    len(set(y))
)


# =========================================
# CHECK DATA
# =========================================

if len(X) == 0:

    print(
        "Dataset dang rong!"
    )

    exit()


if len(set(y)) < 2:

    print(
        "Can it nhat 2 label de train."
    )

    print(
        "Vi du: XIN_CHAO va CAM_ON"
    )

    exit()


# =========================================
# LABEL ENCODING
# =========================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(
    y
)


labels = label_encoder.classes_

print("\nLabels:")

for index, label in enumerate(labels):

    print(
        f"{index} -> {label}"
    )


# =========================================
# ONE-HOT ENCODE
# =========================================

y_categorical = to_categorical(
    y_encoded,
    num_classes=len(labels)
)


# =========================================
# TRAIN / TEST SPLIT
# =========================================

X_train, X_test, y_train, y_test = \
    train_test_split(
        X,
        y_categorical,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded
    )


print("\nTraining samples:")

print(
    X_train.shape
)


print("\nTesting samples:")

print(
    X_test.shape
)


# =========================================
# CREATE MODEL
# =========================================

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

    Dropout(0.3),

    LSTM(
        64,
        return_sequences=False
    ),

    Dropout(0.3),

    Dense(
        64,
        activation="relu"
    ),

    Dropout(0.2),

    Dense(
        len(labels),
        activation="softmax"
    )
])


# =========================================
# COMPILE
# =========================================

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


print("\n==============================")
print("MODEL SUMMARY")
print("==============================")

model.summary()


# =========================================
# CREATE MODEL DIRECTORY
# =========================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# =========================================
# CALLBACKS
# =========================================

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=10,

    restore_best_weights=True,

    verbose=1
)


checkpoint = ModelCheckpoint(

    MODEL_PATH,

    monitor="val_accuracy",

    save_best_only=True,

    verbose=1
)


# =========================================
# TRAIN
# =========================================

print("\n==============================")
print("BAT DAU TRAIN LSTM")
print("==============================")


history = model.fit(

    X_train,

    y_train,

    validation_split=0.2,

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=[
        early_stopping,
        checkpoint
    ],

    verbose=1
)


# =========================================
# LOAD BEST MODEL
# =========================================

from tensorflow.keras.models import load_model

model = load_model(
    MODEL_PATH
)


# =========================================
# EVALUATE
# =========================================

print("\n==============================")
print("DANH GIA MODEL")
print("==============================")


loss, accuracy = model.evaluate(

    X_test,

    y_test,

    verbose=0
)


print(
    f"Test Loss: {loss:.4f}"
)

print(
    f"Test Accuracy: "
    f"{accuracy * 100:.2f}%"
)





# =========================================
# PREDICT
# =========================================

y_pred_prob = model.predict(
    X_test,
    verbose=0
)


y_pred = np.argmax(
    y_pred_prob,
    axis=1
)


y_true = np.argmax(
    y_test,
    axis=1
)


# =========================================
# CLASSIFICATION REPORT
# =========================================

print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=labels,
        zero_division=0
    )
)


# =========================================
# CONFUSION MATRIX
# =========================================

print("Confusion Matrix:")

print(
    confusion_matrix(
        y_true,
        y_pred
    )
)


# =========================================
# SAVE LABELS
# =========================================

label_dict = {

    str(index): label

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


print("\n==============================")
print("HOAN TAT")
print("==============================")


print(
    f"Model:"
)

print(
    MODEL_PATH
)


print(
    "\nLabels:"
)

print(
    LABEL_PATH
)