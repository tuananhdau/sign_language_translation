import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==============================
# CONFIG
# ==============================

DATASET_PATH = "dataset/landmarks.csv"
MODEL_PATH = "models/sign_model.pkl"

# ==============================
# CHECK DATASET
# ==============================

if not os.path.exists(DATASET_PATH):
    print(f"Khong tim thay file dataset: {DATASET_PATH}")
    exit()

# ==============================
# LOAD DATASET
# ==============================

print("Dang doc dataset...")

df = pd.read_csv(DATASET_PATH)

print("Kich thuoc dataset:", df.shape)

print("\nSo mau tung nhan:")
print(df["label"].value_counts())

# ==============================
# SPLIT FEATURES / LABEL
# ==============================

X = df.drop("label", axis=1)

y = df["label"]

print("\nSo features:", X.shape[1])
print("So classes:", y.nunique())

# ==============================
# TRAIN / TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ==============================
# CREATE MODEL
# ==============================

print("\nDang train Random Forest...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

# ==============================
# TRAIN MODEL
# ==============================

model.fit(X_train, y_train)

print("Train hoan tat!")

# ==============================
# PREDICT
# ==============================

y_pred = model.predict(X_test)

# ==============================
# EVALUATION
# ==============================

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("KET QUA DANH GIA")
print("==============================")

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ==============================
# SAVE MODEL
# ==============================

os.makedirs("models", exist_ok=True)

joblib.dump(model, MODEL_PATH)

print("\n==============================")
print("DA LUU MODEL")
print("==============================")

print(MODEL_PATH)