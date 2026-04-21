import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------
# PATHS
# ---------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "behavior_final_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "behavior_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("Dataset Loaded Successfully")
print("Rows:", len(df))
print("Columns:", list(df.columns))

# ---------------------------------------------------
# FEATURES / LABEL
# ---------------------------------------------------

X = df.drop("fatigue_label", axis=1)
y = df["fatigue_label"]

# ---------------------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ---------------------------------------------------
# SCALE FEATURES
# ---------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Full dataset scaled for CV
X_scaled = scaler.transform(X)

# ---------------------------------------------------
# MODEL
# ---------------------------------------------------

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Model...\n")

model.fit(X_train_scaled, y_train)

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------

y_pred = model.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)

print("===================================")
print("RANDOM FOREST RESULTS")
print("===================================")
print("Accuracy:", round(acc * 100, 2), "%")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# ---------------------------------------------------
# CROSS VALIDATION
# ---------------------------------------------------

cv_scores = cross_val_score(
    model,
    X_scaled,
    y,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

print("\n5-Fold Cross Validation Scores:")
print(cv_scores)

print("Mean CV Accuracy:",
      round(cv_scores.mean() * 100, 2), "%")

print("Std Dev:",
      round(cv_scores.std() * 100, 2), "%")

# ---------------------------------------------------
# FEATURE IMPORTANCE
# ---------------------------------------------------

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\nTop Important Features:\n")
print(importance)

# ---------------------------------------------------
# SAVE MODEL
# ---------------------------------------------------

joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print("\nSaved model to:", MODEL_PATH)
print("Saved scaler to:", SCALER_PATH)