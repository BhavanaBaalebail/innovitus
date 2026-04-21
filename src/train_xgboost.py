import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from xgboost import XGBClassifier

# ---------------------------------------------------
# PATHS
# ---------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "behavior_final_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "behavior_xgb.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "xgb_scaler.pkl")

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

df = pd.read_csv(DATA_PATH)

X = df.drop("fatigue_label", axis=1)
y = df["fatigue_label"]

# ---------------------------------------------------
# SPLIT
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ---------------------------------------------------
# SCALE
# ---------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------
# MODEL
# ---------------------------------------------------

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    random_state=42
)

print("Training XGBoost...\n")
model.fit(X_train_scaled, y_train)

# ---------------------------------------------------
# TEST
# ---------------------------------------------------

y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

print("===================================")
print("XGBOOST RESULTS")
print("===================================")
print("Accuracy:", round(acc * 100, 2), "%")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# ---------------------------------------------------
# CROSS VALIDATION
# ---------------------------------------------------

scores = cross_val_score(
    model,
    scaler.transform(X),
    y,
    cv=5,
    scoring="accuracy"
)

print("\n5-Fold Cross Validation Scores:")
print(scores)
print("Mean CV Accuracy:", round(scores.mean() * 100, 2), "%")

# ---------------------------------------------------
# SAVE
# ---------------------------------------------------

joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print("\nSaved:", MODEL_PATH)