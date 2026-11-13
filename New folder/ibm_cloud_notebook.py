"""
ibm_cloud_notebook.py
=====================
Power System Fault Detection — Watson Studio / Jupyter notebook version.
Paste each cell between "# --- CELL ---" markers into a separate notebook cell.

Requirements (available in Watson Studio default Python 3.10 kernel):
    pandas, numpy, scikit-learn, matplotlib, xgboost (optional)
"""

# --- CELL ---
# Cell 1: Imports and configuration
# -----------------------------------
import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from sklearn.utils.class_weight import compute_sample_weight

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available — will train Random Forest only.")

warnings.filterwarnings("ignore")

# IMPORTANT: Update DATA_PATH to match where you uploaded the CSV
# in your Watson Studio project storage.
DATA_PATH    = "fault_data.csv"   # or "/home/wsuser/work/fault_data.csv"
RANDOM_STATE = 42
TEST_SIZE    = 0.20
CV_FOLDS     = 5

print("Imports complete.")

# --- CELL ---
# Cell 2: Load and inspect data
# -----------------------------------
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nDtypes:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())

# --- CELL ---
# Cell 3: Detect target column and show class distribution
# -----------------------------------
TARGET_CANDIDATES = ["Fault Type", "fault_type", "fault type", "label", "class", "target"]
target_col = None
for candidate in TARGET_CANDIDATES:
    for col in df.columns:
        if col.strip().lower() == candidate.lower():
            target_col = col
            break
    if target_col:
        break

if target_col is None:
    str_cols = df.select_dtypes(include="object").columns.tolist()
    target_col = min(str_cols, key=lambda c: df[c].nunique())

print(f"Target column: '{target_col}'")
dist = df[target_col].value_counts()
print("\nClass distribution:")
print(dist)

dist.plot(kind="bar", color="#4C72B0", edgecolor="black")
plt.title("Fault Type Distribution")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=150)
plt.show()

# --- CELL ---
# Cell 4: Clean data
# -----------------------------------
# Drop duplicates
before = len(df)
df = df.drop_duplicates()
print(f"Dropped {before - len(df)} duplicate rows.")

# Drop ID / location columns
DROP_PATTERNS = ["id", "location", "latitude", "longitude", "timestamp", "date"]
drop_cols = [c for c in df.columns
             if c != target_col and any(p in c.lower() for p in DROP_PATTERNS)]
if drop_cols:
    df = df.drop(columns=drop_cols)
    print(f"Dropped non-feature columns: {drop_cols}")

# Fill missing values
for col in df.columns:
    if df[col].isnull().any():
        if df[col].dtype in [np.float64, np.int64]:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
print("Missing values filled.")

# One-hot encode categorical features
cat_cols = [c for c in df.columns if c != target_col and df[c].dtype == "object"]
if cat_cols:
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    print(f"One-hot encoded: {cat_cols}")

# Encode target
le = LabelEncoder()
df["__label__"] = le.fit_transform(df[target_col])
label_classes = le.classes_.tolist()
print(f"Classes: {label_classes}")

# --- CELL ---
# Cell 5: Leakage check
# -----------------------------------
y_raw = df["__label__"]
feature_cols = [c for c in df.columns if c not in [target_col, "__label__"]]

leaky_cols = []
for col in feature_cols:
    try:
        corr = abs(np.corrcoef(df[col].astype(float), y_raw)[0, 1])
        if corr > 0.999:
            leaky_cols.append(col)
    except Exception:
        pass

if leaky_cols:
    print(f"WARNING — dropping leaky columns: {leaky_cols}")
    df = df.drop(columns=leaky_cols)
    feature_cols = [c for c in feature_cols if c not in leaky_cols]
else:
    print("No data leakage detected.")

X = df[feature_cols].astype(float)
y = df["__label__"].values
print(f"Feature matrix: {X.shape}  |  Classes: {label_classes}")

# --- CELL ---
# Cell 6: Train / test split
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

class_counts = pd.Series(y_train).value_counts()
imbalanced = (class_counts.max() / class_counts.min()) > 1.5
sample_weights = compute_sample_weight("balanced", y_train) if imbalanced else None
print(f"Imbalanced: {imbalanced}")

# --- CELL ---
# Cell 7: Train Random Forest
# -----------------------------------
rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train, y_train, sample_weight=sample_weights)
rf_pred = rf.predict(X_test)

rf_acc = accuracy_score(y_test, rf_pred)
rf_f1  = f1_score(y_test, rf_pred, average="macro", zero_division=0)
print(f"Random Forest — Accuracy: {rf_acc:.4f}  Macro-F1: {rf_f1:.4f}")
print(classification_report(y_test, rf_pred, target_names=label_classes))

# --- CELL ---
# Cell 8: Train XGBoost (if available)
# -----------------------------------
if XGBOOST_AVAILABLE:
    xgb = XGBClassifier(
        n_estimators=200, random_state=RANDOM_STATE,
        eval_metric="mlogloss", use_label_encoder=False,
        verbosity=0, n_jobs=-1
    )
    xgb.fit(X_train, y_train, sample_weight=sample_weights)
    xgb_pred = xgb.predict(X_test)
    xgb_acc  = accuracy_score(y_test, xgb_pred)
    xgb_f1   = f1_score(y_test, xgb_pred, average="macro", zero_division=0)
    print(f"XGBoost — Accuracy: {xgb_acc:.4f}  Macro-F1: {xgb_f1:.4f}")
    print(classification_report(y_test, xgb_pred, target_names=label_classes))

    # Pick best model
    if xgb_f1 >= rf_f1:
        best_model, best_name, best_pred = xgb, "XGBoost", xgb_pred
    else:
        best_model, best_name, best_pred = rf, "Random Forest", rf_pred
else:
    best_model, best_name, best_pred = rf, "Random Forest", rf_pred

print(f"\nBest model: {best_name}")

# --- CELL ---
# Cell 9: Cross-validation
# -----------------------------------
cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_validate(
    best_model, X, y,
    cv=cv,
    scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"},
    n_jobs=-1,
)
print(f"{CV_FOLDS}-fold CV on {best_name}")
print(f"  Accuracy : {cv_scores['test_accuracy'].mean():.4f} ± {cv_scores['test_accuracy'].std():.4f}")
print(f"  Macro-F1 : {cv_scores['test_f1_macro'].mean():.4f}  ± {cv_scores['test_f1_macro'].std():.4f}")

# --- CELL ---
# Cell 10: Confusion matrix plot
# -----------------------------------
import seaborn as sns

cm = confusion_matrix(y_test, best_pred)
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_classes, yticklabels=label_classes, ax=ax)
ax.set_title(f"Confusion Matrix — {best_name}")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

# --- CELL ---
# Cell 11: Feature importance plot
# -----------------------------------
importances = best_model.feature_importances_
fi = pd.Series(importances, index=feature_cols).sort_values(ascending=False).head(20)
fi.plot(kind="barh", color="#DD8452", edgecolor="black", figsize=(8, 6))
plt.gca().invert_yaxis()
plt.title(f"Feature Importances — {best_name}")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()

# --- CELL ---
# Cell 12: Save model and metadata
# -----------------------------------
import joblib

joblib.dump(best_model, "model.pkl")
print("Saved model.pkl")

feature_stats = {
    col: {"min": float(X[col].min()), "max": float(X[col].max()), "mean": float(X[col].mean())}
    for col in feature_cols
}
with open("features.json", "w") as f:
    json.dump(feature_stats, f, indent=2)
print("Saved features.json")

with open("label_classes.json", "w") as f:
    json.dump(label_classes, f, indent=2)
print("Saved label_classes.json")

# --- CELL ---
# Cell 13: Quick sanity-check prediction
# -----------------------------------
sample = X_test[:1]
pred_idx   = best_model.predict(sample)[0]
pred_label = label_classes[pred_idx]
pred_proba = best_model.predict_proba(sample)[0]
print(f"Sample prediction: {pred_label}  (confidence: {pred_proba[pred_idx]*100:.1f}%)")
print("All probabilities:", dict(zip(label_classes, [round(float(p), 4) for p in pred_proba])))
print("\nPipeline complete.")
