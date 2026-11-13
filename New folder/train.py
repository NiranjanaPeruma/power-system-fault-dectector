"""
train.py - Power System Fault Detection and Classification
==========================================================
Trains Random Forest and XGBoost classifiers on the fault dataset,
evaluates both, saves the best model, metrics, and plots.

Usage:
    python train.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # headless rendering
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

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
    print("WARNING: xgboost not installed. Skipping XGBoost training.")

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1.  CONFIGURATION
# ---------------------------------------------------------------------------
DATA_PATH     = r"c:\Users\NIRANJANA P\Downloads\archive\fault_data.csv"
OUT_DIR       = os.path.dirname(os.path.abspath(__file__))   # same folder as train.py
RANDOM_STATE  = 42
TEST_SIZE     = 0.20
CV_FOLDS      = 5

# ---------------------------------------------------------------------------
# 2.  LOAD DATA
# ---------------------------------------------------------------------------
print("=" * 60)
print("  Power System Fault Detection — Training Pipeline")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\n[DATA] Loaded CSV  →  shape: {df.shape}")
print(f"       Columns : {df.columns.tolist()}")

# ---------------------------------------------------------------------------
# 3.  INSPECT
# ---------------------------------------------------------------------------
print("\n[DATA] dtypes:")
print(df.dtypes.to_string())

print("\n[DATA] Missing values per column:")
print(df.isnull().sum().to_string())

# ---------------------------------------------------------------------------
# 4.  AUTO-DETECT TARGET COLUMN
# ---------------------------------------------------------------------------
TARGET_CANDIDATES = ["Fault Type", "fault_type", "fault type", "label", "class", "target",
                     "Condition", "fault condition"]
target_col = None
for c in TARGET_CANDIDATES:
    for col in df.columns:
        if col.strip().lower() == c.lower():
            target_col = col
            break
    if target_col:
        break

# Fallback: pick the column with fewest unique string values
if target_col is None:
    str_cols = df.select_dtypes(include="object").columns.tolist()
    if str_cols:
        target_col = min(str_cols, key=lambda c: df[c].nunique())

assert target_col is not None, "Could not detect target column automatically."
print(f"\n[DATA] Target column detected: '{target_col}'")

print("\n[DATA] Class distribution:")
dist = df[target_col].value_counts()
print(dist.to_string())

# ---------------------------------------------------------------------------
# 5.  CLEAN DATA
# ---------------------------------------------------------------------------
# Drop exact duplicates
before = len(df)
df = df.drop_duplicates()
print(f"\n[CLEAN] Dropped {before - len(df)} duplicate rows.")

# Drop non-informative ID / location columns (strings not useful as features)
DROP_PATTERNS = ["id", "location", "latitude", "longitude", "timestamp", "date"]
drop_cols = [
    c for c in df.columns
    if c != target_col
    and any(p in c.lower() for p in DROP_PATTERNS)
]
if drop_cols:
    df = df.drop(columns=drop_cols)
    print(f"[CLEAN] Dropped ID/location columns: {drop_cols}")

# Fill numeric NaN with median, categorical NaN with mode
for col in df.columns:
    if df[col].isnull().any():
        if df[col].dtype in [np.float64, np.int64, float, int]:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
print("[CLEAN] Filled missing values (numeric→median, categorical→mode).")

# Encode categorical features (one-hot, drop first to avoid dummy trap)
cat_cols = [c for c in df.columns if c != target_col and df[c].dtype == "object"]
if cat_cols:
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    print(f"[CLEAN] One-hot encoded categorical columns: {cat_cols}")

# Encode target
le = LabelEncoder()
df["__label__"] = le.fit_transform(df[target_col])
label_classes = le.classes_.tolist()
print(f"[CLEAN] Label classes: {label_classes}")

# ---------------------------------------------------------------------------
# 6.  LEAKAGE CHECK
# ---------------------------------------------------------------------------
y_raw = df["__label__"]
feature_cols = [c for c in df.columns if c not in [target_col, "__label__"]]

leaky_cols = []
for col in feature_cols:
    # correlation of numeric column with encoded label
    try:
        corr = abs(np.corrcoef(df[col].astype(float), y_raw)[0, 1])
        if corr > 0.999:
            leaky_cols.append(col)
    except Exception:
        pass

if leaky_cols:
    print(f"\n[LEAKAGE] WARNING — dropping near-perfect predictors: {leaky_cols}")
    df = df.drop(columns=leaky_cols)
    feature_cols = [c for c in feature_cols if c not in leaky_cols]
else:
    print("[LEAKAGE] No leakage detected.")

# ---------------------------------------------------------------------------
# 7.  PREPARE X / y
# ---------------------------------------------------------------------------
X = df[feature_cols].astype(float)
y = df["__label__"].values

print(f"\n[PREP] Feature matrix: {X.shape}  |  Classes: {label_classes}")

# ---------------------------------------------------------------------------
# 8.  TRAIN-TEST SPLIT
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"[SPLIT] Train: {X_train.shape}  |  Test: {X_test.shape}")

# Check class balance
class_counts = pd.Series(y_train).value_counts()
imbalanced = (class_counts.max() / class_counts.min()) > 1.5
print(f"[SPLIT] Imbalanced classes: {imbalanced}")
sample_weights = compute_sample_weight("balanced", y_train) if imbalanced else None

# ---------------------------------------------------------------------------
# 9.  TRAIN MODELS
# ---------------------------------------------------------------------------
results = {}

# --- Random Forest ---
print("\n[TRAIN] Training Random Forest ...")
rf_kw = dict(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
rf = RandomForestClassifier(**rf_kw)
rf.fit(X_train, y_train, sample_weight=sample_weights)
rf_pred = rf.predict(X_test)
results["Random Forest"] = {
    "model": rf,
    "preds": rf_pred,
    "acc":   accuracy_score(y_test, rf_pred),
    "prec":  precision_score(y_test, rf_pred, average="macro", zero_division=0),
    "rec":   recall_score(y_test, rf_pred, average="macro", zero_division=0),
    "f1":    f1_score(y_test, rf_pred, average="macro", zero_division=0),
    "report": classification_report(y_test, rf_pred, target_names=label_classes)
}
print(f"  RF  Accuracy={results['Random Forest']['acc']:.4f}  Macro-F1={results['Random Forest']['f1']:.4f}")

# --- XGBoost ---
if XGBOOST_AVAILABLE:
    print("[TRAIN] Training XGBoost ...")
    xgb_kw = dict(
        n_estimators=200, random_state=RANDOM_STATE,
        eval_metric="mlogloss", use_label_encoder=False,
        verbosity=0, n_jobs=-1
    )
    if imbalanced:
        xgb_kw["scale_pos_weight"] = 1   # multi-class: use sample_weight
    xgb = XGBClassifier(**xgb_kw)
    xgb.fit(X_train, y_train, sample_weight=sample_weights)
    xgb_pred = xgb.predict(X_test)
    results["XGBoost"] = {
        "model": xgb,
        "preds": xgb_pred,
        "acc":   accuracy_score(y_test, xgb_pred),
        "prec":  precision_score(y_test, xgb_pred, average="macro", zero_division=0),
        "rec":   recall_score(y_test, xgb_pred, average="macro", zero_division=0),
        "f1":    f1_score(y_test, xgb_pred, average="macro", zero_division=0),
        "report": classification_report(y_test, xgb_pred, target_names=label_classes)
    }
    print(f"  XGB Accuracy={results['XGBoost']['acc']:.4f}  Macro-F1={results['XGBoost']['f1']:.4f}")

# ---------------------------------------------------------------------------
# 10.  SELECT BEST MODEL
# ---------------------------------------------------------------------------
best_name = max(results, key=lambda k: results[k]["f1"])
best_result = results[best_name]
best_model  = best_result["model"]
print(f"\n[SELECT] Best model: {best_name}  (Macro-F1={best_result['f1']:.4f})")

# ---------------------------------------------------------------------------
# 11.  CROSS-VALIDATION ON BEST MODEL
# ---------------------------------------------------------------------------
print(f"\n[CV] {CV_FOLDS}-fold cross-validation on {best_name} ...")
cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_validate(
    best_model, X, y,
    cv=cv,
    scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"},
    n_jobs=-1
)
cv_acc_mean = cv_scores["test_accuracy"].mean()
cv_acc_std  = cv_scores["test_accuracy"].std()
cv_f1_mean  = cv_scores["test_f1_macro"].mean()
cv_f1_std   = cv_scores["test_f1_macro"].std()
print(f"  CV Accuracy : {cv_acc_mean:.4f} ± {cv_acc_std:.4f}")
print(f"  CV Macro-F1 : {cv_f1_mean:.4f} ± {cv_f1_std:.4f}")

# ---------------------------------------------------------------------------
# 12.  CLASSIFICATION REPORTS
# ---------------------------------------------------------------------------
print("\n[EVAL] Classification Reports")
for name, res in results.items():
    print(f"\n--- {name} ---")
    print(res["report"])

# ---------------------------------------------------------------------------
# 13.  SAVE MODEL ARTIFACTS
# ---------------------------------------------------------------------------
# model.pkl
joblib.dump(best_model, os.path.join(OUT_DIR, "model.pkl"))
print("[SAVE] model.pkl")

# features.json  — ordered feature names with min, max, mean
feature_stats = {
    col: {
        "min":  float(X[col].min()),
        "max":  float(X[col].max()),
        "mean": float(X[col].mean())
    }
    for col in feature_cols
}
with open(os.path.join(OUT_DIR, "features.json"), "w") as f:
    json.dump(feature_stats, f, indent=2)
print("[SAVE] features.json")

# label_classes.json
with open(os.path.join(OUT_DIR, "label_classes.json"), "w") as f:
    json.dump(label_classes, f, indent=2)
print("[SAVE] label_classes.json")

# metrics.json — all metrics for all models + CV
metrics_out = {}
for name, res in results.items():
    metrics_out[name] = {
        "accuracy":  round(res["acc"], 4),
        "precision": round(res["prec"], 4),
        "recall":    round(res["rec"], 4),
        "macro_f1":  round(res["f1"], 4),
    }
metrics_out["best_model"] = best_name
metrics_out["cross_validation"] = {
    "model":     best_name,
    "folds":     CV_FOLDS,
    "accuracy":  {"mean": round(cv_acc_mean, 4), "std": round(cv_acc_std, 4)},
    "macro_f1":  {"mean": round(cv_f1_mean, 4),  "std": round(cv_f1_std, 4)},
}
with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics_out, f, indent=2)
print("[SAVE] metrics.json")

# ---------------------------------------------------------------------------
# 14.  PLOTS
# ---------------------------------------------------------------------------

# --- class_distribution.png ---
fig, ax = plt.subplots(figsize=(7, 4))
dist.plot(kind="bar", ax=ax, color="#4C72B0", edgecolor="black")
ax.set_title("Class Distribution of Fault Types", fontsize=13)
ax.set_xlabel("Fault Type")
ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=20)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "class_distribution.png"), dpi=150)
plt.close(fig)
print("[PLOT] class_distribution.png")

# --- confusion_matrix.png ---
cm = confusion_matrix(y_test, best_result["preds"])
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=label_classes, yticklabels=label_classes, ax=ax
)
ax.set_title(f"Confusion Matrix — {best_name}", fontsize=13)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150)
plt.close(fig)
print("[PLOT] confusion_matrix.png")

# --- feature_importance.png ---
try:
    importances = best_model.feature_importances_
    fi = pd.Series(importances, index=feature_cols).sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(8, 6))
    fi.plot(kind="barh", ax=ax, color="#DD8452", edgecolor="black")
    ax.invert_yaxis()
    ax.set_title(f"Feature Importances — {best_name} (Top 20)", fontsize=13)
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "feature_importance.png"), dpi=150)
    plt.close(fig)
    print("[PLOT] feature_importance.png")
except AttributeError:
    print("[PLOT] feature_importance.png — skipped (model has no feature_importances_)")

# ---------------------------------------------------------------------------
# 15.  FINAL SUMMARY
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  FINAL SUMMARY")
print("=" * 60)
print(f"  Dataset         : {DATA_PATH}")
print(f"  Rows / Features : {df.shape[0]} rows  |  {len(feature_cols)} features")
print(f"  Target column   : '{target_col}'  ({len(label_classes)} classes)")
print(f"  Best model      : {best_name}")
print(f"  Test Accuracy   : {best_result['acc']:.4f}")
print(f"  Test Precision  : {best_result['prec']:.4f}")
print(f"  Test Recall     : {best_result['rec']:.4f}")
print(f"  Test Macro-F1   : {best_result['f1']:.4f}")
print(f"  CV Accuracy     : {cv_acc_mean:.4f} ± {cv_acc_std:.4f}")
print(f"  CV Macro-F1     : {cv_f1_mean:.4f}  ± {cv_f1_std:.4f}")
print("\n  Saved files:")
for fn in ["model.pkl", "features.json", "label_classes.json", "metrics.json",
           "confusion_matrix.png", "feature_importance.png", "class_distribution.png"]:
    fp = os.path.join(OUT_DIR, fn)
    exists = "✓" if os.path.exists(fp) else "✗"
    print(f"    {exists}  {fn}")
print("=" * 60)
