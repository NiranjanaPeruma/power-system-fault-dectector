"""
app.py — Power System Fault Detection  |  Streamlit Web App
============================================================
Run with:
    streamlit run app.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# PATHS  (same directory as app.py)
# ---------------------------------------------------------------------------
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH     = os.path.join(BASE_DIR, "model.pkl")
FEATURES_PATH  = os.path.join(BASE_DIR, "features.json")
CLASSES_PATH   = os.path.join(BASE_DIR, "label_classes.json")
METRICS_PATH   = os.path.join(BASE_DIR, "metrics.json")
CM_PATH        = os.path.join(BASE_DIR, "confusion_matrix.png")
FI_PATH        = os.path.join(BASE_DIR, "feature_importance.png")
CD_PATH        = os.path.join(BASE_DIR, "class_distribution.png")

# ---------------------------------------------------------------------------
# PLAIN-ENGLISH FAULT DESCRIPTIONS
# Note: these are matched case-insensitively against label names.
# ---------------------------------------------------------------------------
FAULT_DESCRIPTIONS = {
    "line breakage": (
        "**Line Breakage** — A physical break or disconnection in a transmission or "
        "distribution conductor. Causes a sudden drop in current and voltage on the "
        "affected phase, and can create hazardous dangling wires."
    ),
    "transformer failure": (
        "**Transformer Failure** — Breakdown of insulation, windings, or oil inside a "
        "power transformer. Symptoms include abnormal current draw, overheating, and "
        "loss of voltage on the secondary side."
    ),
    "overheating": (
        "**Overheating** — Excess heat build-up in conductors, transformers, or "
        "switchgear due to overcurrent, poor ventilation, or ambient temperature spikes. "
        "Prolonged overheating accelerates insulation degradation and can lead to fire."
    ),
    "line-to-ground": (
        "**Line-to-Ground (L-G) Fault** — A single conductor makes unintended contact "
        "with the ground. The most common fault type (~70 % of all faults). Causes "
        "high zero-sequence current."
    ),
    "line-to-line": (
        "**Line-to-Line (L-L) Fault** — Two conductors make contact with each other "
        "without involving ground. Causes negative-sequence currents and voltage "
        "imbalance."
    ),
    "three-phase": (
        "**Three-Phase (L-L-L) Fault** — All three conductors short together. The "
        "most severe fault type; produces the highest fault current and is the "
        "basis for worst-case protection system design."
    ),
    "double line-to-ground": (
        "**Double Line-to-Ground (DL-G) Fault** — Two conductors short to each "
        "other and to ground simultaneously. Combines characteristics of L-L and L-G "
        "faults."
    ),
    "normal": (
        "**Normal** — No fault condition; system is operating within design parameters."
    ),
    "anomaly": (
        "**Anomaly** — An abnormal operating condition that doesn't fit a specific "
        "fault category but deviates from normal behaviour."
    ),
}


def get_fault_description(label: str) -> str:
    """Return a plain-English description for the fault label, or a generic one."""
    key = label.lower().strip()
    for k, v in FAULT_DESCRIPTIONS.items():
        if k in key or key in k:
            return v
    return f"**{label}** — A power system fault condition detected by the model."


# ---------------------------------------------------------------------------
# LOAD ARTIFACTS  (cached so they are only loaded once)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    errors = []

    model = None
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
        except Exception as e:
            errors.append(f"Could not load model.pkl: {e}")
    else:
        errors.append("model.pkl not found. Please run `python train.py` first.")

    features = {}
    if os.path.exists(FEATURES_PATH):
        with open(FEATURES_PATH) as f:
            features = json.load(f)
    else:
        errors.append("features.json not found.")

    classes = []
    if os.path.exists(CLASSES_PATH):
        with open(CLASSES_PATH) as f:
            classes = json.load(f)
    else:
        errors.append("label_classes.json not found.")

    metrics = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            metrics = json.load(f)

    return model, features, classes, metrics, errors


# ---------------------------------------------------------------------------
# APP LAYOUT
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Power System Fault Detector",
    page_icon="⚡",
    layout="wide",
)

# Sidebar
with st.sidebar:
    st.title("⚡ About This Project")
    st.markdown(
        """
        **Power System Fault Detection**  
        uses machine learning to classify electrical faults
        from sensor measurements.

        **Dataset**  
        Kaggle *Power System Faults* dataset — 506 records of
        electrical measurements (voltage, current, power load,
        temperature, wind speed, weather, maintenance and
        component health status).

        **Model**  
        Random Forest & XGBoost trained with stratified splits
        and class-balanced weights. Best model selected by
        macro F1 score.

        **Fault Classes**  
        - Line Breakage  
        - Transformer Failure  
        - Overheating  

        **How to run**  
        ```
        pip install -r requirements.txt
        python train.py          # train & save model
        streamlit run app.py     # launch this app
        ```
        """
    )
    st.divider()
    st.caption("Built with scikit-learn · XGBoost · Streamlit")

st.title("⚡ Power System Fault Detection & Classification")
st.markdown(
    "Upload sensor readings or enter values manually to detect and classify "
    "power system faults using a trained machine-learning model."
)

# Load artifacts
model, features, classes, metrics, load_errors = load_artifacts()

if load_errors:
    for err in load_errors:
        st.error(err)
    st.stop()

# -----------------------------------------------------------------------
# TABS
# -----------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    ["🔍 Single Prediction", "📂 Batch Prediction", "📊 Model Performance"]
)

# =======================================================================
# TAB 1 — Single Prediction
# =======================================================================
with tab1:
    st.subheader("Single Prediction")
    st.markdown(
        "Enter sensor measurements below and click **Predict** to see the "
        "classified fault type."
    )

    # Build input widgets dynamically from features.json
    input_values = {}
    cols_per_row = 3
    feature_list = list(features.keys())
    rows = [feature_list[i:i+cols_per_row] for i in range(0, len(feature_list), cols_per_row)]

    for row in rows:
        cols = st.columns(len(row))
        for col_widget, feat in zip(cols, row):
            stats = features[feat]
            val = col_widget.number_input(
                label=feat,
                min_value=float(stats["min"]),
                max_value=float(stats["max"]),
                value=float(round(stats["mean"], 4)),
                step=float(max((stats["max"] - stats["min"]) / 100, 0.001)),
                format="%.4f",
                key=f"single_{feat}",
            )
            input_values[feat] = val

    predict_btn = st.button("⚡ Predict Fault Type", type="primary")

    if predict_btn:
        input_df = pd.DataFrame([input_values])
        # Align columns to model's expected order
        input_df = input_df[feature_list]

        try:
            pred_idx   = model.predict(input_df)[0]
            pred_label = classes[pred_idx]
            proba      = model.predict_proba(input_df)[0]
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

        st.markdown("---")
        conf = float(proba[pred_idx]) * 100

        if conf >= 75:
            st.success(f"### ⚡ Predicted Fault: **{pred_label}**  ({conf:.1f}% confidence)")
        elif conf >= 50:
            st.warning(f"### ⚡ Predicted Fault: **{pred_label}**  ({conf:.1f}% confidence)")
        else:
            st.info(f"### ⚡ Predicted Fault: **{pred_label}**  ({conf:.1f}% confidence)")

        # Probability bar chart
        fig = go.Figure(go.Bar(
            x=classes,
            y=[float(p) * 100 for p in proba],
            marker_color=["#3b82f6" if c == pred_label else "#94a3b8" for c in classes],
            text=[f"{p*100:.1f}%" for p in proba],
            textposition="outside",
        ))
        fig.update_layout(
            title="Prediction Probabilities (%)",
            yaxis_title="Probability (%)",
            xaxis_title="Fault Class",
            yaxis_range=[0, 115],
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Plain-English description
        st.markdown("---")
        st.markdown("**What does this mean?**")
        st.markdown(get_fault_description(pred_label))


# =======================================================================
# TAB 2 — Batch Prediction
# =======================================================================
with tab2:
    st.subheader("Batch Prediction")
    st.markdown(
        "Upload a CSV file with the same feature columns as the training data. "
        "The app will add a **Predicted Fault** column and let you download the results."
    )

    with st.expander("Required columns"):
        st.write(list(features.keys()))

    uploaded = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

        # Validate columns
        missing_cols = [c for c in features.keys() if c not in batch_df.columns]
        if missing_cols:
            st.error(
                f"The uploaded file is missing {len(missing_cols)} required column(s): "
                f"{missing_cols}"
            )
            st.stop()

        # Predict
        try:
            X_batch = batch_df[list(features.keys())].astype(float)
            preds   = model.predict(X_batch)
            probas  = model.predict_proba(X_batch)
        except Exception as e:
            st.error(f"Batch prediction error: {e}")
            st.stop()

        batch_df["Predicted Fault"]  = [classes[p] for p in preds]
        batch_df["Confidence (%)"]   = (probas.max(axis=1) * 100).round(2)

        st.success(f"Predictions complete for {len(batch_df)} rows.")
        st.dataframe(batch_df, use_container_width=True)

        # Fault distribution of predictions
        pred_dist = batch_df["Predicted Fault"].value_counts()
        st.bar_chart(pred_dist)

        # Download button
        csv_out = batch_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇ Download Results CSV",
            data=csv_out,
            file_name="fault_predictions.csv",
            mime="text/csv",
        )


# =======================================================================
# TAB 3 — Model Performance
# =======================================================================
with tab3:
    st.subheader("Model Performance")

    if metrics:
        st.markdown("#### Metrics on Hold-Out Test Set (80/20 split)")

        # Build a tidy table
        table_rows = []
        for model_name, vals in metrics.items():
            if isinstance(vals, dict) and "accuracy" in vals and "macro_f1" in vals:
                row = {
                    "Model":     model_name,
                    "Accuracy":  vals.get("accuracy", "—"),
                    "Precision": vals.get("precision", "—"),
                    "Recall":    vals.get("recall", "—"),
                    "Macro F1":  vals.get("macro_f1", "—"),
                }
                table_rows.append(row)

        if table_rows:
            metrics_df = pd.DataFrame(table_rows).set_index("Model")
            st.dataframe(
                metrics_df.style.format(lambda v: f"{v:.4f}" if isinstance(v, (int, float)) else v),
                use_container_width=True,
            )

        # Cross-validation
        cv = metrics.get("cross_validation", {})
        if cv:
            st.markdown(
                f"#### {cv['folds']}-Fold Cross-Validation — {cv['model']}"
            )
            cv_df = pd.DataFrame({
                "Metric":    ["Accuracy", "Macro F1"],
                "Mean":      [cv["accuracy"]["mean"], cv["macro_f1"]["mean"]],
                "Std Dev":   [cv["accuracy"]["std"],  cv["macro_f1"]["std"]],
            }).set_index("Metric")
            st.dataframe(
                cv_df.style.format(lambda v: f"{v:.4f}" if isinstance(v, (int, float)) else v),
                use_container_width=True,
            )

        st.markdown(f"**Best model selected:** `{metrics.get('best_model', 'N/A')}`")

    # Plots
    st.markdown("---")
    st.markdown("#### Saved Training Plots")

    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists(CM_PATH):
            st.image(CM_PATH, caption="Confusion Matrix", width=700)
        else:
            st.info("confusion_matrix.png not found. Run train.py first.")
    with col2:
        if os.path.exists(FI_PATH):
            st.image(FI_PATH, caption="Feature Importance", width=700)
        else:
            st.info("feature_importance.png not found. Run train.py first.")

    if os.path.exists(CD_PATH):
        st.image(CD_PATH, caption="Class Distribution", width=1000)
    else:
        st.info("class_distribution.png not found. Run train.py first.")
