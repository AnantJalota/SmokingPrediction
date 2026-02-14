
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_curve
)

from model import SmokerPrediction

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Smoking Prediction App",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Smoking Prediction – Model Evaluation Dashboard")
st.markdown(
    "Evaluate trained machine learning models on unseen test data."
)

# --------------------------------------------------
# Model registry
# --------------------------------------------------
MODEL_REGISTRY = {
    "Random Forest": {
        "artifact_path": "models/smoker_model_rf.pkl",
        "feature_fn": "create_features"
    },
    "XGBoost": {
        "artifact_path": "models/smoker_model_xgb.pkl",
        "feature_fn": "create_features"
    },
    "Logistic Regression": {
        "artifact_path": "models/smoker_model_lr.pkl",
        "feature_fn": "create_basic_features"
    },
    "Decision Tree": {
        "artifact_path": "models/smoker_model_dt.pkl",
        "feature_fn": "decision_features"
    },
    "Naive Bayes": {
        "artifact_path": "models/smoker_model_nb.pkl",
        "feature_fn": "create_basic_features"
    },
    "KNN": {
        "artifact_path": "models/smoker_model_knn.pkl",
        "feature_fn": "load_and_preprocess"
    }
}

# --------------------------------------------------
# Sidebar Controls
# --------------------------------------------------
st.sidebar.header("Configuration")

uploaded_file = st.sidebar.file_uploader(
    "Upload TEST dataset (CSV only)",
    type=["csv"]
)

model_name = st.sidebar.selectbox(
    "Select trained model",
    options=list(MODEL_REGISTRY.keys())
)

run_button = st.sidebar.button("Run Evaluation")

if uploaded_file is None:
    st.warning("Upload a test dataset to begin evaluation.")
    st.stop()

test_df = pd.read_csv(uploaded_file)
st.success(f"Dataset loaded — {test_df.shape[0]} rows, {test_df.shape[1]} columns")

if run_button:

    config = MODEL_REGISTRY[model_name]

    artifact = joblib.load(config["artifact_path"])
    model = artifact["model"]
    feature_list = artifact["features"]
    threshold = artifact.get("threshold", 0.5)

    # Feature engineering
    sp = SmokerPrediction(test_df, drop=False)
    feature_method = getattr(sp, config["feature_fn"])
    processed_df = feature_method()

    if "smoking" not in processed_df.columns:
        st.error("Target column 'smoking' not found after preprocessing.")
        st.stop()

    y_true = processed_df["smoking"].values
    X = processed_df.drop(columns=["smoking"])
    X = X[feature_list]

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    else:
        y_prob = model.predict(X)

    y_pred = (y_prob >= threshold).astype(int)

    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    st.header("Model Performance")

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Accuracy", f"{accuracy:.3f}")
    m2.metric("AUC", f"{auc:.3f}")
    m3.metric("Precision", f"{precision:.3f}")
    m4.metric("Recall", f"{recall:.3f}")
    m5.metric("F1 Score", f"{f1:.3f}")
    m6.metric("MCC", f"{mcc:.3f}")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)

    fig_cm, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Non Smoker", "Smoker"],
        yticklabels=["Non Smoker", "Smoker"],
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig_cm)

    # ROC Curve
    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    fig_roc, ax = plt.subplots()
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    st.pyplot(fig_roc)

    # Classification Report
    with st.expander("Detailed Classification Report"):
        report_df = pd.DataFrame(
            classification_report(
                y_true,
                y_pred,
                target_names=["Non Smoker", "Smoker"],
                output_dict=True
            )
        ).T
        st.dataframe(report_df.style.format("{:.3f}"))
