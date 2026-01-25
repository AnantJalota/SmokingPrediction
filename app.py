import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

from model import SmokerPrediction

st.set_page_config(page_title="Smoking Prediction", layout="wide")
st.title("Smoking Prediction – Model Evaluation App")

# ---------------------------
# Model registry
# ---------------------------
MODEL_REGISTRY = {
    "XGBoost": {
        "artifact": "smoker_model_xgb.pkl",
        "feature_fn": "create_features"
    },
    "Random Forest": {
        "artifact": "smoker_model_rf.pkl",
        "feature_fn": "create_features"
    },
    "Logistic Regression": {
        "artifact": "smoker_model_lr.pkl",
        "feature_fn": "create_basic_features"
    },
    "Decision Tree": {
        "artifact": "smoker_model_dt.pkl",
        "feature_fn": "decision_features"
    },
    "Naive Bayes": {
        "artifact": "smoker_model_nb.pkl",
        "feature_fn": "create_basic_features"
    },
    "KNN": {
        "artifact": "smoker_model_knn.pkl",
        "feature_fn": "load_and_preprocess"
    }
}

# ---------------------------
# Upload test data
# ---------------------------
uploaded_file = st.file_uploader("Upload test dataset (CSV)", type=["csv"])

if uploaded_file:
    test_df = pd.read_csv(uploaded_file)
    st.success("Dataset uploaded successfully")

    model_name = st.selectbox("Select Model", MODEL_REGISTRY.keys())

    if st.button("Run Evaluation"):
        config = MODEL_REGISTRY[model_name]

        # Load artifact
        with open(config["artifact"], "rb") as f:
            artifact = pickle.load(f)

        model = artifact["model"]
        threshold = artifact.get("threshold", 0.5)

        # Feature engineering
        sp = SmokerPrediction(test_df, drop=False)
        df_feat = getattr(sp, config["feature_fn"])()

        # Split X / y
        y_true = df_feat["smoking"]
        X = df_feat.drop("smoking", axis=1)

        # Align columns (VERY IMPORTANT)
        required_cols = artifact.get("features") or artifact.get("columns")
        X = X[required_cols]

        # Prediction + thresholding
        y_proba = model.predict_proba(X)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        # Metrics
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Evaluation Metrics")
            st.write(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
            st.write(f"Precision: {precision_score(y_true, y_pred):.4f}")
            st.write(f"Recall: {recall_score(y_true, y_pred):.4f}")
            st.write(f"F1 Score: {f1_score(y_true, y_pred):.4f}")
            st.write(f"MCC: {matthews_corrcoef(y_true, y_pred):.4f}")

        with col2:
            st.subheader("Confusion Matrix")
            st.write(confusion_matrix(y_true, y_pred))

        st.subheader("Classification Report")
        st.text(classification_report(y_true, y_pred))
