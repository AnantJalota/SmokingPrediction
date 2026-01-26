import streamlit as st
import pandas as pd
import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

from model import SmokerPrediction

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Smoking Prediction App",
    layout="wide"
)

st.title("Smoking Prediction – Model Evaluation Dashboard")

st.markdown(
    """
    This app allows evaluation of trained machine learning models on an uploaded
    **test dataset**. Predictions are generated using the same preprocessing,
    feature engineering, and probability thresholding as used during training.
    """
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
# Dataset upload
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload TEST dataset (CSV only)",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload a CSV file containing test data.")
    st.stop()

# Load dataset
test_df = pd.read_csv(uploaded_file)
st.success("Test dataset uploaded successfully")

# --------------------------------------------------
# Model selection
# --------------------------------------------------
model_name = st.selectbox(
    "Select a trained model",
    options=list(MODEL_REGISTRY.keys())
)

if st.button("Run Evaluation"):

    config = MODEL_REGISTRY[model_name]

    # --------------------------------------------------
    # Load artifact
    # --------------------------------------------------
    artifact = joblib.load(config["artifact_path"])

    model = artifact["model"]
    feature_list = artifact["features"]
    threshold = artifact.get("threshold", 0.5)

    # --------------------------------------------------
    # Feature engineering
    # --------------------------------------------------
    sp = SmokerPrediction(test_df, drop=False)

    feature_method = getattr(sp, config["feature_fn"])
    processed_df = feature_method()

    # --------------------------------------------------
    # Split X / y
    # --------------------------------------------------
    if "smoking" not in processed_df.columns:
        st.error("Target column 'smoking' not found after preprocessing.")
        st.stop()

    y_true = processed_df["smoking"].values
    X = processed_df.drop(columns=["smoking"])

    # Align features EXACTLY as during training
    X = X[feature_list]

    # --------------------------------------------------
    # Prediction + probability manipulation
    # --------------------------------------------------
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    else:
        # Fallback (should rarely happen)
        y_prob = model.predict(X)

    y_pred = (y_prob >= threshold).astype(int)

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Evaluation Metrics")
        st.write(f"Accuracy: **{accuracy_score(y_true, y_pred):.2f}**")
        st.write(f"AUC: **{roc_auc_score(y_true, y_pred):.2f}**")
        st.write(f"Precision: **{precision_score(y_true, y_pred):.2f}**")
        st.write(f"Recall: **{recall_score(y_true, y_pred):.2f}**")
        st.write(f"F1 Score: **{f1_score(y_true, y_pred):.2f}**")
        st.write(f"MCC: **{matthews_corrcoef(y_true, y_pred):.2f}**")

    with col2:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        st.dataframe(
            pd.DataFrame(
                cm,
                index=["Actual 0", "Actual 1"],
                columns=["Predicted 0", "Predicted 1"]
            )
        )

    # --------------------------------------------------
    # Classification report
    # --------------------------------------------------
    st.subheader("Classification Report")
    st.text(classification_report(y_true, y_pred))

    # --------------------------------------------------
    # Optional: probability inspection
    # --------------------------------------------------
    with st.expander("View prediction probabilities"):
        prob_df = pd.DataFrame({
            "Actual": y_true,
            "Predicted": y_pred,
            "Probability": y_prob
        })
        st.dataframe(prob_df.head(50))

