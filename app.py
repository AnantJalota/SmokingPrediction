import streamlit as st
import pandas as pd
import joblib
import requests
import matplotlib.pyplot as plt

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
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.weight": "normal"
})
# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Smoking Prediction App",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Smoking Prediction – Model Evaluation Dashboard")
st.markdown("Evaluate trained machine learning models on unseen test data.")

# --------------------------------------------------
# GitHub dataset download
# --------------------------------------------------
GITHUB_RAW_URL = "https://raw.githubusercontent.com/AnantJalota/SmokingPrediction/main/Smoking_test.csv"

@st.cache_data
def fetch_github_csv(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.content

st.sidebar.header("Dataset Options")

try:
    csv_data = fetch_github_csv(GITHUB_RAW_URL)
    st.sidebar.download_button(
        label="Click to Download CSV",
        data=csv_data,
        file_name="Smoking_test.csv",
        mime="text/csv"
    )
except Exception as e:
    st.sidebar.error(f"Error fetching file: {e}")

uploaded_file = st.sidebar.file_uploader(
    "Upload TEST dataset (CSV only)",
    type=["csv"]
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

    # Confusion Matrix (Matplotlib only)
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)

    fig_cm, ax = plt.subplots(figsize=(2.5, 2.5), dpi=150)

    im = ax.imshow(cm)
    
    # Smaller, softer tick labels
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Non Smoker", "Smoker"], fontsize=5)
    
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Non Smoker", "Smoker"], fontsize=5)
    
    # Softer axis labels
    ax.set_xlabel("Predicted", fontsize=5)
    ax.set_ylabel("Actual", fontsize=5)
    
    # Subtle title
    ax.set_title("Confusion Matrix", fontsize=7)
    
    # Smaller, non-bold cell values
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, cm[i, j],
                ha="center",
                va="center",
                fontsize=9,
                fontweight="normal"
            )
    
    # Remove thick border effect
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    st.pyplot(fig_cm, use_container_width=False)
   
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







