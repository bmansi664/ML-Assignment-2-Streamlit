import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

# Page Configuration
st.set_page_config(
    page_title="Classification Model Benchmark Dashboard",
    page_icon="🧠",
    layout="wide"
)

def load_evaluation_dataset(file_buffer):
    """Parse uploaded CSV file and separate features from target labels."""
    dataset = pd.read_csv(file_buffer)
    if "target" not in dataset.columns:
        st.error("Uploaded CSV missing required column: 'target'.")
        st.stop()
    return dataset.drop(columns=["target"]), dataset["target"]

def compute_evaluation_metrics(ground_truth, predictions, probabilities):
    """Calculate key classification performance metrics."""
    return {
        "Accuracy": accuracy_score(ground_truth, predictions),
        "AUC Score": roc_auc_score(ground_truth, probabilities),
        "Precision": precision_score(ground_truth, predictions),
        "Recall": recall_score(ground_truth, predictions),
        "F1 Score": f1_score(ground_truth, predictions),
        "MCC Score": matthews_corrcoef(ground_truth, predictions)
    }

# Main Banner
st.title("🧠 Diagnostic Model Performance Evaluator")
st.write("Upload test datasets to execute predictions across pre-trained classifiers and inspect performance statistics.")

# Sidebar File Uploader (Defined strictly ONCE)
st.sidebar.header("📁 Data Source Setup")
uploaded_csv = st.sidebar.file_uploader("Upload test_data.csv", type=["csv"], key="test_data_uploader")

if uploaded_csv is not None:
    st.sidebar.success("Test dataset loaded successfully!")
    feature_matrix, target_labels = load_evaluation_dataset(uploaded_csv)

    st.sidebar.header("⚙️ Algorithm Configuration")
    chosen_algorithm = st.sidebar.selectbox(
        "Select Classifier Algorithm",
        ("Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "Gradient Boosting"),
        key="algorithm_select_box"
    )

    model_key = chosen_algorithm.lower().replace(" ", "_") + ".pkl"

    try:
        fitted_model = joblib.load(f"model/{model_key}")
        feature_scaler = joblib.load("model/scaler.pkl")
    except Exception as load_err:
        st.error(f"Error reading model artifacts from model/ directory: {load_err}")
        st.stop()

    # Apply scaling conditionally for distance/linear models
    scaled_algorithms = ["Logistic Regression", "KNN", "Naive Bayes"]
    if chosen_algorithm in scaled_algorithms:
        inference_features = feature_scaler.transform(feature_matrix)
    else:
        inference_features = feature_matrix

    # Generate inference
    predicted_labels = fitted_model.predict(inference_features)
    try:
        class_probabilities = fitted_model.predict_proba(inference_features)[:, 1]
    except (AttributeError, IndexError):
        class_probabilities = predicted_labels

    # Metric Evaluation
    metrics_dict = compute_evaluation_metrics(target_labels, predicted_labels, class_probabilities)

    st.subheader(f"📊 Assessment Summary: {chosen_algorithm}")
    metric_columns = st.columns(6)
    for idx, (metric_name, metric_val) in enumerate(metrics_dict.items()):
        metric_columns[idx].metric(metric_name, f"{metric_val:.4f}")

    st.markdown("---")

    # Visual Diagnostics
    chart_col, table_col = st.columns(2)

    with chart_col:
        st.subheader("Confusion Matrix")
        conf_mat = confusion_matrix(target_labels, predicted_labels)
        fig_handle, ax_handle = plt.subplots(figsize=(5, 3.5))
        sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", ax=ax_handle, cbar=False)
        ax_handle.set_xlabel("Predicted Output")
        ax_handle.set_ylabel("Actual Label")
        st.pyplot(fig_handle)

    with table_col:
        st.subheader("Detailed Classification Report")
        report_data = classification_report(target_labels, predicted_labels, output_dict=True)
        st.dataframe(pd.DataFrame(report_data).transpose().style.format("{:.4f}"), use_container_width=True)

else:
    st.info("💡 Please upload `test_data.csv` using the sidebar to begin model evaluation.")