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

st.set_page_config(page_title="ML Performance Evaluator", layout="wide")

st.title("📊 Machine Learning Classifier Dashboard")
st.write("Upload test data to compute metrics, confusion matrix, and classification reports across pre-trained models.")

# Sidebar - Dataset Upload
st.sidebar.header("1. Dataset Upload")
uploaded_file = st.sidebar.file_uploader("Upload test_data.csv", type=["csv"])

if uploaded_file is not None:
    df_test = pd.read_csv(uploaded_file)
    st.sidebar.success("CSV Uploaded Successfully!")

    if "target" in df_test.columns:
        X_test = df_test.drop(columns=["target"])
        y_test = df_test["target"]
    else:
        st.error("Uploaded CSV must contain a 'target' column.")
        st.stop()

    # Sidebar - Model Selection Dropdown
    st.sidebar.header("2. Model Selection")
    model_choice = st.sidebar.selectbox(
        "Select Classification Algorithm",
        ("Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "Gradient Boosting")
    )

    model_filename = model_choice.lower().replace(" ", "_") + ".pkl"

    try:
        model = joblib.load(f"model/{model_filename}")
    except Exception:
        st.error(f"Could not load `model/{model_filename}`. Check project directory.")
        st.stop()

    # Model Predictions
    y_pred = model.predict(X_test)
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
    except AttributeError:
        y_prob = y_pred

    # Compute Metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    # Display Metrics Grid
    st.subheader(f"📈 Evaluation Metrics: {model_choice}")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Accuracy", f"{acc:.4f}")
    c2.metric("AUC Score", f"{auc:.4f}")
    c3.metric("Precision", f"{prec:.4f}")
    c4.metric("Recall", f"{rec:.4f}")
    c5.metric("F1 Score", f"{f1:.4f}")
    c6.metric("MCC Score", f"{mcc:.4f}")

    st.markdown("---")

    # Visualizations
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        st.pyplot(fig)

    with col_right:
        st.subheader("Classification Report")
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report_dict).transpose().style.format("{:.4f}"))

else:
    st.info("💡 Upload `test_data.csv` from the sidebar to begin evaluation.")