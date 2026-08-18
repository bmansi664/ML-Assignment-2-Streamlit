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

# Custom Page Configuration
st.set_page_config(
    page_title="Classification Model Benchmark Dashboard",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Breast Cancer Classification Analytics Portal")
st.markdown("Upload test datasets to dynamically compute performance metrics, view confusion matrices, and analyze classification reports.")

# Sidebar - Dataset Upload
st.sidebar.header("📁 Step 1: Upload Dataset")
user_csv = st.sidebar.file_uploader("Upload test_data.csv file", type=["csv"])

if user_csv is not None:
    test_data_df = pd.read_csv(user_csv)
    st.sidebar.success("Dataset successfully loaded!")

    if "target" in test_data_df.columns:
        features_df = test_data_df.drop(columns=["target"])
        labels_series = test_data_df["target"]
    else:
        st.error("The uploaded CSV file must contain a column named 'target'.")
        st.stop()

    # Sidebar - Model Selection Dropdown
    st.sidebar.header("⚙️ Step 2: Select Classifier")
    algorithm_option = st.sidebar.selectbox(
        "Choose Model Algorithm",
        ("Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "Gradient Boosting")
    )

    formatted_model_filename = algorithm_option.lower().replace(" ", "_") + ".pkl"

    # Model and Scaler Loader
    try:
        classifier_model = joblib.load(f"model/{formatted_model_filename}")
        data_scaler = joblib.load("model/scaler.pkl")
    except Exception as err:
        st.error(f"Failed to load required model artifacts: {err}. Verify file paths inside the model/ directory.")
        st.stop()

    # Apply conditional feature transformation
    if algorithm_option in ["Logistic Regression", "KNN", "Naive Bayes"]:
        eval_features = data_scaler.transform(features_df)
    else:
        eval_features = features_df

    # Model Inference
    predicted_labels = classifier_model.predict(eval_features)
    try:
        predicted_probs = classifier_model.predict_proba(eval_features)[:, 1]
    except AttributeError:
        predicted_probs = predicted_labels

    # Evaluation Metrics Computation
    metric_acc = accuracy_score(labels_series, predicted_labels)
    metric_auc = roc_auc_score(labels_series, predicted_probs)
    metric_prec = precision_score(labels_series, predicted_labels)
    metric_rec = recall_score(labels_series, predicted_labels)
    metric_f1 = f1_score(labels_series, predicted_labels)
    metric_mcc = matthews_corrcoef(labels_series, predicted_labels)

    # Display Metrics Cards
    st.subheader(f"📊 Performance Evaluation Summary: {algorithm_option}")
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    col1.metric("Accuracy", f"{metric_acc:.4f}")
    col2.metric("AUC", f"{metric_auc:.4f}")
    col3.metric("Precision", f"{metric_prec:.4f}")
    col4.metric("Recall", f"{metric_rec:.4f}")
    col5.metric("F1 Score", f"{metric_f1:.4f}")
    col6.metric("MCC Score", f"{metric_mcc:.4f}")

    st.divider()

    # Visualization Display
    left_plot_col, right_report_col = st.columns(2)

    with left_plot_col:
        st.subheader("Matrix Analysis")
        raw_cm = confusion_matrix(labels_series, predicted_labels)
        fig_plot, ax_canvas = plt.subplots(figsize=(5, 3.5))
        sns.heatmap(raw_cm, annot=True, fmt="d", cmap="Blues", ax=ax_canvas, cbar=False)
        ax_canvas.set_xlabel("Predicted Class")
        ax_canvas.set_ylabel("True Class")
        st.pyplot(fig_plot)

    with right_report_col:
        st.subheader("Detailed Classification Report")
        detailed_report = classification_report(labels_series, predicted_labels, output_dict=True)
        st.dataframe(pd.DataFrame(detailed_report).transpose().style.format("{:.4f}"), use_container_width=True)

else:
    st.info("📌 Please upload `test_data.csv` using the sidebar file loader to evaluate model outputs.")
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

# Custom Page Configuration
st.set_page_config(
    page_title="Classification Model Benchmark Dashboard",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Breast Cancer Classification Analytics Portal")
st.markdown("Upload test datasets to dynamically compute performance metrics, view confusion matrices, and analyze classification reports.")

# Sidebar - Dataset Upload
st.sidebar.header("📁 Step 1: Upload Dataset")
user_csv = st.sidebar.file_uploader("Upload test_data.csv file", type=["csv"])

if user_csv is not None:
    test_data_df = pd.read_csv(user_csv)
    st.sidebar.success("Dataset successfully loaded!")

    if "target" in test_data_df.columns:
        features_df = test_data_df.drop(columns=["target"])
        labels_series = test_data_df["target"]
    else:
        st.error("The uploaded CSV file must contain a column named 'target'.")
        st.stop()

    # Sidebar - Model Selection Dropdown
    st.sidebar.header("⚙️ Step 2: Select Classifier")
    algorithm_option = st.sidebar.selectbox(
        "Choose Model Algorithm",
        ("Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "Gradient Boosting")
    )

    formatted_model_filename = algorithm_option.lower().replace(" ", "_") + ".pkl"

    # Model and Scaler Loader
    try:
        classifier_model = joblib.load(f"model/{formatted_model_filename}")
        data_scaler = joblib.load("model/scaler.pkl")
    except Exception as err:
        st.error(f"Failed to load required model artifacts: {err}. Verify file paths inside the model/ directory.")
        st.stop()

    # Apply conditional feature transformation
    if algorithm_option in ["Logistic Regression", "KNN", "Naive Bayes"]:
        eval_features = data_scaler.transform(features_df)
    else:
        eval_features = features_df

    # Model Inference
    predicted_labels = classifier_model.predict(eval_features)
    try:
        predicted_probs = classifier_model.predict_proba(eval_features)[:, 1]
    except AttributeError:
        predicted_probs = predicted_labels

    # Evaluation Metrics Computation
    metric_acc = accuracy_score(labels_series, predicted_labels)
    metric_auc = roc_auc_score(labels_series, predicted_probs)
    metric_prec = precision_score(labels_series, predicted_labels)
    metric_rec = recall_score(labels_series, predicted_labels)
    metric_f1 = f1_score(labels_series, predicted_labels)
    metric_mcc = matthews_corrcoef(labels_series, predicted_labels)

    # Display Metrics Cards
    st.subheader(f"📊 Performance Evaluation Summary: {algorithm_option}")
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    col1.metric("Accuracy", f"{metric_acc:.4f}")
    col2.metric("AUC", f"{metric_auc:.4f}")
    col3.metric("Precision", f"{metric_prec:.4f}")
    col4.metric("Recall", f"{metric_rec:.4f}")
    col5.metric("F1 Score", f"{metric_f1:.4f}")
    col6.metric("MCC Score", f"{metric_mcc:.4f}")

    st.divider()

    # Visualization Display
    left_plot_col, right_report_col = st.columns(2)

    with left_plot_col:
        st.subheader("Matrix Analysis")
        raw_cm = confusion_matrix(labels_series, predicted_labels)
        fig_plot, ax_canvas = plt.subplots(figsize=(5, 3.5))
        sns.heatmap(raw_cm, annot=True, fmt="d", cmap="Blues", ax=ax_canvas, cbar=False)
        ax_canvas.set_xlabel("Predicted Class")
        ax_canvas.set_ylabel("True Class")
        st.pyplot(fig_plot)

    with right_report_col:
        st.subheader("Detailed Classification Report")
        detailed_report = classification_report(labels_series, predicted_labels, output_dict=True)
        st.dataframe(pd.DataFrame(detailed_report).transpose().style.format("{:.4f}"), use_container_width=True)

else:
    st.info("📌 Please upload `test_data.csv` using the sidebar file loader to evaluate model outputs.")
