# Machine Learning Assignment 2: Classification Models & Web Deployment

## a. Problem Statement
The objective of this assignment is to train, evaluate, and deploy multiple machine learning classification algorithms on a structured dataset. The models are evaluated using six performance metrics (Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient) and served via an interactive Streamlit web application on Streamlit Community Cloud.

## b. Dataset Description
- **Dataset Name:** Breast Cancer Wisconsin (Diagnostic) Dataset
- **Source:** UCI Machine Learning Repository / Scikit-Learn
- **Feature Size:** 30 continuous numerical features (Exceeds the mandatory minimum of 12 features)
- **Instance Size:** 569 instances (Exceeds the mandatory minimum of 500 instances)
- **Target Variable:** Binary classification (`0`: Malignant, `1`: Benign)

## c. Mandatory Submission Links
- **GitHub Repository:** https://github.com/bmansi664/ML-Assignment-2-Streamlit
- **Live Streamlit App:** https://ml-assignment-2-app-lwehasfgckykjss2zt3bhq.streamlit.app/

## d. Models Used & Comparison Table

The evaluation metrics calculated across all classification models on the test dataset are summarized below:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| **Decision Tree** | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| **kNN** | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| **Naive Bayes** | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| **Random Forest (Ensemble)** | 0.9561 | 0.9937 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| **Gradient Boosting (Ensemble)** | 0.9561 | 0.9907 | 0.9467 | 0.9861 | 0.9660 | 0.9058 |

## e. Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Performed best overall across all primary metrics (Accuracy: 0.9825, F1: 0.9861, MCC: 0.9623), taking full advantage of feature standardization and clear linear decision boundaries. |
| **Decision Tree** | Showed higher split variance leading to lower MCC (0.8174) and Accuracy (0.9123) compared to ensemble variations. |
| **kNN** | Delivered strong distance-based metrics (F1: 0.9655, MCC: 0.9054) following StandardScaler feature normalization. |
| **Naive Bayes** | Maintained robust class probability output and AUC (0.9868) despite class conditional feature independence assumptions. |
| **Random Forest (Ensemble)** | Reduced decision tree variance effectively, achieving strong AUC (0.9937) and stable balance across metrics. |
| **Gradient Boosting (Ensemble)** | Achieved highest recall (0.9861) tied with Logistic Regression along with high overall AUC (0.9907). |
| **Overall Winner for your dataset?** | **Logistic Regression** is the clear winner for this standardized dataset, achieving top scores in Accuracy (**0.9825**), AUC (**0.9954**), F1 Score (**0.9861**), and MCC (**0.9623**). |
