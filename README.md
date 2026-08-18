## d. Models Used & Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| **Decision Tree** | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| **KNN** | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| **Naive Bayes** | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| **Random Forest (Ensemble)** | 0.9561 | 0.9937 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| **Gradient Boosting (Ensemble)** | 0.9561 | 0.9907 | 0.9467 | 0.9861 | 0.9660 | 0.9058 |

## e. Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Performed best overall across all primary metrics (Accuracy: 0.9825, F1: 0.9861, MCC: 0.9623), leveraging standardized features and high feature linear separability. |
| **Decision Tree** | Showed lowest MCC (0.8174) and Accuracy (0.9123) due to single-tree decision boundary variance. |
| **KNN** | Delivered high balance (F1: 0.9655, MCC: 0.9054) following feature standardization. |
| **Naive Bayes** | Maintained strong AUC performance (0.9868) despite class conditional feature independence assumptions. |
| **Random Forest (Ensemble)** | Reduced tree variance effectively, achieving high AUC (0.9937) and stable F1 score (0.9655). |
| **Gradient Boosting (Ensemble)** | Achieved peak recall (0.9861) tied with Logistic Regression along with robust AUC (0.9907). |
| **Overall Winner** | **Logistic Regression** is the top performer on this dataset, leading in Accuracy (**0.9825**), AUC (**0.9954**), and MCC (**0.9623**). |
