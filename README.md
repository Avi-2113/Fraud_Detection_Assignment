# Fraud_Detection_Assignment
Detect fraudulent financial transactions using Decision Tree ML with feature engineering, SMOTE, and hyperparameter tuning.
# Fraud Detection Using Decision Tree

This project detects fraudulent financial transactions using a Decision Tree classifier. It includes data preprocessing, feature engineering, handling class imbalance with SMOTE, hyperparameter tuning using GridSearchCV, and model evaluation.

## 📁 Dataset
- **File:** `fraud_detection.csv`
- **Columns:**
  - `transaction_id` — Unique ID for each transaction
  - `amount` — Transaction amount
  - `merchant_type` — Type of merchant
  - `device_type` — Device used for transaction
  - `label` — Target variable (0 = Legitimate, 1 = Fraud)
  - Derived features after preprocessing:
    - `log_amount` — Log-transformed amount
    - `high_amount` — Binary feature for high-value transactions

## 🛠 Features & Techniques
- **Feature Engineering:**  
  - Created `log_amount` and `high_amount` for better model interpretation  
  - Added transaction frequency per ID
- **Encoding:** One-hot encoding for categorical variables
- **Class Imbalance Handling:** SMOTE for oversampling minority class
- **Model:** Decision Tree Classifier
- **Hyperparameter Tuning:** GridSearchCV to optimize `criterion`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `class_weight`
- **Evaluation Metrics:** Precision, Recall, F1-score, ROC-AUC
- **Visualization:** ROC curve, Precision-Recall curve, Decision Tree plot

## 📝 Installation
Run the following in your Jupyter environment:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn joblib
