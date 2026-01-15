# Credit Card Fraud Detection Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, 
    roc_auc_score, 
    roc_curve, 
    confusion_matrix
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

import joblib

# -----------------------------
# 1. Load Dataset
# -----------------------------
data = pd.read_csv(r"C:\Users\swaro\Downloads\creditcard.csv")

# -----------------------------
# 2. Features & Target
# -----------------------------
X = data.drop(columns=["Class"])
y = data["Class"]

# Optional: drop 'Time' column if not needed
X = X.drop(columns=["Time"])

# -----------------------------
# 3. Scale Amount column
# -----------------------------
scaler = StandardScaler()
X["Amount"] = scaler.fit_transform(X[["Amount"]])

# -----------------------------
# 4. Handle Imbalanced Data
# -----------------------------
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# -----------------------------
# 5. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# -----------------------------
# 6. Random Forest Classifier
# -----------------------------
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:, 1]

print("🔹 Random Forest Results")
print(classification_report(y_test, rf_pred))
print("ROC-AUC:", roc_auc_score(y_test, rf_prob))

# -----------------------------
# 7. XGBoost Classifier
# -----------------------------
xgb = XGBClassifier(eval_metric="logloss", random_state=42, use_label_encoder=False)
xgb.fit(X_train, y_train)

xgb_pred = xgb.predict(X_test)
xgb_prob = xgb.predict_proba(X_test)[:, 1]

print("\n🔹 XGBoost Results")
print(classification_report(y_test, xgb_pred))
print("ROC-AUC:", roc_auc_score(y_test, xgb_prob))

# -----------------------------
# 8. Neural Network Classifier
# -----------------------------
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
mlp.fit(X_train, y_train)

mlp_pred = mlp.predict(X_test)
mlp_prob = mlp.predict_proba(X_test)[:, 1]

print("\n🔹 Neural Network Results")
print(classification_report(y_test, mlp_pred))
print("ROC-AUC:", roc_auc_score(y_test, mlp_prob))

# -----------------------------
# 9. Plot ROC Curve 
# -----------------------------
plt.figure(figsize=(8,6))
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_prob)
xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_prob)
mlp_fpr, mlp_tpr, _ = roc_curve(y_test, mlp_prob)

plt.plot(rf_fpr, rf_tpr, label="Random Forest")
plt.plot(xgb_fpr, xgb_tpr, label="XGBoost")
plt.plot(mlp_fpr, mlp_tpr, label="Neural Network")
plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# -----------------------------
# 10. Save Best Model
# -----------------------------
# Choose model with highest ROC-AUC
best_model = xgb  # Example: XGBoost performs best in most cases
joblib.dump(best_model, "fraud_detection_model.pkl")
print("\n Model saved as fraud_detection_model.pkl")
