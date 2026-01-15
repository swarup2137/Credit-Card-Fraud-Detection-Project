# Credit Card Fraud Detection Project

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

import joblib

# 1. Load Dataset
data = pd.read_csv(r"C:\Users\swaro\Downloads\creditcard.csv")

# 2. Features & Target
X = data.drop(columns=["Class"])
y = data["Class"]

# 3. Scale Amount column (safe copy)
scaler = StandardScaler()
X = X.copy()
X["Amount"] = scaler.fit_transform(X[["Amount"]])

# 4. Handle Imbalanced Data
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled,
    test_size=0.2,
    random_state=42,
    stratify=y_resampled
)

# 6. Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:, 1]

print("🔹 Random Forest Results")
print(classification_report(y_test, rf_pred))
print("ROC-AUC:", roc_auc_score(y_test, rf_prob))

# 7. XGBoost
xgb = XGBClassifier(
    eval_metric="logloss",
    random_state=42,
    use_label_encoder=False
)
xgb.fit(X_train, y_train)

xgb_pred = xgb.predict(X_test)
xgb_prob = xgb.predict_proba(X_test)[:, 1]

print("\n🔹 XGBoost Results")
print(classification_report(y_test, xgb_pred))
print("ROC-AUC:", roc_auc_score(y_test, xgb_prob))

# 8. Neural Network
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    max_iter=300,
    random_state=42
)
mlp.fit(X_train, y_train)

mlp_pred = mlp.predict(X_test)
mlp_prob = mlp.predict_proba(X_test)[:, 1]

print("\n🔹 Neural Network Results")
print(classification_report(y_test, mlp_pred))
print("ROC-AUC:", roc_auc_score(y_test, mlp_prob))

# 9. Save Best Model
joblib.dump(xgb, "fraud_detection_model.pkl")
print("\n  Model saved as fraud_detection_model.pkl")
