# Credit-Card-Fraud-Detection-Project
Overview

Detects fraudulent credit card transactions using Random Forest, XGBoost, and Neural Networks. Handles imbalanced data with SMOTE and evaluates models using classification metrics and ROC-AUC.

Dataset

Source: Kaggle Credit Card Fraud Dataset

Transactions: 284,807 (492 fraud)

Features: V1–V28 (PCA), Amount, Time, Class (target)

Installation
C:/Users/swaro/AppData/Local/Programs/Python/Python312/python.exe -m pip install numpy==1.26.4 scipy==1.11.4 scikit-learn==1.4.2 xgboost==1.7.6 imbalanced-learn==0.14.1 joblib pandas matplotlib seaborn

Usage

Place creditcard.csv in the script folder.

Run:

C:/Users/swaro/AppData/Local/Programs/Python/Python312/python.exe fraud_detection.py


Outputs metrics for all models and saves best model as fraud_detection_model.pkl.
<img width="998" height="829" alt="image" src="https://github.com/user-attachments/assets/135da312-3854-47bd-a3ac-12647ef419f4" />


Models

Random Forest

XGBoost

Neural Network (MLP)

Metrics

Accuracy, Precision, Recall, F1-score, ROC-AUC

Future Improvements

Hyperparameter tuning

Real-time detection system

Additional feature engineering
