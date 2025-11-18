"""
Performs SHAP analysis: global feature importance for XGBoost, and five local case studies.
Local case selection includes:
- 2 high-confidence misclassifications (prob>0.8 or <0.2 and wrong)
- 2 model-disagreement cases (models disagree strongly)
- 1 borderline surprising case (highest absolute difference between model probs)


Outputs are saved under outputs/ (plots + CSV summaries)
"""
import os
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from src.utils import save_df


os.makedirs('outputs', exist_ok=True)


# load artifacts
preprocessor = joblib.load('models/preprocessor.joblib')
pca = joblib.load('models/pca.joblib')
log = joblib.load('models/logistic_pca.joblib')
xgb = joblib.load('models/xgb_prepped.joblib')


# load test set
test_df = pd.read_csv('outputs/test_set_with_preds.csv')
X_test = test_df.drop(columns=['target', 'log_proba', 'xgb_proba'])
y_test = test_df['target']


# prepare data for xgb (preprocessed matrix)
X_test_prep = preprocessor.transform(X_test)


# SHAP for XGBoost
explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_test_prep)


# Global importance
plt.figure(figsize=(8,6))
shap.summary_plot(shap_values, X_test_prep, show=False)
plt.tight_layout()
plt.savefig('outputs/shap_summary_global.png')
plt.close()


# compute interesting local cases
xgb_proba = test_df['xgb_proba'].values
os.makedirs('outp
