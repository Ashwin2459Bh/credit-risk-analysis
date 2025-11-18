Credit Risk Analysis, Prediction & Explainability

This project focuses on predicting credit risk using machine learning techniques and explaining model behavior using SHAP. The objective is to classify loan applicants into defaulters and non-defaulters, interpret the drivers of credit risk, and discuss how these findings can support policy improvements for lending institutions.

Project Overview

The project builds a credit risk prediction model using structured financial and personal data. It includes exploratory data analysis, data preprocessing, feature engineering, oversampling for class imbalance, modeling using XGBoost and Random Forest, and interpretability using both global and local SHAP explanations.

The results highlight the most influential features contributing to credit default and provide insights that can improve risk management practices.

Repository Structure
credit-default-shap-project/
│
├── project.ipynb
├── dataset.csv
├── report.md
├── summary.md
├── local explanation.md
│
└── plots/
    ├── 1.Correlation matrix of features.png
    ├── 2.Global SHAP Plot.png
    ├── 3.Global features summary SHAP Plot.png
    ├── 4.Feature importance plot.png
    ├── 5.SHAP Dependence plot.png
    ├── 6.Force plot for global features.png
    ├── 7.High risk 1 plot.png
    ├── 8.High risk 2 plot.png
    ├── 9.High risk borderline case plot.png
    ├── 10.Low risk 1 plot.png
    ├── 11.Low risk 2 plot.png

Dataset

The dataset contains information on loan applicants. Key categories include:

Personal Attributes

Age

Annual Income

Employment Length

Loan Attributes

Loan Amount

Interest Rate

Loan Grade

Loan Purpose

Credit History

Credit History Length

Default on File

Target Variable

0 = No Default

1 = Default

Dataset Source: Public datasets from platforms such as Kaggle, Lending Club, or Home Credit.

Data Analysis

Exploratory Data Analysis (EDA) was performed to understand the structure and identify inconsistencies in the dataset.

Missing Value Handling

Employment length was imputed using the median.

Interest rate was imputed using the mean.

Outlier Removal

Removed age values greater than 80.

Removed employment length values over 60 years.

Feature Engineering

Several meaningful financial ratios were created to capture risk indicators:

Loan-to-Income Ratio

Employment-to-Loan Ratio

Interest-to-Loan Ratio

Class Imbalance Handling

SMOTE (Synthetic Minority Oversampling Technique) was applied to balance the default and non-default classes.

Model Training

Two models were trained and evaluated to identify the best-performing classifier for credit risk prediction:

Models Used

XGBoost Classifier

Random Forest Classifier

Evaluation Metrics

AUC Score

F1 Score

Precision

Recall

Both models were tested, and the results were compared to identify the most reliable predictive approach.

Model Interpretability

SHAP (Shapley Additive Explanations) was used to interpret both global and local model behavior.

Global SHAP Analysis

SHAP summary plot

Feature importance ranking

Dependence plots

Global force plot

These visualizations highlight the overall contribution of each feature to the model’s predictions.

Local SHAP Analysis

Five individual credit applicants were analyzed to demonstrate how specific features influenced their predicted risk level.
This includes:

High-risk cases

Borderline cases

Low-risk cases

These local explanations help expose model decisions and support transparent decision-making.

How to Run the Project
Option 1: Google Colab

Upload both the notebook and dataset.

Install the necessary dependencies using the requirements file.

Run the notebook cells sequentially.

Option 2: Local Execution

Install dependencies:

pip install -r requirements.txt


Launch the notebook:

jupyter notebook project.ipynb

Key Insights

The Loan-to-Income Ratio is the most influential factor determining default risk.

Applicants with longer credit histories are less likely to default.

Higher interest rates are strongly correlated with increased risk.

Loan purpose significantly affects classification outcomes.

Local explanations highlight prediction strengths, weaknesses, and borderline decisions.
