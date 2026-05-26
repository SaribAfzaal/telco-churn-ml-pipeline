# Telco Customer Churn Prediction — ML Pipeline

## Objective
Build a reusable, production-ready ML pipeline using scikit-learn
to predict customer churn for a telecom company.

## Methodology
1. Loaded and cleaned the Telco Churn dataset
2. Built a ColumnTransformer for scaling and encoding
3. Wrapped preprocessing and model into a single sklearn Pipeline
4. Trained Logistic Regression and Random Forest models
5. Tuned Random Forest using GridSearchCV with 5-fold cross validation
6. Exported the final pipeline using joblib

## Key Results
- Best ROC-AUC: ~0.85 (Tuned Random Forest)
- Best Accuracy: ~81%
- Pipeline exported to models/churn_pipeline.pkl

## Tools Used
Python, scikit-learn, pandas, matplotlib, seaborn, joblib
