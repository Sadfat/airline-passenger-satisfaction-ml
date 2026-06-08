This project builds a complete machine learning pipeline to predict whether 
an airline passenger will be satisfied or dissatisfied with their flight experience. 
Starting from raw survey data, it covers the full ML lifecycle — exploratory data 
analysis, feature engineering, multi-model training, SHAP-based explainability, 
Bayesian hyperparameter tuning with Optuna, passenger segment analysis, and 
production deployment via a containerised FastAPI REST API.

The final XGBoost model achieves 99.4% ROC-AUC and 96.1% F1 on a held-out 
test set of 25,976 passengers. SHAP analysis reveals that inflight entertainment 
and seat comfort are the dominant satisfaction drivers, while disloyal economy 
passengers represent the highest-risk segment for airlines to target.
