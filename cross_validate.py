import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import joblib # To load data info

# Load data info (X, y, numerical_features, categorical_features)
try:
    data_info = joblib.load('data_info.pkl')
    X = data_info['X']
    y = data_info['y']
    numerical_features = data_info['numerical_features']
    categorical_features = data_info['categorical_features']
    print("Data info loaded successfully.")
except FileNotFoundError:
    print("Error: 'data_info.pkl' not found. Please run 'train_model.py' first to generate it.")
    exit()

# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create a preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the model pipeline using the BEST PARAMETERS found by tune_model.py
# These parameters were: {'regressor__max_depth': 10, 'regressor__max_features': 'sqrt', 'regressor__min_samples_split': 2, 'regressor__n_estimators': 50}
model_pipeline_cv = Pipeline(steps=[('preprocessor', preprocessor),
                                      ('regressor', RandomForestRegressor(
                                          n_estimators=50,
                                          max_depth=10,
                                          max_features='sqrt',
                                          min_samples_split=2,
                                          random_state=42 # Keep for reproducibility
                                      ))])

print("\n--- Performing Cross-Validation (with Tuned Parameters) ---")
# Perform 5-fold cross-validation using Mean Absolute Error as scoring metric
# 'neg_mean_absolute_error' is used because cross_val_score maximizes the score,
# so we negate MAE to make it a maximization problem.
cv_scores = cross_val_score(model_pipeline_cv, X, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)

# Convert negative MAE scores to positive
mae_scores = -cv_scores

print(f"Cross-validation MAE scores: {mae_scores}")
print(f"Average MAE across folds: ${np.mean(mae_scores):,.2f}")
print(f"Standard deviation of MAE across folds: ${np.std(mae_scores):,.2f}")
