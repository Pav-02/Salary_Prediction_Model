import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib # To save the model

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

# Split data into training and testing sets (re-split for consistency)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create a preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the model pipeline
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', RandomForestRegressor(random_state=42))])

# Define the parameter grid for GridSearchCV
param_grid = {
    'regressor__n_estimators': [50, 100, 200], # Number of trees in the forest
    'regressor__max_features': ['sqrt', 'log2', None], # Number of features to consider when looking for the best split
    'regressor__max_depth': [10, 20, None], # Maximum depth of the tree
    'regressor__min_samples_split': [2, 5, 10] # Minimum number of samples required to split an internal node
}

# Initialize GridSearchCV
grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1)

print("\n--- Starting Hyperparameter Tuning with GridSearchCV ---")
grid_search.fit(X_train, y_train)

print("\nBest parameters found:")
print(grid_search.best_params_)

print("\nBest MAE score (negative because GridSearchCV minimizes):")
print(-grid_search.best_score_) # Convert back to positive MAE

# Get the best model
best_model = grid_search.best_estimator_

# Save the best model
joblib.dump(best_model, 'salary_prediction_model_tuned.pkl')
print("Tuned model saved as salary_prediction_model_tuned.pkl")

# Evaluate the best model on the test set
y_pred_tuned = best_model.predict(X_test)

mae_tuned = mean_absolute_error(y_test, y_pred_tuned)
mse_tuned = mean_squared_error(y_test, y_pred_tuned)
rmse_tuned = np.sqrt(mse_tuned)
r2_tuned = r2_score(y_test, y_pred_tuned)

print(f"\n--- Tuned Model Evaluation on Test Set ---")
print(f"Mean Absolute Error (MAE): ${mae_tuned:,.2f}")
print(f"Root Mean Squared Error (RMSE): ${rmse_tuned:,.2f}")
print(f"R-squared (R2): {r2_tuned:.4f}")
