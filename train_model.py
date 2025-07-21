import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib # To save the model

# 1. Load the dataset
# Make sure 'employee_salary_dataset.csv' is in the same directory as your script
try:
    df = pd.read_csv('employee_salary_dataset.csv')
    print("Dataset loaded successfully.")
    # Clean the 'Salary' column: remove commas and convert to numeric
    df['Salary'] = df['Salary'].astype(str).str.replace(',', '').str.strip().astype(float)
    print("Salary column cleaned and converted to numeric.")
except FileNotFoundError:
    print("Error: 'employee_salary_dataset.csv' not found. Creating a dummy dataset for demonstration.")
    # For demonstration, we'll create a dummy dataset if not found
    data = {
        'Years of Experience': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Education Level': ['Bachelor', 'Master', 'PhD', 'Bachelor', 'Master', 'PhD', 'Bachelor', 'Master', 'PhD', 'Bachelor', 'Master', 'PhD', 'Bachelor', 'Master', 'PhD', 'Bachelor', 'Master', 'PhD', 'Bachelor', 'Master'],
        'Job Role': ['Software Engineer', 'Data Scientist', 'HR Manager', 'Software Engineer', 'Data Scientist', 'HR Manager', 'Software Engineer', 'Data Scientist', 'HR Manager', 'Software Engineer', 'Data Scientist', 'HR Manager', 'Software Engineer', 'Data Scientist', 'HR Manager', 'Software Engineer', 'Data Scientist', 'HR Manager', 'Software Engineer', 'Data Scientist'],
        'City': ['New York', 'London', 'Paris', 'New York', 'London', 'Paris', 'New York', 'London', 'Paris', 'New York', 'London', 'Paris', 'New York', 'London', 'Paris', 'New York', 'London', 'Paris', 'New York', 'London'],
        'Age': [25, 28, 30, 26, 29, 32, 27, 30, 33, 28, 25, 28, 30, 26, 29, 32, 27, 30, 33, 28],
        'Salary': [50000, 70000, 90000, 55000, 75000, 95000, 60000, 80000, 100000, 65000, 52000, 72000, 92000, 57000, 77000, 97000, 62000, 82000, 102000, 67000]
    }
    df = pd.DataFrame(data)
    # Clean the 'Salary' column for the dummy dataset as well
    df['Salary'] = df['Salary'].astype(str).str.replace(',', '').str.strip().astype(float)
    df.to_csv('employee_salary_dataset.csv', index=False)
    print("Dummy 'employee_salary_dataset.csv' created for demonstration.")


# 2. Separate features (X) and target (y)
X = df.drop('Salary', axis=1)
y = df['Salary']
print("\nFeatures (X) and target (y) separated.")

# 3. Identify numerical and categorical features
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

print(f"Numerical features identified: {list(numerical_features)}")
print(f"Categorical features identified: {list(categorical_features)}")

# 4. Create preprocessing pipelines for numerical and categorical features
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

print("\nTransformers (StandardScaler, OneHotEncoder) initialized.")

# 5. Create a preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])
print("ColumnTransformer created to apply different transformations to different feature types.")

# 6. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nData split into training and testing sets:")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

print("\nData preprocessing setup complete. The 'preprocessor' object is ready to transform data.")

# --- Model Selection and Training (from previous steps, now integrated) ---
print("\n--- Starting Model Training ---")
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])

# Train the model
model_pipeline.fit(X_train, y_train)

print("Model training complete.")

# Save the trained model
joblib.dump(model_pipeline, 'salary_prediction_model.pkl')
print("Model saved as salary_prediction_model.pkl")

# --- Model Evaluation (from previous steps, now integrated) ---
print("\n--- Evaluating Model Performance ---")
y_pred = model_pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Evaluation:")
print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
print(f"Mean Squared Error (MSE): ${mse:,.2f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
print(f"R-squared (R2): {r2:.4f}")
print("\n--- Model Training and Evaluation Script Finished ---")

# Save X, y, numerical_features, categorical_features for later use in other scripts
joblib.dump({'X': X, 'y': y, 'numerical_features': numerical_features, 'categorical_features': categorical_features}, 'data_info.pkl')
print("Data info (X, y, features) saved as data_info.pkl for subsequent scripts.")
