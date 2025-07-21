import pandas as pd
import joblib  # To load the model

# Load the trained model
try:
    loaded_model = joblib.load('salary_prediction_model.pkl')
    print("Model loaded successfully from salary_prediction_model.pkl")
except FileNotFoundError:
    print("Error: 'salary_prediction_model.pkl' not found. Please ensure the training script was run successfully.")
    exit()

# Load employee data from CSV
try:
    csv_file_path = 'employee_salary_dataset.csv'  # Update this path if needed
    new_employee_data = pd.read_csv(csv_file_path)
    print(f"\nNew employee data loaded from {csv_file_path}:")
    print(new_employee_data)
except FileNotFoundError:
    print(f"Error: '{csv_file_path}' not found. Please ensure the CSV file is present.")
    exit()

# Make predictions
predicted_salaries = loaded_model.predict(new_employee_data)

# Add predicted salaries to the DataFrame
new_employee_data['Predicted Salary'] = predicted_salaries
new_employee_data['Predicted Salary'] = new_employee_data['Predicted Salary'].map('${:,.2f}'.format)

# Display the results
print("\nEmployee Data with Predicted Salaries:")
print(new_employee_data)
