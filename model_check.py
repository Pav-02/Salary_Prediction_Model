import pandas as pd

try:
    df = pd.read_csv('employee_salary_dataset.csv')
    # Clean the 'Salary' column just like in train_model.py
    df['Salary'] = df['Salary'].astype(str).str.replace(',', '').str.strip().astype(float)

    print("\n--- Checking 'Salary' column variability ---")
    print(f"Number of unique salary values: {df['Salary'].nunique()}")
    print(f"Unique salary values: {df['Salary'].unique()}")
    print(f"Standard deviation of salary: {df['Salary'].std()}")

    if df['Salary'].nunique() <= 1:
        print("\n**Observation: The 'Salary' column has very few or only one unique value.**")
        print("This explains the 0.0 MAE and 1.0 R2. The model is predicting perfectly because there's no variation to predict.")
        print("To build a model that predicts varying salaries, your dataset needs a 'Salary' column with diverse values.")
    else:
        print("\nObservation: The 'Salary' column has varying values. The perfect scores might indicate a very simple relationship or a small, clean dataset.")

except FileNotFoundError:
    print("Error: 'employee_salary_dataset.csv' not found.")
except Exception as e:
    print(f"An error occurred: {e}")

