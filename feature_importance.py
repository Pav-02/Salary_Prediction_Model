import pandas as pd
import joblib # To load the model

# Load the trained model (loading the TUNED model now)
try:
    loaded_model = joblib.load('salary_prediction_model_tuned.pkl')
    print("Tuned model loaded successfully.")
except FileNotFoundError:
    print("Error: 'salary_prediction_model_tuned.pkl' not found. Please run 'tune_model.py' first.")
    exit()

# Load data info (X, y, numerical_features, categorical_features)
try:
    data_info = joblib.load('data_info.pkl')
    X = data_info['X']
    numerical_features = data_info['numerical_features']
    categorical_features = data_info['categorical_features']
    print("Data info loaded successfully.")
except FileNotFoundError:
    print("Error: 'data_info.pkl' not found. Please run 'train_model.py' first to generate it.")
    exit()

# Access the regressor and preprocessor from the pipeline
regressor = loaded_model.named_steps['regressor']
preprocessor = loaded_model.named_steps['preprocessor']

# Get feature names after one-hot encoding
# The preprocessor needs to be fitted to get feature names out
# If the loaded_model is already fitted, its preprocessor is also fitted.
categorical_features_encoded = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
all_features = list(numerical_features) + list(categorical_features_encoded)

# Get feature importances
importances = regressor.feature_importances_

# Create a DataFrame for better visualization
feature_importance_df = pd.DataFrame({
    'Feature': all_features,
    'Importance': importances
})

# Sort by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print("\n--- Feature Importances (from Tuned Model) ---")
print(feature_importance_df)

# Optional: Visualize feature importances (requires matplotlib and seaborn)
# If you want to run this part, install them: pip install matplotlib seaborn
# import matplotlib.pyplot as plt
# import seaborn as sns
# plt.figure(figsize=(12, 8))
# sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(15))
# plt.title('Top 15 Feature Importances (Tuned Model)')
# plt.tight_layout()
# plt.show()
