import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('salary_prediction_model_tuned.pkl')

st.set_page_config(page_title="Salary Predictor", layout="centered")
st.title("💼 Employee Salary Prediction")

st.markdown("Fill the details below to predict an employee's salary.")

# Input fields
age = st.number_input("Age", min_value=18, max_value=65, step=1)
experience = st.number_input("Years of Experience", min_value=0, max_value=40, step=1)
education = st.selectbox("Education Level", ["Bachelor's", "Master's", "PhD"])
gender = st.selectbox("Gender", ["Male", "Female"])
job_role = st.selectbox("Job Role", [
    "Software Engineer", "Data Scientist", "HR Manager", "Project Manager", "Others"
])

# Encoding (must match your training encoding logic!)
def encode_inputs(age, experience, education, gender, job_role):
    education_map = {"Bachelor's": 0, "Master's": 1, "PhD": 2}
    gender_map = {"Male": 0, "Female": 1}
    job_role_map = {
        "Software Engineer": 0,
        "Data Scientist": 1,
        "HR Manager": 2,
        "Project Manager": 3,
        "Others": 4
    }

    return [
        age,
        experience,
        education_map[education],
        gender_map[gender],
        job_role_map[job_role]
    ]

if st.button("Predict Salary"):
    features = encode_inputs(age, experience, education, gender, job_role)
    salary = model.predict([features])[0]
    st.success(f"Estimated Salary: ₹ {salary:,.2f}")
