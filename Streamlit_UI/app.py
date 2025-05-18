import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load('interview_model.pkl')

st.title("🎯 B.Tech/MCA Placement Predictor")

st.write("Enter candidate details to predict selection:")

# Collect inputs
aptitude = st.slider("Aptitude Score", 0, 100, 50)
coding = st.slider("Coding Skill Score", 0, 100, 50)
communication = st.slider("Communication Skill", 0, 100, 50)
gpa = st.number_input("enter your GPA", min_value=0.0, max_value=10.0, step=0.1)
#gpa = st.slider("GPA", 0, 100, 50)
#internships = st.slider("Internships", 0, 100, 50)
project_score = st.slider("Project Quality", 0, 100, 50)
#english = st.slider("English_Fluency", 0, 100, 50)

# Convert to DataFrame
input_data = pd.DataFrame({
    'aptitude_score': [aptitude],
    'programming_score': [coding],
    'communication_score': [communication],
    'gpa': [gpa],
    #'internships' : [internships],
    'project_score': [project_score],
    #'english_fluency' : [english]

})

# Prediction
if st.button("Predict Selection"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"✅ Candidate likely to be **SELECTED** with probability {probability:.2f}")
    else:
        st.error(f"❌ Candidate likely to be **REJECTED** with probability {1 - probability:.2f}")

