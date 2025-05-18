# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle

# Load the model
with open('interview_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Create FastAPI app
app = FastAPI(title="Student Selection Predictor")

# Define input schema
class InputData(BaseModel):
    aptitude_score: float
    programming_score: float
    communication_score: float
    gpa: float
    project_score: float

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Student Selection Predictor API"}

# Prediction endpoint
@app.post("/predict/")
def predict(data: InputData):
    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Make prediction
    prediction = model.predict(input_df)[0]
    
    probability = model.predict_proba(input_df)[0][1]

    result = {
        "prediction": "Selected" if prediction == 1 else "Rejected",
        "probability_of_selection": round(probability, 2)
    }
    return result
