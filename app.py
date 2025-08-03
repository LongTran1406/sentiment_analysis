from fastapi import FastAPI, Request
from pydantic import BaseModel
import pickle
import numpy as np

with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("best_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post('/predict')
def predict(input: TextInput):
    text = input.text
    vectorized = vectorizer.transform([text])
    proba = model.predict_proba(vectorized)[0][1]
    
    return {
        "text": text,
        "toxicity_probability": float(proba),
        "prediction": int(proba >= 0.5)  # or your best threshold
    }