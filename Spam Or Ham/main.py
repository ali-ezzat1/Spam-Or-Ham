from fastapi import FastAPI
from pydantic import BaseModel
import joblib

model = joblib.load("XG_Boost.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = FastAPI()

class Review(BaseModel):
    text: str
    
@app.post("/predict")
def predict(review: Review):
    # Transform the input text using the vectorizer
    transformed_text = vectorizer.transform([review.text])
    
    # Make prediction using the loaded model
    prediction = model.predict(transformed_text)
    
    label = "spam" if prediction == 1 else "ham"
    return(f"Prediction: {label}")

    
  