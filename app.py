import joblib
from fastapi import FastAPI
from pydantic import BaseModel

# Load the saved model and vectorizer
model = joblib.load("news_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Initialize FastAPI app
app = FastAPI()

# Define request body format
class NewsInput(BaseModel):
    headline: str

@app.get("/")
def home():
    return {"message": "FastAPI is running!"}

@app.post("/predict/")
async def predict_news(input_data: NewsInput):
    tfidf_input = vectorizer.transform([input_data.headline])
    prediction = model.predict(tfidf_input)[0]
    return {"headline": input_data.headline, "prediction": "Real News" if prediction == 1 else "Fake News"}

# Run the API server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
