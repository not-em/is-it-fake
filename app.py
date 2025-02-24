import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load the saved model and vectorizer
model = joblib.load("news_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Initialize FastAPI app
app = FastAPI()

# Define request body format
class NewsInput(BaseModel):
    headline: str

# Allow requests from your GitHub Pages domain
origins = [
    "https://not-em.github.io",  # Your GitHub Pages URL
    "https://is-it-fake.onrender.com",  # Your own API (optional but good practice)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow only these domains
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

@app.get("/")
def home():
    return {"message": "FastAPI is running with CORS enabled!"}

@app.post("/predict/")
async def predict_news(input_data: NewsInput):
    tfidf_input = vectorizer.transform([input_data.headline])
    prediction = model.predict(tfidf_input)[0]
    return {"headline": input_data.headline, "prediction": "Real News" if prediction == 1 else "Fake News"}

# Run the API server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
