from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

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
async def predict(data: dict):
    headline = data.get("headline", "")
    return {"headline": headline, "prediction": "Fake" if "fake" in headline.lower() else "Real"}
