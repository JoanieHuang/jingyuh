from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import logging
import joblib
from sentence_transformers import SentenceTransformer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize app
app = FastAPI()

# Load models only once
try:
    logger.info("Loading model and embedder...")
    svm_model = joblib.load("svm.joblib")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("Models loaded successfully.")
except Exception as e:
    logger.critical(f"Failed to load model: {e}")
    raise RuntimeError("Model loading failed") from e

# Define input schema
class HeadlineRequest(BaseModel):
    headlines: List[str]

@app.get("/status")
def get_status():
    return {"status": "OK"}

@app.post("/score_headlines")
def score_headlines(request: HeadlineRequest):
    headlines = request.headlines
    if not headlines:
        logger.warning("Empty headline list received.")
        raise HTTPException(status_code=400, detail="No headlines provided.")

    logger.info(f"Scoring {len(headlines)} headlines...")
    try:
        embeddings = embedder.encode(headlines)
        preds = svm_model.predict(embeddings)
        return {"labels": preds.tolist()}
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction error.")
