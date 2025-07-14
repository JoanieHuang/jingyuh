from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import logging
import joblib
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Loading models...")
svm_model = joblib.load("svm.joblib")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
logger.info("Models loaded")

app = FastAPI()

class HeadlinesRequest(BaseModel):
    headlines: List[str]

@app.get("/status")
def get_status():
    logger.info("GET /status")
    return {"status": "OK"}

@app.post("/score_headlines")
def score_headlines(request: HeadlinesRequest):
    try:
        logger.info(f"POST /score_headlines with {len(request.headlines)} headlines")
        vectors = embedder.encode(request.headlines)
        predictions = svm_model.predict(vectors)
        labels = predictions.tolist()
        return {"labels": labels}
    except Exception as e:
        logger.error(f"Error during scoring: {e}")
        raise HTTPException(status_code=500, detail="Scoring failed")
