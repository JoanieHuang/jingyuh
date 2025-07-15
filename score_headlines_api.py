from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import logging
import joblib
from sentence_transformers import SentenceTransformer
import uvicorn
import traceback

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize app
app = FastAPI(title="Headline Sentiment Analysis API", version="1.0.0")

# Load models only once
try:
    logger.info("Loading model and embedder...")
    svm_model = joblib.load("svm.joblib")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("Models loaded successfully.")
except Exception as e:
    logger.critical(f"Failed to load models: {str(e)}")
    logger.critical(f"Traceback: {traceback.format_exc()}")
    raise e

# Define request/response models
class HeadlineRequest(BaseModel):
    headlines: List[str]

class HeadlineResponse(BaseModel):
    labels: List[str]

class StatusResponse(BaseModel):
    status: str

def encode_and_predict(headlines: List[str]) -> List[str]:
    """Encode headlines and predict sentiment"""
    try:
        if not headlines:
            logger.warning("Empty headlines list provided")
            return []
        
        logger.info(f"Processing {len(headlines)} headlines")
        
        # Encode headlines using sentence transformer
        embeddings = embedder.encode(headlines)
        logger.info(f"Embeddings shape: {embeddings.shape}")
        
        # Predict using SVM model
        predictions = svm_model.predict(embeddings)
        
        # FORCE DEBUG OUTPUT
        print("=" * 60)
        print(f"DEBUG - RAW PREDICTIONS: {predictions}")
        print(f"DEBUG - PREDICTION TYPE: {type(predictions)}")
        print(f"DEBUG - FIRST ELEMENT: {predictions[0] if len(predictions) > 0 else 'empty'}")
        print(f"DEBUG - FIRST ELEMENT TYPE: {type(predictions[0]) if len(predictions) > 0 else 'empty'}")
        print(f"DEBUG - UNIQUE VALUES: {set(predictions)}")
        print(f"DEBUG - PREDICTIONS LIST: {list(predictions)}")
        print("=" * 60)
        
        # Log the same info
        logger.info(f"Raw predictions: {predictions}")
        logger.info(f"Prediction type: {type(predictions[0]) if len(predictions) > 0 else 'empty'}")
        logger.info(f"Unique prediction values: {set(predictions)}")
        
        # Convert predictions to labels
        # Try different possible mappings
        label_mapping = {'Pessimistic': 'Pessimistic', 'Neutral': 'Neutral', 'Optimistic': 'Optimistic'}
        labels = []
        
        for i, pred in enumerate(predictions):
            print(f"Processing prediction {i}: {pred} (type: {type(pred)})")
            if pred in label_mapping:
                labels.append(label_mapping[pred])
                print(f"  -> Mapped to: {label_mapping[pred]}")
            else:
                labels.append('Unknown')
                print(f"  -> Mapped to: Unknown (pred {pred} not in mapping)")
        
        print(f"Final labels: {labels}")
        logger.info(f"Final labels: {labels}")
        logger.info(f"Successfully processed {len(headlines)} headlines")
        return labels
        
    except Exception as e:
        logger.error(f"Error in encode_and_predict: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        print(f"ERROR in encode_and_predict: {str(e)}")
        raise e

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Health check endpoint"""
    try:
        logger.info("Status check requested")
        return StatusResponse(status="OK")
    except Exception as e:
        logger.error(f"Error in status endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/score_headlines", response_model=HeadlineResponse)
async def score_headlines(request: HeadlineRequest):
    """Score headlines for sentiment analysis"""
    try:
        logger.info(f"Received request to score {len(request.headlines)} headlines")
        
        # Validate input
        if not request.headlines:
            logger.warning("Empty headlines list in request")
            raise HTTPException(status_code=400, detail="Headlines list cannot be empty")
        
        # Check if models are loaded
        if svm_model is None or embedder is None:
            logger.error("Models not loaded")
            raise HTTPException(status_code=500, detail="Models not loaded")
        
        # Predict sentiment
        labels = encode_and_predict(request.headlines)
        
        logger.info(f"Successfully returned {len(labels)} labels")
        return HeadlineResponse(labels=labels)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in score_headlines endpoint: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    # Run the server on port 8008 (Jingyu Huang's assigned port)
    logger.info("Starting Headline Sentiment Analysis API on port 8008")
    uvicorn.run(app, host="0.0.0.0", port=8008, log_level="info")