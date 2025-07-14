"""
Headline Sentiment Analysis API Service
Author: Jingyu Huang
Port: 8008
"""

import logging
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Headline Sentiment Analysis API", version="1.0.0")

# Global variables for model and tokenizer
model = None
tokenizer = None
device = None
label_mapping = {0: 'Pessimistic', 1: 'Neutral', 2: 'Optimistic'}

class HeadlineRequest(BaseModel):
    headlines: List[str]

class HeadlineResponse(BaseModel):
    labels: List[str]

class StatusResponse(BaseModel):
    status: str

def load_model():
    """Load the transformer model and tokenizer once at startup"""
    global model, tokenizer, device
    
    try:
        logger.info("Loading transformer model and tokenizer...")
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Load model and tokenizer
        model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to(device)
        model.eval()
        
        logger.info("Model and tokenizer loaded successfully")
        
    except Exception as e:
        logger.critical(f"Failed to load model: {str(e)}")
        logger.critical(f"Traceback: {traceback.format_exc()}")
        raise e

def predict_sentiment(headlines: List[str]) -> List[str]:
    """Predict sentiment for a list of headlines"""
    try:
        if not headlines:
            logger.warning("Empty headlines list provided")
            return []
        
        logger.info(f"Processing {len(headlines)} headlines")
        
        # Tokenize all headlines
        encoded = tokenizer(
            headlines,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Move to device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        # Predict
        with torch.no_grad():
            outputs = model(**encoded)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_labels = torch.argmax(predictions, dim=-1)
        
        # Convert to labels
        labels = [label_mapping[label.item()] for label in predicted_labels]
        
        logger.info(f"Successfully processed {len(headlines)} headlines")
        return labels
        
    except Exception as e:
        logger.error(f"Error in predict_sentiment: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise e

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

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
        
        # Check if model is loaded
        if model is None or tokenizer is None:
            logger.error("Model not loaded")
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Predict sentiment
        labels = predict_sentiment(request.headlines)
        
        logger.info(f"Successfully returned {len(labels)} labels")
        return HeadlineResponse(labels=labels)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in score_headlines endpoint: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    # Run the server
    logger.info("Starting Headline Sentiment Analysis API on port 8008")
    uvicorn.run(app, host="0.0.0.0", port=8008, log_level="info")