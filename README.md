# Headline Sentiment Analysis Web Service and Streamlit App

## Project Overview
This project implements a headline sentiment analysis system with two components:
1. **API Service**: A FastAPI-based real-time sentiment analysis API running on port 8008.
2. **Streamlit Frontend**: A web app built with Streamlit where users can input multiple headlines and get their sentiment labels by calling the API (running on port 9008).

## How to Run

### 1. Start the API Service
```bash
python score_headlines_api.py
```

· The API listens on port 8008 by default.
· Check status at: http://localhost:8008/status
· POST headlines to: http://localhost:8008/score_headlines

### 2. Start the Streamlit App
```bash
streamlit run headline_streamlit_app.py --server.port 9008
```

· Open browser: http://localhost:9008
· Enter headlines (one per line) and click Analyze Sentiment

### Dependencies
Install required Python packages:
```bash
pip install -r requirements.txt
```

### Project Structure
· score_headlines_api.py — API server code
· headline_streamlit_app.py — Streamlit frontend code
· svm.joblib — Pre-trained sentiment model file
· requirements.txt — Python dependencies

### Notes
· Start API server before running Streamlit app
· Change ports if 8008 or 9008 are busy
· Use Ctrl + C to stop running services