import sys
import os
import argparse
import datetime
import joblib
from sentence_transformers import SentenceTransformer

def parse_args():
    parser = argparse.ArgumentParser(description="Classify sentiment of headlines.")
    parser.add_argument("input_file", help="Path to the text file containing headlines")
    parser.add_argument("source", help="Source of the headlines, e.g., nyt or chicagotribune")
    return parser.parse_args()

# load headlines
def load_headlines(filepath):
    if not os.path.isfile(filepath):
        print(f"Error: File '{filepath}' does not exist.")
        sys.exit(1)

    with open(filepath, "r", encoding="utf-8") as f:
        headlines = [line.strip() for line in f if line.strip()]
    return headlines

# load model
def load_models():
    svm_model = joblib.load("svm.joblib")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return svm_model, embedder


# prediction
def predict_labels(headlines, svm_model, embedder):
    vectors = embedder.encode(headlines)
    predictions = svm_model.predict(vectors)
    return predictions

# output
def write_output(headlines, predictions, source):
    today = datetime.date.today()
    filename = f"headline_scores_{source}_{today.year}_{today.month:02d}_{today.day:02d}.txt"

    with open(filename, "w", encoding="utf-8") as f:
        for label, headline in zip(predictions, headlines):
            f.write(f"{label},{headline}\n")

    print(f"Results written to: {filename}")

def main():
    args = parse_args()
    headlines = load_headlines(args.input_file)
    svm_model, embedder = load_models()
    predictions = predict_labels(headlines, svm_model, embedder)
    write_output(headlines, predictions, args.source)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Error: Please provide both an input file and a source (e.g. nyt or chicagotribune).")
        print("Example: python score_headlines.py todaysheadlines.txt nyt")
        sys.exit(1)
    main()
    

