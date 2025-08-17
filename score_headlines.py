"""
Module for classifying sentiment of headlines using a pretrained SVM model
and sentence transformer embeddings.
"""

import sys
import os
import argparse
import datetime
import joblib
from sentence_transformers import SentenceTransformer

def parse_args():
    """
    Parse command line arguments: input file path and source name.
    """
    parser = argparse.ArgumentParser(description="Classify sentiment of headlines.")
    parser.add_argument("input_file", help="Path to the text file containing headlines")
    parser.add_argument("source", help="Source of the headlines, e.g., nyt or chicagotribune")
    return parser.parse_args()

def load_headlines(filepath):
    """
    Load headlines from a text file, stripping empty lines.

    Args:
        filepath (str): Path to the headlines text file.

    Returns:
        list of str: List of headline strings.
    """
    if not os.path.isfile(filepath):
        print(f"Error: File '{filepath}' does not exist.")
        sys.exit(1)

    with open(filepath, "r", encoding="utf-8") as f:
        headlines = [line.strip() for line in f if line.strip()]
    return headlines

def load_models():
    """
    Load pretrained SVM model and sentence transformer embedder.

    Returns:
        tuple: (svm_model, embedder)
    """
    svm_model = joblib.load("svm.joblib")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return svm_model, embedder

def predict_labels(headlines, svm_model, embedder):
    """
    Predict sentiment labels for the list of headlines.

    Args:
        headlines (list of str): Headlines to classify.
        svm_model: Loaded SVM model.
        embedder: Sentence transformer model.

    Returns:
        numpy.ndarray: Predicted labels.
    """
    vectors = embedder.encode(headlines)
    predictions = svm_model.predict(vectors)
    return predictions

def write_output(headlines, predictions, source):
    """
    Write the predictions and headlines to an output file.

    Args:
        headlines (list of str): List of headline strings.
        predictions (array-like): Predicted labels.
        source (str): Source name for filename.
    """
    today = datetime.date.today()
    filename = f"headline_scores_{source}_{today.year}_{today.month:02d}_{today.day:02d}.txt"

    with open(filename, "w", encoding="utf-8") as f:
        for label, headline in zip(predictions, headlines):
            f.write(f"{label},{headline}\n")

    print(f"Results written to: {filename}")

def main():
    """
    Main function to run the sentiment classification pipeline.
    """
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
