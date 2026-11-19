#!/usr/bin/env python3
"""
predict_from_mlflow.py
MLflow Model Serving & Inference on Unseen Data (Labs 8 & 11)
Course: Skill Lab (MLOps with MLflow) - Mini Project

Key requirements:
1. Does NOT import Hugging Face or pipeline directly.
2. Loads the model strictly via the alias URI: models:/sentiment-model@champion
3. Resolves and prints model version, run ID, and Hugging Face model ID from MLflow registry metadata.
4. Evaluates on brand-new unseen test data (data/new_data.csv).
5. Saves predictions to outputs/new_data_predictions.csv.
"""

import os
import sys

# Ensure offline mode so local cached weights load immediately without network delay
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import pandas as pd
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient(tracking_uri=TRACKING_URI)

REGISTERED_MODEL_NAME = "sentiment-model"
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "new_data.csv")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "outputs", "new_data_predictions.csv")

def main():
    print("=" * 80)
    print(" MLFLOW CHAMPION MODEL INFERENCE PIPELINE (Lab 8 & 11)")
    print("=" * 80)
    print(f"Tracking URI: {TRACKING_URI}")
    print(f"Target Model: models:/{REGISTERED_MODEL_NAME}@champion\n")

    # 1. Inspect Registry Metadata without hardcoding versions
    try:
        model_version_info = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "champion")
        version_num = model_version_info.version
        run_id = model_version_info.run_id
        description = model_version_info.description

        # Fetch underlying run parameters to get the exact HF model ID
        run = client.get_run(run_id)
        hf_model_id = run.data.params.get("hf_model_id", "Unknown")

        print(f"[Registry Inspection]")
        print(f"  • Registered Model Name : {REGISTERED_MODEL_NAME}")
        print(f"  • Resolved Champion Alias: Version {version_num}")
        print(f"  • Source Run ID          : {run_id}")
        print(f"  • Hugging Face Model ID  : {hf_model_id}")
        print(f"  • Description            : {description}\n")
    except Exception as e:
        print(f"Warning: Could not fetch alias metadata via client ({e}). Proceeding with URI load...")
        version_num = "Unknown"
        hf_model_id = "Unknown"

    # 2. Load model dynamically using MLflow Model URI
    model_uri = f"models:/{REGISTERED_MODEL_NAME}@champion"
    print(f"[Model Loading] Loading PyFunc model from URI: {model_uri}...")
    try:
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        print("  --> Model loaded successfully through MLflow PyFunc runtime!\n")
    except Exception as e:
        print(f"Error loading model from {model_uri}: {e}")
        sys.exit(1)

    # 3. Load Brand-New Unseen Test Data
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file {DATA_PATH} not found.")
        sys.exit(1)

    new_data = pd.read_csv(DATA_PATH)
    print(f"[Data Loading] Loaded {len(new_data)} brand-new unseen examples from {DATA_PATH}\n")

    # 4. Predict via MLflow
    preds_df = loaded_model.predict(new_data)

    # Combine with original metadata
    results_df = new_data.copy()
    results_df["predicted_label"] = preds_df["predicted_label"]
    results_df["confidence_score"] = preds_df["confidence_score"]
    if "expected_sentiment" in results_df.columns:
        results_df["match"] = (results_df["expected_sentiment"] == results_df["predicted_label"])

    # 5. Display Clean Formatted Table
    print("[Inference Results Table]")
    print("-" * 105)
    header = f"{'ID':<3} | {'Input Text':<58} | {'Predicted':<10} | {'Score':<6} | {'Expected':<10} | {'Match'}"
    print(header)
    print("-" * 105)

    for _, row in results_df.iterrows():
        idx = row["id"]
        txt = (row["text"][:55] + "...") if len(row["text"]) > 55 else row["text"]
        pred = row["predicted_label"]
        score = f"{row['confidence_score']:.4f}"
        expected = row.get("expected_sentiment", "N/A")
        match = "YES" if row.get("match", False) else "NO"
        print(f"{idx:<3} | {txt:<58} | {pred:<10} | {score:<6} | {expected:<10} | {match}")
    print("-" * 105)

    # 6. Save Output CSV
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[Artifact] Successfully saved new predictions to: {OUTPUT_FILE}")
    print("=" * 80)

if __name__ == "__main__":
    main()
