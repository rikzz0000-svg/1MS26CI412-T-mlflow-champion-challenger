#!/usr/bin/env python3
"""
register_champion.py
Model Registry & Champion-Challenger Alias Assignment
Course: Skill Lab (MLOps with MLflow) - Mini Project (Lab 11)

This script:
1. Connects to the MLflow Model Registry using MlflowClient
2. Registers both Model A and Model B under 'sentiment-model'
3. Sets the 'champion' alias on the winning model version
4. Sets the 'challenger' alias on the other version
5. Displays the active aliases and registered model metadata
"""

import os
import sys
import json
import time

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import mlflow
from mlflow.tracking import MlflowClient

# Setup tracking
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient(tracking_uri=TRACKING_URI)

REGISTERED_MODEL_NAME = "sentiment-model"
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
RUN_IDS_FILE = os.path.join(OUTPUTS_DIR, "run_ids.json")

def main():
    print(f"[Registry] Connecting to MLflow at {TRACKING_URI}...")

    # Load run IDs saved by evaluate.py
    if not os.path.exists(RUN_IDS_FILE):
        print(f"Error: {RUN_IDS_FILE} not found. Please run evaluate.py first.")
        sys.exit(1)

    with open(RUN_IDS_FILE, "r") as f:
        run_data = json.load(f)

    champ_run_id = run_data["champion_run_id"]
    chall_run_id = run_data["challenger_run_id"]
    winner = run_data["winner"]
    winning_run_id = run_data["winning_run_id"]

    print(f"[Registry] Registering candidate models under '{REGISTERED_MODEL_NAME}'...")

    # Register Model A (DistilBERT)
    model_a_uri = f"runs:/{champ_run_id}/model"
    print(f"  Registering Model A from URI: {model_a_uri}")
    mv_a = mlflow.register_model(
        model_uri=model_a_uri,
        name=REGISTERED_MODEL_NAME
    )
    print(f"  -> Model A registered as Version {mv_a.version}")
    client.update_model_version(
        name=REGISTERED_MODEL_NAME,
        version=mv_a.version,
        description=f"Model A baseline: {run_data['model_a_id']}"
    )

    # Register Model B (BERT-base)
    model_b_uri = f"runs:/{chall_run_id}/model"
    print(f"  Registering Model B from URI: {model_b_uri}")
    mv_b = mlflow.register_model(
        model_uri=model_b_uri,
        name=REGISTERED_MODEL_NAME
    )
    print(f"  -> Model B registered as Version {mv_b.version}")
    client.update_model_version(
        name=REGISTERED_MODEL_NAME,
        version=mv_b.version,
        description=f"Model B candidate: {run_data['model_b_id']}"
    )

    # Assign Aliases based on decision rule outcome
    if winning_run_id == champ_run_id:
        champ_ver = mv_a.version
        chall_ver = mv_b.version
        winner_name = run_data['model_a_id']
    else:
        champ_ver = mv_b.version
        chall_ver = mv_a.version
        winner_name = run_data['model_b_id']

    print(f"\n[Registry] Setting aliases based on evaluation outcome:")
    print(f"  Winner was: {winner} ({winner_name})")

    # Set alias 'champion'
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="champion",
        version=champ_ver
    )
    print(f"  [SUCCESS] Set alias '@champion'   -> Version {champ_ver}")

    # Set alias 'challenger'
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="challenger",
        version=chall_ver
    )
    print(f"  [SUCCESS] Set alias '@challenger' -> Version {chall_ver}")

    # Verify and inspect registered model details
    reg_model = client.get_registered_model(REGISTERED_MODEL_NAME)
    aliases = reg_model.aliases

    print(f"\n[Registry Summary]")
    print(f"  Model Name: {reg_model.name}")
    print(f"  Active Aliases:")
    for alias_name, ver in aliases.items():
        print(f"    - @{alias_name:<10} -> Version {ver}")

    # Save registry status
    reg_summary = {
        "registered_model_name": REGISTERED_MODEL_NAME,
        "aliases": dict(aliases),
        "versions": {
            mv_a.version: {"role": "champion_run", "model_id": run_data['model_a_id'], "run_id": champ_run_id},
            mv_b.version: {"role": "challenger_run", "model_id": run_data['model_b_id'], "run_id": chall_run_id}
        }
    }
    with open(os.path.join(OUTPUTS_DIR, "registry_status.json"), "w") as f:
        json.dump(reg_summary, f, indent=2)

if __name__ == "__main__":
    main()
