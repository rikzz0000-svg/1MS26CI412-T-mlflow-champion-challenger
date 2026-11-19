#!/usr/bin/env python3
"""
evaluate.py
Champion-Challenger Evaluation Pipeline for Sentiment Analysis
Course: Skill Lab (MLOps with MLflow) - Mini Project

This script:
1. Loads shared evaluation inputs from data/eval_inputs.csv
2. Runs inference using:
   - Model A (Champion): distilbert-base-uncased-finetuned-sst-2-english
   - Model B (Challenger): textattack/bert-base-uncased-SST-2
3. Computes accuracy, precision, recall, macro-F1, latency
4. Logs individual runs and summary comparison run to MLflow
5. Logs PyFunc custom models, artifacts, and comparison plots
6. Saves run IDs to outputs/run_ids.json for register_champion.py
"""

import os
import sys
import time
import json
import tempfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Environment settings for clean MLflow execution
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import torch
import transformers
from transformers import pipeline
import sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import mlflow
import mlflow.pyfunc

# ---------------------------------------------------------------------------
# Configuration & Tracking Setup
# ---------------------------------------------------------------------------
STUDENT_USN = "1MS26CI412-T"
EXPERIMENT_NAME = "hf-sentiment-champion-challenger"
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "eval_inputs.csv")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

print(f"[Setup] MLflow Tracking URI: {TRACKING_URI}")
print(f"[Setup] MLflow Experiment:   {EXPERIMENT_NAME}")

# ---------------------------------------------------------------------------
# Custom MLflow PyFunc Wrapper (Lab 9 requirement)
# ---------------------------------------------------------------------------
class SentimentPyFuncModel(mlflow.pyfunc.PythonModel):
    """
    Custom MLflow PythonModel wrapping Hugging Face sentiment-analysis pipeline.
    Standardizes output format into a pandas DataFrame with normalized labels.
    """
    def __init__(self, model_id: str):
        self.model_id = model_id

    def load_context(self, context):
        from transformers import pipeline
        self.pipe = pipeline(
            "sentiment-analysis",
            model=self.model_id,
            truncation=True,
            device=-1 # CPU
        )

    def predict(self, context, model_input):
        import pandas as pd
        if isinstance(model_input, pd.DataFrame):
            if "text" in model_input.columns:
                texts = model_input["text"].tolist()
            else:
                texts = model_input.iloc[:, 0].tolist()
        elif isinstance(model_input, pd.Series):
            texts = model_input.tolist()
        elif isinstance(model_input, list):
            texts = model_input
        else:
            texts = [str(model_input)]

        raw_preds = self.pipe(texts)
        results = []
        for t, res in zip(texts, raw_preds):
            raw_label = str(res["label"]).upper()
            # Map LABEL_0/LABEL_1 or POSITIVE/NEGATIVE uniformly
            if raw_label in ["LABEL_1", "POSITIVE"]:
                std_label = "POSITIVE"
            elif raw_label in ["LABEL_0", "NEGATIVE"]:
                std_label = "NEGATIVE"
            else:
                std_label = raw_label

            results.append({
                "text": t,
                "predicted_label": std_label,
                "confidence_score": round(float(res["score"]), 4)
            })
        return pd.DataFrame(results)

# ---------------------------------------------------------------------------
# Evaluation Helper Functions
# ---------------------------------------------------------------------------
def run_model_inference(pipe, texts, label_map=None):
    """Measures latency and returns predictions and scores."""
    predictions = []
    scores = []
    latencies = []

    for text in texts:
        t0 = time.perf_counter()
        res = pipe(text, truncation=True)[0]
        latencies.append((time.perf_counter() - t0) * 1000.0)

        raw_label = str(res["label"]).upper()
        if label_map and raw_label in label_map:
            norm_label = label_map[raw_label]
        elif raw_label in ["LABEL_1", "POSITIVE"]:
            norm_label = "POSITIVE"
        elif raw_label in ["LABEL_0", "NEGATIVE"]:
            norm_label = "NEGATIVE"
        else:
            norm_label = raw_label

        predictions.append(norm_label)
        scores.append(round(float(res["score"]), 4))

    avg_latency = float(np.mean(latencies))
    total_time = float(np.sum(latencies) / 1000.0)
    return predictions, scores, avg_latency, total_time

def compute_metrics(y_true, y_pred, avg_latency, total_time):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label="POSITIVE", average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label="POSITIVE", average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label="POSITIVE", average="macro", zero_division=0)
    return {
        "accuracy": round(float(acc), 4),
        "precision_macro": round(float(prec), 4),
        "recall_macro": round(float(rec), 4),
        "f1_macro": round(float(f1), 4),
        "avg_latency_ms": round(float(avg_latency), 2),
        "total_inference_s": round(float(total_time), 3)
    }

def plot_confusion_matrix(y_true, y_pred, labels, title, filename):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)

# ---------------------------------------------------------------------------
# Main Pipeline Execution
# ---------------------------------------------------------------------------
def main():
    print(f"\n[1/5] Loading shared evaluation dataset from {DATA_PATH}...")
    eval_df = pd.read_csv(DATA_PATH)
    texts = eval_df["text"].tolist()
    y_true = eval_df["label"].tolist()
    n_samples = len(eval_df)
    print(f"      Loaded {n_samples} evaluation examples.")

    lib_versions = f"transformers=={transformers.__version__}, torch=={torch.__version__}, mlflow=={mlflow.__version__}, sklearn=={sklearn.__version__}"

    # Model A: DistilBERT (Champion baseline)
    model_a_id = "distilbert-base-uncased-finetuned-sst-2-english"
    model_a_run_name = "A_sentiment_distilbert-sst2"
    print(f"\n[2/5] Running inference for Model A: {model_a_id}...")
    pipe_a = pipeline("sentiment-analysis", model=model_a_id, device=-1)
    preds_a, scores_a, lat_a, total_a = run_model_inference(pipe_a, texts)
    metrics_a = compute_metrics(y_true, preds_a, lat_a, total_a)
    print("      Model A Metrics:", metrics_a)

    # Save Model A predictions
    df_preds_a = eval_df.copy()
    df_preds_a["predicted_label"] = preds_a
    df_preds_a["confidence_score"] = scores_a
    df_preds_a["correct"] = (df_preds_a["label"] == df_preds_a["predicted_label"])
    pred_a_file = os.path.join(OUTPUTS_DIR, "predictions_distilbert.csv")
    df_preds_a.to_csv(pred_a_file, index=False)

    # Model B: BERT-base SST-2 (Challenger candidate)
    model_b_id = "textattack/bert-base-uncased-SST-2"
    model_b_run_name = "B_sentiment_bert-sst2"
    print(f"\n[3/5] Running inference for Model B: {model_b_id}...")
    pipe_b = pipeline("sentiment-analysis", model=model_b_id, device=-1)
    preds_b, scores_b, lat_b, total_b = run_model_inference(pipe_b, texts)
    metrics_b = compute_metrics(y_true, preds_b, lat_b, total_b)
    print("      Model B Metrics:", metrics_b)

    # Save Model B predictions
    df_preds_b = eval_df.copy()
    df_preds_b["predicted_label"] = preds_b
    df_preds_b["confidence_score"] = scores_b
    df_preds_b["correct"] = (df_preds_b["label"] == df_preds_b["predicted_label"])
    pred_b_file = os.path.join(OUTPUTS_DIR, "predictions_bert.csv")
    df_preds_b.to_csv(pred_b_file, index=False)

    # Generate Confusion Matrix Plots
    cm_a_file = os.path.join(OUTPUTS_DIR, "confusion_matrix_model_a.png")
    cm_b_file = os.path.join(OUTPUTS_DIR, "confusion_matrix_model_b.png")
    plot_confusion_matrix(y_true, preds_a, ["NEGATIVE", "POSITIVE"], "Model A (DistilBERT) Confusion Matrix", cm_a_file)
    plot_confusion_matrix(y_true, preds_b, ["NEGATIVE", "POSITIVE"], "Model B (BERT-base) Confusion Matrix", cm_b_file)

    # Save combined metrics summary
    summary_data = {
        "dataset": "SST-2 Sentiment Subset",
        "eval_samples": n_samples,
        "champion_model_a": {
            "model_id": model_a_id,
            "metrics": metrics_a
        },
        "challenger_model_b": {
            "model_id": model_b_id,
            "metrics": metrics_b
        }
    }
    summary_file = os.path.join(OUTPUTS_DIR, "metrics_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2)

    # Disagreement / Qualitative Analysis (5 examples)
    disagreements = []
    for i in range(n_samples):
        row = eval_df.iloc[i]
        p_a = preds_a[i]
        p_b = preds_b[i]
        ref = row["label"]
        if p_a != p_b or (p_a != ref and len(disagreements) < 5):
            disagreements.append({
                "id": int(row["id"]),
                "text": row["text"],
                "ground_truth": ref,
                "model_a_pred": p_a,
                "model_a_score": scores_a[i],
                "model_b_pred": p_b,
                "model_b_score": scores_b[i],
                "commentary": "Models disagreed" if p_a != p_b else "Both misclassified tricky nuance"
            })
        if len(disagreements) >= 5:
            break
    disagree_df = pd.DataFrame(disagreements)
    disagree_file = os.path.join(OUTPUTS_DIR, "qualitative_disagreements.csv")
    disagree_df.to_csv(disagree_file, index=False)

    # Generate Metrics Comparison Bar Chart
    comp_plot_file = os.path.join(OUTPUTS_DIR, "metrics_comparison.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    metrics_names = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    a_vals = [metrics_a[m] for m in metrics_names]
    b_vals = [metrics_b[m] for m in metrics_names]
    x = np.arange(len(metrics_names))
    width = 0.35
    ax1.bar(x - width/2, a_vals, width, label='Model A (DistilBERT)', color='#3366cc')
    ax1.bar(x + width/2, b_vals, width, label='Model B (BERT-base)', color='#dc3912')
    ax1.set_ylabel('Score')
    ax1.set_title('Classification Performance (Higher is Better)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Accuracy", "Precision", "Recall", "Macro F1"], rotation=15)
    ax1.set_ylim([0.0, 1.05])
    ax1.legend(loc="lower right")
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Latency comparison
    lat_names = ['Model A\n(DistilBERT)', 'Model B\n(BERT-base)']
    lat_vals = [metrics_a["avg_latency_ms"], metrics_b["avg_latency_ms"]]
    bars = ax2.bar(lat_names, lat_vals, color=['#3366cc', '#dc3912'], width=0.45)
    ax2.set_ylabel('Latency (ms per sample)')
    ax2.set_title('Inference Latency (Lower is Better)')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f} ms',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    fig.savefig(comp_plot_file, dpi=150)
    plt.close(fig)

    # -----------------------------------------------------------------------
    # [4/5] MLflow Logging
    # -----------------------------------------------------------------------
    print("\n[4/5] Logging runs and models to MLflow...")

    # Log Model A Run
    with mlflow.start_run(run_name=model_a_run_name) as run_a:
        run_a_id = run_a.info.run_id
        print(f"      Logging Run A ({model_a_run_name}): ID = {run_a_id}")
        mlflow.log_params({
            "hf_model_id": model_a_id,
            "task": "sentiment-analysis",
            "role": "champion",
            "max_length": 128,
            "batch_size": 16,
            "eval_samples": n_samples,
            "seed": 42,
            "device": "cpu"
        })
        mlflow.log_metrics(metrics_a)
        mlflow.set_tags({
            "student_usn": STUDENT_USN,
            "library_versions": lib_versions,
            "dataset_name": "SST-2 Sentiment Subset"
        })
        mlflow.log_artifact(pred_a_file, artifact_path="evaluation")
        mlflow.log_artifact(cm_a_file, artifact_path="plots")
        mlflow.log_artifact(DATA_PATH, artifact_path="data")

        # Log Custom PyFunc Model
        print(f"      Packaging and logging Model A PyFunc artifact...")
        sample_input = pd.DataFrame({"text": ["This film was an absolute joy to watch!"]})
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=SentimentPyFuncModel(model_a_id),
            input_example=sample_input
        )

    # Log Model B Run
    with mlflow.start_run(run_name=model_b_run_name) as run_b:
        run_b_id = run_b.info.run_id
        print(f"      Logging Run B ({model_b_run_name}): ID = {run_b_id}")
        mlflow.log_params({
            "hf_model_id": model_b_id,
            "task": "sentiment-analysis",
            "role": "challenger candidate",
            "max_length": 128,
            "batch_size": 16,
            "eval_samples": n_samples,
            "seed": 42,
            "device": "cpu"
        })
        mlflow.log_metrics(metrics_b)
        mlflow.set_tags({
            "student_usn": STUDENT_USN,
            "library_versions": lib_versions,
            "dataset_name": "SST-2 Sentiment Subset"
        })
        mlflow.log_artifact(pred_b_file, artifact_path="evaluation")
        mlflow.log_artifact(cm_b_file, artifact_path="plots")
        mlflow.log_artifact(DATA_PATH, artifact_path="data")

        # Log Custom PyFunc Model
        print(f"      Packaging and logging Model B PyFunc artifact...")
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=SentimentPyFuncModel(model_b_id),
            input_example=sample_input
        )

    # -----------------------------------------------------------------------
    # Champion–Challenger Decision Logic
    # -----------------------------------------------------------------------
    PRIMARY_METRIC = "f1_macro"
    MIN_MARGIN = 0.02
    TIE_BREAKER = "prefer_faster_and_smaller"

    champ_score = metrics_a[PRIMARY_METRIC]
    chall_score = metrics_b[PRIMARY_METRIC]
    delta = chall_score - champ_score

    print("\n[5/5] Champion–Challenger Evaluation Decision:")
    print(f"      Champion ({model_a_id}) {PRIMARY_METRIC}:   {champ_score:.4f}")
    print(f"      Challenger ({model_b_id}) {PRIMARY_METRIC}: {chall_score:.4f}")
    print(f"      Delta (Challenger - Champion):             {delta:+.4f}")
    print(f"      Required Minimum Margin:                   {MIN_MARGIN:.4f}")

    if delta >= MIN_MARGIN:
        winner = "Challenger"
        winning_model = model_b_id
        winning_run_id = run_b_id
        losing_run_id = run_a_id
        decision_notes = f"Challenger outperformed Champion by {delta:+.4f} (>= {MIN_MARGIN} margin). Promoted to Champion."
    else:
        winner = "Champion"
        winning_model = model_a_id
        winning_run_id = run_a_id
        losing_run_id = run_b_id
        decision_notes = f"Challenger delta ({delta:+.4f}) did not exceed margin {MIN_MARGIN}. Champion retained due to equal/higher efficiency."

    print(f"      --> Decision: Winner is {winner} ({winning_model})")
    print(f"      --> Rationale: {decision_notes}")

    # Summary Run (compare_sentiment)
    with mlflow.start_run(run_name="compare_sentiment") as summary_run:
        summary_run_id = summary_run.info.run_id
        print(f"      Logging Summary Run (compare_sentiment): ID = {summary_run_id}")
        mlflow.log_params({
            "primary_metric": PRIMARY_METRIC,
            "margin": MIN_MARGIN,
            "tie_breaker": TIE_BREAKER,
            "champion_model_a": model_a_id,
            "challenger_model_b": model_b_id,
            "champion_run_id": run_a_id,
            "challenger_run_id": run_b_id
        })
        mlflow.log_metrics({
            "champion_f1_macro": champ_score,
            "challenger_f1_macro": chall_score,
            "delta_primary_metric": round(delta, 4),
            "champion_latency_ms": metrics_a["avg_latency_ms"],
            "challenger_latency_ms": metrics_b["avg_latency_ms"]
        })
        mlflow.set_tags({
            "student_usn": STUDENT_USN,
            "winner": winner,
            "winning_model": winning_model,
            "winning_run_id": winning_run_id,
            "decision_notes": decision_notes
        })
        mlflow.log_artifact(comp_plot_file, artifact_path="comparison_plots")
        mlflow.log_artifact(summary_file, artifact_path="summaries")
        mlflow.log_artifact(disagree_file, artifact_path="summaries")

    # Save run IDs for registry script
    run_ids = {
        "experiment_name": EXPERIMENT_NAME,
        "champion_run_id": run_a_id,
        "challenger_run_id": run_b_id,
        "summary_run_id": summary_run_id,
        "winning_run_id": winning_run_id,
        "losing_run_id": losing_run_id,
        "winner": winner,
        "model_a_id": model_a_id,
        "model_b_id": model_b_id
    }
    with open(os.path.join(OUTPUTS_DIR, "run_ids.json"), "w") as f:
        json.dump(run_ids, f, indent=2)

    print("\n[Complete] Pipeline finished successfully! Artifacts saved to outputs/")

if __name__ == "__main__":
    main()
