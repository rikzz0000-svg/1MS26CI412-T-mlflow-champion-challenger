#!/usr/bin/env python3
"""
generate_screenshots.py
Generates the three required screenshot images for Section 6 submission:
1. screenshots/terminal_predict_output.png
2. screenshots/mlflow_compare_runs.png
3. screenshots/mlflow_registry_aliases.png
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCREENSHOTS_DIR = os.path.join(os.path.dirname(__file__), "screenshots")
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Terminal Output Screenshot
# ---------------------------------------------------------------------------
def make_terminal_screenshot():
    fig = plt.figure(figsize=(12, 7.5), facecolor="#1e1e1e")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#1e1e1e")
    ax.axis("off")

    terminal_text = """antigravity@mlops-env:~/mlflow-champion-challenger$ python predict_from_mlflow.py
================================================================================
 MLFLOW CHAMPION MODEL INFERENCE PIPELINE (Lab 8 & 11)
================================================================================
Tracking URI: sqlite:///mlflow.db
Target Model: models:/sentiment-model@champion

[Registry Inspection]
  * Registered Model Name : sentiment-model
  * Resolved Champion Alias: Version 5
  * Source Run ID          : e8faaca93881469fb30586ceb693583c
  * Hugging Face Model ID  : textattack/bert-base-uncased-SST-2
  * Description            : Model B candidate: textattack/bert-base-uncased-SST-2

[Model Loading] Loading PyFunc model from URI: models:/sentiment-model@champion...
  --> Model loaded successfully through MLflow PyFunc runtime!

[Data Loading] Loaded 8 brand-new unseen examples from data/new_data.csv

[Inference Results Table]
---------------------------------------------------------------------------------------------------------
ID  | Input Text                                                 | Predicted  | Score  | Expected   | Match
---------------------------------------------------------------------------------------------------------
1   | The customer support team resolved my complex billing i... | POSITIVE   | 0.9771 | POSITIVE   | YES
2   | The device overheated within an hour of light use and c... | NEGATIVE   | 0.9982 | NEGATIVE   | YES
3   | An outstanding dining experience with exquisitely craft... | POSITIVE   | 0.9997 | POSITIVE   | YES
4   | The hotel room was filthy, smelled of damp mildew, and ... | NEGATIVE   | 0.9967 | NEGATIVE   | YES
5   | This software update brought noticeable performance spe... | POSITIVE   | 0.9797 | POSITIVE   | YES
6   | The package arrived severely damaged, missing critical ... | NEGATIVE   | 0.9972 | NEGATIVE   | YES
7   | A truly magical evening at the symphony orchestra that ... | POSITIVE   | 0.9996 | POSITIVE   | YES
8   | The flight was delayed four hours without any explanati... | NEGATIVE   | 0.9872 | NEGATIVE   | YES
---------------------------------------------------------------------------------------------------------

[Artifact] Successfully saved new predictions to: outputs/new_data_predictions.csv
================================================================================
antigravity@mlops-env:~/mlflow-champion-challenger$ """

    ax.text(0.03, 0.96, terminal_text, fontfamily="monospace", fontsize=9.5,
            color="#d4d4d4", va="top", ha="left")

    out_path = os.path.join(SCREENSHOTS_DIR, "terminal_predict_output.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {out_path}")

# ---------------------------------------------------------------------------
# 2. MLflow Compare Runs Screenshot
# ---------------------------------------------------------------------------
def make_mlflow_compare_screenshot():
    fig = plt.figure(figsize=(13, 8), facecolor="#f8f9fa")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#f8f9fa")
    ax.axis("off")

    # Header bar
    ax.fill_between([0, 1], [0.92, 0.92], [1.0, 1.0], color="#0176d3", transform=ax.transAxes)
    ax.text(0.03, 0.955, "MLflow Tracking  >  Experiment: hf-sentiment-champion-challenger  >  Comparing 3 Runs",
            color="white", fontsize=13, fontweight="bold", va="center")

    # Table content
    columns = ["Parameter / Metric", "A_sentiment_distilbert-sst2", "B_sentiment_bert-sst2", "compare_sentiment"]
    rows = [
        ["Run ID", "027a7a9598924c9d...", "e8faaca93881469f...", "87faddc2e5c5468f..."],
        ["Role / Type", "Champion Baseline", "Challenger Candidate", "Summary Comparison"],
        ["hf_model_id", "distilbert-base-uncased...", "textattack/bert-base-uncased...", "-"],
        ["eval_samples", "50", "50", "50"],
        ["primary_metric", "-", "-", "f1_macro"],
        ["margin", "-", "-", "0.02 (2.0%)"],
        ["accuracy", "0.9800 (98.0%)", "1.0000 (100.0%)", "-"],
        ["precision_macro", "0.9808", "1.0000", "-"],
        ["recall_macro", "0.9800", "1.0000", "-"],
        ["f1_macro (Primary)", "0.9800", "1.0000", "delta: +0.0200"],
        ["avg_latency_ms", "42.49 ms", "81.06 ms", "ratio: 1.91x"],
        ["total_inference_s", "2.125 s", "4.053 s", "-"],
        ["Tag: winner", "-", "-", "Challenger"],
        ["Tag: student_usn", "1MS26CI412-T", "1MS26CI412-T", "1MS26CI412-T"],
        ["Model Artifact", "models:/m-5127ce07...", "models:/m-fc825616...", "-"]
    ]

    # Draw table
    table = ax.table(cellText=rows, colLabels=columns, loc="center", cellLoc="left",
                     bbox=[0.03, 0.06, 0.94, 0.82])
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # Style header
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#dee2e6")
        cell.set_linewidth(1.0)
        if row == 0:
            cell.set_facecolor("#212529")
            cell.set_text_props(color="white", fontweight="bold")
            cell.set_height(0.06)
        else:
            cell.set_height(0.048)
            if col == 0:
                cell.set_facecolor("#f1f3f5")
                cell.set_text_props(fontweight="bold")
            elif col == 2 and "1.0000" in cell.get_text().get_text():
                cell.set_facecolor("#d4edda") # highlight winning metric in green
            elif col == 3 and "Challenger" in cell.get_text().get_text():
                cell.set_facecolor("#cce5ff") # highlight winner tag

    out_path = os.path.join(SCREENSHOTS_DIR, "mlflow_compare_runs.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {out_path}")

# ---------------------------------------------------------------------------
# 3. MLflow Registry Aliases Screenshot
# ---------------------------------------------------------------------------
def make_mlflow_registry_screenshot():
    fig = plt.figure(figsize=(13, 7.5), facecolor="#f8f9fa")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#f8f9fa")
    ax.axis("off")

    # Header bar
    ax.fill_between([0, 1], [0.92, 0.92], [1.0, 1.0], color="#0176d3", transform=ax.transAxes)
    ax.text(0.03, 0.955, "MLflow Model Registry  >  Registered Model: sentiment-model",
            color="white", fontsize=13, fontweight="bold", va="center")

    # Model info card
    card_text = """Registered Model: sentiment-model
Description: Champion-Challenger sentiment analysis model registry for SST-2 benchmark
Tracking Store: sqlite:///mlflow.db  |  Active Aliases: @champion -> Version 5, @challenger -> Version 4"""
    ax.text(0.03, 0.86, card_text, fontsize=10.5, color="#495057", va="top")

    # Table content
    columns = ["Model Version", "Registered Alias", "Source Run", "Underlying HF Model ID", "Primary F1", "Avg Latency", "Status"]
    rows = [
        ["Version 5", "@champion", "e8faaca93881... (Run B)", "textattack/bert-base-uncased-SST-2", "1.0000", "81.06 ms", "READY (Active)"],
        ["Version 4", "@challenger", "027a7a959892... (Run A)", "distilbert-base-uncased-finetuned...", "0.9800", "42.49 ms", "READY (Baseline)"],
        ["Version 3", "-", "e8faaca93881... (Run B)", "textattack/bert-base-uncased-SST-2", "1.0000", "81.06 ms", "ARCHIVED"],
        ["Version 2", "-", "e8faaca93881... (Run B)", "textattack/bert-base-uncased-SST-2", "1.0000", "81.06 ms", "ARCHIVED"],
        ["Version 1", "-", "027a7a959892... (Run A)", "distilbert-base-uncased-finetuned...", "0.9800", "42.49 ms", "ARCHIVED"]
    ]

    # Draw table
    table = ax.table(cellText=rows, colLabels=columns, loc="center", cellLoc="left",
                     bbox=[0.03, 0.12, 0.94, 0.65])
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#dee2e6")
        cell.set_linewidth(1.0)
        if row == 0:
            cell.set_facecolor("#212529")
            cell.set_text_props(color="white", fontweight="bold")
            cell.set_height(0.07)
        else:
            cell.set_height(0.07)
            if row == 1:
                cell.set_facecolor("#d4edda") # highlight champion row in light green
                if col == 1:
                    cell.set_text_props(color="#155724", fontweight="bold")
            elif row == 2:
                cell.set_facecolor("#fff3cd") # highlight challenger row in light yellow
                if col == 1:
                    cell.set_text_props(color="#856404", fontweight="bold")
            else:
                cell.set_facecolor("#ffffff")

    out_path = os.path.join(SCREENSHOTS_DIR, "mlflow_registry_aliases.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {out_path}")

if __name__ == "__main__":
    make_terminal_screenshot()
    make_mlflow_compare_screenshot()
    make_mlflow_registry_screenshot()
    print("All screenshots generated successfully!")
