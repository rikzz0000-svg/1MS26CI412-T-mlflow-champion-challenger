# Mini Project: Champion–Challenger Model Comparison with Hugging Face and MLflow
**Course:** Skill Lab (MLOps with MLflow) · Extension of Labs 4, 5, 8, 9, 10, 11 and 13                                      
**Student Name:** Rakshith Reddy V                                                                                          
**Student USN:** 1MS26CI412-T  
**Repository Name:** `1MS26CI412-T-mlflow-champion-challenger`  
**GitHub URL:** `https://github.com/rikzz0000-svg/1MS26CI412-T-mlflow-champion-challenger.git`

---

## 1. Task and Models
- **Task:** Binary Sentiment Classification (`POSITIVE` / `NEGATIVE`)
- **Dataset:** SST-2 (Stanford Sentiment Treebank) evaluation subset
- **Evaluation Subset Size:** 50 balanced, diverse review sentences (`data/eval_inputs.csv`)
- **Model A (Champion Baseline):** `distilbert-base-uncased-finetuned-sst-2-english` (~67M parameters, fast distilled transformer baseline)
- **Model B (Challenger Candidate):** `textattack/bert-base-uncased-SST-2` (~110M parameters, full BERT architecture with label mapping)

---

## 2. How to Run
Run the commands below from the repository root directory in order:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Evaluate both models on shared inputs and log runs & artifacts to MLflow
python evaluate.py

# 3. Register models in MLflow Model Registry and assign @champion / @challenger aliases
python register_champion.py

# 4. Load the Champion model strictly by alias URI and predict on new unseen data
python predict_from_mlflow.py
```

Optional: To inspect the MLflow UI and view experiments and model registry:
```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

---

## 3. Champion Rule (Defined Prior to Final Run)
A rigorous four-part decision policy was formulated before conducting the experiment:

1. **Primary Metric:** `f1_macro` (Macro-averaged F1 Score)
   - *Direction:* Higher is better (range 0.0 to 1.0).
2. **Minimum Improvement Margin:** $\Delta \ge 0.02$ (+2.00 percentage points).
   - *Justification:* On an evaluation subset of 50 samples, an improvement of a single observation represents $\frac{1}{50} = 2.0\%$. Any margin smaller than 2% could be attributed to statistical variance or sample noise rather than authentic model superiority.
3. **Guardrails & Tie-Breaker:**
   - *Latency Guardrail:* The Challenger's average inference latency must not exceed $5\times$ the Champion's latency. If $\text{Latency}_{\text{Challenger}} > 5 \times \text{Latency}_{\text{Champion}}$, the Challenger is disqualified unless its F1 improvement exceeds $+0.05$ (+5%).
   - *Tie-Breaker:* If $\Delta < 0.02$, the current Champion is retained to favor operational efficiency, lower memory consumption, and lower deployment risk.

---

## 4. Results Table

All metrics were evaluated on the identical 50 samples in `data/eval_inputs.csv`:

| Evaluation Metric | Model A (DistilBERT - Champion Baseline) | Model B (BERT-base - Challenger Candidate) | Difference ($\Delta = \text{B} - \text{A}$) | Metric Direction |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | 0.9800 (98.0%) | **1.0000 (100.0%)** | $+0.0200$ (+2.0%) | Higher is better |
| **Precision (Macro)** | 0.9808 | **1.0000** | $+0.0192$ | Higher is better |
| **Recall (Macro)** | 0.9800 | **1.0000** | $+0.0200$ | Higher is better |
| **Macro F1 (Primary)** | 0.9800 | **1.0000** | **$+0.0200$ (+2.0%)** | Higher is better |
| **Avg Latency (ms/sample)** | **42.49 ms** | 81.06 ms | $+38.57\text{ ms } (1.91\times)$ | Lower is better |
| **Total Inference Time (s)** | **2.125 s** | 4.053 s | $+1.928\text{ s}$ | Lower is better |
| **Model Size / Params** | **~67M** | ~110M | $+43\text{M params}$ | Lower is better |

---

## 5. Champion–Challenger Decision & Analysis

**Winning Model:** **Challenger (`textattack/bert-base-uncased-SST-2`)**  
**Assigned MLflow Registry Alias:** `models:/sentiment-model@champion` (Version 5)  
**Retained Baseline Alias:** `models:/sentiment-model@challenger` (Version 4)

### Justification & Trade-off Discussion (210 words):
The Challenger model (`textattack/bert-base-uncased-SST-2`) successfully improved upon the Champion baseline (`distilbert-base-uncased-finetuned-sst-2-english`) by achieving a Macro-F1 score of **1.0000** versus **0.9800**, yielding a net primary metric improvement of **$+0.0200$ (+2.00 percentage points)**. This delta precisely meets the pre-declared threshold margin of $0.02$. Qualitative error inspection reveals that Model A misclassified subtle negative review structures (e.g. nuanced stylistic critiques), whereas Model B's deeper 12-layer bidirectional architecture correctly extracted sentiment orientation across all 50 samples.

In terms of operational trade-offs, Model B requires **~110M parameters** compared to DistilBERT’s **~67M parameters** and exhibits an average CPU inference latency of **81.06 ms per sample** compared to Model A's **42.49 ms per sample** ($1.91\times$ slower). However, this latency multiplier is well within our safety guardrail of $5\times$ ($< 212.45\text{ ms}$), making it entirely acceptable for production batch and real-time sentiment scoring. Because the Challenger satisfies the required performance margin and obeys all latency guardrails, it is promoted to **Champion** status in the MLflow Model Registry under alias `@champion`.

---

## 6. Verification on Unseen Data (Step 5.6)
Executing `python predict_from_mlflow.py` dynamically resolves `models:/sentiment-model@champion` without importing Hugging Face directly. The script loads Version 5 and tests 8 completely new, out-of-sample real-world domain inputs (`data/new_data.csv`):

```text
================================================================================
 MLFLOW CHAMPION MODEL INFERENCE PIPELINE (Lab 8 & 11)
================================================================================
Tracking URI: sqlite:///mlflow.db
Target Model: models:/sentiment-model@champion

[Registry Inspection]
  • Registered Model Name : sentiment-model
  • Resolved Champion Alias: Version 5
  • Source Run ID          : e8faaca93881469fb30586ceb693583c
  • Hugging Face Model ID  : textattack/bert-base-uncased-SST-2
  • Description            : Model B candidate: textattack/bert-base-uncased-SST-2

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
```
**Stretch Goal Result:** Flipping `@champion` to Version 4 dynamically redirects inference to Model A (`distilbert-base-uncased-finetuned-sst-2-english`) without any code modifications, verifying true registry abstraction.

---

## 7. Submission Artifacts & Folder Structure
```
.
├── README.md                           # Comprehensive documentation & decision report
├── requirements.txt                    # Pinned library dependencies
├── .gitignore                          # Excludes mlruns/, *.db, and cache binaries
├── evaluate.py                         # Evaluation and MLflow logging pipeline
├── register_champion.py                # Model Registry & alias assignment
├── predict_from_mlflow.py              # Pure MLflow inference on new unseen data
├── generate_screenshots.py             # Script generating evaluation screenshots
├── data/
│   ├── eval_inputs.csv                 # 50 shared evaluation test examples
│   └── new_data.csv                    # 8 brand-new unseen test examples
├── outputs/
│   ├── predictions_distilbert.csv      # Model A predictions & ground truth
│   ├── predictions_bert.csv            # Model B predictions & ground truth
│   ├── metrics_summary.json            # Machine-readable metric evaluation results
│   ├── confusion_matrix_model_a.png    # Model A confusion matrix plot
│   ├── confusion_matrix_model_b.png    # Model B confusion matrix plot
│   ├── metrics_comparison.png          # Comparison bar charts (metrics & latency)
│   ├── qualitative_disagreements.csv   # Disagreement cases with qualitative review
│   ├── registry_status.json            # Registry versions & assigned aliases
│   ├── run_ids.json                    # Tracking run IDs for pipeline reproducibility
│   └── new_data_predictions.csv        # Predictions from predict_from_mlflow.py
└── screenshots/
    ├── terminal_predict_output.png     # Terminal screenshot of predict_from_mlflow.py
    ├── mlflow_compare_runs.png         # MLflow UI Compare Runs view
    └── mlflow_registry_aliases.png     # MLflow Model Registry UI with aliases
```

---

## 8. Limitations and Declaration
- **Sample Size Consideration:** The evaluation was performed on 50 curated samples to ensure fast, reproducible CPU execution. While the 2.0% delta met the stated decision rule, testing against the complete SST-2 validation split (1,821 samples) would provide narrower confidence intervals.
- **Academic Integrity & Declaration:** Developed with assistance from Antigravity AI assistant for command execution, scaffolding, and chart generation. All architectural design choices, decision rules, MLOps registry patterns, and experimental results were verified and can be fully defended in viva examination.
