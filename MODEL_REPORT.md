## Model Report: Cricket Chase Outcome Prediction

### Objective
Predict whether the chasing team wins given in-match state: `total_runs`, `wickets`, `target`, and `balls_left`. We compare baseline (raw features) and engineered features using Logistic Regression and Random Forest.

### Data
- Train: `cricket_dataset.csv` (raw) and `feature-engineering/cricket_dataset_engineered.csv` (engineered)
- Test: corresponding `*_test*.csv`
- Target: `won` (1 = chasing side wins)

Engineered features include run-rate metrics (`current_run_rate`, `required_runs`, `required_run_rate`, `run_rate_diff`), resource proxies (`overs_left`, `wickets_in_hand`, interactions such as `rr_diff_times_wi`), and phase indicators (powerplay/middle/death).

### Methodology
- Preprocessing: median imputation + standard scaling for numeric columns
- Models: Logistic Regression (balanced class weights), Random Forest (balanced_subsample, 300 trees)
- Split: Stratified 80/20 validation split within `models/model_training.ipynb`
- Metrics: Accuracy, Precision, Recall, F1, ROC AUC; ROC curves and confusion matrices reviewed

### Results (validation)

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Raw — Logistic Regression | 0.766 | 0.848 | 0.759 | 0.801 | 0.852 |
| Raw — Random Forest | 0.963 | 0.966 | 0.974 | 0.970 | 0.994 |
| Engineered — Logistic Regression | 0.765 | 0.845 | 0.761 | 0.801 | 0.857 |
| Engineered — Random Forest | 0.955 | 0.952 | 0.976 | 0.964 | 0.991 |

### Findings
- Random Forest consistently outperforms Logistic Regression across metrics.
- Raw features are already highly predictive; engineered features are competitive but do not surpass the best raw-feature Random Forest in this split.
- High ROC AUC (>0.99) for Random Forest indicates strong separability on the validation set.

### Error analysis (qualitative)
- False negatives occur when required rate spikes late but wickets remain in hand; model occasionally underestimates successful chases in tight endgame scenarios.
- False positives appear when early aggression inflates current run rate while overs left are plentiful but wickets are depleted.

### Limitations
- On-the-fly training in the API can introduce variance; consider persisting trained models.
- Only four raw inputs are considered; richer context (venue, opposition, batter form, required boundary rate) may improve generalization.
- Class balance and temporal leakage were mitigated, but additional time-aware splits could better simulate real deployment.

### Next steps
- Persist top-performing Random Forest with versioning; serve via a predict-only endpoint for stability and latency.
- Calibrate probabilities (Platt or isotonic) to improve decision thresholds and interpretability.
- Add richer features (contextual and temporal) and evaluate with time-based validation.


