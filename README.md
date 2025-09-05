## Cricket Prediction API

Production-ready FastAPI service for predicting the outcome of a T20/ODI-style run chase and generating concise natural‑language explanations. The project includes:

- **Data and EDA**: Raw and engineered datasets with exploratory notebooks
- **Feature engineering**: Reproducible transformations for model inputs
- **Model training & inference**: On-demand fitting and prediction with scikit‑learn pipelines
- **LLM explanations**: Optional OpenAI-powered explanations of predictions


### Repository structure

```
c:\Users\Admin\Desktop\work\cricket-prediction-task
├─ api/
│  ├─ __init__.py
│  ├─ routes/
│  │  ├─ llm_routes.py
│  │  └─ model_routes.py
│  └─ utils/
│     ├─ features.py
│     └─ llm.py
├─ eda/
│  └─ exploratory_analysis.ipynb
├─ feature-engineering/
│  ├─ cricket_dataset_engineered.csv
│  ├─ cricket_dataset_test_engineered.csv
│  └─ feature_engineering.ipynb
├─ models/
│  └─ model_training.ipynb
├─ cricket_dataset.csv
├─ cricket_dataset_test.csv
├─ main.py
└─ README.md
```


### Quickstart

1) Create and activate a virtual environment, then install dependencies.

```bash
python -m venv .venv
.\.venv\Scripts\activate  # on Windows
pip install -U pip
pip install fastapi uvicorn scikit-learn pandas numpy python-dotenv openai
```

2) (Optional) Configure OpenAI for LLM explanations.

- Create a `.env` file in the project root and add:
  ```
  OPENAI_API_KEY=your_openai_api_key
  ```

3) Run the API.

```bash
uvicorn main:app --reload
```

The server will expose the root health endpoint at `/` and routers under `/model` and `/llm`.


### Data

- `cricket_dataset.csv`: Raw training dataset with columns including `total_runs`, `wickets`, `target`, `balls_left`, and `won`.
- `feature-engineering/cricket_dataset_engineered.csv`: Engineered training dataset (features derived from the raw columns).
- `cricket_dataset_test.csv` and `feature-engineering/cricket_dataset_test_engineered.csv`: Test splits corresponding to the above.

You can train on either raw columns or engineered features at inference time (see API flags below). When `use_engineered=true`, the service expects the engineered training file to be present; otherwise it falls back to the raw dataset.


### Feature engineering

Feature transformations live in `api/utils/features.py`.

- Basic raw feature set: `['total_runs', 'wickets', 'target', 'balls_left']`
- Target column: `won`
- Engineered examples: `overs_left`, `wickets_in_hand`, `current_run_rate`, `required_runs`, `required_run_rate`, `run_rate_diff`, `resources_proxy`, `req_per_ball`, powerplay/middle/death overs flags, and interaction ratios.

The function `add_cricket_features(df, is_train=True)` returns a transformed `DataFrame`. If `is_train` is `True` and the target exists, it keeps `won` as the first column.


### API overview

Root

- `GET /`
  - Returns service status and available routes.


Model endpoints

- `POST /model/predict`
  - Train a fresh classifier on the fly (filtered dataset) and predict a single input.
  - Request body (JSON):
    - `total_runs` (float)
    - `wickets` (float)
    - `target` (int)
    - `balls_left` (float)
    - `use_engineered` (bool, default `true`): Use engineered features and the engineered training CSV
    - `model_name` ("logreg" | "random_forest", default `random_forest`)
    - Optional training filters (all nullable): `min_balls_left`, `max_balls_left`, `min_target`, `max_target`, `min_wickets_in_hand`, `won`
  - Behavior:
    - Loads training data from `feature-engineering/cricket_dataset_engineered.csv` when `use_engineered=true`, else from `cricket_dataset.csv`
    - Applies optional filters before fitting
    - Builds a numeric preprocessing pipeline with imputation and scaling
    - Fits either Logistic Regression or Random Forest and returns the prediction
  - Response body (JSON):
    - `prediction` (0/1)
    - `probability` (float probability of class 1)
    - `model` (string)
    - `engineered` (bool)
    - `train_rows` (int rows used after filters)


LLM explanation endpoint

- `POST /llm/predict`
  - Generate a succinct explanation of a model outcome using OpenAI.
  - Requires `OPENAI_API_KEY` in environment.
  - Request body (JSON):
    - `total_runs` (float)
    - `wickets` (float)
    - `target` (int)
    - `balls_left` (float)
    - `predicted` (int) — model outcome you want explained
    - `probability` (float) — probability associated with the prediction
  - Response body (JSON):
    - `explanation` (string)


### Notes on modeling

- Pipelines: Numeric imputation via median and standard scaling; classifiers include Logistic Regression and Random Forest.
- Class imbalance: Models are configured with balanced class weights.
- Stateless training: The API trains per request to respect optional filters; introduce persistence or a model registry if you need stable models across requests.

### Model performance summary

Validated on a held-out split in `models/model_training.ipynb` (stratified 80/20, same preprocessing as the API). Metrics rounded to three decimals.

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Raw — Logistic Regression | 0.766 | 0.848 | 0.759 | 0.801 | 0.852 |
| Raw — Random Forest | 0.963 | 0.966 | 0.974 | 0.970 | 0.994 |
| Engineered — Logistic Regression | 0.765 | 0.845 | 0.761 | 0.801 | 0.857 |
| Engineered — Random Forest | 0.955 | 0.952 | 0.976 | 0.964 | 0.991 |

Observations:

- Random Forest outperforms Logistic Regression on both raw and engineered features.
- Engineered features are competitive but do not surpass the best raw-feature Random Forest in this split.


### Local development

- Linting and type checking: Add your preferred tools (e.g., `ruff`, `mypy`) as needed.
- Notebooks: See `eda/exploratory_analysis.ipynb`, `feature-engineering/feature_engineering.ipynb`, and `models/model_training.ipynb` for analysis and experiments.


### Environment variables

- `OPENAI_API_KEY`: Required only for `/llm/predict`.


### Run examples

With the server running locally at `http://127.0.0.1:8000`:

```bash
curl -s -X POST http://127.0.0.1:8000/model/predict \
  -H "Content-Type: application/json" \
  -d '{
    "total_runs": 120,
    "wickets": 5,
    "target": 160,
    "balls_left": 24,
    "use_engineered": true,
    "model_name": "random_forest"
  }'
```

```bash
curl -s -X POST http://127.0.0.1:8000/llm/predict \
  -H "Content-Type: application/json" \
  -d '{
    "total_runs": 120,
    "wickets": 5,
    "target": 160,
    "balls_left": 24,
    "predicted": 0,
    "probability": 0.42
  }'
```


### Troubleshooting

- If `/llm/predict` fails with authentication errors, ensure `.env` is present and `OPENAI_API_KEY` is set.
- If `/model/predict` returns 400 with "No training rows...", relax your training filters or switch `use_engineered`.
- If file paths fail in production, ensure the working directory matches the project root; the service resolves training data paths relative to this directory.


### License

Proprietary or internal-use by default. Update this section if you plan to release under an open-source license.


