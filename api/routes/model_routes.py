 from __future__ import annotations
 
 from fastapi import APIRouter, HTTPException
 from pydantic import BaseModel, Field
 from typing import Optional, Literal, List
 
 import pandas as pd
 
 from sklearn.linear_model import LogisticRegression
 from sklearn.ensemble import RandomForestClassifier
 from sklearn.model_selection import train_test_split
 from sklearn.pipeline import Pipeline
 from sklearn.compose import ColumnTransformer
 from sklearn.preprocessing import StandardScaler
 from sklearn.impute import SimpleImputer
 
 from api.utils.features import add_cricket_features, BASIC_RAW_COLUMNS, TARGET_COLUMN
 
 router = APIRouter(prefix="/model", tags=["model"])
 
 
 class PredictionRequest(BaseModel):
     """Request schema for a single prediction.

     Includes optional flags to control feature set selection and training
     filters applied to the dataset before model fitting.
     """
     # prediction inputs (single sample)
     total_runs: float
     wickets: float
     target: int
     balls_left: float
 
     # modeling flags
     use_engineered: bool = Field(default=True)
     model_name: Literal['logreg', 'random_forest'] = Field(default='random_forest')
 
     # optional filters applied to the training data prior to fitting
     min_balls_left: Optional[float] = None
     max_balls_left: Optional[float] = None
     min_target: Optional[int] = None
     max_target: Optional[int] = None
     min_wickets_in_hand: Optional[int] = None
     won: Optional[int] = Field(default=None)
 
 
 def build_pipelines(feature_columns: List[str]):
     """Create preprocessing + classifier pipelines for numeric features.

     The preprocessing step imputes missing values and standardizes features.
     Returns pipelines for Logistic Regression and Random Forest.

     Parameters
     ----------
     feature_columns : list[str]
         The numeric feature columns to include in the pipelines.

     Returns
     -------
     dict[str, Pipeline]
         Mapping from model name to pipeline objects.
     """
     ct = ColumnTransformer([
         ('num', Pipeline([
             ('impute', SimpleImputer(strategy='median')),
             ('scale', StandardScaler())
         ]), feature_columns)
     ])
 
     logreg = Pipeline([
         ('prep', ct),
         ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', solver='lbfgs'))
     ])
 
     rf = Pipeline([
         ('prep', ct),
         ('clf', RandomForestClassifier(
             n_estimators=300,
             random_state=42,
             n_jobs=-1,
             class_weight='balanced_subsample'
         ))
     ])
     return {'logreg': logreg, 'random_forest': rf}
 
 
 @router.post('/predict')
 def predict(payload: PredictionRequest):
     """Train a model on-the-fly (with optional filters) and predict one case.

     Loads either the engineered or raw training dataset depending on
     `use_engineered`, applies the provided filters, fits the selected model,
     and returns the predicted class with probability.
     """
     try:
         # single-row input
         input_df = pd.DataFrame([{
             'total_runs': payload.total_runs,
             'wickets': payload.wickets,
             'target': payload.target,
             'balls_left': payload.balls_left
         }])
 
         # transform input if engineered features requested
         if payload.use_engineered:
             X_infer = add_cricket_features(input_df, is_train=False)
         else:
             X_infer = input_df[BASIC_RAW_COLUMNS]
 
         feature_columns = X_infer.columns.tolist()
         pipelines = build_pipelines(feature_columns)
         clf = pipelines[payload.model_name]
 
         # load training data
         from pathlib import Path
         BASE_DIR = Path(__file__).resolve().parents[2]
         if payload.use_engineered:
             df_train = pd.read_csv(BASE_DIR / 'feature-engineering' / 'cricket_dataset_engineered.csv')
         else:
             df_train = pd.read_csv(BASE_DIR / 'cricket_dataset.csv')
 
         # apply optional filters to training set
         if payload.min_balls_left is not None:
             df_train = df_train[df_train['balls_left'] >= payload.min_balls_left]
         if payload.max_balls_left is not None:
             df_train = df_train[df_train['balls_left'] <= payload.max_balls_left]
         if payload.min_target is not None:
             df_train = df_train[df_train['target'] >= payload.min_target]
         if payload.max_target is not None:
             df_train = df_train[df_train['target'] <= payload.max_target]
         if payload.won is not None and 'won' in df_train.columns:
             df_train = df_train[df_train['won'] == payload.won]
         if payload.min_wickets_in_hand is not None:
             if 'wickets_in_hand' not in df_train.columns:
                 df_eng = add_cricket_features(df_train, is_train='won' in df_train.columns)
                 df_train = df_train.merge(df_eng[['wickets_in_hand']], left_index=True, right_index=True)
             df_train = df_train[df_train['wickets_in_hand'] >= payload.min_wickets_in_hand]
 
         # prepare training matrices
         y = df_train[TARGET_COLUMN].astype(int)
         if payload.use_engineered:
             X_train = df_train.drop(columns=[TARGET_COLUMN])
         else:
             X_train = df_train[BASIC_RAW_COLUMNS]
 
         # fit
         if len(X_train) == 0:
             raise HTTPException(status_code=400, detail='No training rows after applying filters')
         clf.fit(X_train, y)
 
         # predict
         proba = float(clf.predict_proba(X_infer)[:, 1][0])
         pred = int(proba >= 0.5)
         return {
             "prediction": pred,
             "probability": proba,
             "model": payload.model_name,
             "engineered": payload.use_engineered,
             "train_rows": int(len(df_train))
         }
     except HTTPException:
         raise
     except Exception as exc:
         raise HTTPException(status_code=500, detail=str(exc)) from exc
 
 
 # Note: No separate /filter endpoint; filtering is handled inside /model/predict


