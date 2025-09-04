from __future__ import annotations

import numpy as np
import pandas as pd

BASIC_RAW_COLUMNS = ['total_runs', 'wickets', 'target', 'balls_left']
TARGET_COLUMN = 'won'

def add_cricket_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    out = df.copy()
    out['wickets'] = out['wickets'].clip(lower=0, upper=10)
    out['balls_left'] = out['balls_left'].clip(lower=0, upper=120)
    out['overs_left'] = out['balls_left'] / 6.0
    out['wickets_in_hand'] = 10 - out['wickets']
    overs_faced = (120 - out['balls_left']) / 6.0
    out['current_run_rate'] = np.where(overs_faced > 0, out['total_runs'] / overs_faced, 0.0)
    out['required_runs'] = np.maximum(out['target'] - out['total_runs'], 0)
    out['required_run_rate'] = np.where(out['overs_left'] > 0, out['required_runs'] / out['overs_left'], np.inf)
    out['run_rate_diff'] = out['current_run_rate'] - out['required_run_rate']
    out['resources_proxy'] = out['wickets_in_hand'] * out['overs_left']
    out['req_per_ball'] = np.where(out['balls_left'] > 0, out['required_runs'] / out['balls_left'], np.inf)
    out['is_powerplay'] = (overs_faced <= 6).astype(int)
    out['is_middle_overs'] = ((overs_faced > 6) & (overs_faced <= 15)).astype(int)
    out['is_death_overs'] = (overs_faced > 15).astype(int)
    out['tight_chase'] = ((out['required_runs'] <= 20) & (out['overs_left'] <= 4)).astype(int)
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        out[col] = out[col].replace([np.inf, -np.inf], np.nan)
    out['rrr_over_wickets'] = np.where(out['wickets_in_hand'] > 0, out['required_run_rate'] / out['wickets_in_hand'], np.nan)
    out['rr_diff_times_wi'] = out['run_rate_diff'] * out['wickets_in_hand']
    out['crr_over_wickets'] = np.where(out['wickets_in_hand'] > 0, out['current_run_rate'] / out['wickets_in_hand'], np.nan)
    if is_train and TARGET_COLUMN in out.columns:
        cols = [TARGET_COLUMN] + [c for c in out.columns if c != TARGET_COLUMN]
        out = out[cols]
    return out


