import numpy as np
import pandas as pd

from api.utils.features import add_cricket_features, BASIC_RAW_COLUMNS, TARGET_COLUMN


def test_add_cricket_features_adds_expected_columns_and_clips_values():
    # include target to verify training layout behavior
    df = pd.DataFrame({
        'total_runs': [0.0, 150.0],
        'wickets': [0.0, 15.0],  # will be clipped to 10
        'target': [160, 140],
        'balls_left': [0.0, 200.0],  # will be clipped to 120
        TARGET_COLUMN: [0, 1],
    })

    out = add_cricket_features(df, is_train=True)

    # Target should be the first column in train mode
    assert out.columns[0] == TARGET_COLUMN

    # Engineered columns should exist
    expected_cols = [
        'overs_left', 'wickets_in_hand', 'current_run_rate', 'required_runs',
        'required_run_rate', 'run_rate_diff', 'resources_proxy', 'req_per_ball',
        'is_powerplay', 'is_middle_overs', 'is_death_overs',
        'rrr_over_wickets', 'rr_diff_times_wi', 'crr_over_wickets'
    ]
    for col in expected_cols:
        assert col in out.columns, f"Missing engineered column: {col}"

    # Values are clipped within expected ranges
    assert (out['wickets'] <= 10).all()
    assert (out['wickets'] >= 0).all()
    assert (out['balls_left'] <= 120).all()
    assert (out['balls_left'] >= 0).all()

    # No infinite values remain (they are converted to NaN)
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    assert np.isfinite(out[numeric_cols].to_numpy()).sum() + out[numeric_cols].isna().to_numpy().sum() == out[numeric_cols].size


def test_add_cricket_features_works_without_target_for_inference():
    df = pd.DataFrame({
        'total_runs': [120.0],
        'wickets': [5.0],
        'target': [160],
        'balls_left': [24.0],
    })

    out = add_cricket_features(df, is_train=False)

    # Basic raw columns should still be present (clipped) and engineered columns added
    for col in BASIC_RAW_COLUMNS:
        assert col in out.columns
    assert 'overs_left' in out.columns
    assert TARGET_COLUMN not in out.columns


