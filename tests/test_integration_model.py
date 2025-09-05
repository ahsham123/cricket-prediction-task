from fastapi.testclient import TestClient
import os

from main import app


client = TestClient(app)


def test_model_predict_endpoint_with_engineered_features():
    payload = {
        "total_runs": 120.0,
        "wickets": 5.0,
        "target": 160,
        "balls_left": 24.0,
        "use_engineered": True,
        "model_name": "random_forest"
    }

    res = client.post("/model/predict", json=payload)
    assert res.status_code == 200, res.text
    data = res.json()

    # Validate response schema
    assert set(["prediction", "probability", "model", "engineered", "train_rows"]).issubset(data.keys())
    assert data["model"] == "random_forest"
    assert data["engineered"] is True
    assert isinstance(data["prediction"], int)
    assert 0.0 <= float(data["probability"]) <= 1.0
    assert data["train_rows"] > 0


