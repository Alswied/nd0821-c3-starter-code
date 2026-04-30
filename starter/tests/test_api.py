"""
API tests for FastAPI Census Income service.

Rubric requirements covered:
- GET / test (status code + response body)
- POST /predict test for <=50K
- POST /predict test for >50K
"""

from pathlib import Path
import sys

import pandas as pd
from fastapi.testclient import TestClient

# --- Make sanitycheck.py happy ---
# sanitycheck.py adds starter/tests to sys.path, so we add the parent 'starter/'
# directory so that `import main` resolves to starter/main.py
STARTER_DIR = Path(__file__).resolve().parents[1]  # .../starter
sys.path.append(str(STARTER_DIR))

from main import app  # imports starter/main.py

client = TestClient(app)

# Absolute path to data, independent of current working directory
REPO_ROOT = Path(__file__).resolve().parents[2]     # repo root
DATA_PATH = REPO_ROOT / "starter" / "data" / "census.csv"


def _row_to_payload(row: pd.Series) -> dict:
    """Convert a labeled census row into a JSON payload for POST /predict."""
    payload = row.drop("salary").to_dict()

    # Convert numpy types -> native Python types (json-serializable)
    for k, v in payload.items():
        if hasattr(v, "item"):
            payload[k] = v.item()
    return payload


def test_get_root():
    """GET / must return status code 200 and the greeting JSON."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Census Income Prediction API"}


def test_post_predict_leq_50k():
    """POST /predict must be able to return <=50K."""
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    row = df[df["salary"] == "<=50K"].iloc[0]
    payload = _row_to_payload(row)

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["prediction"] == "<=50K"


def test_post_predict_gt_50k():
    """POST /predict must be able to return >50K."""
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    row = df[df["salary"] == ">50K"].iloc[0]
    payload = _row_to_payload(row)

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["prediction"] == ">50K"
