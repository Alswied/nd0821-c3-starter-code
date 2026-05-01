"""
FastAPI app for Census Income prediction.

Implements:
- GET / : greeting
- POST /predict : model inference
"""

# ------------------------------------------------------------------
# Standard library imports
# ------------------------------------------------------------------
from pathlib import Path
import sys
from typing import Any, Dict

# ------------------------------------------------------------------
# Third-party imports
# ------------------------------------------------------------------
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# ------------------------------------------------------------------
# Local application imports
# ------------------------------------------------------------------
from starter.ml.data import process_data

# ------------------------------------------------------------------
# Path setup
# ------------------------------------------------------------------
STARTER_DIR = Path(__file__).resolve().parent
sys.path.append(str(STARTER_DIR))

app = FastAPI()

# ------------------------------------------------------------------
# Model artifacts
# ------------------------------------------------------------------
REPO_ROOT = STARTER_DIR.parent
MODEL_DIR = REPO_ROOT / "starter" / "model"

MODEL_PATH = MODEL_DIR / "model.pkl"
ENCODER_PATH = MODEL_DIR / "encoder.pkl"
LB_PATH = MODEL_DIR / "lb.pkl"

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)
lb = joblib.load(LB_PATH)

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


class CensusRecord(BaseModel):
    """Input schema for census income prediction."""

    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=284582)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., alias="education-num", example=13)
    marital_status: str = Field(
        ..., alias="marital-status", example="Married-civ-spouse"
    )
    occupation: str = Field(..., example="Exec-managerial")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., alias="capital-gain", example=0)
    capital_loss: int = Field(..., alias="capital-loss", example=0)
    hours_per_week: int = Field(..., alias="hours-per-week", example=40)
    native_country: str = Field(
        ..., alias="native-country", example="United-States"
    )

    class Config:
        """Pydantic configuration."""

        populate_by_name = True
        json_schema_extra = {
            "example": {
                "age": 37,
                "workclass": "Private",
                "fnlgt": 284582,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Married-civ-spouse",
                "occupation": "Exec-managerial",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States",
            }
        }


@app.get("/")
def root() -> Dict[str, str]:
    """Root endpoint returning a welcome message."""
    return {"message": "Welcome to the Census Income Prediction API"}


@app.post("/predict")
def predict(record: CensusRecord) -> Dict[str, Any]:
    """Run model inference and return income prediction."""
    payload = (
        record.model_dump(by_alias=True)
        if hasattr(record, "model_dump")
        else record.dict(by_alias=True)
    )

    df = pd.DataFrame([payload])

    x, _, _, _ = process_data(
        df,
        categorical_features=CAT_FEATURES,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    prediction = int(model.predict(x)[0])
    label = lb.classes_[prediction]

    return {"prediction": label}