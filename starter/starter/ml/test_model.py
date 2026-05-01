"""
Unit tests for model training, inference, and evaluation functions.

This module tests core functionality in starter.starter.ml.model,
including:
- model training
- prediction inference
- computation of evaluation metrics
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from starter.starter.ml.model import (
    train_model,
    inference,
    compute_model_metrics,
)


def test_train_model_returns_model():
    """
    Test that train_model returns a fitted RandomForestClassifier.
    """
    X = np.random.rand(20, 5)
    y = np.random.randint(0, 2, size=20)

    model = train_model(X, y)

    assert isinstance(model, RandomForestClassifier)


def test_compute_model_metrics_range():
    """
    Test that compute_model_metrics returns valid metric values.
    """
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])

    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= fbeta <= 1.0


def test_inference_output_shape():
    """
    Test that inference returns predictions of correct shape.
    """
    X = np.random.rand(10, 5)
    y = np.random.randint(0, 2, size=10)

    model = train_model(X, y)
    preds = inference(model, X)

    assert len(preds) == len(X)
