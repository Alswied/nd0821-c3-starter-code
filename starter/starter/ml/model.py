"""
Model training, inference, and evaluation utilities.

This module contains:
- Training logic for a RandomForestClassifier
- Inference helper for generating predictions
- Evaluation metrics (precision, recall, F1)
"""
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Labels.
    Returns
    -------
    model : RandomForestClassifier
        Trained machine learning model.
    """

    # Initialize the RandomForestClassifier
    # - n_estimators=100: number of decision trees in the ensemble
    # - random_state=42: ensures reproducibility across runs
    # - n_jobs=-1: use all available CPU cores for training

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    # Train (fit) the model on the training data
    model.fit(X_train, y_train)

    # Return the trained model so it can be used for inference or evaluation
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.ndarray
        Known labels, binarized.
    preds : np.ndarray
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.ndarray
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    # Use the trained model to generate predictions
    preds = model.predict(X)
    return preds
