"""
Model training, inference, and evaluation utilities.

This module contains:
- Training logic for a RandomForestClassifier
- Inference helper for generating predictions
- Evaluation metrics (precision, recall, F1)
- Slice-based performance evaluation for categorical features
"""
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier

# Import process_data for use in compute_slice_metrics
from starter.starter.ml.data import process_data


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
    Validates the trained machine learning model
    using precision, recall, and F1.

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


def compute_slice_metrics(
        data,
        feature,
        categorical_features,
        model,
        encoder,
        lb):
    """
    Compute model performance metrics for slices of the data based on
    a categorical feature.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset containing features and labels.
    feature : str
        Categorical feature name to slice on.
    categorical_features : list
        List of all categorical feature names.
    model : trained model
        Trained machine learning model.
    encoder : OneHotEncoder
        Fitted encoder from training.
    lb : LabelBinarizer
        Fitted label binarizer.

    Returns
    -------
    dict
        Mapping from feature value to (precision, recall, fbeta).
    """
    slice_metrics = {}

    for value in data[feature].unique():
        slice_data = data[data[feature] == value]

        if slice_data.empty:
            continue

        X_slice, y_slice, _, _ = process_data(
            slice_data,
            categorical_features=categorical_features,
            label="salary",
            training=False,
            encoder=encoder,
            lb=lb,
        )

        preds = model.predict(X_slice)
        precision, recall, fbeta = compute_model_metrics(y_slice, preds)

        slice_metrics[value] = (precision, recall, fbeta)

    return slice_metrics
