"""
Training script for the Census Income model.

This script loads the cleaned census dataset, trains a RandomForest model
using the preprocessing utilities, evaluates performance, and saves all
artifacts required for inference.
"""
# Script to train machine learning model.

# Add the necessary imports for the starter code.
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

from starter.starter.ml.data import process_data
from starter.starter.ml.model import train_model, compute_model_metrics

# Add code to load in the data.
data = pd.read_csv("starter/data/census.csv")

# Double cleanup of column names
data.columns = data.columns.str.strip()

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=42)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Process training data with the process_data function.
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True,
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train and save a model:
# Train the model
model = train_model(X_train, y_train)

# Evaluate model performance
precision, recall, fbeta = compute_model_metrics(
    y_test,
    model.predict(X_test),
)

print("Model performance on test set:")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {fbeta:.4f}")

# Save trained model and preprocessing artifacts
joblib.dump(model, "starter/model/model.pkl")
joblib.dump(encoder, "starter/model/encoder.pkl")
joblib.dump(lb, "starter/model/lb.pkl")
