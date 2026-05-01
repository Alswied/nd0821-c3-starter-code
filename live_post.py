"""
live_post.py

This script sends a POST request to the deployed FastAPI Census Income
Prediction API and prints the HTTP status code and prediction result.

"""

import requests

# --------------------------------------
# Configuration
# ---------------------------------------------------------------------

# URL of the live deployed API endpoint
API_URL = "https://udacity-census-api-tjn5.onrender.com/predict"

# Example payload matching the CensusRecord schema
# (Uses the same structure as API tests and Swagger examples)
payload = {
    "age": 40,
    "workclass": "Private",
    "fnlgt": 284582,
    "education": "Bachelors",
    "education-num": 16,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 28,
    "native-country": "United-States",
}

# --------------------------------------
# Send request and print response
# --------------------------------------

# Send POST request to the live API
response = requests.post(API_URL, json=payload, timeout=30)

# Output HTTP status code and JSON response body
print("Status code:", response.status_code)
print("Response body:", response.json())
