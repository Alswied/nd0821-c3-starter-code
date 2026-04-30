# Model Card

## Model Details
The model is a RandomForestClassifier trained to predict whether an individual earns more than 50K per year based on census data. 
The model is implemented using scikit-learn and uses one-hot encoding for categorical features.

## Intended Use
This model is intended for educational purposes as part of Udacity course "Deploying a Scalable ML Pipeline in Production". It can be used to demonstrate end-to-end machine learning workflows including training, evaluation, testing, and deployment.

## Training Data
The model is trained on the UCI Census Income dataset. The dataset contains demographic and employment-related features such as age, workclass, education, marital status, occupation, race, sex, and native country. The target variable indicates whether income exceeds 50K per year.

## Evaluation Data
The data is split into training and test sets using an 80/20 split. All evaluation metrics are computed on the held-out test set.

## Metrics
The following metrics are used to evaluate model performance:
- Precision
- Recall
- F1-score

Overall model performance on the test set:
- Precision: 0.7419  
- Recall: 0.6384  
- F1-score: 0.6863  

## Slice Performance
Model performance was also evaluated on data slices based on the categorical feature **sex**.

- sex = Female  
  - Precision: 0.7229  
  - Recall: 0.5150  
  - F1-score: 0.6015  

- sex = Male  
  - Precision: 0.7445  
  - Recall: 0.6599  
  - F1-score: 0.6997  

These results show that model performance varies across subgroups.

## Ethical Considerations
The dataset includes sensitive demographic attributes. Differences in performance across slices indicate potential bias. The model should not be used in sensitive real-world applications without further fairness analysis and mitigation.

## Caveats and Recommendations
The model performance is limited by the quality and representativeness of the training data. The model uses default hyperparameters and does not include bias mitigation techniques. Future work could include hyperparameter tuning, additional slice analysis, and fairness-aware training methods.