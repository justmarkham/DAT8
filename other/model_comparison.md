# Comparison of Machine Learning Models

## K-nearest neighbors (KNN)

**Advantages:**

- Simple to understand and explain
- Model training is fast
- Can be used for classification and regression

**Disadvantages:**

- Must store all of the training data
- Prediction phase can be slow when n is large
- Sensitive to irrelevant features
- Sensitive to the scale of the data
- Accuracy is (generally) not competitive with the best supervised learning methods

## Linear Regression

**Advantages:**

- Simple to explain
- Highly interpretable
- Model training and prediction are fast
- No tuning is required (excluding regularization)
- Features don't need scaling
- Can perform well with a small number of observations
- Well-understood

**Disadvantages:**

- Presumes a linear relationship between the features and the response
- Performance is (generally) not competitive with the best supervised learning methods due to high bias
- Can't automatically learn feature interactions

## Logistic Regression

**Advantages:**

- Highly interpretable (if you remember how)
- Model training and prediction are fast
- No tuning is required (excluding regularization)
- Features don't need scaling
- Can perform well with a small number of observations
- Outputs well-calibrated predicted probabilities

**Disadvantages:**

- Presumes a linear relationship between the features and the log-odds of the response
- Performance is (generally) not competitive with the best supervised learning methods
- Can't automatically learn feature interactions

## Naive Bayes

**Advantages:**

- Model training and prediction are very fast
- Somewhat interpretable
- No tuning is required
- Features don't need scaling
- Insensitive to irrelevant features (with enough observations)
- Performs better than logistic regression when the training set is very small

**Disadvantages:**

- Predicted probabilities are not well-calibrated
- Correlated features can be problematic (due to the independence assumption)
- Can't handle negative features (with Multinomial Naive Bayes)
- Has a higher "asymptotic error" than logistic regression
