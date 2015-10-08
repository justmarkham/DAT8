## Comparing Model Evaluation Procedures

**Training and testing on the same data**

- Goal is to estimate likely performance of a model on out-of-sample data
- But, maximizing training performance rewards overly complex models that won't necessarily generalize
- Unnecessarily complex models overfit the training data:
    - Will do well when tested using the in-sample data
    - May do poorly on out-of-sample data
    - Learns the "noise" in the data rather than the "signal"

**Train/test split**

- Split the dataset into two pieces, so that the model can be trained and tested on different data
- Testing performance is a better estimate of out-of-sample performance (compared to training performance)
- But, it provides a high variance estimate since changing which observations happen to be in the testing set can significantly change testing performance
- Allows you to easily inspect your testing results (via confusion matrix or ROC curve)

**K-fold cross-validation**

- Systematically create "K" train/test splits and average the results together
- Cross-validated performance is a more reliable estimate of out-of-sample performance (compared to testing performance)
- Runs "K" times slower than train/test split

## Comparing Evaluation Metrics for Classification Problems

**Classification accuracy/error**

- Classification accuracy is the percentage of correct predictions (higher is better)
- Classification error is the percentage of incorrect predictions (lower is better)
- Easiest classification metric to understand

**Confusion matrix**

- Confusion matrix gives you a better understanding of how your classifier is performing
- Allows you to calculate sensitivity, specificity, and many other metrics that might match your business objective better than accuracy

**ROC curves and Area Under the Curve (AUC)**

- Allows you to visualize the performance of your classifier across all possible classification thresholds, thus helping you to choose a threshold that appropriately balances sensitivity and specificity
- Still useful when there is high class imbalance (unlike classification accuracy/error)
- Harder to use when there are more than two response classes

**Log loss**

- Most useful when well-calibrated predicted probabilities are important to your business objective

## Comparing Evaluation Metrics for Regression Problems

**Mean Absolute Error (MAE)**

- Mean of the absolute value of the errors
- Easiest regression metric to understand

**Mean Squared Error (MSE)**

- Mean of the squared errors
- More popular than MAE, because MSE "punishes" larger errors, which tends to be useful in the real world

**Root Mean Squared Error (RMSE)**

- Square root of the mean of the squared errors
- Even more popular than MSE, because RMSE is interpretable in the "y" units
