# # Logistic regression exercise with Titanic data

# ## Introduction
# 
# - Data from Kaggle's Titanic competition: [data](https://github.com/justmarkham/DAT8/blob/master/data/titanic.csv), [data dictionary](https://www.kaggle.com/c/titanic/data)
# - **Goal**: Predict survival based on passenger characteristics
# - `titanic.csv` is already in our repo, so there is no need to download the data from the Kaggle website

# ## Step 1: Read the data into Pandas

import pandas as pd
url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/titanic.csv'
titanic = pd.read_csv(url, index_col='PassengerId')
titanic.head()


# ## Step 2: Create X and y
# 
# Define **Pclass** and **Parch** as the features, and **Survived** as the response.

feature_cols = ['Pclass', 'Parch']
X = titanic[feature_cols]
y = titanic.Survived


# ## Step 3: Split the data into training and testing sets

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


# ## Step 4: Fit a logistic regression model and examine the coefficients
# 
# Confirm that the coefficients make intuitive sense.

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=1e9)
logreg.fit(X_train, y_train)
zip(feature_cols, logreg.coef_[0])


# ## Step 5: Make predictions on the testing set and calculate the accuracy

# class predictions (not predicted probabilities)
y_pred_class = logreg.predict(X_test)


# calculate classification accuracy
from sklearn import metrics
print metrics.accuracy_score(y_test, y_pred_class)


# ## Step 6: Compare your testing accuracy to the null accuracy

# this works regardless of the number of classes
y_test.value_counts().head(1) / len(y_test)


# this only works for binary classification problems coded as 0/1
max(y_test.mean(), 1 - y_test.mean())


# # Confusion matrix of Titanic predictions

# print confusion matrix
print metrics.confusion_matrix(y_test, y_pred_class)


# save confusion matrix and slice into four pieces
confusion = metrics.confusion_matrix(y_test, y_pred_class)
TP = confusion[1][1]
TN = confusion[0][0]
FP = confusion[0][1]
FN = confusion[1][0]


print 'True Positives:', TP
print 'True Negatives:', TN
print 'False Positives:', FP
print 'False Negatives:', FN


# calculate the sensitivity
print TP / float(TP + FN)
print 44 / float(44 + 51)


# calculate the specificity
print TN / float(TN + FP)
print 105 / float(105 + 23)


# store the predicted probabilities
y_pred_prob = logreg.predict_proba(X_test)[:, 1]


# histogram of predicted probabilities
import matplotlib.pyplot as plt
plt.hist(y_pred_prob)
plt.xlim(0, 1)
plt.xlabel('Predicted probability of survival')
plt.ylabel('Frequency')


# increase sensitivity by lowering the threshold for predicting survival
import numpy as np
y_pred_class = np.where(y_pred_prob > 0.3, 1, 0)


# old confusion matrix
print confusion


# new confusion matrix
print metrics.confusion_matrix(y_test, y_pred_class)


# new sensitivity (higher than before)
print 63 / float(63 + 32)


# new specificity (lower than before)
print 72 / float(72 + 56)
