# # Data Preparation and Advanced Model Evaluation

# ## Agenda
# 
# **Data preparation**
# 
# - Handling missing values
# - Handling categorical features (review)
# 
# **Advanced model evaluation**
# 
# - ROC curves and AUC
# - Bonus: ROC curve is only sensitive to rank order of predicted probabilities
# - Cross-validation

# ## Part 1: Handling missing values

# scikit-learn models expect that all values are **numeric** and **hold meaning**. Thus, missing values are not allowed by scikit-learn.

# read the Titanic data
import pandas as pd
url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/titanic.csv'
titanic = pd.read_csv(url, index_col='PassengerId')
titanic.shape


# check for missing values
titanic.isnull().sum()


# One possible strategy is to **drop missing values**:

# drop rows with any missing values
titanic.dropna().shape


# drop rows where Age is missing
titanic[titanic.Age.notnull()].shape


# Sometimes a better strategy is to **impute missing values**:

# mean Age
titanic.Age.mean()


# median Age
titanic.Age.median()


# most frequent Age
titanic.Age.mode()


# fill missing values for Age with the median age
titanic.Age.fillna(titanic.Age.median(), inplace=True)


# Another strategy would be to build a **KNN model** just to impute missing values. How would we do that?
# 
# If values are missing from a categorical feature, we could treat the missing values as **another category**. Why might that make sense?
# 
# How do we **choose** between all of these strategies?

# ## Part 2: Handling categorical features (Review)

# How do we include a categorical feature in our model?
# 
# - **Ordered categories:** transform them to sensible numeric values (example: small=1, medium=2, large=3)
# - **Unordered categories:** use dummy encoding (0/1)

titanic.head(10)


# encode Sex_Female feature
titanic['Sex_Female'] = titanic.Sex.map({'male':0, 'female':1})


# create a DataFrame of dummy variables for Embarked
embarked_dummies = pd.get_dummies(titanic.Embarked, prefix='Embarked')
embarked_dummies.drop(embarked_dummies.columns[0], axis=1, inplace=True)

# concatenate the original DataFrame and the dummy DataFrame
titanic = pd.concat([titanic, embarked_dummies], axis=1)


titanic.head(1)


# - How do we **interpret** the encoding for Embarked?
# - Why didn't we just encode Embarked using a **single feature** (C=0, Q=1, S=2)?
# - Does it matter which category we choose to define as the **baseline**?
# - Why do we only need **two dummy variables** for Embarked?

# define X and y
feature_cols = ['Pclass', 'Parch', 'Age', 'Sex_Female', 'Embarked_Q', 'Embarked_S']
X = titanic[feature_cols]
y = titanic.Survived

# train/test split
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# train a logistic regression model
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=1e9)
logreg.fit(X_train, y_train)

# make predictions for testing set
y_pred_class = logreg.predict(X_test)

# calculate testing accuracy
from sklearn import metrics
print metrics.accuracy_score(y_test, y_pred_class)


# ## Part 3: ROC curves and AUC

# predict probability of survival
y_pred_prob = logreg.predict_proba(X_test)[:, 1]


import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 14


# plot ROC curve
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')


# calculate AUC
print metrics.roc_auc_score(y_test, y_pred_prob)


# Besides allowing you to calculate AUC, seeing the ROC curve can help you to choose a threshold that **balances sensitivity and specificity** in a way that makes sense for the particular context.

# histogram of predicted probabilities grouped by actual response value
df = pd.DataFrame({'probability':y_pred_prob, 'actual':y_test})
df.hist(column='probability', by='actual', sharex=True, sharey=True)


# What would have happened if you had used **y_pred_class** instead of **y_pred_prob** when drawing the ROC curve or calculating AUC?

# ROC curve using y_pred_class - WRONG!
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_class)
plt.plot(fpr, tpr)


# AUC using y_pred_class - WRONG!
print metrics.roc_auc_score(y_test, y_pred_class)


# If you use **y_pred_class**, it will interpret the zeros and ones as predicted probabilities of 0% and 100%.

# ## Bonus: ROC curve is only sensitive to rank order of predicted probabilities

# print the first 10 predicted probabilities
y_pred_prob[:10]


# take the square root of predicted probabilities (to make them all bigger)
import numpy as np
y_pred_prob_new = np.sqrt(y_pred_prob)

# print the modified predicted probabilities
y_pred_prob_new[:10]


# histogram of predicted probabilities has changed
df = pd.DataFrame({'probability':y_pred_prob_new, 'actual':y_test})
df.hist(column='probability', by='actual', sharex=True, sharey=True)


# ROC curve did not change
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob_new)
plt.plot(fpr, tpr)


# AUC did not change
print metrics.roc_auc_score(y_test, y_pred_prob_new)


# ## Part 4: Cross-validation

# calculate cross-validated AUC
from sklearn.cross_validation import cross_val_score
cross_val_score(logreg, X, y, cv=10, scoring='roc_auc').mean()


# add Fare to the model
feature_cols = ['Pclass', 'Parch', 'Age', 'Sex_Female', 'Embarked_Q', 'Embarked_S', 'Fare']
X = titanic[feature_cols]

# recalculate AUC
cross_val_score(logreg, X, y, cv=10, scoring='roc_auc').mean()
