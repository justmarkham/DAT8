# # Exercise with bank marketing data

# ## Introduction
# 
# - Data from the UCI Machine Learning Repository: [data](https://github.com/justmarkham/DAT8/blob/master/data/bank-additional.csv), [data dictionary](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
# - **Goal:** Predict whether a customer will purchase a bank product marketed over the phone
# - `bank-additional.csv` is already in our repo, so there is no need to download the data from the UCI website

# ## Step 1: Read the data into Pandas

import pandas as pd
url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/bank-additional.csv'
bank = pd.read_csv(url, sep=';')
bank.head()


# ## Step 2: Prepare at least three features
# 
# - Include both numeric and categorical features
# - Choose features that you think might be related to the response (based on intuition or exploration)
# - Think about how to handle missing values (encoded as "unknown")

# list all columns (for reference)
bank.columns


# ### y (response)

# convert the response to numeric values and store as a new column
bank['outcome'] = bank.y.map({'no':0, 'yes':1})


# ### age

# probably not a great feature
bank.boxplot(column='age', by='outcome')


# ### job

# looks like a useful feature
bank.groupby('job').outcome.mean()


# create job_dummies (we will add it to the bank DataFrame later)
job_dummies = pd.get_dummies(bank.job, prefix='job')
job_dummies.drop(job_dummies.columns[0], axis=1, inplace=True)


# ### default

# looks like a useful feature
bank.groupby('default').outcome.mean()


# but only one person in the dataset has a status of yes
bank.default.value_counts()


# so, let's treat this as a 2-class feature rather than a 3-class feature
bank['default'] = bank.default.map({'no':0, 'unknown':1, 'yes':1})


# ### contact

# looks like a useful feature
bank.groupby('contact').outcome.mean()


# convert the feature to numeric values
bank['contact'] = bank.contact.map({'cellular':0, 'telephone':1})


# ### month

# looks like a useful feature at first glance
bank.groupby('month').outcome.mean()


# but, it looks like their success rate is actually just correlated with number of calls
# thus, the month feature is unlikely to generalize
bank.groupby('month').outcome.agg(['count', 'mean']).sort('count')


# ### duration

# looks like an excellent feature, but you can't know the duration of a call beforehand, thus it can't be used in your model
bank.boxplot(column='duration', by='outcome')


# ### previous

# looks like a useful feature
bank.groupby('previous').outcome.mean()


# ### poutcome

# looks like a useful feature
bank.groupby('poutcome').outcome.mean()


# create poutcome_dummies
poutcome_dummies = pd.get_dummies(bank.poutcome, prefix='poutcome')
poutcome_dummies.drop(poutcome_dummies.columns[0], axis=1, inplace=True)


# concatenate bank DataFrame with job_dummies and poutcome_dummies
bank = pd.concat([bank, job_dummies, poutcome_dummies], axis=1)


# ### euribor3m

# looks like an excellent feature
bank.boxplot(column='euribor3m', by='outcome')


# ## Step 3: Model building
# 
# - Use cross-validation to evaluate the AUC of a logistic regression model with your chosen features
# - Try to increase the AUC by selecting different sets of features

# new list of columns (including dummy columns)
bank.columns


# create X (including 13 dummy columns)
feature_cols = ['default', 'contact', 'previous', 'euribor3m'] + list(bank.columns[-13:])
X = bank[feature_cols]


# create y
y = bank.outcome


# calculate cross-validated AUC
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
logreg = LogisticRegression(C=1e9)
cross_val_score(logreg, X, y, cv=10, scoring='roc_auc').mean()
