'''
CLASS: Kaggle Stack Overflow competition
'''

# read in the file and set the first column as the index
import pandas as pd
train = pd.read_csv('train.csv', index_col=0)
train.head()


'''
What are some assumptions and theories to test?

OwnerUserId: not unique within the dataset, assigned in order
OwnerCreationDate: users with older accounts have more open questions
ReputationAtPostCreation: higher reputation users have more open questions
OwnerUndeletedAnswerCountAtPostTime: users with more answers have more open questions
Title and BodyMarkdown: well-written questions are more likely to be open
Tags: 1 to 5 tags are required, many unique tags
OpenStatus: most questions should be open (encoded as 1)
'''

## OPEN STATUS

# dataset is perfectly balanced in terms of OpenStatus (not a representative sample)
train.OpenStatus.value_counts()


## USER ID

# OwnerUserId is not unique within the dataset, let's examine the top user
train.OwnerUserId.value_counts()

# mostly closed questions, few answers, all lowercase, grammatical mistakes
train[train.OwnerUserId==466534].describe()
train[train.OwnerUserId==466534].head(10)

# let's find a user with a high proportion of open questions
train.groupby('OwnerUserId').OpenStatus.mean()
train.groupby('OwnerUserId').OpenStatus.agg(['mean','count']).sort('count')

# lots of answers, better grammar, multiple tags, all .net
train[train.OwnerUserId==185593].head(10)


## REPUTATION

# ReputationAtPostCreation is higher for open questions: possibly use as a feature
train.groupby('OpenStatus').ReputationAtPostCreation.describe().unstack()

# not a useful histogram
train.ReputationAtPostCreation.plot(kind='hist')

# much more useful histogram
train[train.ReputationAtPostCreation < 1000].ReputationAtPostCreation.plot(kind='hist')

# grouped histogram
train[train.ReputationAtPostCreation < 1000].hist(column='ReputationAtPostCreation', by='OpenStatus', sharey=True)

# grouped box plot
train[train.ReputationAtPostCreation < 1000].boxplot(column='ReputationAtPostCreation', by='OpenStatus')


## ANSWER COUNT

# rename column
train.rename(columns={'OwnerUndeletedAnswerCountAtPostTime':'Answers'}, inplace=True)

# Answers is higher for open questions: possibly use as a feature
train.groupby('OpenStatus').Answers.describe().unstack()


## TITLE LENGTH

# create a new feature that represents the length of the title (in characters)
train['TitleLength'] = train.Title.apply(len)

# Title is longer for open questions: possibly use as a feature
train.groupby('OpenStatus').TitleLength.describe().unstack()
train.boxplot(column='TitleLength', by='OpenStatus')


'''
Define a function that takes a raw CSV file and returns a DataFrame that
includes all created features (and any other modifications)
'''

# define the function
def make_features(filename):
    df = pd.read_csv(filename, index_col=0)
    df.rename(columns={'OwnerUndeletedAnswerCountAtPostTime':'Answers'}, inplace=True)
    df['TitleLength'] = df.Title.apply(len)
    return df

# apply function to both training and testing files
train = make_features('train.csv')
test = make_features('test.csv')


'''
Evaluate a model with three features
'''

# define X and y
feature_cols = ['ReputationAtPostCreation', 'Answers', 'TitleLength']
X = train[feature_cols]
y = train.OpenStatus

# split into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# fit a logistic regression model
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=1e9)
logreg.fit(X_train, y_train)

# examine the coefficients to check that they makes sense
logreg.coef_

# predict response classes and predict class probabilities
y_pred_class = logreg.predict(X_test)
y_pred_prob = logreg.predict_proba(X_test)[:, 1]

# check how well we did
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_class)    # 0.564 (better than guessing)
metrics.confusion_matrix(y_test, y_pred_class)  # predicts closed a lot of the time
metrics.roc_auc_score(y_test, y_pred_prob)      # 0.591 (not horrible)
metrics.log_loss(y_test, y_pred_prob)           # 0.684 (what is this?)

# let's see if cross-validation gives us similar results
from sklearn.cross_validation import cross_val_score
scores = cross_val_score(logreg, X, y, scoring='log_loss', cv=10)
scores.mean()       # 0.684 (identical to train/test split)
scores.std()        # very small


'''
Understanding log loss
'''

# 5 pretend response values
y_test = [0, 0, 0, 1, 1]

# 5 sets of predicted probabilities for those observations
y_pred_prob_sets = [[0.1, 0.2, 0.3, 0.8, 0.9],
                    [0.4, 0.4, 0.4, 0.6, 0.6],
                    [0.4, 0.4, 0.7, 0.6, 0.6],
                    [0.4, 0.4, 0.9, 0.6, 0.6],
                    [0.5, 0.5, 0.5, 0.5, 0.5]]

# calculate AUC for each set of predicted probabilities
for y_pred_prob in y_pred_prob_sets:
    print y_pred_prob, metrics.roc_auc_score(y_test, y_pred_prob)

# calculate log loss for each set of predicted probabilities
for y_pred_prob in y_pred_prob_sets:
    print y_pred_prob, metrics.log_loss(y_test, y_pred_prob)


'''
Create a submission file
'''

# train the model on ALL data (not X_train and y_train)
logreg.fit(X, y)

# predict class probabilities for the actual testing data (not X_test)
X_oos = test[feature_cols]
oos_pred_prob = logreg.predict_proba(X_oos)[:, 1]

# sample submission file indicates we need two columns: PostId and predicted probability
test.index      # PostId
oos_pred_prob   # predicted probability

# create a DataFrame that has 'id' as the index, then export to a CSV file
sub = pd.DataFrame({'id':test.index, 'OpenStatus':oos_pred_prob}).set_index('id')
sub.to_csv('sub1.csv')  # 0.687


'''
Explore data and create more features
'''

## TAGS

# Tag1 is required, and the rest are optional
train.isnull().sum()

# create a new feature that represents the number of tags for each question
train['NumTags'] = train.loc[:, 'Tag1':'Tag5'].notnull().sum(axis=1)

# NumTags is higher for open questions: possibly use as a feature
train.groupby('OpenStatus').NumTags.mean()
train.groupby('NumTags').OpenStatus.mean()


## USER ID

# OwnerUserId is assigned in numerical order
train.sort('OwnerUserId').OwnerCreationDate

# OwnerUserId is lower for open questions: possibly use as a feature
train.groupby('OpenStatus').OwnerUserId.mean()

# account age at time of question is probably a better feature
train['OwnerCreationDate'] = pd.to_datetime(train.OwnerCreationDate)
train['PostCreationDate'] = pd.to_datetime(train.PostCreationDate)
train['OwnerAge'] = (train.PostCreationDate - train.OwnerCreationDate).dt.days

# check that it worked
train.head()
train.boxplot(column='OwnerAge', by='OpenStatus')
train[train.OwnerAge < 0].head()

# fix errors
import numpy as np
train['OwnerAge'] = np.where(train.OwnerAge < 0, 0, train.OwnerAge)
train.boxplot(column='OwnerAge', by='OpenStatus')


'''
Evaluate new set of features using cross-validation
'''

feature_cols = ['ReputationAtPostCreation', 'Answers', 'TitleLength', 'NumTags', 'OwnerAge']
X = train[feature_cols]
cross_val_score(logreg, X, y, scoring='log_loss', cv=10).mean()     # 0.675


'''
Update make_features and create another submission file
'''

# update the function
def make_features(filename):
    df = pd.read_csv(filename, index_col=0, parse_dates=['OwnerCreationDate', 'PostCreationDate'])
    df.rename(columns={'OwnerUndeletedAnswerCountAtPostTime':'Answers'}, inplace=True)
    df['TitleLength'] = df.Title.apply(len)
    df['NumTags'] = df.loc[:, 'Tag1':'Tag5'].notnull().sum(axis=1)
    df['OwnerAge'] = (df.PostCreationDate - df.OwnerCreationDate).dt.days
    df['OwnerAge'] = np.where(df.OwnerAge < 0, 0, df.OwnerAge)
    return df

# apply function to both training and testing files
train = make_features('train.csv')
test = make_features('test.csv')

# train the model on ALL data
feature_cols = ['ReputationAtPostCreation', 'Answers', 'TitleLength', 'NumTags', 'OwnerAge']
X = train[feature_cols]
logreg.fit(X, y)

# predict class probabilities for the actual testing data
X_oos = test[feature_cols]
oos_pred_prob = logreg.predict_proba(X_oos)[:, 1]

# create submission file
sub = pd.DataFrame({'id':test.index, 'OpenStatus':oos_pred_prob}).set_index('id')
sub.to_csv('sub2.csv')  # 0.650


'''
Build a document-term matrix from Title using CountVectorizer
'''

# use CountVectorizer with the default settings
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
dtm = vect.fit_transform(train.Title)

# define X and y
X = dtm
y = train.OpenStatus

# slightly improper cross-validation of a Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
cross_val_score(nb, X, y, scoring='log_loss', cv=10).mean()    # 0.657

# try tuning CountVectorizer and repeat Naive Bayes
vect = CountVectorizer(stop_words='english')
dtm = vect.fit_transform(train.Title)
X = dtm
cross_val_score(nb, X, y, scoring='log_loss', cv=10).mean()    # 0.635

# build document-term matrix for the actual testing data and make predictions
nb.fit(X, y)
oos_dtm = vect.transform(test.Title)
oos_pred_prob = nb.predict_proba(oos_dtm)[:, 1]
sub = pd.DataFrame({'id':test.index, 'OpenStatus':oos_pred_prob}).set_index('id')
sub.to_csv('sub3.csv')  # 0.544


'''
BONUS: Dummy encoding of Tag1
'''

# number of unique tags for Tag1 (over 5000)
train.Tag1.nunique()

# percentage of open questions varies widely by tag
train.groupby('Tag1').OpenStatus.agg(['mean','count']).sort('count')

# convert Tag1 from strings to integers
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train['Tag1_enc'] = le.fit_transform(train.Tag1)

# confirm that the conversion worked
train.Tag1.value_counts().head()
train.Tag1_enc.value_counts().head()

# create a dummy column for each value of Tag1_enc (returns a sparse matrix)
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
tag1_dummies = ohe.fit_transform(train[['Tag1_enc']])
tag1_dummies

# define X and y
X = tag1_dummies
y = train.OpenStatus

# try a Naive Bayes model with tag1_dummies as the features
cross_val_score(nb, X, y, scoring='log_loss', cv=10).mean()   # 0.650

# adjust Tag1 on testing set since LabelEncoder errors on new values during a transform
test['Tag1'] = test['Tag1'].map(lambda s: '<unknown>' if s not in le.classes_ else s)
le.classes_ = np.append(le.classes_, '<unknown>')

# apply the same encoding to the actual testing data and make predictions
nb.fit(X, y)
test['Tag1_enc'] = le.transform(test.Tag1)
oos_tag1_dummies = ohe.transform(test[['Tag1_enc']])
oos_pred_prob = nb.predict_proba(oos_tag1_dummies)[:, 1]
sub = pd.DataFrame({'id':test.index, 'OpenStatus':oos_pred_prob}).set_index('id')
sub.to_csv('sub4.csv')  # 0.652
