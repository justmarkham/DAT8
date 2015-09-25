# # Working with Text Data and Naive Bayes in scikit-learn

# ## Agenda
# 
# **Working with text data**
# 
# - Representing text as data
# - Reading SMS data
# - Vectorizing SMS data
# - Examining the tokens and their counts
# - Bonus: Calculating the "spamminess" of each token
# 
# **Naive Bayes classification**
# 
# - Building a Naive Bayes model
# - Comparing Naive Bayes with logistic regression

# ## Part 1: Representing text as data
# 
# From the [scikit-learn documentation](http://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction):
# 
# > Text Analysis is a major application field for machine learning algorithms. However the raw data, a sequence of symbols cannot be fed directly to the algorithms themselves as most of them expect **numerical feature vectors with a fixed size** rather than the **raw text documents with variable length**.
# 
# We will use [CountVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) to "convert text into a matrix of token counts":

from sklearn.feature_extraction.text import CountVectorizer


# start with a simple example
simple_train = ['call you tonight', 'Call me a cab', 'please call me... PLEASE!']


# learn the 'vocabulary' of the training data
vect = CountVectorizer()
vect.fit(simple_train)
vect.get_feature_names()


# transform training data into a 'document-term matrix'
simple_train_dtm = vect.transform(simple_train)
simple_train_dtm


# print the sparse matrix
print simple_train_dtm


# convert sparse matrix to a dense matrix
simple_train_dtm.toarray()


# examine the vocabulary and document-term matrix together
import pandas as pd
pd.DataFrame(simple_train_dtm.toarray(), columns=vect.get_feature_names())


# From the [scikit-learn documentation](http://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction):
# 
# > In this scheme, features and samples are defined as follows:
# 
# > - Each individual token occurrence frequency (normalized or not) is treated as a **feature**.
# > - The vector of all the token frequencies for a given document is considered a multivariate **sample**.
# 
# > A **corpus of documents** can thus be represented by a matrix with **one row per document** and **one column per token** (e.g. word) occurring in the corpus.
# 
# > We call **vectorization** the general process of turning a collection of text documents into numerical feature vectors. This specific strategy (tokenization, counting and normalization) is called the **Bag of Words** or "Bag of n-grams" representation. Documents are described by word occurrences while completely ignoring the relative position information of the words in the document.

# transform testing data into a document-term matrix (using existing vocabulary)
simple_test = ["please don't call me"]
simple_test_dtm = vect.transform(simple_test)
simple_test_dtm.toarray()


# examine the vocabulary and document-term matrix together
pd.DataFrame(simple_test_dtm.toarray(), columns=vect.get_feature_names())


# **Summary:**
# 
# - `vect.fit(train)` learns the vocabulary of the training data
# - `vect.transform(train)` uses the fitted vocabulary to build a document-term matrix from the training data
# - `vect.transform(test)` uses the fitted vocabulary to build a document-term matrix from the testing data (and ignores tokens it hasn't seen before)

# ## Part 2: Reading SMS data

# read tab-separated file
url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv'
col_names = ['label', 'message']
sms = pd.read_table(url, sep='\t', header=None, names=col_names)
print sms.shape


sms.head(20)


sms.label.value_counts()


# convert label to a numeric variable
sms['label'] = sms.label.map({'ham':0, 'spam':1})


# define X and y
X = sms.message
y = sms.label


# split into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print X_train.shape
print X_test.shape


# ## Part 3: Vectorizing SMS data

# instantiate the vectorizer
vect = CountVectorizer()


# learn training data vocabulary, then create document-term matrix
vect.fit(X_train)
X_train_dtm = vect.transform(X_train)
X_train_dtm


# alternative: combine fit and transform into a single step
X_train_dtm = vect.fit_transform(X_train)
X_train_dtm


# transform testing data (using fitted vocabulary) into a document-term matrix
X_test_dtm = vect.transform(X_test)
X_test_dtm


# ## Part 4: Examining the tokens and their counts

# store token names
X_train_tokens = vect.get_feature_names()


# first 50 tokens
print X_train_tokens[:50]


# last 50 tokens
print X_train_tokens[-50:]


# view X_train_dtm as a dense matrix
X_train_dtm.toarray()


# count how many times EACH token appears across ALL messages in X_train_dtm
import numpy as np
X_train_counts = np.sum(X_train_dtm.toarray(), axis=0)
X_train_counts


X_train_counts.shape


# create a DataFrame of tokens with their counts
pd.DataFrame({'token':X_train_tokens, 'count':X_train_counts}).sort('count')


# ## Bonus: Calculating the "spamminess" of each token

# create separate DataFrames for ham and spam
sms_ham = sms[sms.label==0]
sms_spam = sms[sms.label==1]


# learn the vocabulary of ALL messages and save it
vect.fit(sms.message)
all_tokens = vect.get_feature_names()


# create document-term matrices for ham and spam
ham_dtm = vect.transform(sms_ham.message)
spam_dtm = vect.transform(sms_spam.message)


# count how many times EACH token appears across ALL ham messages
ham_counts = np.sum(ham_dtm.toarray(), axis=0)


# count how many times EACH token appears across ALL spam messages
spam_counts = np.sum(spam_dtm.toarray(), axis=0)


# create a DataFrame of tokens with their separate ham and spam counts
token_counts = pd.DataFrame({'token':all_tokens, 'ham':ham_counts, 'spam':spam_counts})


# add one to ham and spam counts to avoid dividing by zero (in the step that follows)
token_counts['ham'] = token_counts.ham + 1
token_counts['spam'] = token_counts.spam + 1


# calculate ratio of spam-to-ham for each token
token_counts['spam_ratio'] = token_counts.spam / token_counts.ham
token_counts.sort('spam_ratio')


# ## Part 5: Building a Naive Bayes model
# 
# We will use [Multinomial Naive Bayes](http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html):
# 
# > The multinomial Naive Bayes classifier is suitable for classification with **discrete features** (e.g., word counts for text classification). The multinomial distribution normally requires integer feature counts. However, in practice, fractional counts such as tf-idf may also work.

# train a Naive Bayes model using X_train_dtm
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train_dtm, y_train)


# make class predictions for X_test_dtm
y_pred_class = nb.predict(X_test_dtm)


# calculate accuracy of class predictions
from sklearn import metrics
print metrics.accuracy_score(y_test, y_pred_class)


# confusion matrix
print metrics.confusion_matrix(y_test, y_pred_class)


# predict (poorly calibrated) probabilities
y_pred_prob = nb.predict_proba(X_test_dtm)[:, 1]
y_pred_prob


# calculate AUC
print metrics.roc_auc_score(y_test, y_pred_prob)


# print message text for the false positives
X_test[y_test < y_pred_class]


# print message text for the false negatives
X_test[y_test > y_pred_class]


# what do you notice about the false negatives?
X_test[3132]


# ## Part 6: Comparing Naive Bayes with logistic regression

# import/instantiate/fit
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=1e9)
logreg.fit(X_train_dtm, y_train)


# class predictions and predicted probabilities
y_pred_class = logreg.predict(X_test_dtm)
y_pred_prob = logreg.predict_proba(X_test_dtm)[:, 1]


# calculate accuracy and AUC
print metrics.accuracy_score(y_test, y_pred_class)
print metrics.roc_auc_score(y_test, y_pred_prob)
