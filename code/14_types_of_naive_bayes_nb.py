# # Comparing Multinomial and Gaussian Naive Bayes
# 
# scikit-learn documentation: [MultinomialNB](http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html) and [GaussianNB](http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)
# 
# Dataset: [Pima Indians Diabetes](https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes) from the UCI Machine Learning Repository

# read the data
import pandas as pd
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data'
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv(url, header=None, names=col_names)


# notice that all features are continuous
pima.head()


# create X and y
X = pima.drop('label', axis=1)
y = pima.label


# split into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


# import both Multinomial and Gaussian Naive Bayes
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import metrics


# testing accuracy of Multinomial Naive Bayes
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
y_pred_class = mnb.predict(X_test)
print metrics.accuracy_score(y_test, y_pred_class)


# testing accuracy of Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_class = gnb.predict(X_test)
print metrics.accuracy_score(y_test, y_pred_class)


# **Conclusion:** When applying Naive Bayes classification to a dataset with **continuous features**, it is better to use Gaussian Naive Bayes than Multinomial Naive Bayes. The latter is suitable for datasets containing **discrete features** (e.g., word counts).
# 
# Wikipedia has a short [description](https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Gaussian_naive_Bayes) of Gaussian Naive Bayes, as well as an excellent [example](https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Sex_classification) of its usage.
