# # Linear regression homework with Yelp votes

# ## Introduction
# 
# This assignment uses a small subset of the data from Kaggle's [Yelp Business Rating Prediction](https://www.kaggle.com/c/yelp-recsys-2013) competition.
# 
# **Description of the data:**
# 
# - `yelp.json` is the original format of the file. `yelp.csv` contains the same data, in a more convenient format. Both of the files are in this repo, so there is no need to download the data from the Kaggle website.
# - Each observation in this dataset is a review of a particular business by a particular user.
# - The "stars" column is the number of stars (1 through 5) assigned by the reviewer to the business. (Higher stars is better.) In other words, it is the rating of the business by the person who wrote the review.
# - The "cool" column is the number of "cool" votes this review received from other Yelp users. All reviews start with 0 "cool" votes, and there is no limit to how many "cool" votes a review can receive. In other words, it is a rating of the review itself, not a rating of the business.
# - The "useful" and "funny" columns are similar to the "cool" column.

# ## Task 1
# 
# Read `yelp.csv` into a DataFrame.

# access yelp.csv using a relative path
import pandas as pd
yelp = pd.read_csv('../data/yelp.csv')
yelp.head(1)


# ## Task 1 (Bonus)
# 
# Ignore the `yelp.csv` file, and construct this DataFrame yourself from `yelp.json`. This involves reading the data into Python, decoding the JSON, converting it to a DataFrame, and adding individual columns for each of the vote types.

# read the data from yelp.json into a list of rows
# each row is decoded into a dictionary using using json.loads()
import json
with open('../data/yelp.json', 'rU') as f:
    data = [json.loads(row) for row in f]


# show the first review
data[0]


# convert the list of dictionaries to a DataFrame
yelp = pd.DataFrame(data)
yelp.head(1)


# add DataFrame columns for cool, useful, and funny
yelp['cool'] = [row['votes']['cool'] for row in data]
yelp['useful'] = [row['votes']['useful'] for row in data]
yelp['funny'] = [row['votes']['funny'] for row in data]


# drop the votes column
yelp.drop('votes', axis=1, inplace=True)
yelp.head(1)


# ## Task 2
# 
# Explore the relationship between each of the vote types (cool/useful/funny) and the number of stars.

# treat stars as a categorical variable and look for differences between groups
yelp.groupby('stars').mean()


# correlation matrix
import seaborn as sns
sns.heatmap(yelp.corr())


# multiple scatter plots
sns.pairplot(yelp, x_vars=['cool', 'useful', 'funny'], y_vars='stars', size=6, aspect=0.7, kind='reg')


# ## Task 3
# 
# Define cool/useful/funny as the features, and stars as the response.

feature_cols = ['cool', 'useful', 'funny']
X = yelp[feature_cols]
y = yelp.stars


# ## Task 4
# 
# Fit a linear regression model and interpret the coefficients. Do the coefficients make intuitive sense to you? Explore the Yelp website to see if you detect similar trends.

from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X, y)
zip(feature_cols, linreg.coef_)


# ## Task 5
# 
# Evaluate the model by splitting it into training and testing sets and computing the RMSE. Does the RMSE make intuitive sense to you?

from sklearn.cross_validation import train_test_split
from sklearn import metrics
import numpy as np


# define a function that accepts a list of features and returns testing RMSE
def train_test_rmse(feature_cols):
    X = yelp[feature_cols]
    y = yelp.stars
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    y_pred = linreg.predict(X_test)
    return np.sqrt(metrics.mean_squared_error(y_test, y_pred))


# calculate RMSE with all three features
train_test_rmse(['cool', 'useful', 'funny'])


# ## Task 6
# 
# Try removing some of the features and see if the RMSE improves.

print train_test_rmse(['cool', 'useful'])
print train_test_rmse(['cool', 'funny'])
print train_test_rmse(['useful', 'funny'])


# ## Task 7 (Bonus)
# 
# Think of some new features you could create from the existing data that might be predictive of the response. Figure out how to create those features in Pandas, add them to your model, and see if the RMSE improves.

# new feature: review length (number of characters)
yelp['length'] = yelp.text.apply(len)


# new features: whether or not the review contains 'love' or 'hate'
yelp['love'] = yelp.text.str.contains('love', case=False).astype(int)
yelp['hate'] = yelp.text.str.contains('hate', case=False).astype(int)


# add new features to the model and calculate RMSE
train_test_rmse(['cool', 'useful', 'funny', 'length', 'love', 'hate'])


# ## Task 8 (Bonus)
# 
# Compare your best RMSE on the testing set with the RMSE for the "null model", which is the model that ignores all features and simply predicts the mean response value in the testing set.

# split the data (outside of the function)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


# create a NumPy array with the same shape as y_test
y_null = np.zeros_like(y_test, dtype=float)


# fill the array with the mean of y_test
y_null.fill(y_test.mean())


# calculate null RMSE
print np.sqrt(metrics.mean_squared_error(y_test, y_null))


# ## Task 9 (Bonus)
# 
# Instead of treating this as a regression problem, treat it as a classification problem and see what testing accuracy you can achieve with KNN.

# import and instantiate KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=50)


# classification models will automatically treat the response value (1/2/3/4/5) as unordered categories
knn.fit(X_train, y_train)
y_pred_class = knn.predict(X_test)
print metrics.accuracy_score(y_test, y_pred_class)


# ## Task 10 (Bonus)
# 
# Figure out how to use linear regression for classification, and compare its classification accuracy with KNN's accuracy.

# use linear regression to make continuous predictions
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)


# round its predictions to the nearest integer
y_pred_class = y_pred.round()


# calculate classification accuracy of the rounded predictions
print metrics.accuracy_score(y_test, y_pred_class)
