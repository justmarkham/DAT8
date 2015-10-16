# # Exercise with Capital Bikeshare data

# ## Introduction
# 
# - Capital Bikeshare dataset from Kaggle: [data](https://github.com/justmarkham/DAT8/blob/master/data/bikeshare.csv), [data dictionary](https://www.kaggle.com/c/bike-sharing-demand/data)
# - Each observation represents the bikeshare rentals initiated during a given hour of a given day

import pandas as pd
import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, export_graphviz


# read the data and set "datetime" as the index
url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/bikeshare.csv'
bikes = pd.read_csv(url, index_col='datetime', parse_dates=True)


# "count" is a method, so it's best to rename that column
bikes.rename(columns={'count':'total'}, inplace=True)


# create "hour" as its own feature
bikes['hour'] = bikes.index.hour


bikes.head()


bikes.tail()


# - **hour** ranges from 0 (midnight) through 23 (11pm)
# - **workingday** is either 0 (weekend or holiday) or 1 (non-holiday weekday)

# ## Task 1
# 
# Run these two `groupby` statements and figure out what they tell you about the data.

# mean rentals for each value of "workingday"
bikes.groupby('workingday').total.mean()


# mean rentals for each value of "hour"
bikes.groupby('hour').total.mean()


# ## Task 2
# 
# Run this plotting code, and make sure you understand the output. Then, separate this plot into two separate plots conditioned on "workingday". (In other words, one plot should display the hourly trend for "workingday=0", and the other should display the hourly trend for "workingday=1".)

# mean rentals for each value of "hour"
bikes.groupby('hour').total.mean().plot()


# hourly rental trend for "workingday=0"
bikes[bikes.workingday==0].groupby('hour').total.mean().plot()


# hourly rental trend for "workingday=1"
bikes[bikes.workingday==1].groupby('hour').total.mean().plot()


# combine the two plots
bikes.groupby(['hour', 'workingday']).total.mean().unstack().plot()


# ## Task 3
# 
# Fit a linear regression model to the entire dataset, using "total" as the response and "hour" and "workingday" as the only features. Then, print the coefficients and interpret them. What are the limitations of linear regression in this instance?

# create X and y
feature_cols = ['hour', 'workingday']
X = bikes[feature_cols]
y = bikes.total


# fit a linear regression model and print coefficients
linreg = LinearRegression()
linreg.fit(X, y)
linreg.coef_


# ## Task 4
# 
# Use 10-fold cross-validation to calculate the RMSE for the linear regression model.

# save the 10 MSE scores output by cross_val_score
scores = cross_val_score(linreg, X, y, cv=10, scoring='mean_squared_error')


# convert MSE to RMSE, and then calculate the mean of the 10 RMSE scores
np.mean(np.sqrt(-scores))


# ## Task 5
# 
# Use 10-fold cross-validation to evaluate a decision tree model with those same features (fit to any "max_depth" you choose).

# evaluate a decision tree model with "max_depth=7"
treereg = DecisionTreeRegressor(max_depth=7, random_state=1)
scores = cross_val_score(treereg, X, y, cv=10, scoring='mean_squared_error')
np.mean(np.sqrt(-scores))


# ## Task 6
# 
# Fit a decision tree model to the entire dataset using "max_depth=3", and create a tree diagram using Graphviz. Then, figure out what each leaf represents. What did the decision tree learn that a linear regression model could not learn?

# fit a decision tree model with "max_depth=3"
treereg = DecisionTreeRegressor(max_depth=3, random_state=1)
treereg.fit(X, y)


# create a Graphviz file
export_graphviz(treereg, out_file='tree_bikeshare.dot', feature_names=feature_cols)

# At the command line, run this to convert to PNG:
#   dot -Tpng tree_bikeshare.dot -o tree_bikeshare.png


# ![Tree for bikeshare data](images/tree_bikeshare.png)
