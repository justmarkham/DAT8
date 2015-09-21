# # Linear Regression

# ## Agenda
# 
# 1. Introducing the bikeshare dataset
#     - Reading in the data
#     - Visualizing the data
# 2. Linear regression basics
#     - Form of linear regression
#     - Building a linear regression model
#     - Using the model for prediction
#     - Does the scale of the features matter?
# 3. Working with multiple features
#     - Visualizing the data (part 2)
#     - Adding more features to the model
# 4. Choosing between models
#     - Feature selection
#     - Evaluation metrics for regression problems
#     - Comparing models with train/test split and RMSE
#     - Comparing testing RMSE with null RMSE
# 5. Creating features
#     - Handling categorical features
#     - Feature engineering
# 6. Comparing linear regression with other models

# ## Reading in the data
# 
# We'll be working with a dataset from Capital Bikeshare that was used in a Kaggle competition ([data dictionary](https://www.kaggle.com/c/bike-sharing-demand/data)).

# read the data and set the datetime as the index
import pandas as pd
url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/bikeshare.csv'
bikes = pd.read_csv(url, index_col='datetime', parse_dates=True)


bikes.head()


# **Questions:**
# 
# - What does each observation represent?
# - What is the response variable (as defined by Kaggle)?
# - How many features are there?

# "count" is a method, so it's best to name that column something else
bikes.rename(columns={'count':'total'}, inplace=True)


# ## Visualizing the data

import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 14


# Pandas scatter plot
bikes.plot(kind='scatter', x='temp', y='total', alpha=0.2)


# Seaborn scatter plot with regression line
sns.lmplot(x='temp', y='total', data=bikes, aspect=1.5, scatter_kws={'alpha':0.2})


# ## Form of linear regression
# 
# $y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n$
# 
# - $y$ is the response
# - $\beta_0$ is the intercept
# - $\beta_1$ is the coefficient for $x_1$ (the first feature)
# - $\beta_n$ is the coefficient for $x_n$ (the nth feature)
# 
# The $\beta$ values are called the **model coefficients**:
# 
# - These values are estimated (or "learned") during the model fitting process using the **least squares criterion**.
# - Specifically, we are find the line (mathematically) which minimizes the **sum of squared residuals** (or "sum of squared errors").
# - And once we've learned these coefficients, we can use the model to predict the response.
# 
# ![Estimating coefficients](images/estimating_coefficients.png)
# 
# In the diagram above:
# 
# - The black dots are the **observed values** of x and y.
# - The blue line is our **least squares line**.
# - The red lines are the **residuals**, which are the vertical distances between the observed values and the least squares line.

# ## Building a linear regression model

# create X and y
feature_cols = ['temp']
X = bikes[feature_cols]
y = bikes.total


# import, instantiate, fit
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X, y)


# print the coefficients
print linreg.intercept_
print linreg.coef_


# Interpreting the **intercept** ($\beta_0$):
# 
# - It is the value of $y$ when $x$=0.
# - Thus, it is the estimated number of rentals when the temperature is 0 degrees Celsius.
# - **Note:** It does not always make sense to interpret the intercept. (Why?)
# 
# Interpreting the **"temp" coefficient** ($\beta_1$):
# 
# - It is the change in $y$ divided by change in $x$, or the "slope".
# - Thus, a temperature increase of 1 degree Celsius is **associated with** a rental increase of 9.17 bikes.
# - This is not a statement of causation.
# - $\beta_1$ would be **negative** if an increase in temperature was associated with a **decrease** in rentals.

# ## Using the model for prediction
# 
# How many bike rentals would we predict if the temperature was 25 degrees Celsius?

# manually calculate the prediction
linreg.intercept_ + linreg.coef_*25


# use the predict method
linreg.predict(25)


# ## Does the scale of the features matter?
# 
# Let's say that temperature was measured in Fahrenheit, rather than Celsius. How would that affect the model?

# create a new column for Fahrenheit temperature
bikes['temp_F'] = bikes.temp * 1.8 + 32
bikes.head()


# Seaborn scatter plot with regression line
sns.lmplot(x='temp_F', y='total', data=bikes, aspect=1.5, scatter_kws={'alpha':0.2})


# create X and y
feature_cols = ['temp_F']
X = bikes[feature_cols]
y = bikes.total

# instantiate and fit
linreg = LinearRegression()
linreg.fit(X, y)

# print the coefficients
print linreg.intercept_
print linreg.coef_


# convert 25 degrees Celsius to Fahrenheit
25 * 1.8 + 32


# predict rentals for 77 degrees Fahrenheit
linreg.predict(77)


# **Conclusion:** The scale of the features is **irrelevant** for linear regression models. When changing the scale, we simply change our **interpretation** of the coefficients.

# remove the temp_F column
bikes.drop('temp_F', axis=1, inplace=True)


# ## Visualizing the data (part 2)

# explore more features
feature_cols = ['temp', 'season', 'weather', 'humidity']


# multiple scatter plots in Seaborn
sns.pairplot(bikes, x_vars=feature_cols, y_vars='total', kind='reg')


# multiple scatter plots in Pandas
fig, axs = plt.subplots(1, len(feature_cols), sharey=True)
for index, feature in enumerate(feature_cols):
    bikes.plot(kind='scatter', x=feature, y='total', ax=axs[index], figsize=(16, 3))


# Are you seeing anything that you did not expect?

# cross-tabulation of season and month
pd.crosstab(bikes.season, bikes.index.month)


# box plot of rentals, grouped by season
bikes.boxplot(column='total', by='season')


# Notably:
# 
# - A line can't capture a non-linear relationship.
# - There are more rentals in winter than in spring (?)

# line plot of rentals
bikes.total.plot()


# What does this tell us?
# 
# There are more rentals in the winter than the spring, but only because the system is experiencing **overall growth** and the winter months happen to come after the spring months.

# correlation matrix (ranges from 1 to -1)
bikes.corr()


# visualize correlation matrix in Seaborn using a heatmap
sns.heatmap(bikes.corr())


# What relationships do you notice?

# ## Adding more features to the model

# create a list of features
feature_cols = ['temp', 'season', 'weather', 'humidity']


# create X and y
X = bikes[feature_cols]
y = bikes.total

# instantiate and fit
linreg = LinearRegression()
linreg.fit(X, y)

# print the coefficients
print linreg.intercept_
print linreg.coef_


# pair the feature names with the coefficients
zip(feature_cols, linreg.coef_)


# Interpreting the coefficients:
# 
# - Holding all other features fixed, a 1 unit increase in **temperature** is associated with a **rental increase of 7.86 bikes**.
# - Holding all other features fixed, a 1 unit increase in **season** is associated with a **rental increase of 22.5 bikes**.
# - Holding all other features fixed, a 1 unit increase in **weather** is associated with a **rental increase of 6.67 bikes**.
# - Holding all other features fixed, a 1 unit increase in **humidity** is associated with a **rental decrease of 3.12 bikes**.
# 
# Does anything look incorrect?

# ## Feature selection
# 
# How do we choose which features to include in the model? We're going to use **train/test split** (and eventually **cross-validation**).
# 
# Why not use of **p-values** or **R-squared** for feature selection?
# 
# - Linear models rely upon **a lot of assumptions** (such as the features being independent), and if those assumptions are violated, p-values and R-squared are less reliable. Train/test split relies on fewer assumptions.
# - Features that are unrelated to the response can still have **significant p-values**.
# - Adding features to your model that are unrelated to the response will always **increase the R-squared value**, and adjusted R-squared does not sufficiently account for this.
# - p-values and R-squared are **proxies** for our goal of generalization, whereas train/test split and cross-validation attempt to **directly estimate** how well the model will generalize to out-of-sample data.
# 
# More generally:
# 
# - There are different methodologies that can be used for solving any given data science problem, and this course follows a **machine learning methodology**.
# - This course focuses on **general purpose approaches** that can be applied to any model, rather than model-specific approaches.

# ## Evaluation metrics for regression problems
# 
# Evaluation metrics for classification problems, such as **accuracy**, are not useful for regression problems. We need evaluation metrics designed for comparing **continuous values**.
# 
# Here are three common evaluation metrics for regression problems:
# 
# **Mean Absolute Error** (MAE) is the mean of the absolute value of the errors:
# 
# $$\frac 1n\sum_{i=1}^n|y_i-\hat{y}_i|$$
# 
# **Mean Squared Error** (MSE) is the mean of the squared errors:
# 
# $$\frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2$$
# 
# **Root Mean Squared Error** (RMSE) is the square root of the mean of the squared errors:
# 
# $$\sqrt{\frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2}$$

# example true and predicted response values
true = [10, 7, 5, 5]
pred = [8, 6, 5, 10]


# calculate these metrics by hand!
from sklearn import metrics
import numpy as np
print 'MAE:', metrics.mean_absolute_error(true, pred)
print 'MSE:', metrics.mean_squared_error(true, pred)
print 'RMSE:', np.sqrt(metrics.mean_squared_error(true, pred))


# Comparing these metrics:
# 
# - **MAE** is the easiest to understand, because it's the average error.
# - **MSE** is more popular than MAE, because MSE "punishes" larger errors, which tends to be useful in the real world.
# - **RMSE** is even more popular than MSE, because RMSE is interpretable in the "y" units.
# 
# All of these are **loss functions**, because we want to minimize them.
# 
# Here's an additional example, to demonstrate how MSE/RMSE punish larger errors:

# same true values as above
true = [10, 7, 5, 5]

# new set of predicted values
pred = [10, 7, 5, 13]

# MAE is the same as before
print 'MAE:', metrics.mean_absolute_error(true, pred)

# MSE and RMSE are larger than before
print 'MSE:', metrics.mean_squared_error(true, pred)
print 'RMSE:', np.sqrt(metrics.mean_squared_error(true, pred))


# ## Comparing models with train/test split and RMSE

from sklearn.cross_validation import train_test_split

# define a function that accepts a list of features and returns testing RMSE
def train_test_rmse(feature_cols):
    X = bikes[feature_cols]
    y = bikes.total
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    y_pred = linreg.predict(X_test)
    return np.sqrt(metrics.mean_squared_error(y_test, y_pred))


# compare different sets of features
print train_test_rmse(['temp', 'season', 'weather', 'humidity'])
print train_test_rmse(['temp', 'season', 'weather'])
print train_test_rmse(['temp', 'season', 'humidity'])


# using these as features is not allowed!
print train_test_rmse(['casual', 'registered'])


# ## Comparing testing RMSE with null RMSE
# 
# Null RMSE is the RMSE that could be achieved by **always predicting the mean response value**. It is a benchmark against which you may want to measure your regression model.

# split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)

# create a NumPy array with the same shape as y_test
y_null = np.zeros_like(y_test, dtype=float)

# fill the array with the mean value of y_test
y_null.fill(y_test.mean())
y_null


# compute null RMSE
np.sqrt(metrics.mean_squared_error(y_test, y_null))


# ## Handling categorical features
# 
# scikit-learn expects all features to be numeric. So how do we include a categorical feature in our model?
# 
# - **Ordered categories:** transform them to sensible numeric values (example: small=1, medium=2, large=3)
# - **Unordered categories:** use dummy encoding (0/1)
# 
# What are the categorical features in our dataset?
# 
# - **Ordered categories:** weather (already encoded with sensible numeric values)
# - **Unordered categories:** season (needs dummy encoding), holiday (already dummy encoded), workingday (already dummy encoded)
# 
# For season, we can't simply leave the encoding as 1 = spring, 2 = summer, 3 = fall, and 4 = winter, because that would imply an **ordered relationship**. Instead, we create **multiple dummy variables:**

# create dummy variables
season_dummies = pd.get_dummies(bikes.season, prefix='season')

# print 5 random rows
season_dummies.sample(n=5, random_state=1)


# However, we actually only need **three dummy variables (not four)**, and thus we'll drop the first dummy variable.
# 
# Why? Because three dummies captures all of the "information" about the season feature, and implicitly defines spring (season 1) as the **baseline level:**

# drop the first column
season_dummies.drop(season_dummies.columns[0], axis=1, inplace=True)

# print 5 random rows
season_dummies.sample(n=5, random_state=1)


# In general, if you have a categorical feature with **k possible values**, you create **k-1 dummy variables**.
# 
# If that's confusing, think about why we only need one dummy variable for holiday, not two dummy variables (holiday_yes and holiday_no).

# concatenate the original DataFrame and the dummy DataFrame (axis=0 means rows, axis=1 means columns)
bikes = pd.concat([bikes, season_dummies], axis=1)

# print 5 random rows
bikes.sample(n=5, random_state=1)


# include dummy variables for season in the model
feature_cols = ['temp', 'season_2', 'season_3', 'season_4', 'humidity']
X = bikes[feature_cols]
y = bikes.total
linreg = LinearRegression()
linreg.fit(X, y)
zip(feature_cols, linreg.coef_)


# How do we interpret the season coefficients? They are **measured against the baseline (spring)**:
# 
# - Holding all other features fixed, **summer** is associated with a **rental decrease of 3.39 bikes** compared to the spring.
# - Holding all other features fixed, **fall** is associated with a **rental decrease of 41.7 bikes** compared to the spring.
# - Holding all other features fixed, **winter** is associated with a **rental increase of 64.4 bikes** compared to the spring.
# 
# Would it matter if we changed which season was defined as the baseline?
# 
# - No, it would simply change our **interpretation** of the coefficients.
# 
# **Important:** Dummy encoding is relevant for all machine learning models, not just linear regression models.

# compare original season variable with dummy variables
print train_test_rmse(['temp', 'season', 'humidity'])
print train_test_rmse(['temp', 'season_2', 'season_3', 'season_4', 'humidity'])


# ## Feature engineering
# 
# See if you can create the following features:
# 
# - **hour:** as a single numeric feature (0 through 23)
# - **hour:** as a categorical feature (use 23 dummy variables)
# - **daytime:** as a single categorical feature (daytime=1 from 7am to 8pm, and daytime=0 otherwise)
# 
# Then, try using each of the three features (on its own) with `train_test_rmse` to see which one performs the best!

# hour as a numeric feature
bikes['hour'] = bikes.index.hour


# hour as a categorical feature
hour_dummies = pd.get_dummies(bikes.hour, prefix='hour')
hour_dummies.drop(hour_dummies.columns[0], axis=1, inplace=True)
bikes = pd.concat([bikes, hour_dummies], axis=1)


# daytime as a categorical feature
bikes['daytime'] = ((bikes.hour > 6) & (bikes.hour < 21)).astype(int)


print train_test_rmse(['hour'])
print train_test_rmse(bikes.columns[bikes.columns.str.startswith('hour_')])
print train_test_rmse(['daytime'])


# ## Comparing linear regression with other models
# 
# Advantages of linear regression:
# 
# - Simple to explain
# - Highly interpretable
# - Model training and prediction are fast
# - No tuning is required (excluding regularization)
# - Features don't need scaling
# - Can perform well with a small number of observations
# - Well-understood
# 
# Disadvantages of linear regression:
# 
# - Presumes a linear relationship between the features and the response
# - Performance is (generally) not competitive with the best supervised learning methods due to high bias
# - Can't automatically learn feature interactions
