# # Logistic Regression

# ## Agenda
# 
# 1. Refresh your memory on how to do linear regression in scikit-learn
# 2. Attempt to use linear regression for classification
# 3. Show you why logistic regression is a better alternative for classification
# 4. Brief overview of probability, odds, e, log, and log-odds
# 5. Explain the form of logistic regression
# 6. Explain how to interpret logistic regression coefficients
# 7. Demonstrate how logistic regression works with categorical features
# 8. Compare logistic regression with other models

# ## Part 1: Predicting a Continuous Response

# glass identification dataset
import pandas as pd
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data'
col_names = ['id','ri','na','mg','al','si','k','ca','ba','fe','glass_type']
glass = pd.read_csv(url, names=col_names, index_col='id')
glass.sort('al', inplace=True)
glass.head()


# **Question:** Pretend that we want to predict **ri**, and our only feature is **al**. How could we do it using machine learning?
# 
# **Answer:** We could frame it as a regression problem, and use a linear regression model with **al** as the only feature and **ri** as the response.
# 
# **Question:** How would we **visualize** this model?
# 
# **Answer:** Create a scatter plot with **al** on the x-axis and **ri** on the y-axis, and draw the line of best fit.

import seaborn as sns
import matplotlib.pyplot as plt
sns.set(font_scale=1.5)


sns.lmplot(x='al', y='ri', data=glass, ci=None)


# **Question:** How would we draw this plot without using Seaborn?

# scatter plot using Pandas
glass.plot(kind='scatter', x='al', y='ri')


# equivalent scatter plot using Matplotlib
plt.scatter(glass.al, glass.ri)
plt.xlabel('al')
plt.ylabel('ri')


# fit a linear regression model
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
feature_cols = ['al']
X = glass[feature_cols]
y = glass.ri
linreg.fit(X, y)


# make predictions for all values of X
glass['ri_pred'] = linreg.predict(X)
glass.head()


# plot those predictions connected by a line
plt.plot(glass.al, glass.ri_pred, color='red')
plt.xlabel('al')
plt.ylabel('Predicted ri')


# put the plots together
plt.scatter(glass.al, glass.ri)
plt.plot(glass.al, glass.ri_pred, color='red')
plt.xlabel('al')
plt.ylabel('ri')


# ### Refresher: interpreting linear regression coefficients

# Linear regression equation: $y = \beta_0 + \beta_1x$

# compute prediction for al=2 using the equation
linreg.intercept_ + linreg.coef_ * 2


# compute prediction for al=2 using the predict method
linreg.predict(2)


# examine coefficient for al
zip(feature_cols, linreg.coef_)


# **Interpretation:** A 1 unit increase in 'al' is associated with a 0.0025 unit decrease in 'ri'.

# increasing al by 1 (so that al=3) decreases ri by 0.0025
1.51699012 - 0.0024776063874696243


# compute prediction for al=3 using the predict method
linreg.predict(3)


# ## Part 2: Predicting a Categorical Response

# examine glass_type
glass.glass_type.value_counts().sort_index()


# types 1, 2, 3 are window glass
# types 5, 6, 7 are household glass
glass['household'] = glass.glass_type.map({1:0, 2:0, 3:0, 5:1, 6:1, 7:1})
glass.head()


# Let's change our task, so that we're predicting **household** using **al**. Let's visualize the relationship to figure out how to do this:

plt.scatter(glass.al, glass.household)
plt.xlabel('al')
plt.ylabel('household')


# Let's draw a **regression line**, like we did before:

# fit a linear regression model and store the predictions
feature_cols = ['al']
X = glass[feature_cols]
y = glass.household
linreg.fit(X, y)
glass['household_pred'] = linreg.predict(X)


# scatter plot that includes the regression line
plt.scatter(glass.al, glass.household)
plt.plot(glass.al, glass.household_pred, color='red')
plt.xlabel('al')
plt.ylabel('household')


# If **al=3**, what class do we predict for household? **1**
# 
# If **al=1.5**, what class do we predict for household? **0**
# 
# We predict the 0 class for **lower** values of al, and the 1 class for **higher** values of al. What's our cutoff value? Around **al=2**, because that's where the linear regression line crosses the midpoint between predicting class 0 and class 1.
# 
# Therefore, we'll say that if **household_pred >= 0.5**, we predict a class of **1**, else we predict a class of **0**.

# understanding np.where
import numpy as np
nums = np.array([5, 15, 8])

# np.where returns the first value if the condition is True, and the second value if the condition is False
np.where(nums > 10, 'big', 'small')


# transform household_pred to 1 or 0
glass['household_pred_class'] = np.where(glass.household_pred >= 0.5, 1, 0)
glass.head()


# plot the class predictions
plt.scatter(glass.al, glass.household)
plt.plot(glass.al, glass.household_pred_class, color='red')
plt.xlabel('al')
plt.ylabel('household')


# ## Part 3: Using Logistic Regression Instead
# 
# Logistic regression can do what we just did:

# fit a logistic regression model and store the class predictions
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=1e9)
feature_cols = ['al']
X = glass[feature_cols]
y = glass.household
logreg.fit(X, y)
glass['household_pred_class'] = logreg.predict(X)


# plot the class predictions
plt.scatter(glass.al, glass.household)
plt.plot(glass.al, glass.household_pred_class, color='red')
plt.xlabel('al')
plt.ylabel('household')


# What if we wanted the **predicted probabilities** instead of just the **class predictions**, to understand how confident we are in a given prediction?

# store the predicted probabilites of class 1
glass['household_pred_prob'] = logreg.predict_proba(X)[:, 1]


# plot the predicted probabilities
plt.scatter(glass.al, glass.household)
plt.plot(glass.al, glass.household_pred_prob, color='red')
plt.xlabel('al')
plt.ylabel('household')


# examine some example predictions
print logreg.predict_proba(1)
print logreg.predict_proba(2)
print logreg.predict_proba(3)


# The first column indicates the predicted probability of **class 0**, and the second column indicates the predicted probability of **class 1**.

# ## Part 4: Probability, odds, e, log, log-odds
# 
# $$probability = \frac {one\ outcome} {all\ outcomes}$$
# 
# $$odds = \frac {one\ outcome} {all\ other\ outcomes}$$
# 
# Examples:
# 
# - Dice roll of 1: probability = 1/6, odds = 1/5
# - Even dice roll: probability = 3/6, odds = 3/3 = 1
# - Dice roll less than 5: probability = 4/6, odds = 4/2 = 2
# 
# $$odds = \frac {probability} {1 - probability}$$
# 
# $$probability = \frac {odds} {1 + odds}$$

# create a table of probability versus odds
table = pd.DataFrame({'probability':[0.1, 0.2, 0.25, 0.5, 0.6, 0.8, 0.9]})
table['odds'] = table.probability/(1 - table.probability)
table


# What is **e**? It is the base rate of growth shared by all continually growing processes:

# exponential function: e^1
np.exp(1)


# What is a **(natural) log**? It gives you the time needed to reach a certain level of growth:

# time needed to grow 1 unit to 2.718 units
np.log(2.718)


# It is also the **inverse** of the exponential function:

np.log(np.exp(5))


# add log-odds to the table
table['logodds'] = np.log(table.odds)
table


# ## Part 5: What is Logistic Regression?

# **Linear regression:** continuous response is modeled as a linear combination of the features:
# 
# $$y = \beta_0 + \beta_1x$$
# 
# **Logistic regression:** log-odds of a categorical response being "true" (1) is modeled as a linear combination of the features:
# 
# $$\log \left({p\over 1-p}\right) = \beta_0 + \beta_1x$$
# 
# This is called the **logit function**.
# 
# Probability is sometimes written as pi:
# 
# $$\log \left({\pi\over 1-\pi}\right) = \beta_0 + \beta_1x$$
# 
# The equation can be rearranged into the **logistic function**:
# 
# $$\pi = \frac{e^{\beta_0 + \beta_1x}} {1 + e^{\beta_0 + \beta_1x}}$$

# In other words:
# 
# - Logistic regression outputs the **probabilities of a specific class**
# - Those probabilities can be converted into **class predictions**
# 
# The **logistic function** has some nice properties:
# 
# - Takes on an "s" shape
# - Output is bounded by 0 and 1
# 
# We have covered how this works for **binary classification problems** (two response classes). But what about **multi-class classification problems** (more than two response classes)?
# 
# - Most common solution for classification models is **"one-vs-all"** (also known as **"one-vs-rest"**): decompose the problem into multiple binary classification problems
# - **Multinomial logistic regression** can solve this as a single problem

# ## Part 6: Interpreting Logistic Regression Coefficients

# plot the predicted probabilities again
plt.scatter(glass.al, glass.household)
plt.plot(glass.al, glass.household_pred_prob, color='red')
plt.xlabel('al')
plt.ylabel('household')


# compute predicted log-odds for al=2 using the equation
logodds = logreg.intercept_ + logreg.coef_[0] * 2
logodds


# convert log-odds to odds
odds = np.exp(logodds)
odds


# convert odds to probability
prob = odds/(1 + odds)
prob


# compute predicted probability for al=2 using the predict_proba method
logreg.predict_proba(2)[:, 1]


# examine the coefficient for al
zip(feature_cols, logreg.coef_[0])


# **Interpretation:** A 1 unit increase in 'al' is associated with a 4.18 unit increase in the log-odds of 'household'.

# increasing al by 1 (so that al=3) increases the log-odds by 4.18
logodds = 0.64722323 + 4.1804038614510901
odds = np.exp(logodds)
prob = odds/(1 + odds)
prob


# compute predicted probability for al=3 using the predict_proba method
logreg.predict_proba(3)[:, 1]


# **Bottom line:** Positive coefficients increase the log-odds of the response (and thus increase the probability), and negative coefficients decrease the log-odds of the response (and thus decrease the probability).

# examine the intercept
logreg.intercept_


# **Interpretation:** For an 'al' value of 0, the log-odds of 'household' is -7.71.

# convert log-odds to probability
logodds = logreg.intercept_
odds = np.exp(logodds)
prob = odds/(1 + odds)
prob


# That makes sense from the plot above, because the probability of household=1 should be very low for such a low 'al' value.

# ![Logistic regression beta values](images/logistic_betas.png)

# Changing the $\beta_0$ value shifts the curve **horizontally**, whereas changing the $\beta_1$ value changes the **slope** of the curve.

# ## Part 7: Using Logistic Regression with Categorical Features

# Logistic regression can still be used with **categorical features**. Let's see what that looks like:

# create a categorical feature
glass['high_ba'] = np.where(glass.ba > 0.5, 1, 0)


# Let's use Seaborn to draw the logistic curve:

# original (continuous) feature
sns.lmplot(x='ba', y='household', data=glass, ci=None, logistic=True)


# categorical feature
sns.lmplot(x='high_ba', y='household', data=glass, ci=None, logistic=True)


# categorical feature, with jitter added
sns.lmplot(x='high_ba', y='household', data=glass, ci=None, logistic=True, x_jitter=0.05, y_jitter=0.05)


# fit a logistic regression model
feature_cols = ['high_ba']
X = glass[feature_cols]
y = glass.household
logreg.fit(X, y)


# examine the coefficient for high_ba
zip(feature_cols, logreg.coef_[0])


# **Interpretation:** Having a high 'ba' value is associated with a 4.43 unit increase in the log-odds of 'household' (as compared to a low 'ba' value).

# ## Part 8: Comparing Logistic Regression with Other Models
# 
# Advantages of logistic regression:
# 
# - Highly interpretable (if you remember how)
# - Model training and prediction are fast
# - No tuning is required (excluding regularization)
# - Features don't need scaling
# - Can perform well with a small number of observations
# - Outputs well-calibrated predicted probabilities
# 
# Disadvantages of logistic regression:
# 
# - Presumes a linear relationship between the features and the log-odds of the response
# - Performance is (generally) not competitive with the best supervised learning methods
# - Can't automatically learn feature interactions
