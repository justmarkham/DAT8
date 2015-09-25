# # Exercise with bank marketing data

# ## Introduction
# 
# - Data from the UCI Machine Learning Repository: [data](https://github.com/justmarkham/DAT8/blob/master/data/bank-additional.csv), [data dictionary](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
# - **Goal:** Predict whether a customer will purchase a bank product marketed over the phone
# - `bank-additional.csv` is already in our repo, so there is no need to download the data from the UCI website

# ## Step 1: Read the data into Pandas

# ## Step 2: Prepare at least three features
# 
# - Include both numeric and categorical features
# - Choose features that you think might be related to the response (based on intuition or exploration)
# - Think about how to handle missing values (encoded as "unknown")

# ## Step 3: Model building
# 
# - Use cross-validation to evaluate the AUC of a logistic regression model with your chosen features
# - Try to increase the AUC by selecting different sets of features
