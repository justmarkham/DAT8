# # Model Evaluation

# ## Review of last class
# 
# - Goal was to predict the **response value** of an **unknown observation**
#     - predict the species of an unknown iris
#     - predict the position of an unknown NBA player
# - Made predictions using KNN models with **different values of K**
# - Need a way to choose the **"best" model**: the one that "generalizes" to "out-of-sample" data
# 
# **Solution:** Create a procedure that **estimates** how well a model is likely to perform on out-of-sample data and use that to choose between models.
# 
# **Note:** These procedures can be used with **any machine learning model**, not only KNN.

# ## Evaluation procedure #1: Train and test on the entire dataset

# 1. Train the model on the **entire dataset**.
# 2. Test the model on the **same dataset**, and evaluate how well we did by comparing the **predicted** response values with the **true** response values.

# read the NBA data into a DataFrame
import pandas as pd
url = 'https://raw.githubusercontent.com/justmarkham/DAT4-students/master/kerry/Final/NBA_players_2015.csv'
nba = pd.read_csv(url, index_col=0)


# map positions to numbers
nba['pos_num'] = nba.pos.map({'C':0, 'F':1, 'G':2})


# create feature matrix (X)
feature_cols = ['ast', 'stl', 'blk', 'tov', 'pf']
X = nba[feature_cols]


# create response vector (y)
y = nba.pos_num


# ### KNN (K=50)

# import the class
from sklearn.neighbors import KNeighborsClassifier

# instantiate the model
knn = KNeighborsClassifier(n_neighbors=50)

# train the model on the entire dataset
knn.fit(X, y)

# predict the response values for the observations in X ("test the model")
knn.predict(X)


# store the predicted response values
y_pred_class = knn.predict(X)


# To evaluate a model, we also need an **evaluation metric:**
# 
# - Numeric calculation used to **quantify** the performance of a model
# - Appropriate metric depends on the **goals** of your problem
# 
# Most common choices for classification problems:
# 
# - **Classification accuracy**: percentage of correct predictions ("reward function" since higher is better)
# - **Classification error**: percentage of incorrect predictions ("loss function" since lower is better)
# 
# In this case, we'll use classification accuracy.

# compute classification accuracy
from sklearn import metrics
print metrics.accuracy_score(y, y_pred_class)


# This is known as **training accuracy** because we are evaluating the model on the same data we used to train the model.

# ### KNN (K=1)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X, y)
y_pred_class = knn.predict(X)
print metrics.accuracy_score(y, y_pred_class)


# ### Problems with training and testing on the same data
# 
# - Goal is to estimate likely performance of a model on **out-of-sample data**
# - But, maximizing training accuracy rewards **overly complex models** that won't necessarily generalize
# - Unnecessarily complex models **overfit** the training data:
#     - Will do well when tested using the in-sample data
#     - May do poorly on out-of-sample data
#     - Learns the "noise" in the data rather than the "signal"
#     - From Quora: [What is an intuitive explanation of overfitting?](http://www.quora.com/What-is-an-intuitive-explanation-of-overfitting/answer/Jessica-Su)
# 
# **Thus, training accuracy is not a good estimate of out-of-sample accuracy.**

# ![1NN classification map](images/iris_01nn_map.png)

# ## Evaluation procedure #2: Train/test split

# 1. Split the dataset into two pieces: a **training set** and a **testing set**.
# 2. Train the model on the **training set**.
# 3. Test the model on the **testing set**, and evaluate how well we did.
# 
# What does this accomplish?
# 
# - Model can be trained and tested on **different data** (we treat testing data like out-of-sample data).
# - Response values are known for the testing set, and thus **predictions can be evaluated**.
# 
# This is known as **testing accuracy** because we are evaluating the model on an independent "test set" that was not used during model training.
# 
# **Testing accuracy is a better estimate of out-of-sample performance than training accuracy.**

# ### Understanding "unpacking"

def min_max(nums):
    smallest = min(nums)
    largest = max(nums)
    return [smallest, largest]


min_and_max = min_max([1, 2, 3])
print min_and_max
print type(min_and_max)


the_min, the_max = min_max([1, 2, 3])
print the_min
print type(the_min)
print the_max
print type(the_max)


# ### Understanding the `train_test_split` function

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)


# before splitting
print X.shape

# after splitting
print X_train.shape
print X_test.shape


# before splitting
print y.shape

# after splitting
print y_train.shape
print y_test.shape


# ![train_test_split](images/train_test_split.png)

# ### Understanding the `random_state` parameter

# WITHOUT a random_state parameter
X_train, X_test, y_train, y_test = train_test_split(X, y)

# print the first element of each object
print X_train.head(1)
print X_test.head(1)
print y_train.head(1)
print y_test.head(1)


# WITH a random_state parameter
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=99)

# print the first element of each object
print X_train.head(1)
print X_test.head(1)
print y_train.head(1)
print y_test.head(1)


# ### Using the train/test split procedure (K=1)

# STEP 1: split X and y into training and testing sets (using random_state for reproducibility)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=99)


# STEP 2: train the model on the training set (using K=1)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)


# STEP 3: test the model on the testing set, and check the accuracy
y_pred_class = knn.predict(X_test)
print metrics.accuracy_score(y_test, y_pred_class)


# ### Repeating for K=50

knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(X_train, y_train)
y_pred_class = knn.predict(X_test)
print metrics.accuracy_score(y_test, y_pred_class)


# ![Bias-variance tradeoff](images/bias_variance.png)

# ### Comparing testing accuracy with null accuracy

# Null accuracy is the accuracy that could be achieved by **always predicting the most frequent class**. It is a benchmark against which you may want to measure your classification model.

# examine the class distribution
y_test.value_counts()


# compute null accuracy
y_test.value_counts().head(1) / len(y_test)


# ### Searching for the "best" value of K

# calculate TRAINING ERROR and TESTING ERROR for K=1 through 100

k_range = range(1, 101)
training_error = []
testing_error = []

for k in k_range:

    # instantiate the model with the current K value
    knn = KNeighborsClassifier(n_neighbors=k)

    # calculate training error
    knn.fit(X, y)
    y_pred_class = knn.predict(X)
    training_accuracy = metrics.accuracy_score(y, y_pred_class)
    training_error.append(1 - training_accuracy)
    
    # calculate testing error
    knn.fit(X_train, y_train)
    y_pred_class = knn.predict(X_test)
    testing_accuracy = metrics.accuracy_score(y_test, y_pred_class)
    testing_error.append(1 - testing_accuracy)


# allow plots to appear in the notebook
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# create a DataFrame of K, training error, and testing error
column_dict = {'K': k_range, 'training error':training_error, 'testing error':testing_error}
df = pd.DataFrame(column_dict).set_index('K').sort_index(ascending=False)
df.head()


# plot the relationship between K (HIGH TO LOW) and TESTING ERROR
df.plot(y='testing error')
plt.xlabel('Value of K for KNN')
plt.ylabel('Error (lower is better)')


# find the minimum testing error and the associated K value
df.sort('testing error').head()


# alternative method
min(zip(testing_error, k_range))


# What could we conclude?
# 
# - When using KNN on this dataset with these features, the **best value for K** is likely to be around 14.
# - Given the statistics of an **unknown player**, we estimate that we would be able to correctly predict his position about 74% of the time.

# ### Training error versus testing error

# plot the relationship between K (HIGH TO LOW) and both TRAINING ERROR and TESTING ERROR
df.plot()
plt.xlabel('Value of K for KNN')
plt.ylabel('Error (lower is better)')


# - **Training error** decreases as model complexity increases (lower value of K)
# - **Testing error** is minimized at the optimum model complexity

# ![Bias-variance tradeoff](images/training_testing_error.png)

# ## Making predictions on out-of-sample data

# Given the statistics of a (truly) unknown player, how do we predict his position?

# instantiate the model with the best known parameters
knn = KNeighborsClassifier(n_neighbors=14)

# re-train the model with X and y (not X_train and y_train) - why?
knn.fit(X, y)

# make a prediction for an out-of-sample observation
knn.predict([1, 1, 0, 1, 2])


# ## Disadvantages of train/test split?

# What would happen if the `train_test_split` function had split the data differently? Would we get the same exact results as before?

# try different values for random_state
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=98)
knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(X_train, y_train)
y_pred_class = knn.predict(X_test)
print metrics.accuracy_score(y_test, y_pred_class)


# - Testing accuracy is a **high-variance estimate** of out-of-sample accuracy
# - **K-fold cross-validation** overcomes this limitation and provides more reliable estimates
# - But, train/test split is still useful because of its **flexibility and speed**
