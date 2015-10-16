# # Advanced scikit-learn

# ## Agenda
# 
# - StandardScaler
# - Pipeline (bonus content)

# ## StandardScaler
# 
# ### What is the problem we're trying to solve?

# fake data
import pandas as pd
train = pd.DataFrame({'id':[0,1,2], 'length':[0.9,0.3,0.6], 'mass':[0.1,0.2,0.8], 'rings':[40,50,60]})
test = pd.DataFrame({'length':[0.59], 'mass':[0.79], 'rings':[54]})


# training data
train


# testing data
test


# define X and y
feature_cols = ['length', 'mass', 'rings']
X = train[feature_cols]
y = train.id


# KNN with K=1
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X, y)


# what "should" it predict?
knn.predict(test)


# allow plots to appear in the notebook
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14
plt.rcParams['figure.figsize'] = (5, 5)


# create a "colors" array for plotting
import numpy as np
colors = np.array(['red', 'green', 'blue'])


# scatter plot of training data, colored by id (0=red, 1=green, 2=blue)
plt.scatter(train.mass, train.rings, c=colors[train.id], s=50)

# testing data
plt.scatter(test.mass, test.rings, c='white', s=50)

# add labels
plt.xlabel('mass')
plt.ylabel('rings')
plt.title('How we interpret the data')


# adjust the x-limits
plt.scatter(train.mass, train.rings, c=colors[train.id], s=50)
plt.scatter(test.mass, test.rings, c='white', s=50)
plt.xlabel('mass')
plt.ylabel('rings')
plt.title('How KNN interprets the data')
plt.xlim(0, 30)


# ### How does StandardScaler solve the problem?
# 
# [StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) is used for the "standardization" of features, also known as "center and scale" or "z-score normalization".

# standardize the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)


# original values
X.values


# standardized values
X_scaled


# figure out how it standardized
print scaler.mean_
print scaler.std_


# manually standardize
(X.values - scaler.mean_) / scaler.std_


# ### Applying StandardScaler to a real dataset
# 
# - Wine dataset from the UCI Machine Learning Repository: [data](http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data), [data dictionary](http://archive.ics.uci.edu/ml/datasets/Wine)
# - **Goal:** Predict the origin of wine using chemical analysis

# read three columns from the dataset into a DataFrame
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
col_names = ['label', 'color', 'proline']
wine = pd.read_csv(url, header=None, names=col_names, usecols=[0, 10, 13])


wine.head()


wine.describe()


# define X and y
feature_cols = ['color', 'proline']
X = wine[feature_cols]
y = wine.label


# split into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


# standardize X_train
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)


# check that it standardized properly
print X_train_scaled[:, 0].mean()
print X_train_scaled[:, 0].std()
print X_train_scaled[:, 1].mean()
print X_train_scaled[:, 1].std()


# standardize X_test
X_test_scaled = scaler.transform(X_test)


# is this right?
print X_test_scaled[:, 0].mean()
print X_test_scaled[:, 0].std()
print X_test_scaled[:, 1].mean()
print X_test_scaled[:, 1].std()


# KNN accuracy on original data
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred_class = knn.predict(X_test)
from sklearn import metrics
print metrics.accuracy_score(y_test, y_pred_class)


# KNN accuracy on scaled data
knn.fit(X_train_scaled, y_train)
y_pred_class = knn.predict(X_test_scaled)
print metrics.accuracy_score(y_test, y_pred_class)


# ## Pipeline (bonus content)
# 
# ### What is the problem we're trying to solve?

# define X and y
feature_cols = ['color', 'proline']
X = wine[feature_cols]
y = wine.label


# proper cross-validation on the original (unscaled) data
knn = KNeighborsClassifier(n_neighbors=3)
from sklearn.cross_validation import cross_val_score
cross_val_score(knn, X, y, cv=5, scoring='accuracy').mean()


# why is this improper cross-validation on the scaled data?
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
cross_val_score(knn, X_scaled, y, cv=5, scoring='accuracy').mean()


# ### How does Pipeline solve the problem?
# 
# [Pipeline](http://scikit-learn.org/stable/modules/pipeline.html) is used for chaining steps together:

# fix the cross-validation process using Pipeline
from sklearn.pipeline import make_pipeline
pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3))
cross_val_score(pipe, X, y, cv=5, scoring='accuracy').mean()


# Pipeline can also be used with [GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html) for parameter searching:

# search for an optimal n_neighbors value using GridSearchCV
neighbors_range = range(1, 21)
param_grid = dict(kneighborsclassifier__n_neighbors=neighbors_range)
from sklearn.grid_search import GridSearchCV
grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
grid.fit(X, y)
print grid.best_score_
print grid.best_params_
