# # Exercise: "Human learning" with iris data
# 
# **Question:** Can you predict the species of an iris using petal and sepal measurements?
# 
# 1. Read the iris data into a Pandas DataFrame, including column names.
# 2. Gather some basic information about the data.
# 3. Use sorting, split-apply-combine, and/or visualization to look for differences between species.
# 4. Write down a set of rules that could be used to predict species based on iris measurements.
# 
# **BONUS:** Define a function that accepts a row of data and returns a predicted species. Then, use that function to make predictions for all existing rows of data, and check the accuracy of your predictions.

import pandas as pd
import matplotlib.pyplot as plt

# display plots in the notebook

# increase default figure and font sizes for easier viewing
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 14


# ## Task 1
# 
# Read the iris data into a pandas DataFrame, including column names.

# define a list of column names (as strings)
col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# define the URL from which to retrieve the data (as a string)
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

# retrieve the CSV file and add the column names
iris = pd.read_csv(url, header=None, names=col_names)


# ## Task 2
# 
# Gather some basic information about the data.

iris.shape


iris.head()


iris.dtypes


iris.describe()


iris.species.value_counts()


iris.isnull().sum()


# ## Task 3
# 
# Use sorting, split-apply-combine, and/or visualization to look for differences between species.

# ### sorting

# sort the DataFrame by petal_width and display the NumPy array
print iris.sort('petal_width').values


# ### split-apply-combine

# mean of sepal_length grouped by species
iris.groupby('species').sepal_length.mean()


# mean of all numeric columns grouped by species
iris.groupby('species').mean()


# description of all numeric columns grouped by species
iris.groupby('species').describe()


# ### visualization

# histogram of petal_width grouped by species
iris.hist(column='petal_width', by='species', sharex=True)


# box plot of petal_width grouped by species
iris.boxplot(column='petal_width', by='species')


# box plot of all numeric columns grouped by species
iris.boxplot(by='species')


# map species to a numeric value so that plots can be colored by species
iris['species_num'] = iris.species.map({'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2})

# alternative method
iris['species_num'] = iris.species.factorize()[0]


# scatter plot of petal_length vs petal_width colored by species
iris.plot(kind='scatter', x='petal_length', y='petal_width', c='species_num', colormap='brg')


# scatter matrix of all features colored by species
pd.scatter_matrix(iris.drop('species_num', axis=1), c=iris.species_num, figsize=(12, 10))


# ## Task 4
# 
# Write down a set of rules that could be used to predict species based on iris measurements.

# define a new feature that represents petal area ("feature engineering")
iris['petal_area'] = iris.petal_length * iris.petal_width


# description of petal_area grouped by species
iris.groupby('species').petal_area.describe().unstack()


# box plot of petal_area grouped by species
iris.boxplot(column='petal_area', by='species')


# only show irises with a petal_area between 7 and 9
iris[(iris.petal_area > 7) & (iris.petal_area < 9)].sort('petal_area')


# My set of rules for predicting species:
# 
# - If petal_area is less than 2, predict **setosa**.
# - Else if petal_area is less than 7.4, predict **versicolor**.
# - Otherwise, predict **virginica**.

# ## Bonus
# 
# Define a function that accepts a row of data and returns a predicted species. Then, use that function to make predictions for all existing rows of data, and check the accuracy of your predictions.

# given a row of data, return a predicted species_num (0/1/2)
def classify_iris(row):

    # calculate the petal_area
    petal_area = row[2] * row[3]
    
    # predict the species based on the rules above    
    if petal_area < 2:
        prediction = 'setosa'
    elif petal_area < 7.4:
        prediction = 'versicolor'
    else:
        prediction = 'virginica'
    
    # map the species name to a numeric value
    species_to_num = {'setosa':0, 'versicolor':1, 'virginica':2}
    
    # return that value
    return species_to_num[prediction]


# print the first row
iris.iloc[0, :]


# print the last row
iris.iloc[149, :]


# test the function on the first and last rows
print classify_iris(iris.iloc[0, :])
print classify_iris(iris.iloc[149, :])


# make predictions for all rows and store them in the DataFrame
iris['prediction'] = [classify_iris(row) for index, row in iris.iterrows()]


# calculate the percentage of correct predictions
sum(iris.species_num == iris.prediction) / 150.
