'''
CLASS: Pandas for Exploratory Data Analysis

MovieLens 100k movie rating data:
    main page: http://grouplens.org/datasets/movielens/
    data dictionary: http://files.grouplens.org/datasets/movielens/ml-100k-README.txt
    files: u.user, u.user_original (no header row)

WHO alcohol consumption data:
    article: http://fivethirtyeight.com/datalab/dear-mona-followup-where-do-people-drink-the-most-beer-wine-and-spirits/    
    original data: https://github.com/fivethirtyeight/data/tree/master/alcohol-consumption
    file: drinks.csv (with additional 'continent' column)

National UFO Reporting Center data:
    main page: http://www.nuforc.org/webreports.html
    file: ufo.csv
'''

import pandas as pd

'''
Reading Files, Selecting Columns, and Summarizing
'''

# can read a file from local computer or directly from a URL
pd.read_table('u.user')
pd.read_table('https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user')

# read 'u.user' into 'users'
users = pd.read_table('u.user', sep='|', index_col='user_id')

# examine the users data
users                   # print the first 30 and last 30 rows
type(users)             # DataFrame
users.head()            # print the first 5 rows
users.head(10)          # print the first 10 rows
users.tail()            # print the last 5 rows
users.index             # "the index" (aka "the labels")
users.columns           # column names (which is "an index")
users.dtypes            # data types of each column
users.shape             # number of rows and columns
users.values            # underlying numpy array

# select a column
users['gender']         # select one column
type(users['gender'])   # Series
users.gender            # select one column using the DataFrame attribute

# summarize (describe) the DataFrame
users.describe()                    # describe all numeric columns
users.describe(include=['object'])  # describe all object columns
users.describe(include='all')       # describe all columns

# summarize a Series
users.gender.describe()             # describe a single column
users.age.mean()                    # only calculate the mean

# count the number of occurrences of each value
users.gender.value_counts()     # most useful for categorical variables
users.age.value_counts()        # can also be used with numeric variables

'''
EXERCISE ONE
'''

# read drinks.csv into a DataFrame called 'drinks'
drinks = pd.read_table('drinks.csv', sep=',')
drinks = pd.read_csv('drinks.csv')              # assumes separator is comma

# print the head and the tail
drinks.head()
drinks.tail()

# examine the default index, data types, and shape
drinks.index
drinks.dtypes
drinks.shape

# print the 'beer_servings' Series
drinks['beer_servings']
drinks.beer_servings

# calculate the mean 'beer_servings' for the entire dataset
drinks.describe()                   # summarize all numeric columns
drinks.beer_servings.describe()     # summarize only the 'beer_servings' Series
drinks.beer_servings.mean()         # only calculate the mean

# count the number of occurrences of each 'continent' value and see if it looks correct
drinks.continent.value_counts()

# BONUS: display only the number of rows of the 'users' DataFrame
users.shape[0]

# BONUS: display the 3 most frequent occupations in 'users'
users.occupation.value_counts().head(3)
users.occupation.value_counts()[:3]

# BONUS: create the 'users' DataFrame from the u.user_original file (which lacks a header row)
# Hint: read the pandas.read_table documentation
user_cols = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
users = pd.read_table('u.user_original', sep='|', header=None, names=user_cols, index_col='user_id')

'''
Filtering and Sorting
'''

# boolean filtering: only show users with age < 20
young_bool = users.age < 20         # create a Series of booleans...
users[young_bool]                   # ...and use that Series to filter rows
users[users.age < 20]               # or, combine into a single step
users[users.age < 20].occupation    # select one column from the filtered results
users[users.age < 20].occupation.value_counts()     # value_counts of resulting Series

# boolean filtering with multiple conditions
users[(users.age < 20) & (users.gender=='M')]       # ampersand for AND condition
users[(users.age < 20) | (users.age > 60)]          # pipe for OR condition

# sorting
users.age.order()                   # sort a column
users.sort('age')                   # sort a DataFrame by a single column
users.sort('age', ascending=False)  # use descending order instead

'''
EXERCISE TWO
'''

# filter 'drinks' to only include European countries
drinks[drinks.continent=='EU']

# filter 'drinks' to only include European countries with wine_servings > 300
drinks[(drinks.continent=='EU') & (drinks.wine_servings > 300)]

# calculate the mean 'beer_servings' for all of Europe
drinks[drinks.continent=='EU'].beer_servings.mean()

# determine which 10 countries have the highest total_litres_of_pure_alcohol
drinks.sort('total_litres_of_pure_alcohol').tail(10)

# BONUS: sort 'users' by 'occupation' and then by 'age' (in a single command)
users.sort(['occupation', 'age'])

# BONUS: filter 'users' to only include doctors and lawyers without using a |
# Hint: read the pandas.Series.isin documentation
users[users.occupation.isin(['doctor', 'lawyer'])]

'''
Renaming, Adding, and Removing Columns
'''

# rename one or more columns
drinks.rename(columns={'beer_servings':'beer', 'wine_servings':'wine'})
drinks.rename(columns={'beer_servings':'beer', 'wine_servings':'wine'}, inplace=True)

# replace all column names
drink_cols = ['country', 'beer', 'spirit', 'wine', 'liters', 'continent']
drinks.columns = drink_cols

# replace all column names when reading the file
drinks = pd.read_csv('drinks.csv', header=0, names=drink_cols)

# add a new column as a function of existing columns
drinks['servings'] = drinks.beer + drinks.spirit + drinks.wine
drinks['mL'] = drinks.liters * 1000

# removing columns
drinks.drop('mL', axis=1)                               # axis=0 for rows, 1 for columns
drinks.drop(['mL', 'servings'], axis=1, inplace=True)   # drop multiple columns

'''
Handling Missing Values
'''

# missing values are usually excluded by default
drinks.continent.value_counts()              # excludes missing values
drinks.continent.value_counts(dropna=False)  # includes missing values

# find missing values in a Series
drinks.continent.isnull()           # True if missing
drinks.continent.notnull()          # True if not missing

# use a boolean Series to filter DataFrame rows
drinks[drinks.continent.isnull()]   # only show rows where continent is missing
drinks[drinks.continent.notnull()]  # only show rows where continent is not missing

# side note: understanding axes
drinks.sum()            # sums "down" the 0 axis (rows)
drinks.sum(axis=0)      # equivalent (since axis=0 is the default)
drinks.sum(axis=1)      # sums "across" the 1 axis (columns)

# side note: adding booleans
pd.Series([True, False, True])          # create a boolean Series
pd.Series([True, False, True]).sum()    # converts False to 0 and True to 1

# find missing values in a DataFrame
drinks.isnull()             # DataFrame of booleans
drinks.isnull().sum()       # count the missing values in each column

# drop missing values
drinks.dropna()             # drop a row if ANY values are missing
drinks.dropna(how='all')    # drop a row only if ALL values are missing

# fill in missing values
drinks.continent.fillna(value='NA', inplace=True)   # fill in missing values with 'NA'

# turn off the missing value filter
drinks = pd.read_csv('drinks.csv', header=0, names=drink_cols, na_filter=False)

'''
EXERCISE THREE
'''

# read ufo.csv into a DataFrame called 'ufo'
ufo = pd.read_table('ufo.csv', sep=',')
ufo = pd.read_csv('ufo.csv')

# check the shape of the DataFrame
ufo.shape

# calculate the most frequent value for each of the columns (in a single command)
ufo.describe()

# what are the four most frequent colors reported?
ufo['Colors Reported'].value_counts().head(4)

# for reports in VA, what's the most frequent city?
ufo[ufo.State=='VA'].City.value_counts().head(1)

# show only the UFO reports from Arlington, VA
ufo[(ufo.City=='Arlington') & (ufo.State=='VA')]

# count the number of missing values in each column
ufo.isnull().sum()

# show only the UFO reports in which the City is missing
ufo[ufo.City.isnull()]

# how many rows remain if you drop all rows with any missing values?
ufo.dropna().shape[0]

# replace any spaces in the column names with an underscore
ufo.rename(columns={'Colors Reported':'Colors_Reported', 'Shape Reported':'Shape_Reported'}, inplace=True)

# BONUS: redo the task above, writing generic code to replace spaces with underscores
# In other words, your code should not reference the specific column names
ufo.columns = [col.replace(' ', '_') for col in ufo.columns]
ufo.columns = ufo.columns.str.replace(' ', '_')

# BONUS: create a new column called 'Location' that includes both City and State
# For example, the 'Location' for the first row would be 'Ithaca, NY'
ufo['Location'] = ufo.City + ', ' + ufo.State

'''
Split-Apply-Combine
Diagram: http://i.imgur.com/yjNkiwL.png
'''

# for each continent, calculate the mean beer servings
drinks.groupby('continent').beer.mean()

# for each continent, count the number of occurrences
drinks.continent.value_counts()

# for each continent, describe beer servings
drinks.groupby('continent').beer.describe()

# similar, but outputs a DataFrame and can be customized
drinks.groupby('continent').beer.agg(['count', 'mean', 'min', 'max'])
drinks.groupby('continent').beer.agg(['count', 'mean', 'min', 'max']).sort('mean')

# if you don't specify a column to which the aggregation function should be applied,
# it will be applied to all numeric columns
drinks.groupby('continent').mean()
drinks.groupby('continent').describe()

'''
EXERCISE FOUR
'''

# for each occupation in 'users', count the number of occurrences
users.occupation.value_counts()

# for each occupation, calculate the mean age
users.groupby('occupation').age.mean()

# BONUS: for each occupation, calculate the minimum and maximum ages
users.groupby('occupation').age.agg(['min', 'max'])

# BONUS: for each combination of occupation and gender, calculate the mean age
users.groupby(['occupation', 'gender']).age.mean()

'''
Selecting Multiple Columns and Filtering Rows
'''

# select multiple columns
my_cols = ['City', 'State']     # create a list of column names...
ufo[my_cols]                    # ...and use that list to select columns
ufo[['City', 'State']]          # or, combine into a single step

# use loc to select columns by name
ufo.loc[:, 'City']              # colon means "all rows", then select one column
ufo.loc[:, ['City', 'State']]   # select two columns
ufo.loc[:, 'City':'State']      # select a range of columns

# loc can also filter rows by "name" (the index)
ufo.loc[0, :]                   # row 0, all columns
ufo.loc[0:2, :]                 # rows 0/1/2, all columns
ufo.loc[0:2, 'City':'State']    # rows 0/1/2, range of columns

# use iloc to filter rows and select columns by integer position
ufo.iloc[:, [0, 3]]             # all rows, columns in position 0/3
ufo.iloc[:, 0:4]                # all rows, columns in position 0/1/2/3
ufo.iloc[0:3, :]                # rows in position 0/1/2, all columns

'''
Other Frequently Used Features
'''

# map existing values to a different set of values
users['is_male'] = users.gender.map({'F':0, 'M':1})

# encode strings as integer values (automatically starts at 0)
users['occupation_num'] = users.occupation.factorize()[0]

# determine unique values in a column
users.occupation.nunique()      # count the number of unique values
users.occupation.unique()       # return the unique values

# replace all instances of a value in a column (must match entire value)
ufo.State.replace('Fl', 'FL', inplace=True)

# string methods are accessed via 'str'
ufo.State.str.upper()                               # converts to uppercase
ufo.Colors_Reported.str.contains('RED', na='False') # checks for a substring

# convert a string to the datetime format
ufo['Time'] = pd.to_datetime(ufo.Time)
ufo.Time.dt.hour                        # datetime format exposes convenient attributes
(ufo.Time.max() - ufo.Time.min()).days  # also allows you to do datetime "math"
ufo[ufo.Time > pd.datetime(2014, 1, 1)] # boolean filtering with datetime format

# setting and then removing an index
ufo.set_index('Time', inplace=True)
ufo.reset_index(inplace=True)

# sort a column by its index
ufo.State.value_counts().sort_index()

# change the data type of a column
drinks['beer'] = drinks.beer.astype('float')

# change the data type of a column when reading in a file
pd.read_csv('drinks.csv', dtype={'beer_servings':float})

# create dummy variables for 'continent' and exclude first dummy column
continent_dummies = pd.get_dummies(drinks.continent, prefix='cont').iloc[:, 1:]

# concatenate two DataFrames (axis=0 for rows, axis=1 for columns)
drinks = pd.concat([drinks, continent_dummies], axis=1)

'''
Less Frequently Used Features
'''

# create a DataFrame from a dictionary
pd.DataFrame({'capital':['Montgomery', 'Juneau', 'Phoenix'], 'state':['AL', 'AK', 'AZ']})

# create a DataFrame from a list of lists
pd.DataFrame([['Montgomery', 'AL'], ['Juneau', 'AK'], ['Phoenix', 'AZ']], columns=['capital', 'state'])

# detecting duplicate rows
users.duplicated()          # True if a row is identical to a previous row
users.duplicated().sum()    # count of duplicates
users[users.duplicated()]   # only show duplicates
users.drop_duplicates()     # drop duplicate rows
users.age.duplicated()      # check a single column for duplicates
users.duplicated(['age', 'gender', 'zip_code']).sum()   # specify columns for finding duplicates

# display a cross-tabulation of two Series
pd.crosstab(users.occupation, users.gender)

# alternative syntax for boolean filtering (noted as "experimental" in the documentation)
users.query('age < 20')                 # users[users.age < 20]
users.query("age < 20 and gender=='M'") # users[(users.age < 20) & (users.gender=='M')]
users.query('age < 20 or age > 60')     # users[(users.age < 20) | (users.age > 60)]

# display the memory usage of a DataFrame
ufo.info()          # total usage
ufo.memory_usage()  # usage by column

# change a Series to the 'category' data type (reduces memory usage and increases performance)
ufo['State'] = ufo.State.astype('category')

# temporarily define a new column as a function of existing columns
drinks.assign(servings = drinks.beer + drinks.spirit + drinks.wine)

# limit which rows are read when reading in a file
pd.read_csv('drinks.csv', nrows=10)           # only read first 10 rows
pd.read_csv('drinks.csv', skiprows=[1, 2])    # skip the first two rows of data

# write a DataFrame out to a CSV
drinks.to_csv('drinks_updated.csv')                 # index is used as first column
drinks.to_csv('drinks_updated.csv', index=False)    # ignore index

# save a DataFrame to disk (aka 'pickle') and read it from disk (aka 'unpickle')
drinks.to_pickle('drinks_pickle')
pd.read_pickle('drinks_pickle')

# randomly sample a DataFrame
train = drinks.sample(frac=0.75, random_state=1)    # will contain 75% of the rows
test = drinks[~drinks.index.isin(train.index)]      # will contain the other 25%

# change the maximum number of rows and columns printed ('None' means unlimited)
pd.set_option('max_rows', None)     # default is 60 rows
pd.set_option('max_columns', None)  # default is 20 columns
print drinks

# reset options to defaults
pd.reset_option('max_rows')
pd.reset_option('max_columns')

# change the options temporarily (settings are restored when you exit the 'with' block)
with pd.option_context('max_rows', None, 'max_columns', None):
    print drinks
