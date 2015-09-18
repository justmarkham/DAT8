# # Visualization with Pandas (and Matplotlib)

import pandas as pd
import matplotlib.pyplot as plt

# display plots in the notebook

# increase default figure and font sizes for easier viewing
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 14


# read in the drinks data
drink_cols = ['country', 'beer', 'spirit', 'wine', 'liters', 'continent']
url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/drinks.csv'
drinks = pd.read_csv(url, header=0, names=drink_cols, na_filter=False)


# ## Histogram: show the distribution of a numerical variable

# sort the beer column and mentally split it into 3 groups
drinks.beer.order().values


# compare with histogram
drinks.beer.plot(kind='hist', bins=3)


# try more bins
drinks.beer.plot(kind='hist', bins=20)


# add title and labels
drinks.beer.plot(kind='hist', bins=20, title='Histogram of Beer Servings')
plt.xlabel('Beer Servings')
plt.ylabel('Frequency')


# compare with density plot (smooth version of a histogram)
drinks.beer.plot(kind='density', xlim=(0, 500))


# ## Scatter Plot: show the relationship between two numerical variables

# select the beer and wine columns and sort by beer
drinks[['beer', 'wine']].sort('beer').values


# compare with scatter plot
drinks.plot(kind='scatter', x='beer', y='wine')


# add transparency
drinks.plot(kind='scatter', x='beer', y='wine', alpha=0.3)


# vary point color by spirit servings
drinks.plot(kind='scatter', x='beer', y='wine', c='spirit', colormap='Blues')


# scatter matrix of three numerical columns
pd.scatter_matrix(drinks[['beer', 'spirit', 'wine']])


# increase figure size
pd.scatter_matrix(drinks[['beer', 'spirit', 'wine']], figsize=(10, 8))


# ## Bar Plot: show a numerical comparison across different categories

# count the number of countries in each continent
drinks.continent.value_counts()


# compare with bar plot
drinks.continent.value_counts().plot(kind='bar')


# calculate the mean alcohol amounts for each continent
drinks.groupby('continent').mean()


# side-by-side bar plots
drinks.groupby('continent').mean().plot(kind='bar')


# drop the liters column
drinks.groupby('continent').mean().drop('liters', axis=1).plot(kind='bar')


# stacked bar plots
drinks.groupby('continent').mean().drop('liters', axis=1).plot(kind='bar', stacked=True)


# ## Box Plot: show quartiles (and outliers) for one or more numerical variables
# 
# **Five-number summary:**
# 
# - min = minimum value
# - 25% = first quartile (Q1) = median of the lower half of the data
# - 50% = second quartile (Q2) = median of the data
# - 75% = third quartile (Q3) = median of the upper half of the data
# - max = maximum value
# 
# (More useful than mean and standard deviation for describing skewed distributions)
# 
# **Interquartile Range (IQR)** = Q3 - Q1
# 
# **Outliers:**
# 
# - below Q1 - 1.5 * IQR
# - above Q3 + 1.5 * IQR

# sort the spirit column
drinks.spirit.order().values


# show "five-number summary" for spirit
drinks.spirit.describe()


# compare with box plot
drinks.spirit.plot(kind='box')


# include multiple variables
drinks.drop('liters', axis=1).plot(kind='box')


# ## Line Plot: show the trend of a numerical variable over time

# read in the ufo data
url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/ufo.csv'
ufo = pd.read_csv(url)
ufo['Time'] = pd.to_datetime(ufo.Time)
ufo['Year'] = ufo.Time.dt.year


# count the number of ufo reports each year (and sort by year)
ufo.Year.value_counts().sort_index()


# compare with line plot
ufo.Year.value_counts().sort_index().plot()


# don't use a line plot when there is no logical ordering
drinks.continent.value_counts().plot()


# ## Grouped Box Plots: show one box plot for each group

# reminder: box plot of beer servings
drinks.beer.plot(kind='box')


# box plot of beer servings grouped by continent
drinks.boxplot(column='beer', by='continent')


# box plot of all numeric columns grouped by continent
drinks.boxplot(by='continent')


# ## Grouped Histograms: show one histogram for each group

# reminder: histogram of beer servings
drinks.beer.plot(kind='hist')


# histogram of beer servings grouped by continent
drinks.hist(column='beer', by='continent')


# share the x axes
drinks.hist(column='beer', by='continent', sharex=True)


# share the x and y axes
drinks.hist(column='beer', by='continent', sharex=True, sharey=True)


# change the layout
drinks.hist(column='beer', by='continent', sharex=True, layout=(2, 3))


# ## Assorted Functionality

# saving a plot to a file
drinks.beer.plot(kind='hist', bins=20, title='Histogram of Beer Servings')
plt.xlabel('Beer Servings')
plt.ylabel('Frequency')
plt.savefig('beer_histogram.png')


# list available plot styles
plt.style.available


# change to a different style
plt.style.use('ggplot')
