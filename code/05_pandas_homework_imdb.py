'''
Pandas Homework with IMDb data
'''

'''
BASIC LEVEL
'''

import pandas as pd
import matplotlib.pyplot as plt

# read in 'imdb_1000.csv' and store it in a DataFrame named movies
movies = pd.read_csv('imdb_1000.csv')

# check the number of rows and columns
movies.shape

# check the data type of each column
movies.dtypes

# calculate the average movie duration
movies.duration.mean()

# sort the DataFrame by duration to find the shortest and longest movies
movies.sort('duration').head(1)
movies.sort('duration').tail(1)

# create a histogram of duration, choosing an "appropriate" number of bins
movies.duration.plot(kind='hist', bins=20)

# use a box plot to display that same data
movies.duration.plot(kind='box')

'''
INTERMEDIATE LEVEL
'''

# count how many movies have each of the content ratings
movies.content_rating.value_counts()

# use a visualization to display that same data, including a title and x and y labels
movies.content_rating.value_counts().plot(kind='bar', title='Top 1000 Movies by Content Rating')
plt.xlabel('Content Rating')
plt.ylabel('Number of Movies')

# convert the following content ratings to "UNRATED": NOT RATED, APPROVED, PASSED, GP
movies.content_rating.replace(['NOT RATED', 'APPROVED', 'PASSED', 'GP'], 'UNRATED', inplace=True)

# convert the following content ratings to "NC-17": X, TV-MA
movies.content_rating.replace(['X', 'TV-MA'], 'NC-17', inplace=True)

# count the number of missing values in each column
movies.isnull().sum()

# if there are missing values: examine them, then fill them in with "reasonable" values
movies[movies.content_rating.isnull()]
movies.content_rating.fillna('UNRATED', inplace=True)

# calculate the average star rating for movies 2 hours or longer,
# and compare that with the average star rating for movies shorter than 2 hours
movies[movies.duration >= 120].star_rating.mean()
movies[movies.duration < 120].star_rating.mean()

# use a visualization to detect whether there is a relationship between duration and star rating
movies.plot(kind='scatter', x='duration', y='star_rating', alpha=0.2)

# calculate the average duration for each genre
movies.groupby('genre').duration.mean()

'''
ADVANCED LEVEL
'''

# visualize the relationship between content rating and duration
movies.boxplot(column='duration', by='content_rating')
movies.hist(column='duration', by='content_rating', sharex=True)

# determine the top rated movie (by star rating) for each genre
movies.sort('star_rating', ascending=False).groupby('genre').title.first()
movies.groupby('genre').title.first()   # equivalent, since DataFrame is already sorted by star rating

# check if there are multiple movies with the same title, and if so, determine if they are actually duplicates
dupe_titles = movies[movies.title.duplicated()].title
movies[movies.title.isin(dupe_titles)]

# calculate the average star rating for each genre, but only include genres with at least 10 movies

# option 1: manually create a list of relevant genres, then filter using that list
movies.genre.value_counts()
top_genres = ['Drama', 'Comedy', 'Action', 'Crime', 'Biography', 'Adventure', 'Animation', 'Horror', 'Mystery']
movies[movies.genre.isin(top_genres)].groupby('genre').star_rating.mean()

# option 2: automatically create a list of relevant genres by saving the value_counts and then filtering
genre_counts = movies.genre.value_counts()
top_genres = genre_counts[genre_counts >= 10].index
movies[movies.genre.isin(top_genres)].groupby('genre').star_rating.mean()

# option 3: calculate the average star rating for all genres, then filter using a boolean Series
movies.groupby('genre').star_rating.mean()[movies.genre.value_counts() >= 10]

# option 4: aggregate by count and mean, then filter using the count
genre_ratings = movies.groupby('genre').star_rating.agg(['count', 'mean'])
genre_ratings[genre_ratings['count'] >= 10]

'''
BONUS
'''

# Figure out something "interesting" using the actors data!
