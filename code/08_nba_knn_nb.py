
# coding: utf-8

# # KNN exercise with NBA player data

# ## Introduction
# 
# - NBA player statistics from 2014-2015 (partial season): [data](https://github.com/justmarkham/DAT4-students/blob/master/kerry/Final/NBA_players_2015.csv), [data dictionary](https://github.com/justmarkham/DAT-project-examples/blob/master/pdf/nba_paper.pdf)
# - **Goal:** Predict player position using assists, steals, blocks, turnovers, and personal fouls

# ## Step 1: Read the data into Pandas

# In[1]:

# read the data into a DataFrame
import pandas as pd
url = 'https://raw.githubusercontent.com/justmarkham/DAT4-students/master/kerry/Final/NBA_players_2015.csv'
nba = pd.read_csv(url, index_col=0)


# In[2]:

# examine the columns
nba.columns


# In[3]:

# examine the positions
nba.pos.value_counts()


# ## Step 2: Create X and y
# 
# Use the following features: assists, steals, blocks, turnovers, personal fouls

# In[4]:

# map positions to numbers
nba['pos_num'] = nba.pos.map({'C':0, 'F':1, 'G':2})


# In[5]:

# create feature matrix (X)
feature_cols = ['ast', 'stl', 'blk', 'tov', 'pf']
X = nba[feature_cols]


# In[6]:

# alternative way to create X
X = nba.loc[:, 'ast':'pf']


# In[7]:

# create response vector (y)
y = nba.pos_num


# ## Step 3: Train a KNN model (K=5)

# In[8]:

# import class
from sklearn.neighbors import KNeighborsClassifier


# In[9]:

# instantiate with K=5
knn = KNeighborsClassifier(n_neighbors=5)


# In[10]:

# fit with data
knn.fit(X, y)


# ## Step 4: Predict player position and calculate predicted probability of each position
# 
# Predict for a player with these statistics: 1 assist, 1 steal, 0 blocks, 1 turnover, 2 personal fouls

# In[11]:

# create a list to represent a player
player = [1, 1, 0, 1, 2]


# In[12]:

# make a prediction
knn.predict(player)


# In[13]:

# calculate predicted probabilities
knn.predict_proba(player)


# ## Step 5: Repeat steps 3 and 4 using K=50

# In[14]:

# repeat for K=50
knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(X, y)
knn.predict(player)


# In[15]:

# calculate predicted probabilities
knn.predict_proba(player)


# ## Bonus: Explore the features to decide which ones are predictive

# In[16]:

# allow plots to appear in the notebook
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt

# increase default figure and font sizes for easier viewing
plt.rcParams['figure.figsize'] = (6, 4)
plt.rcParams['font.size'] = 14


# In[17]:

# description of assists grouped by position
nba.groupby('pos').ast.describe().unstack()


# In[18]:

# box plot of assists grouped by position
nba.boxplot(column='ast', by='pos')


# In[19]:

# histogram of assists grouped by position
nba.hist(column='ast', by='pos', sharex=True)

