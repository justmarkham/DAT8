# # Pandas Review

import pandas as pd
url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/drinks.csv'
df = pd.read_csv(url).head(5).copy()
df


# For each of the following lines of code:
# 
# - What the **data type** of the object that is returned?
# - What is the **shape** of the object that is returned?
# 
# 
# 1. `df`
# 2. `df.continent`
# 3. `df['continent']`
# 4. `df[['country', 'continent']]`
# 5. `df[[False, True, False, True, False]]`

# ## Question 1

df


print type(df)
print df.shape


# ## Question 2

df.continent


print type(df.continent)
print df.continent.shape


# ## Question 3

df['continent']


print type(df['continent'])
print df['continent'].shape


# ## Question 4

df[['country', 'continent']]


print type(df[['country', 'continent']])
print df[['country', 'continent']].shape


# equivalent
cols = ['country', 'continent']
df[cols]


# ## Question 5

df[[False, True, False, True, False]]


print type(df[[False, True, False, True, False]])
print df[[False, True, False, True, False]].shape


# equivalent
df[df.continent=='EU']
