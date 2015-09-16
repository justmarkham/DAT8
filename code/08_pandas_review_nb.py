
# coding: utf-8

# # Pandas Review

# In[1]:

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

# In[2]:

df


# In[3]:

print type(df)
print df.shape


# ## Question 2

# In[4]:

df.continent


# In[5]:

print type(df.continent)
print df.continent.shape


# ## Question 3

# In[6]:

df['continent']


# In[7]:

print type(df['continent'])
print df['continent'].shape


# ## Question 4

# In[8]:

df[['country', 'continent']]


# In[9]:

print type(df[['country', 'continent']])
print df[['country', 'continent']].shape


# In[10]:

# equivalent
cols = ['country', 'continent']
df[cols]


# ## Question 5

# In[11]:

df[[False, True, False, True, False]]


# In[12]:

print type(df[[False, True, False, True, False]])
print df[[False, True, False, True, False]].shape


# In[13]:

# equivalent
df[df.continent=='EU']

