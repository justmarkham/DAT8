# # Applying Bayes' theorem to iris classification
# 
# Can **Bayes' theorem** help us to solve a **classification problem**, namely predicting the species of an iris?

# ## Preparing the data
# 
# We'll read the iris data into a DataFrame, and **round up** all of the measurements to the next integer:

import pandas as pd
import numpy as np


# read the iris data into a DataFrame
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris = pd.read_csv(url, header=None, names=col_names)
iris.head()


# apply the ceiling function to the numeric columns
iris.loc[:, 'sepal_length':'petal_width'] = iris.loc[:, 'sepal_length':'petal_width'].apply(np.ceil)
iris.head()


# ## Deciding how to make a prediction
# 
# Let's say that I have an **out-of-sample iris** with the following measurements: **7, 3, 5, 2**. How might I predict the species?

# show all observations with features: 7, 3, 5, 2
iris[(iris.sepal_length==7) & (iris.sepal_width==3) & (iris.petal_length==5) & (iris.petal_width==2)]


# count the species for these observations
iris[(iris.sepal_length==7) & (iris.sepal_width==3) & (iris.petal_length==5) & (iris.petal_width==2)].species.value_counts()


# count the species for all observations
iris.species.value_counts()


# Let's frame this as a **conditional probability problem**: What is the probability of some particular species, given the measurements 7, 3, 5, and 2?
# 
# $$P(species \ | \ 7352)$$
# 
# We could calculate the conditional probability for **each of the three species**, and then predict the species with the **highest probability**:
# 
# $$P(setosa \ | \ 7352)$$
# $$P(versicolor \ | \ 7352)$$
# $$P(virginica \ | \ 7352)$$

# ## Calculating the probability of each species
# 
# **Bayes' theorem** gives us a way to calculate these conditional probabilities.
# 
# Let's start with **versicolor**:
# 
# $$P(versicolor \ | \ 7352) = \frac {P(7352 \ | \ versicolor) \times P(versicolor)} {P(7352)}$$
# 
# We can calculate each of the terms on the right side of the equation:
# 
# $$P(7352 \ | \ versicolor) = \frac {13} {50} = 0.26$$
# 
# $$P(versicolor) = \frac {50} {150} = 0.33$$
# 
# $$P(7352) = \frac {17} {150} = 0.11$$
# 
# Therefore, Bayes' theorem says the **probability of versicolor given these measurements** is:
# 
# $$P(versicolor \ | \ 7352) = \frac {0.26 \times 0.33} {0.11} = 0.76$$
# 
# Let's repeat this process for **virginica** and **setosa**:
# 
# $$P(virginica \ | \ 7352) = \frac {0.08 \times 0.33} {0.11} = 0.24$$
# 
# $$P(setosa \ | \ 7352) = \frac {0 \times 0.33} {0.11} = 0$$
# 
# We predict that the iris is a versicolor, since that species had the **highest conditional probability**.

# ## Summary
# 
# 1. We framed a **classification problem** as three conditional probability problems.
# 2. We used **Bayes' theorem** to calculate those conditional probabilities.
# 3. We made a **prediction** by choosing the species with the highest conditional probability.

# ## Bonus: The intuition behind Bayes' theorem
# 
# Let's make some hypothetical adjustments to the data, to demonstrate how Bayes' theorem makes intuitive sense:
# 
# Pretend that **more of the existing versicolors had measurements of 7352:**
# 
# - $P(7352 \ | \ versicolor)$ would increase, thus increasing the numerator.
# - It would make sense that given an iris with measurements of 7352, the probability of it being a versicolor would also increase.
# 
# Pretend that **most of the existing irises were versicolor:**
# 
# - $P(versicolor)$ would increase, thus increasing the numerator.
# - It would make sense that the probability of any iris being a versicolor (regardless of measurements) would also increase.
# 
# Pretend that **17 of the setosas had measurements of 7352:**
# 
# - $P(7352)$ would double, thus doubling the denominator.
# - It would make sense that given an iris with measurements of 7352, the probability of it being a versicolor would be cut in half.
