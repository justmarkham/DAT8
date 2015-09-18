# # Exponential functions and logarithms

import math
import numpy as np


# ## Exponential functions

# What is **e**? It is simply a number (known as Euler's number):

math.e


# **e** is a significant number, because it is the base rate of growth shared by all continually growing processes.
# 
# For example, if I have **10 dollars**, and it grows 100% in 1 year (compounding continuously), I end up with **10\*e^1 dollars**:

# 100% growth for 1 year
10 * np.exp(1)


# 100% growth for 2 years
10 * np.exp(2)


# Side note: When e is raised to a power, it is known as **the exponential function**. Technically, any number can be the base, and it would still be known as **an exponential function** (such as 2^5). But in our context, the base of the exponential function is assumed to be e.
# 
# Anyway, what if I only have 20% growth instead of 100% growth?

# 20% growth for 1 year
10 * np.exp(0.20)


# 20% growth for 2 years
10 * np.exp(0.20 * 2)


# ## Logarithms

# What is the **(natural) logarithm**? It gives you the time needed to reach a certain level of growth. For example, if I want growth by a factor of 2.718, it will take me 1 unit of time (assuming a 100% growth rate):

# time needed to grow 1 unit to 2.718 units
np.log(2.718)


# If I want growth by a factor of 7.389, it will take me 2 units of time:

# time needed to grow 1 unit to 7.389 units
np.log(7.389)


# If I want growth by a factor of 1, it will take me 0 units of time:

# time needed to grow 1 unit to 1 unit
np.log(1)


# If I want growth by a factor of 0.5, it will take me -0.693 units of time (which is like looking back in time):

# time needed to grow 1 unit to 0.5 units
np.log(0.5)


# ## Connecting the concepts

# As you can see, the exponential function and the natural logarithm are **inverses** of one another:

np.log(np.exp(5))


np.exp(np.log(5))
