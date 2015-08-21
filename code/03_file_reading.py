'''
Lesson on file reading using Airline Safety Data
https://github.com/fivethirtyeight/data/tree/master/airline-safety
'''

# read the whole file at once, return a single string (including newlines)
# 'rU' mode (read universal) converts different line endings into '\n'

# use a context manager to automatically close your file

# read the file into a list (each list element is one row)

# do the same thing using a list comprehension

# side note: splitting strings

# split each string (at the commas) into a list

# do the same thing using the csv module

# separate the header and data


'''
EXERCISES:

1. Create a list containing the average number of incidents per year for each airline.
Example for Aer Lingus: (2 + 0)/30 = 0.07
Expected output: [0.07, 2.73, 0.23, ...]

2. Create a list of airline names (without the star).
Expected output: ['Aer Lingus', 'Aeroflot', 'Aerolineas Argentinas', ...]

3. Create a list (of the same length) that contains 1 if there's a star and 0 if not.
Expected output: [0, 1, 0, ...]

4. BONUS: Create a dictionary in which the key is the airline name (without the star)
   and the value is the average number of incidents.
Expected output: {'Aer Lingus': 0.07, 'Aeroflot': 2.73, ...}
'''


'''
A few extra things that will help you with the homework
'''

# 'set' data structure is useful for gathering unique elements

# 'in' statement is useful for lists

# 'in' is useful for strings (checks for substrings)

# 'in' is useful for dictionaries (checks keys but not values)

# 'count' method for strings counts how many times a character appears
