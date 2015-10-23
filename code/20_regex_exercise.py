'''
EXERCISE: Regular Expressions
'''

# open file and store each line as one list element
with open('homicides.txt', mode='rU') as f:
    data = [row for row in f]


'''
Create a list of ages
'''

import re

ages = []
for row in data:
    match = re.search(r'\d+ years? old', row)
    if match:
        ages.append(match.group())
    else:
        ages.append('0')

# split the string on spaces, only keep the first element, and convert to int
ages = [int(element.split()[0]) for element in ages]

# calculate average age
sum(ages) / float(len(ages))

# check that 'data' and 'ages' are the same length
assert(len(data)==len(ages))


'''
Create a list of ages (using match groups)
'''

ages = []
for row in data:
    match = re.search(r'(\d+)( years? old)', row)
    if match:
        ages.append(int(match.group(1)))
    else:
        ages.append(0)


'''
Create a list of causes
'''

causes = []
for row in data:
    match = re.search(r'Cause: (.+?)<', row)
    if match:
        causes.append(match.group(1).lower())
    else:
        causes.append('unknown')

# tally the causes
from collections import Counter
Counter(causes)
