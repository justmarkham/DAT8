'''
Python Intermediate Workshop
'''

'''
LISTS
'''

# creating
a = [1, 2, 3, 4, 5]     # create lists using brackets

# slicing
a[0]        # returns 1 (Python is zero indexed)
a[1:3]      # returns [2, 3] (inclusive of first index but exclusive of second)
a[-1]       # returns 5 (last element)

# appending
a[5] = 6        # error because you can't assign outside the existing range
a.append(6)     # list method that appends 6 to the end
a = a + [0]     # use plus sign to combine lists

# checking length
len(a)      # returns 7

# checking type
type(a)     # returns list
type(a[0])  # returns int

# sorting
sorted(a)               # sorts the list
sorted(a, reverse=True) # reverse=True is an 'optional argument'
sorted(a, True)         # error because optional arguments must be named


'''
STRINGS
'''

# creating
a = 'hello'     # can use single or double quotes

# slicing
a[0]        # returns 'h' (works like list slicing)
a[1:3]      # returns 'el'
a[-1]       # returns 'o'

# concatenating
a + ' there'        # use plus sign to combine strings
5 + ' there'        # error because they are different types
str(5) + ' there'   # cast 5 to a string in order for this to work

# uppercasing
a[0] = 'H'      # error because strings are immutable (can't overwrite characters)
a.upper()       # string method (this method doesn't exist for lists)

# checking length
len(a)      # returns 5 (number of characters)


'''
EXERCISE:
1. Create a list of the first names of your family members.
2. Print the name of the last person in the list.
3. Print the length of the name of the first person in the list.
4. Change one of the names from their real name to their nickname.
5. Append a new person to the list.
6. Change the name of the new person to lowercase using the string method 'lower'.
7. Sort the list in reverse alphabetical order.
Bonus: Sort the list by the length of the names (shortest to longest).
'''

names = ['Wesley', 'Larry', 'Wan']  # list of names
names[-1]                           # last element
len(names[0])                       # length of first string
names[0] = 'Wes'                    # overwrite existing element
names.append('Gabriel')             # append new element
names[-1] = names[-1].lower()       # change last string to be lowercase
sorted(names, reverse=True)         # sort the list in reverse order
sorted(names, key=len)              # sort the list by length


'''
FOR LOOPS AND LIST COMPREHENSIONS
'''

# for loop to print 1 through 5
nums = range(1, 6)      # create a list of 1 through 5
for num in nums:        # num 'becomes' each list element for one loop
    print num

# for loop to print 1, 3, 5
other = [1, 3, 5]       # create a different list
for x in other:         # name 'x' does not matter, not defined in advance
    print x             # this loop only executes 3 times (not 5)

# for loop to create a list of 2, 4, 6, 8, 10
doubled = []                # create empty list to store results
for num in nums:            # loop through nums (will execute 5 times)
    doubled.append(num*2)   # append the double of the current value of num

# equivalent list comprehension
doubled = [num*2 for num in nums]   # expression (num*2) goes first, brackets
                                    # indicate we are storing results in a list


'''
EXERCISE 1:
Given that: letters = ['a', 'b', 'c']
Write a list comprehension that returns: ['A', 'B', 'C']

EXERCISE 2 (BONUS):
Given that: word = 'abc'
Write a list comprehension that returns: ['A', 'B', 'C']

EXERCISE 3 (BONUS):
Given that: fruits = ['Apple', 'Banana', 'Cherry']
Write a list comprehension that returns: ['A', 'B', 'C']
'''

letters = ['a', 'b', 'c']
[letter.upper() for letter in letters]  # iterate through a list of strings,
                                        # and each string has an 'upper' method
word = 'abc'
[letter.upper() for letter in word]     # iterate through each character

fruits = ['Apple', 'Banana', 'Cherry']
[fruit[0] for fruit in fruits]          # slice the first character from each string


'''
DICTIONARIES
'''

# dictionaries are made of key-value pairs (like a real dictionary)
family = {'dad':'Homer', 'mom':'Marge', 'size':2}

# check the length
len(family)         # returns 3 (number of key-value pairs)

# use the key to look up a value (fast operation regardless of dictionary size)
family['dad']       # returns 'Homer'

# can't use a value to look up a key
family['Homer']     # error

# dictionaries are unordered
family[0]           # error

# add a new entry
family['cat'] = 'snowball'

# keys must be unique, so this edits an existing entry
family['cat'] = 'snowball ii'

# delete an entry
del family['cat']

# keys can be strings or numbers or tuples, values can be any type
family['kids'] = ['bart', 'lisa']   # value can be a list

# accessing a list element within a dictionary
family['kids'][0]   # returns 'bart'

# useful methods
family.keys()       # returns list: ['dad', 'kids', 'mom', 'size']
family.values()     # returns list: ['Homer', ['bart', 'lisa'], 'Marge', 2]
family.items()      # returns list of tuples:
                    # [('dad', 'Homer'), ('kids', ['bart', 'lisa']), ('mom', 'Marge'), ('size', 2)]


'''
EXERCISE:
1. Print the name of the mom.
2. Change the size to 5.
3. Add 'Maggie' to the list of kids.
4. Fix 'bart' and 'lisa' so that the first letter is capitalized.
Bonus: Do this last step using a list comprehension.
'''

family['mom']                       # returns 'Marge'
family['size'] = 5                  # replaces existing value for 'size'
family['kids'].append('Maggie')     # access a list, then append 'Maggie' to it
family['kids'][0] = 'Bart'          # capitalize names by overwriting them
family['kids'][1] = 'Lisa'

# or, capitalize using a list comprehension and the 'capitalize' string method
family['kids'] = [kid.capitalize() for kid in family['kids']]

# or, slice the string, uppercase the first letter, and concatenate with other letters
family['kids'] = [kid[0].upper() + kid[1:] for kid in family['kids']]


'''
REQUESTS
'''

# import module (make its functions available)
import requests

# use requests to talk to the web
r = requests.get('http://www.google.com')
type(r)         # special 'response' object
r.text          # HTML of web page stored as string
type(r.text)    # string is encoded as unicode
r.text[0]       # string can be sliced like any string


'''
APIs

What is an API?
- Application Programming Interface
- Structured way to expose specific functionality and data access to users
- Web APIs usually follow the "REST" standard

How to interact with a REST API:
- Make a "request" to a specific URL (an "endpoint"), and get the data back in a "response"
- Most relevant request method for us is GET (other methods: POST, PUT, DELETE)
- Response is often JSON format
- Web console is sometimes available (allows you to explore an API)

API Providers: https://apigee.com/providers
Echo Nest API Console: https://apigee.com/console/echonest
API key: http://bit.ly/myechonest
'''

# request data from the Echo Nest API
r = requests.get('http://developer.echonest.com/api/v4/artist/top_hottt?api_key=KBGUPZPJZS9PHWNIN&format=json')
r.text          # looks like a dictionary
type(r.text)    # actually stored as a string
r.json()        # decodes JSON
type(r.json())  # JSON can be represented as a dictionary
top = r.json()  # store that dictionary

# store the artist data
artists = top['response']['artists']    # list of 15 dictionaries

# create a list of artist names only
names = [artist['name'] for artist in artists]  # can iterate through list to access dictionaries
