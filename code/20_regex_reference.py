'''
REFERENCE GUIDE: Regular Expressions
'''

'''
Rules for Searching:

Search proceeds through string from start to end, stopping at first match
All of the pattern must be matched

Basic Patterns:

Ordinary characters match themselves exactly
. matches any single character except newline \n
\w matches a word character (letter, digit, underscore)
\W matches any non-word character
\b matches boundary between word and non-word
\s matches single whitespace character (space, newline, return, tab, form)
\S matches single non-whitespace character
\d matches single digit (0 through 9)
\t matches tab
\n matches newline
\r matches return
\ match a special character, such as period: \.

Basic Python Usage:

match = re.search(r'pattern', string_to_search)
Returns match object
If there is a match, access match using match.group()
If there is no match, match is None
Use 'r' in front of pattern to designate a raw string
'''

import re

s = 'my 1st string!!'

match = re.search(r'my', s)     # returns match object
if match:                       # checks whether match was found
    print match.group()         # if match was found, then print result

re.search(r'my', s).group()     # single-line version (without error handling)
re.search(r'st', s).group()     # 'st'
re.search(r'sta', s).group()    # error
re.search(r'\w\w\w', s).group() # '1st'
re.search(r'\W', s).group()     # ' '
re.search(r'\W\W', s).group()   # '!!'
re.search(r'\s', s).group()     # ' '
re.search(r'\s\s', s).group()   # error
re.search(r'..t', s).group()    # '1st'
re.search(r'\s\St', s).group()  # ' st'
re.search(r'\bst', s).group()   # 'st'


'''
Repetition:

+ 1 or more occurrences of the pattern to its left
* 0 or more occurrences of the pattern to its left
? 0 or 1 occurrence of the pattern to its left

+ and * are 'greedy': they try to use up as much of the string as possible

Add ? after + or * to make them 'lazy': +? or *?
'''

s = 'sid is missing class'

re.search(r'miss\w+', s).group()    # 'missing'
re.search(r'is\w+', s).group()      # 'issing'
re.search(r'is\w*', s).group()      # 'is'

s = '<h1>my heading</h1>'

re.search(r'<.+>', s).group()   # '<h1>my heading</h1>'
re.search(r'<.+?>', s).group()  # '<h1>'


'''
Positions:

^ match start of a string
$ match end of a string
'''

s = 'sid is missing class'

re.search(r'^miss', s).group()  # error
re.search(r'..ss', s).group()   # 'miss'
re.search(r'..ss$', s).group()  # 'lass'


'''
Brackets:

[abc] match a or b or c
\w, \s, etc. work inside brackets, except period just means a literal period
[a-z] match any lowercase letter (dash indicates range unless it's last)
[abc-] match a or b or c or -
[^ab] match anything except a or b
'''

s = 'my email is john-doe@gmail.com'

re.search(r'\w+@\w+', s).group()            # 'doe@gmail'
re.search(r'[\w.-]+@[\w.-]+', s).group()    # 'john-doe@gmail.com'


'''
Lookarounds:

Lookahead matches a pattern only if it is followed by another pattern
100(?= dollars) matches '100' only if it is followed by ' dollars'

Lookbehind matches a pattern only if it is preceded by another pattern
(?<=\$)100 matches '100' only if it is preceded by '$'
'''

s = 'Name: Cindy, 30 years old'

re.search(r'\d+(?= years? old)', s).group()     # '30'
re.search(r'(?<=Name: )\w+', s).group()         # 'Cindy'


'''
Match Groups:

Parentheses create logical groups inside of match text
match.group(1) corresponds to first group
match.group(2) corresponds to second group
match.group() corresponds to entire match text (as usual)
'''

s = 'my email is john-doe@gmail.com'

match = re.search(r'([\w.-]+)@([\w.-]+)', s)
if match:
    match.group(1)      # 'john-doe'
    match.group(2)      # 'gmail.com'
    match.group()       # 'john-doe@gmail.com'


'''
Finding All Matches:

re.findall() finds all matches and returns them as a list of strings
list_of_strings = re.findall(r'pattern', string_to_search)

If pattern includes parentheses, a list of tuples is returned
'''

s = 'emails: joe@gmail.com, bob@gmail.com'

re.findall(r'[\w.-]+@[\w.-]+', s)       # ['joe@gmail.com', 'bob@gmail.com']
re.findall(r'([\w.-]+)@([\w.-]+)', s)   # [('joe', 'gmail.com'), ('bob', 'gmail.com')]


'''
Option Flags:

Options flags modify the behavior of the pattern matching

default: matching is case sensitive
re.IGNORECASE: ignore uppercase/lowercase differences ('a' matches 'a' or 'A')

default: period matches any character except newline
re.DOTALL: allow period to match newline

default: within a string of many lines, ^ and $ match start and end of entire string
re.MULTILINE: allow ^ and $ to match start and end of each line

Option flag is third argument to re.search() or re.findall():
re.search(r'pattern', string_to_search, re.IGNORECASE)
re.findall(r'pattern', string_to_search, re.IGNORECASE)
'''

s = 'emails: nicole@ga.co, joe@gmail.com, PAT@GA.CO'

re.findall(r'\w+@ga\.co', s)                # ['nicole@ga.co']
re.findall(r'\w+@ga\.co', s, re.IGNORECASE) # ['nicole@ga.co', 'PAT@GA.CO']


'''
Substitution:

re.sub() finds all matches and replaces them with a specified string
new_string = re.sub(r'pattern', r'replacement', string_to_search)

Replacement string can refer to text from matching groups:
\1 refers to group(1)
\2 refers to group(2)
etc.
'''

s = 'sid is missing class'

re.sub(r'is ', r'was ', s)                          # 'sid was missing class'

s = 'emails: joe@gmail.com, bob@gmail.com'

re.sub(r'([\w.-]+)@([\w.-]+)', r'\1@yahoo.com', s)  # 'emails: joe@yahoo.com, bob@yahoo.com'


'''
Useful to know, but not covered above:

re.split() splits a string by the occurrences of a pattern
re.compile() compiles a pattern (for improved performance if it's used many times)
A|B indicates a pattern that can match A or B
'''
