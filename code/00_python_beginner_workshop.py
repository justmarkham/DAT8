'''
Python Beginner Workshop
'''

'''
Multi-line comments go between 3 quotation marks.
You can use single or double quotes.
'''

# One-line comments are preceded by the pound symbol


# BASIC DATA TYPES

x = 5               # creates an object
print type(x)       # check the type: int (not declared explicitly)
type(x)             # automatically prints
type(5)             # assigning it to a variable is not required

type(5.0)           # float
type('five')        # str
type(True)          # bool


# LISTS

nums = [5, 5.0, 'five']     # multiple data types
nums                        # print the list
type(nums)                  # check the type: list
len(nums)                   # check the length: 3
nums[0]                     # print first element
nums[0] = 6                 # replace a list element

nums.append(7)              # list 'method' that modifies the list
help(nums.append)           # help on this method
help(nums)                  # help on a list object
nums.remove('five')         # another list method

sorted(nums)                # 'function' that does not modify the list
nums                        # it was not affected
nums = sorted(nums)         # overwrite the original list
sorted(nums, reverse=True)  # optional argument


# FUNCTIONS

def give_me_five():         # function definition ends with colon
    return 5                # indentation required for function body

give_me_five()              # prints the return value (5)
num = give_me_five()        # assigns return value to a variable, doesn't print it

def calc(x, y, op):         # three parameters (without any defaults)
    if op == 'add':         # conditional statement
        return x + y
    elif op == 'subtract':
        return x - y
    else:
        print 'Valid operations: add, subtract'

calc(5, 3, 'add')
calc(5, 3, 'subtract')
calc(5, 3, 'multiply')
calc(5, 3)


# EXERCISE: Write a function that takes two parameters (hours and rate), and
# returns the total pay.

def compute_pay(hours, rate):
    return hours * rate

compute_pay(40, 10.50)


# FOR LOOPS

# print each list element in uppercase
fruits = ['apple', 'banana', 'cherry']
for fruit in fruits:
    print fruit.upper()
