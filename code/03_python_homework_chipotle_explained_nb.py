# # Python Homework with Chipotle Data - Explained
# 
# *Original version written by [Alex Sherman](https://www.linkedin.com/in/alexjmsherman)*

# ## Part 1
# 
# - Read in the file with csv.reader() and store it in an object called 'file_nested_list'.
# - Hint: This is a TSV file, and csv.reader() needs to be told how to handle it.

# Change the working directory to the 'data' directory


# To use csv.reader, we must import the csv module
import csv

# The csv.reader has a delimeter parameter, which we set to '\t' to indicate that the file is tab-separated
with open('chipotle.tsv', mode='rU') as f:   # We temporarily refer to the file by the variable name f for file
    file_nested_list = [row for row in csv.reader(f, delimiter='\t')]   # Create a list by looping through each line in f


# ### Why use csv.reader?
# 
# As stated in the [CSV file reading and writing documentation](https://docs.python.org/2/library/csv.html):
# 
# > There is no "CSV standard", so the format is operationally defined by the many applications which 
# read and write it. The lack of a standard means that subtle differences often exist in the data 
# produced and consumed by different applications. These differences can make it annoying to process 
# CSV files from multiple sources. Still, while the delimiters and quoting characters vary, the 
# overall format is similar enough that it is possible to write a single module which can efficiently
# manipulate such data, hiding the details of reading and writing the data from the programmer.
# 
# In other words, depending on the source, there may be intricacies in the data format. These are not always easy to distinguish - for instance, non-visible new line characters. The csv.reader module is built to handle these intricacies, and thus provides an efficient way to load data.
# 
# This is why we prefer: `file_nested_list = [row for row in csv.reader(f, delimiter='\t')]`
# 
# Instead of: `file_nested_list = [row.split('\t') for row in f]`

# ## Part 2
# 
# - Separate 'file_nested_list' into the 'header' and the 'data'.

header = file_nested_list[0]
data = file_nested_list[1:]


# ## Part 3
# 
# - Calculate the average price of an order.
# - **Hint:** Examine the data to see if the 'quantity' column is relevant to this calculation.
# - **Hint:** Think carefully about the simplest way to do this!

# We want to find the average price of an order. This means we need the **sum of the price of all orders** and the **total number of orders**.

# ### Calculating the sum of the price of all orders

# After exploring our data for a minute, we find two orders for the same item - Chicken Bowl - differing by the quantity
print header
print data[4]
print data[5]


# We see that the item_price field reflects the quantity ordered. Thus, to calculate the total value of all orders, we can safely ignore the quantity column because the item_price takes quantity into account.

# We want the sum of all the order prices - the last item in each list. Here are two ways we could get this data:

# Option 1
prices = [row[4] for row in data]   # slice to position four

# Option 2
prices = [row[-1] for row in data]   # slice to the last position

# Let's look at the first five results:
prices[0:5]


# Each item in the list is a sting.  We can tell this because the results above are wrapped in quotes.
# To confirm, let's explicity check the type of the first item in the list:
type(prices[0])


# Since we want to do a calculation, we need to change the type from string to float. 
# To do this, we first need to remove the $. Here are two different ways to accomplish this:

# Option 1
prices = [row[4][1:] for row in data]   # remove the dollar sign by slicing

# Option 2
prices = [row[4].replace('$', '') for row in data]   # remove the dollar sign by replacing '$' with an empty string

# Let's look at the first five results:
prices[0:5]


# Now we can convert our results to floats
prices = [float(row[4][1:]) for row in data]

# Let's look at the first five results and check the type of the first item:
print prices[0:5]
print type(prices[0])


# Finally, we calculate our total order sum with the built-in sum function
total_order_sum = sum([float(row[4][1:]) for row in data]) 
total_order_sum


# ### Calculating the total number of orders

# We can look at the first and last items in the list
print header
print data[0]
print data[-1]


# It seems that there are 1834 orders. You could assume this since that's the maximum order_id, but it is best to check, as we are not certain that the data is clean. If the data was not sorted by order or if there was a missing order, then 1834 might not be correct.
# 
# So, let's confirm this assumption:

# First, let's build a list of the order_ids
order_ids = [row[0] for row in data]

# Let's look at the first ten results
order_ids[0:10]


# We only want to count each order once. We can get the distinct order values with the set function:
set(order_ids[0:10])


# Only keep unique order_ids
unique_order_ids = set(order_ids)

# Use the len function to determine the number of unique order_ids
num_orders = len(unique_order_ids)
num_orders


# ### Calculating the average price

# Finally, we answer the question by calculating the average
average_order_price = total_order_sum / num_orders
average_order_price


# Let's recap by looking at the final code:
total_order_sum = sum([float(row[4][1:]) for row in data])
num_orders = len(set([row[0] for row in data]))
average_order_price = round(total_order_sum / num_orders, 2)   # Let's round our result to 2 decimal places
average_order_price


# ## Part 4
# 
# - Create a list (or set) of all unique sodas and soft drinks that they sell.
# - **Note:** Just look for 'Canned Soda' and 'Canned Soft Drink', and ignore other drinks like 'Izze'.

# First let's look at all of the items
distinct_items = set([row[2] for row in data])
distinct_items 


# Our first goal is to reduce the dataset to only soda and soft drink orders.
# 
# It appears that the only items that use the word 'Canned' are 'Canned Soda' and 'Canned Soft Drink.'
# 
# This means we only need to use one filter criteria: **Look for rows with the word 'Canned'**

# Create a list only including soda and soft drink orders
soda_orders = []
for row in data:
    if 'Canned' in row[2]:
        soda_orders.append(row)

# Let's look at the first five results:
soda_orders[0:5]


# This can also be done using a list comprehension with an 'if' condition
soda_orders = [row for row in data if 'Canned' in row[2]]


# Just out of interest, let's look at two other ways we could have filtered the data:

soda_orders = [row for row in data if 'Canned Soda' in row[2] or 'Canned Soft Drink' in row[2]]
soda_orders[0:5]


soda_orders = [row for row in data if 'Canned Soda' == row[2] or 'Canned Soft Drink' == row[2]]
soda_orders[0:5]


# We only want the choice_description (e.g. Sprite, Mountain Dew). This is the fourth item in the list.
# Since Python uses 0-based indexing, we get this by using row[3] as the first argument in our list comprehension:
sodas = [row[3] for row in data if 'Canned' in row[2]]

# Let's look at the first five results
sodas[0:5]


# The results above may look like 5 lists inside of a larger list. Let's assume that's the case, and try to get the first Sprite:

sodas[0][0]


# What is going on?
# 
# The raw data for choice_description includues brackets (e.g. [Sprite]). We loaded this data in as a string, so while it looks like we have lists inside lists, the result is actually just one list. This is indicated by the quotes wrapping each item in the list, which means the list contains strings.

# Print the first list element
print sodas[0]

# Show that it's a string
print type(sodas[0])

# It is 8 characters long, including the brackets
print len(sodas[0])


# Let's strip the brackets at the start and end of each soda name, using [1:-1] to remove the first and last characters
sodas = [row[3][1:-1] for row in data if 'Canned' in row[2]]

# Let's look at the first five results
sodas[0:5]


# Almost done - we just need to get rid of duplicate values
unique_sodas = set([row[3][1:-1] for row in data if 'Canned' in row[2]])   # Success in one line of code!
unique_sodas


# Just for reference, how would this look if we did not use a list comprehension?

# build a list of all sodas
sodas = []
for row in data:
    if 'Canned' in row[2]:
        sodas.append(row[3][1:-1])   # strip the brackets

# create a set of unique sodas
unique_sodas = set(sodas)


# ## Part 5
# 
# - Calculate the average number of toppings per burrito.
# - **Note:** Let's ignore the 'quantity' column to simplify this task.
# - **Hint:** Think carefully about the easiest way to count the number of toppings!

# To calculate the average number of toppings, we simply need to divide the **total number of burritos** by the **total number of toppings**.

# ### Calculating the total number of burritos

# keep a running total
burrito_count = 0

# loop through the data, looking for lines containing 'Burrito'
for row in data:
    if 'Burrito' in row[2]:
        burrito_count = burrito_count + 1


# Like many programming languages, Python allows you to use `x += 1` as a replacement for `x = x + 1`. Let's use that instead:

# keep a running total
burrito_count = 0

# loop through the data, looking for lines containing 'Burrito'
for row in data:
    if 'Burrito' in row[2]:
        burrito_count += 1   # this is the only line that changed


burrito_count


# The count is 1172, which seems reasonable given the total number of orders (1834).

# ### Calculating the total number of toppings

# Let's look at a single burrito order
data[7]


# There appear to be 8 toppings:
data[7][3]


# With all of this formatting within the string, what's the easiest way to count the number of toppings?
# 
# Start by asking yourself: How did you count the number of toppings? You probably looked for **commas**!

# Use the string method 'count' to count the number of commas
data[7][3].count(',')


# And of course, if there are 7 commas, that means there are 8 toppings.
# 
# So, let's revise our original loop:

# keep a running total of burritos and toppings
burrito_count = 0
topping_count = 0

# calculate number of toppings by counting the commas and adding 1
for row in data:
    if 'Burrito' in row[2]:
        burrito_count += 1
        topping_count += (row[3].count(',') + 1)

print burrito_count
print topping_count


# ### Calculating the average number of toppings

# calculate the average topping count and round to 2 digits
round(topping_count / float(burrito_count), 2)


# Just for reference, how would this look if we used list comprehensions?

burrito_count = sum(1 for row in data if 'Burrito' in row[2])
topping_count = sum([row[3].count(',') + 1 for row in data if 'Burrito' in row[2]])
round(topping_count / float(burrito_count), 2)


# ## Part 6
# 
# - Create a dictionary in which the keys represent chip orders and the values represent the total number of orders.
# - **Expected output:** {'Chips and Roasted Chili-Corn Salsa': 18, ... }
# - **Note:** Please take the 'quantity' column into account!
# - **Optional:** Learn how to use 'defaultdict' to simplify your code.

# ### Building a dictionary of names

# Let's pretend I have a list of four names, and I want to make a dictionary in which the **key** is the name, and the **value** is the count of that name.

# This is my list of names
names = ['Ben', 'Victor', 'Laura', 'Victor']


# I want to create a dictionary that looks like this:
# 
# `{'Ben':1, 'Laura':1, 'Victor':2}`
# 
# How would I do that? Here's my first attempt:

# create empty dictionary
name_count = {}

# loop through list of names
for name in names:
    # set the name as the key and 1 as the value
    name_count[name] = 1

name_count


# Well, that creates a dictionary, but it didn't count Victor twice.
# 
# Let's try again:

name_count = {}

for name in names:
    # increment the value
    name_count[name] += 1

name_count


# That doesn't work because the dictionary starts out empty, and you can't tell Python to "increment the Ben value by 1" unless the Ben value starts at 0.
# 
# Let's try to fix that:

name_count = {}

for name in names:
    # initially set every name to 0
    name_count[name] = 0

for name in names:
    # increment the value
    name_count[name] += 1

name_count


# By looping through the list twice, we fixed the problem. But that's kind of clunky.
# 
# Here's what we really want to do:

name_count = {}

for name in names:
    
    # check if the key is already present in the dictionary
    if name not in name_count:
        name_count[name] = 1   # this is a new key, so create key/value pair
    else:
        name_count[name] += 1   # this is an existing key, so add to the value

name_count


# ### Applying this technique to chip orders

# Reminder on the header
header


# Find all the chip orders
chip_orders = [row for row in data if 'Chips' in row[2]]

# Look at the first five
chip_orders[:5]


# The chip quantities are easily accessible
chip_quantities = [row[1] for row in data if 'Chips' in row[2]]

# Look at the first five
chip_quantities[:5]


# Let's put this all together!

# start with an empty dictionary
chips = {}

# if chip order is not in dictionary, then add a new key/value pair
# if chip order is already in dictionary, then update the value for that key
for row in data:
    if 'Chips' in row[2]:
        if row[2] not in chips:
            chips[row[2]] = int(row[1])     # this is a new key, so create key/value pair
        else:
            chips[row[2]] += int(row[1])    # this is an existing key, so add to the value

chips


# ### Using defaultdict instead

# [defaultdict](https://docs.python.org/2/library/collections.html) simplifies this task, because it saves you the trouble of checking whether a key already exists.
# 
# Here's a simple example using the names data:

# This is a tiny variation of our code that previously raised an error

# Create an empty dictionary that will eventually contain integers (and thus the default value is 0)
from collections import defaultdict
name_count = defaultdict(int)

# We no longer have to check if the key is present
for name in names:
    name_count[name] += 1

name_count


# It will print nicely if we convert it to a regular dictionary
dict(name_count)


# Apply this to the chip orders
dchips = defaultdict(int)

for row in data:
    if 'Chips' in row[2]:
        dchips[row[2]] += int(row[1])

dict(dchips)
