## Class 2 Homework: Command Line Chipotle

#### Submitting Your Homework

* Create a Markdown file that includes your answers **and** the code you used to arrive at those answers.
* Add this Markdown file to a GitHub repo that you'll use for all of your coursework.
* Submit a link to your repo using the homework submission form.

#### Command Line Tasks

1. Look at the head and the tail of **chipotle.tsv** in the **data** subdirectory of this repo. Think for a minute about how the data is structured. What do you think each column means? What do you think each row means? Tell me! (If you're unsure, look at more of the file contents.)
2. How many orders do there appear to be?
3. How many lines are in this file?
4. Which burrito is more popular, steak or chicken?
5. Do chicken burritos more often have black beans or pinto beans?
6. Make a list of all of the CSV or TSV files in the DAT8 repo (using a single command). Think about how wildcard characters can help you with this task.
7. Count the approximate number of occurrences of the word "dictionary" (regardless of case) across all files in the DAT8 repo.
8. **Optional:** Use the the command line to discover something "interesting" about the Chipotle data. Try using the commands from the "advanced" section!

#### Solution

1. **order_id** is the unique identifier for each order. **quantity** is the number purchased of a particular item. **item_name** is the primary name for the item being purchased. **choice_description** is list of modifiers for that item. **price** is the price for that entire line (taking **quantity** into account). A given order consists of one or more rows, depending upon the number of unique items being purchased in that order.
    * `head chipotle.tsv`
    * `tail chipotle.tsv`
2. There are 1834 orders (since 1834 is the highest **order_id** number).
3. The file has 4623 lines.
    * `wc -l chipotle.tsv`
4. Chicken burritos are more popular than steak burritos.
    * Compare `grep -i 'chicken burrito' chipotle.tsv | wc -l` with `grep -i 'steak burrito' chipotle.tsv | wc -l`
    * Alternatively, use the 'c' option of `grep` to skip the piping step: `grep -ic 'chicken burrito' chipotle.tsv`
5. Black beans are more popular than pinto beans (on chicken burritos).
    * Compare `grep -i 'chicken burrito' chipotle.tsv | grep -i 'black beans' | wc -l` with `grep -i 'chicken burrito' chipotle.tsv | grep -i 'pinto beans' | wc -l`
    * Alternatively, use the 'c' option of `grep` and a more complex regular expression pattern to skip the piping steps: `grep -ic 'chicken burrito.*black beans' chipotle.tsv`
6. At the moment, the CSV and TSV files in the DAT8 repo are **airlines.csv**, **chipotle.tsv**, and **sms.tsv**, all of which are in the **data** subdirectory.
    * Change your working directory to DAT8, and then use `find . -name *.?sv`
7. At the moment, there are 13 lines in DAT8 files that contain the word 'dictionary', which is a good approximation of the number of occurrences.
    * Change your working directory to DAT8, and then use `grep -ir 'dictionary' . | wc -l`
    * Alternatively, use the 'c' option of `grep` to skip the piping step: `grep -irc 'dictionary' .`
