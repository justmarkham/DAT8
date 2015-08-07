## Setup checklist

This is a checklist to confirm that your laptop is set up properly for DAT8. If at any point you get an error message, please note the error message and we will help you to fix it! If you don't get any error messages, you are properly set up.

### GitHub
* Log into your GitHub account, and "star" the DAT8 repository (the one you are looking at right now) by clicking the Star button in the upper right corner of the screen.

### Git
* Open a command line application:
    * For Windows, we recommend [Git Bash](http://git-scm.com/download/win) instead of Git Shell (which uses Powershell).
    * For Mac, you will probably be using Terminal, or another command line tool of your choice.
* Type `git config --global user.name "YourFirstName YourLastName"` (including the quotes)
* Type `git config --global user.email "youremail@domain.com"` (use the email address associated with your GitHub account)

### Python
* While still at the command line:
    * Type `conda list` (if you choose not to use Anaconda, this will generate an error)
    * Type `pip install textblob`
    * Type `python` to open the Python interpreter
* While in the Python interpreter:
    * Look at the Python version number. It should start with 2.7. If your version number starts with 3, that's fine as long as you are aware of the differences between Python 2 and 3.
    * Type `import pandas`
    * Type `import textblob`
    * Type `exit()` to exit the interpreter. You can now close the command line application.
* Open Spyder (if you can't find Spyder, look for the Anaconda Launcher application)
    * In the console (probably on the right side of the screen), type `import pandas`
    * Type `import textblob`
        * If this worked from the interpreter but not in Spyder, and you're using a Mac, try opening the PYTHONPATH manager (in Spyder) and adding a path to where textblob was installed (such as `/Users/yourname/anaconda/lib/python2.7/site-packages/`). Then, restart Spyder.
