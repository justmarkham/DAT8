# # Natural Language Processing (NLP)

# ## Introduction
# 
# *Adapted from [NLP Crash Course](http://files.meetup.com/7616132/DC-NLP-2013-09%20Charlie%20Greenbacker.pdf) by Charlie Greenbacker and [Introduction to NLP](http://spark-public.s3.amazonaws.com/nlp/slides/intro.pdf) by Dan Jurafsky*

# ### What is NLP?
# 
# - Using computers to process (analyze, understand, generate) natural human languages
# - Most knowledge created by humans is unstructured text, and we need a way to make sense of it
# - Build probabilistic model using data about a language

# ### What are some of the higher level task areas?
# 
# - **Information retrieval**: Find relevant results and similar results
#     - [Google](https://www.google.com/)
# - **Information extraction**: Structured information from unstructured documents
#     - [Events from Gmail](https://support.google.com/calendar/answer/6084018?hl=en)
# - **Machine translation**: One language to another
#     - [Google Translate](https://translate.google.com/)
# - **Text simplification**: Preserve the meaning of text, but simplify the grammar and vocabulary
#     - [Rewordify](https://rewordify.com/)
#     - [Simple English Wikipedia](https://simple.wikipedia.org/wiki/Main_Page)
# - **Predictive text input**: Faster or easier typing
#     - [My application](https://justmarkham.shinyapps.io/textprediction/)
#     - [A much better application](https://farsite.shinyapps.io/swiftkey-cap/)
# - **Sentiment analysis**: Attitude of speaker
#     - [Hater News](http://haternews.herokuapp.com/)
# - **Automatic summarization**: Extractive or abstractive summarization
#     - [autotldr](https://www.reddit.com/r/technology/comments/35brc8/21_million_people_still_use_aol_dialup/cr2zzj0)
# - **Natural Language Generation**: Generate text from data
#     - [How a computer describes a sports match](http://www.bbc.com/news/technology-34204052)
#     - [Publishers withdraw more than 120 gibberish papers](http://www.nature.com/news/publishers-withdraw-more-than-120-gibberish-papers-1.14763)
# - **Speech recognition and generation**: Speech-to-text, text-to-speech
#     - [Google's Web Speech API demo](https://www.google.com/intl/en/chrome/demos/speech.html)
#     - [Vocalware Text-to-Speech demo](https://www.vocalware.com/index/demo)
# - **Question answering**: Determine the intent of the question, match query with knowledge base, evaluate hypotheses
#     - [How did supercomputer Watson beat Jeopardy champion Ken Jennings?](http://blog.ted.com/how-did-supercomputer-watson-beat-jeopardy-champion-ken-jennings-experts-discuss/)
#     - [IBM's Watson Trivia Challenge](http://www.nytimes.com/interactive/2010/06/16/magazine/watson-trivia-game.html)
#     - [The AI Behind Watson](http://www.aaai.org/Magazine/Watson/watson.php)

# ### What are some of the lower level components?
# 
# - **Tokenization**: breaking text into tokens (words, sentences, n-grams)
# - **Stopword removal**: a/an/the
# - **Stemming and lemmatization**: root word
# - **TF-IDF**: word importance
# - **Part-of-speech tagging**: noun/verb/adjective
# - **Named entity recognition**: person/organization/location
# - **Spelling correction**: "New Yrok City"
# - **Word sense disambiguation**: "buy a mouse"
# - **Segmentation**: "New York City subway"
# - **Language detection**: "translate this page"
# - **Machine learning**

# ### Why is NLP hard?
# 
# - **Ambiguity**:
#     - Hospitals are Sued by 7 Foot Doctors
#     - Juvenile Court to Try Shooting Defendant
#     - Local High School Dropouts Cut in Half
# - **Non-standard English**: text messages
# - **Idioms**: "throw in the towel"
# - **Newly coined words**: "retweet"
# - **Tricky entity names**: "Where is A Bug's Life playing?"
# - **World knowledge**: "Mary and Sue are sisters", "Mary and Sue are mothers"
# 
# NLP requires an understanding of the **language** and the **world**.

# ## Part 1: Reading in the Yelp Reviews

# - "corpus" = collection of documents
# - "corpora" = plural form of corpus

import pandas as pd
import numpy as np
import scipy as sp
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from textblob import TextBlob, Word
from nltk.stem.snowball import SnowballStemmer


# read yelp.csv into a DataFrame
url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/yelp.csv'
yelp = pd.read_csv(url)

# create a new DataFrame that only contains the 5-star and 1-star reviews
yelp_best_worst = yelp[(yelp.stars==5) | (yelp.stars==1)]

# define X and y
X = yelp_best_worst.text
y = yelp_best_worst.stars

# split the new DataFrame into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


# ## Part 2: Tokenization

# - **What:** Separate text into units such as sentences or words
# - **Why:** Gives structure to previously unstructured text
# - **Notes:** Relatively easy with English language text, not easy with some languages

# use CountVectorizer to create document-term matrices from X_train and X_test
vect = CountVectorizer()
X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)


# rows are documents, columns are terms (aka "tokens" or "features")
X_train_dtm.shape


# last 50 features
print vect.get_feature_names()[-50:]


# show vectorizer options
vect


# [CountVectorizer documentation](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)

# - **lowercase:** boolean, True by default
# - Convert all characters to lowercase before tokenizing.

# don't convert to lowercase
vect = CountVectorizer(lowercase=False)
X_train_dtm = vect.fit_transform(X_train)
X_train_dtm.shape


# - **ngram_range:** tuple (min_n, max_n)
# - The lower and upper boundary of the range of n-values for different n-grams to be extracted. All values of n such that min_n <= n <= max_n will be used.

# include 1-grams and 2-grams
vect = CountVectorizer(ngram_range=(1, 2))
X_train_dtm = vect.fit_transform(X_train)
X_train_dtm.shape


# last 50 features
print vect.get_feature_names()[-50:]


# **Predicting the star rating:**

# use default options for CountVectorizer
vect = CountVectorizer()

# create document-term matrices
X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)

# use Naive Bayes to predict the star rating
nb = MultinomialNB()
nb.fit(X_train_dtm, y_train)
y_pred_class = nb.predict(X_test_dtm)

# calculate accuracy
print metrics.accuracy_score(y_test, y_pred_class)


# calculate null accuracy
y_test_binary = np.where(y_test==5, 1, 0)
max(y_test_binary.mean(), 1 - y_test_binary.mean())


# define a function that accepts a vectorizer and calculates the accuracy
def tokenize_test(vect):
    X_train_dtm = vect.fit_transform(X_train)
    print 'Features: ', X_train_dtm.shape[1]
    X_test_dtm = vect.transform(X_test)
    nb = MultinomialNB()
    nb.fit(X_train_dtm, y_train)
    y_pred_class = nb.predict(X_test_dtm)
    print 'Accuracy: ', metrics.accuracy_score(y_test, y_pred_class)


# include 1-grams and 2-grams
vect = CountVectorizer(ngram_range=(1, 2))
tokenize_test(vect)


# ## Part 3: Stopword Removal

# - **What:** Remove common words that will likely appear in any text
# - **Why:** They don't tell you much about your text

# show vectorizer options
vect


# - **stop_words:** string {'english'}, list, or None (default)
# - If 'english', a built-in stop word list for English is used.
# - If a list, that list is assumed to contain stop words, all of which will be removed from the resulting tokens.
# - If None, no stop words will be used. max_df can be set to a value in the range [0.7, 1.0) to automatically detect and filter stop words based on intra corpus document frequency of terms.

# remove English stop words
vect = CountVectorizer(stop_words='english')
tokenize_test(vect)


# set of stop words
print vect.get_stop_words()


# ## Part 4: Other CountVectorizer Options

# - **max_features:** int or None, default=None
# - If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus.

# remove English stop words and only keep 100 features
vect = CountVectorizer(stop_words='english', max_features=100)
tokenize_test(vect)


# all 100 features
print vect.get_feature_names()


# include 1-grams and 2-grams, and limit the number of features
vect = CountVectorizer(ngram_range=(1, 2), max_features=100000)
tokenize_test(vect)


# - **min_df:** float in range [0.0, 1.0] or int, default=1
# - When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold. This value is also called cut-off in the literature. If float, the parameter represents a proportion of documents, integer absolute counts.

# include 1-grams and 2-grams, and only include terms that appear at least 2 times
vect = CountVectorizer(ngram_range=(1, 2), min_df=2)
tokenize_test(vect)


# ## Part 5: Introduction to TextBlob

# TextBlob: "Simplified Text Processing"

# print the first review
print yelp_best_worst.text[0]


# save it as a TextBlob object
review = TextBlob(yelp_best_worst.text[0])


# list the words
review.words


# list the sentences
review.sentences


# some string methods are available
review.lower()


# ## Part 6: Stemming and Lemmatization

# **Stemming:**
# 
# - **What:** Reduce a word to its base/stem/root form
# - **Why:** Often makes sense to treat related words the same way
# - **Notes:**
#     - Uses a "simple" and fast rule-based approach
#     - Stemmed words are usually not shown to users (used for analysis/indexing)
#     - Some search engines treat words with the same stem as synonyms

# initialize stemmer
stemmer = SnowballStemmer('english')

# stem each word
print [stemmer.stem(word) for word in review.words]


# **Lemmatization**
# 
# - **What:** Derive the canonical form ('lemma') of a word
# - **Why:** Can be better than stemming
# - **Notes:** Uses a dictionary-based approach (slower than stemming)

# assume every word is a noun
print [word.lemmatize() for word in review.words]


# assume every word is a verb
print [word.lemmatize(pos='v') for word in review.words]


# define a function that accepts text and returns a list of lemmas
def split_into_lemmas(text):
    text = unicode(text, 'utf-8').lower()
    words = TextBlob(text).words
    return [word.lemmatize() for word in words]


# use split_into_lemmas as the feature extraction function (WARNING: SLOW!)
vect = CountVectorizer(analyzer=split_into_lemmas)
tokenize_test(vect)


# last 50 features
print vect.get_feature_names()[-50:]


# ## Part 7: Term Frequency-Inverse Document Frequency (TF-IDF)

# - **What:** Computes "relative frequency" that a word appears in a document compared to its frequency across all documents
# - **Why:** More useful than "term frequency" for identifying "important" words in each document (high frequency in that document, low frequency in other documents)
# - **Notes:** Used for search engine scoring, text summarization, document clustering

# example documents
simple_train = ['call you tonight', 'Call me a cab', 'please call me... PLEASE!']


# Term Frequency
vect = CountVectorizer()
tf = pd.DataFrame(vect.fit_transform(simple_train).toarray(), columns=vect.get_feature_names())
tf


# Document Frequency
vect = CountVectorizer(binary=True)
df = vect.fit_transform(simple_train).toarray().sum(axis=0)
pd.DataFrame(df.reshape(1, 6), columns=vect.get_feature_names())


# Term Frequency-Inverse Document Frequency (simple version)
tf/df


# TfidfVectorizer
vect = TfidfVectorizer()
pd.DataFrame(vect.fit_transform(simple_train).toarray(), columns=vect.get_feature_names())


# **More details:** [TF-IDF is about what matters](http://planspace.org/20150524-tfidf_is_about_what_matters/)

# ## Part 8: Using TF-IDF to Summarize a Yelp Review
# 
# Reddit's autotldr uses the [SMMRY](http://smmry.com/about) algorithm, which is based on TF-IDF!

# create a document-term matrix using TF-IDF
vect = TfidfVectorizer(stop_words='english')
dtm = vect.fit_transform(yelp.text)
features = vect.get_feature_names()
dtm.shape


def summarize():
    
    # choose a random review that is at least 300 characters
    review_length = 0
    while review_length < 300:
        review_id = np.random.randint(0, len(yelp))
        review_text = unicode(yelp.text[review_id], 'utf-8')
        review_length = len(review_text)
    
    # create a dictionary of words and their TF-IDF scores
    word_scores = {}
    for word in TextBlob(review_text).words:
        word = word.lower()
        if word in features:
            word_scores[word] = dtm[review_id, features.index(word)]
    
    # print words with the top 5 TF-IDF scores
    print 'TOP SCORING WORDS:'
    top_scores = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    for word, score in top_scores:
        print word
    
    # print 5 random words
    print '\n' + 'RANDOM WORDS:'
    random_words = np.random.choice(word_scores.keys(), size=5, replace=False)
    for word in random_words:
        print word
    
    # print the review
    print '\n' + review_text


summarize()


# ## Part 9: Sentiment Analysis

print review


# polarity ranges from -1 (most negative) to 1 (most positive)
review.sentiment.polarity


# understanding the apply method
yelp['length'] = yelp.text.apply(len)
yelp.head(1)


# define a function that accepts text and returns the polarity
def detect_sentiment(text):
    return TextBlob(text.decode('utf-8')).sentiment.polarity


# create a new DataFrame column for sentiment (WARNING: SLOW!)
yelp['sentiment'] = yelp.text.apply(detect_sentiment)


# box plot of sentiment grouped by stars
yelp.boxplot(column='sentiment', by='stars')


# reviews with most positive sentiment
yelp[yelp.sentiment == 1].text.head()


# reviews with most negative sentiment
yelp[yelp.sentiment == -1].text.head()


# widen the column display
pd.set_option('max_colwidth', 500)


# negative sentiment in a 5-star review
yelp[(yelp.stars == 5) & (yelp.sentiment < -0.3)].head(1)


# positive sentiment in a 1-star review
yelp[(yelp.stars == 1) & (yelp.sentiment > 0.5)].head(1)


# reset the column display width
pd.reset_option('max_colwidth')


# ## Bonus: Adding Features to a Document-Term Matrix

# create a DataFrame that only contains the 5-star and 1-star reviews
yelp_best_worst = yelp[(yelp.stars==5) | (yelp.stars==1)]

# define X and y
feature_cols = ['text', 'sentiment', 'cool', 'useful', 'funny']
X = yelp_best_worst[feature_cols]
y = yelp_best_worst.stars

# split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


# use CountVectorizer with text column only
vect = CountVectorizer()
X_train_dtm = vect.fit_transform(X_train.text)
X_test_dtm = vect.transform(X_test.text)
print X_train_dtm.shape
print X_test_dtm.shape


# shape of other four feature columns
X_train.drop('text', axis=1).shape


# cast other feature columns to float and convert to a sparse matrix
extra = sp.sparse.csr_matrix(X_train.drop('text', axis=1).astype(float))
extra.shape


# combine sparse matrices
X_train_dtm_extra = sp.sparse.hstack((X_train_dtm, extra))
X_train_dtm_extra.shape


# repeat for testing set
extra = sp.sparse.csr_matrix(X_test.drop('text', axis=1).astype(float))
X_test_dtm_extra = sp.sparse.hstack((X_test_dtm, extra))
X_test_dtm_extra.shape


# use logistic regression with text column only
logreg = LogisticRegression(C=1e9)
logreg.fit(X_train_dtm, y_train)
y_pred_class = logreg.predict(X_test_dtm)
print metrics.accuracy_score(y_test, y_pred_class)


# use logistic regression with all features
logreg = LogisticRegression(C=1e9)
logreg.fit(X_train_dtm_extra, y_train)
y_pred_class = logreg.predict(X_test_dtm_extra)
print metrics.accuracy_score(y_test, y_pred_class)


# ## Bonus: Fun TextBlob Features

# spelling correction
TextBlob('15 minuets late').correct()


# spellcheck
Word('parot').spellcheck()


# definitions
Word('bank').define('v')


# language identification
TextBlob('Hola amigos').detect_language()


# ## Conclusion
# 
# - NLP is a gigantic field
# - Understanding the basics broadens the types of data you can work with
# - Simple techniques go a long way
# - Use scikit-learn for NLP whenever possible
