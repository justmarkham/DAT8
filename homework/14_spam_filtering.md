## Class 14 Pre-work: Spam Filtering

Read Paul Graham's [A Plan for Spam](http://www.paulgraham.com/spam.html).

Here are some questions to think about:

- Should a spam filter optimize for sensitivity or specificity, in Paul's opinion?
    - Specificity, in order to minimize false positives (non-spam being incorrectly marked as spam).
- Before he tried the "statistical approach" to spam filtering, what was his approach?
    - He hand-engineered features and used those features to compute a score.
- What are the key components of his statistical filtering system? In other words, how does it work?
    - Scan the entire text (including headers) and tokenize it.
    - Count the number of occurrences of each token in the ham corpus and the spam corpus (separately).
    - Assign each token a "spam score" based on its relative frequency in the corpora.
    - For new email, only take into account the 15 most "interesting" tokens.
- What did Paul say were some of the benefits of the statistical approach?
    - It works better (almost no false positives).
    - It requires less work because it discovers features automatically.
    - The "spam score" is interpretable.
    - It can easily be tuned to the individual user.
    - It evolves with the spam.
    - It creates an implicit whitelist/blacklist of email addresses, server names, etc.
- How good was his prediction of the "spam of the future"?
    - Great!
