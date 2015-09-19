## Class 13 Pre-work: Cross-validation

Watch my video on [cross-validation](https://www.youtube.com/watch?v=6dbrR-WymjI) (36 minutes), and be prepared to **discuss it in class** on Tuesday. The [notebook](../notebooks/13_cross_validation.ipynb) shown in the video is also in this repository.

Alternatively, read section 5.1 of [An Introduction to Statistical Learning](http://www-bcf.usc.edu/~gareth/ISL/) (11 pages).

Here are some questions to think about:

- What is the purpose of model evaluation?
- What is the drawback of training and testing on the same data?
- How does train/test split work, and what is its primary drawback?
- How does K-fold cross-validation work, and what is the role of "K"?
- Why do we pass X and y, not X_train and y_train, to the `cross_val_score` function?
- Why does `cross_val_score` need a "scoring" parameter?
- What does `cross_val_score` return, and what do we usually do with that object?
- Under what circumstances does `cross_val_score` return negative scores?
- When should you use train/test split, and when should you use cross-validation?
