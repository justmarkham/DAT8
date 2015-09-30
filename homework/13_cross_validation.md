## Class 13 Pre-work: Cross-validation

Watch my video on [cross-validation](https://www.youtube.com/watch?v=6dbrR-WymjI) (36 minutes), and be prepared to **discuss it in class** on Tuesday. The [notebook](../notebooks/13_cross_validation.ipynb) shown in the video is also in this repository.

Alternatively, read section 5.1 of [An Introduction to Statistical Learning](http://www-bcf.usc.edu/~gareth/ISL/) (11 pages).

Here are some questions to think about:

- What is the purpose of model evaluation?
    - The purpose is to estimate the likely performance of a model on out-of-sample data, so that we can choose the model that is most likely to generalize, and so that we can have an idea of how well that model will actually perform.
- What is the drawback of training and testing on the same data?
    - Training accuracy is maximized for overly complex models which overfit the training data, and thus it's not a good measure of how well a model will generalize.
- How does train/test split work, and what is its primary drawback?
    - It splits the data into two pieces, trains the model on the training set, and tests the model on the testing set. Testing accuracy can change a lot depending upon which observations happen to be in the training and testing sets.
- How does K-fold cross-validation work, and what is the role of "K"?
    - First, it splits the data into K equal folds. Then, it trains the model on folds 2 through K, tests the model on fold 1, and calculates the requested evaluation metric. Then, it repeats that process K-1 more times, until every fold has been the testing set exactly once.
- Why do we pass X and y, not X_train and y_train, to the `cross_val_score` function?
    - It will take care of splitting the data into the K folds, so we don't need to split it ourselves.
- Why does `cross_val_score` need a "scoring" parameter?
    - It needs to know what evaluation metric to calculate, since many different metrics are available.
- What does `cross_val_score` return, and what do we usually do with that object?
    - It returns a NumPy array containing the K scores. We usually calculate the mean score, though we might also be interested in the standard deviation.
- Under what circumstances does `cross_val_score` return negative scores?
    - The scores will be negative if the evaluation metric is a loss function (something you want to minimize) rather than a reward function (something you want to maximize).
- When should you use train/test split, and when should you use cross-validation?
    - Train/test split is useful when you want to inspect your testing results (via confusion matrix or ROC curve) and when evaluation speed is a concern. Cross-validation is useful when you are most concerned with the accuracy of your estimation.
