## Class 9 Pre-work: Bias-Variance Tradeoff

Read this excellent article, [Understanding the Bias-Variance Tradeoff](http://scott.fortmann-roe.com/docs/BiasVariance.html), and be prepared to **discuss it in class** on Tuesday.

**Note:** You can ignore sections 4.2 and 4.3.

Here are some questions to think about while you read:
* In the Party Registration example, what are the features? What is the response? Is this a regression or classification problem?
    * The features are wealth and religiousness. The response is voter party registration. This is a classification problem.
* Conceptually, how is KNN being applied to this problem to make a prediction?
    * Find the K most similar voters in the training data (in terms of wealth and religiousness), and use the majority party registration among those "neighbors" as the predicted party registration for the unknown individual.
* How do the four visualizations in section 3 relate to one another? Change the value of K using the slider, and make sure you understand what changed in the visualizations (and why it changed).
    * First viz: training data colored by response value
    * Second viz: classification map for K=1
    * Third viz: out-of-sample data colored by predicted response value, and identification of the neighborhoods used to make that prediction
    * Fourth viz: predicted response value for each hexagon
    * Changing K changes the predictions in the third and fourth viz
* In figures 4 and 5, what do the lighter colors versus the darker colors mean? How is the darkness calculated?
    * Darkness indicates confidence in the prediction, and is calculated using the proportion of nearest neighbors that have the same response value.
* What does the black line in figure 5 represent? What predictions would the best possible machine learning model make, with respect to this line?
    * The black line is the the underlying model that generated the training data. The best possible machine learning model would learn that line as its decision boundary. It would not be a perfect model, but it would be the best possible model.
* Choose a very small value of K, and click the button "Generate New Training Data" a number of times. Do you "see" low variance or high variance, and low bias or high bias?
    * High variance, low bias
* Repeat this with a very large value of K. Do you "see" low variance or high variance, and low bias or high bias?
    * Low variance, high bias
* Try using other values of K. What value of K do you think is "best"? How do you define "best"?
    * A value of K in the middle is best. The best value is the value that results in a model whose predictions most consistently match the decision boundary.
* Does a small value for K cause "overfitting" or "underfitting"?
    * Overfitting
* Why should we care about variance at all? Shouldn't we just minimize bias and ignore variance?
    * If you had all of the possible data (past and future), a model with high complexity (and thus high variance) would be ideal because it would capture all of the complexity in the data and wouldn't need to generalize. But given that we only have a single sample of data, both bias and variance contribute to prediction error and should be appropriately balanced.
