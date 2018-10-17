Solidify your knowledge of ML fundamentals and get up to speed with TensorFlow's
new high-level Estimator API with these
[Datalab](https://cloud.google.com/datalab/) programming exercises (click
[here](https://github.com/google/eng-edu/blob/master/ml/cc/README.md) for
installation instructions):

*   **First Steps with TensorFlow:** An introduction to building models with
    TensorFlow’s [`Estimator`](https://www.tensorflow.org/extend/estimators)
    API. Train and evaluate a
    [`LinearRegressor`](https://www.tensorflow.org/api_docs/python/tf/contrib/learn/LinearRegressor)
    model for predicting California housing prices, and experiment with tuning
    hyperparameters to reduce loss.

*   **Synthetic Features and Outliers:** An overview of feature engineering.
    Refine the California housing model by creating a synthetic feature and
    clipping outliers.

*   **Validation:** A deeper dive into best practices for training and
    evaluation. Add more features to the California housing model, and split the
    data into train/validation/test sets to prevent overfitting.

*   **Feature Sets:** A more in-depth exploration of feature engineering.
    Simplify the California housing model by removing features—without
    decreasing performance!

*   **Feature Crosses:** Advanced feature engineering using TensorFlow
    `Estimator`s. Further improve the California housing model by binning
    features and adding feature crosses.

*   **Logistic Regression:** An introduction to performing logistic regression.
    Convert the California housing model into a binary classifier that predicts
    whether a city block is high-cost.

*   **Sparsity and L1 Regularization:** A walkthrough illustrating how to apply
    regularization to prevent model overfitting and increase efficiency. Find a
    good L1 regularization coefficient to apply to the California housing
    classifier to reduce log-loss.

*   **Intro to Neural Nets:** An introduction to constructing neural networks
    using TensorFlow `Estimator`s. Train a new California housing regression
    model with a
    [DNNRegressor](https://www.tensorflow.org/api_docs/python/tf/contrib/learn/DNNRegressor),
    and tune hidden-layer hyperparameters to minimize loss.

*   **Improving Neural Net Performance:** Advanced neural network training.
    Augment the performance of the California housing `DNNRegressor` by
    normalizing features and experimenting with different training optimization
    algorithms.

*   **Classifying Handwritten Digits with Neural Networks:** An exploration of
    neural networks in the field of computer vision. Build a
    [`DNNClassifier`](https://www.tensorflow.org/api_docs/python/tf/contrib/learn/DNNClassifier)
    that can accurately recognize the handwritten digits in the [MNIST data
    set](http://yann.lecun.com/exdb/mnist/).

*   **Intro to Sparse Data and Embeddings:** An introduction to embeddings.
    Build a sentiment-analysis model for movie-review text using an embedding
    that projects data into two dimensions.

*   **Intro to Fairness:** An introduction to evaluating models for fairness.
    Explore the Adult Census Income data set to proactively identify potential
    sources of bias, and then, post-training, evaluate model performance by
    subgroup.

## Principal Contributors

The following Google employees contributed to the development of these
exercises:

*   Eric Breck, Software Engineer
*   Mig Gerard, Software Engineer
*   Sally Goldman, Research Scientist
*   Mustafa Ispir, Software Engineer
*   Vihan Jain, Software Engineer
*   Sanders Kleinfeld, Technical Writer
*   Dan Moldovan, Software Engineer
*   Nicholas Navaroli, Software Engineer
*   Barry Rosenberg, Technical Writer
*   D. Sculley, Software Engineer
*   Shawn Simister, Developer Programs Engineer
*   Andrew Zaldivar, Developer Advocate

## Licensing

All exercises are made available under the [Apache License Version
2.0](https://github.com/google/eng-edu/blob/master/LICENSE)
