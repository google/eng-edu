# Text Classification

This repository contains end-to-end tutorial-like code samples to help solve
text classification problems using machine learning.

## Prerequisites

*   [TensorFlow](https://www.tensorflow.org/)
*   [Scikit-learn](http://scikit-learn.org/stable/)

## Modules

We have one module for each step in the text classification workflow.

1.  *load_data* - Functions to load data from four different datasets. For each
    of the dataset we:

    +   Read the required fields (texts and labels).
    +   Do any pre-processing if required. For example, make sure all label
        values are in range [0, num_classes-1].
    +   Split the data into training and validation sets.
    +   Shuffle the training data.

2.  *explore_data* - Helper functions to understand datasets.

3.  *vectorize_data* - N-gram and sequence vectorization functions.

4.  *build_model* - Helper functions to create multi-layer perceptron and
    separable convnet models.

5.  *train data* - Demonstrates how to use all of the above modules and train a
    model.

    + *train_ngram_model* - Trains a multi-layer perceptron model on IMDb
    movie reviews sentiment analysis dataset.

    + *train_sequence_model* - Trains a sepCNN model on Rotten Tomatoes movie
    reviews sentiment analysis dataset.

    + *train_fine_tuned_sequence_model* - Trains a sepCNN model with
    pre-trained embeddings that are fine-tuned on Tweet weather topic
    classification dataset.

    + *batch_train_sequence_model* - Same as *train_sequence_model* but here
    we are training data in batches.

6.  *tune_model* - Contains example to demonstrate how you can find the best
    hyper-parameter values for your model.
