"""Module to load data.

Consists of functions to load data from four different datasets (IMDb, Rotten
Tomatoes, Tweet Weather, Amazon Reviews). Each of these functions do the
following:
    - Read the required fields (texts and labels).
    - Do any pre-processing if required. For example, make sure all label
        values are in range [0, num_classes-1].
    - Split the data into training and validation sets.
    - Shuffle the training data.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random

import numpy as np
import pandas as pd


def load_imdb_sentiment_analysis_dataset(data_path, seed=123):
    """Loads the Imdb movie reviews sentiment analysis dataset.

    # Arguments
        data_path: string, path to the data directory.
        seed: int, seed for randomizer.

    # Returns
        A tuple of training and validation data.
        Number of training samples: 25000
        Number of test samples: 25000
        Number of categories: 2 (0 - negative, 1 - positive)

    # References
        Mass et al., http://www.aclweb.org/anthology/P11-1015

        Download and uncompress archive from:
        http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
    """
    imdb_data_path = os.path.join(data_path, 'aclImdb')

    # Load the training data
    train_texts = []
    train_labels = []
    for category in ['pos', 'neg']:
        train_path = os.path.join(imdb_data_path, 'train', category)
        for fname in sorted(os.listdir(train_path)):
            if fname.endswith('.txt'):
                with open(os.path.join(train_path, fname)) as f:
                    train_texts.append(f.read())
                train_labels.append(0 if category == 'neg' else 1)

    # Load the validation data.
    test_texts = []
    test_labels = []
    for category in ['pos', 'neg']:
        test_path = os.path.join(imdb_data_path, 'test', category)
        for fname in sorted(os.listdir(test_path)):
            if fname.endswith('.txt'):
                with open(os.path.join(test_path, fname)) as f:
                    test_texts.append(f.read())
                test_labels.append(0 if category == 'neg' else 1)

    # Shuffle the training data and labels.
    random.seed(seed)
    random.shuffle(train_texts)
    random.seed(seed)
    random.shuffle(train_labels)

    return ((train_texts, np.array(train_labels)),
            (test_texts, np.array(test_labels)))


def load_tweet_weather_topic_classification_dataset(data_path,
                                                    validation_split=0.2,
                                                    seed=123):
    """Loads the tweet weather topic classification dataset.

    # Arguments
        data_path: string, path to the data directory.
        validation_split: float, percentage of data to use for validation.
        seed: int, seed for randomizer.

    # Returns
        A tuple of training and validation data.
        Number of training samples: 62356
        Number of test samples: 15590
        Number of topics: 15

    # References
        https://www.kaggle.com/c/crowdflower-weather-twitter/data

        Download from:
        https://www.kaggle.com/c/3586/download/train.csv
    """
    columns = [1] + [i for i in range(13, 28)]  # 1 - text, 13-28 - topics.
    data = _load_and_shuffle_data(data_path, 'train.csv', columns, seed)

    # Get tweet text and the max confidence score for the weather types.
    texts = list(data['tweet'])
    weather_data = data.iloc[:, 1:]

    labels = []
    for i in range(len(texts)):
        # Pick topic with the max confidence score.
        labels.append(np.argmax(list(weather_data.iloc[i, :].values)))

    return _split_training_and_validation_sets(
        texts, np.array(labels), validation_split)


def load_rotten_tomatoes_sentiment_analysis_dataset(data_path,
                                                    validation_split=0.2,
                                                    seed=123):
    """Loads the rotten tomatoes sentiment analysis dataset.

    # Arguments
        data_path: string, path to the data directory.
        validation_split: float, percentage of data to use for validation.
        seed: int, seed for randomizer.

    # Returns
        A tuple of training and validation data.
        Number of training samples: 124848
        Number of test samples: 31212
        Number of categories: 5 (0 - negative, 1 - somewhat negative,
                2 - neutral, 3 - somewhat positive, 4 - positive)

    # References
        https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data

        Download and uncompress archive from:
        https://www.kaggle.com/c/3810/download/train.tsv.zip
    """
    columns = (2, 3)  # 2 - Phrases, 3 - Sentiment.
    data = _load_and_shuffle_data(data_path, 'train.tsv', columns, seed, '\t')

    # Get the review phrase and sentiment values.
    texts = list(data['Phrase'])
    labels = np.array(data['Sentiment'])
    return _split_training_and_validation_sets(texts, labels, validation_split)


def load_amazon_reviews_sentiment_analysis_dataset(data_path, seed=123):
    """Loads the amazon reviews sentiment analysis dataset.

    # Arguments
        data_path: string, path to the data directory.
        seed: int, seed for randomizer.

    # Returns
        A tuple of training and validation data.
        Number of training samples: 3000000
        Number of test samples: 650000
        Number of categories: 5

    # References
        Zhang et al., https://arxiv.org/abs/1509.01626

        Download and uncompress archive from:
                https://drive.google.com/open?id=0Bz8a_Dbh9QhbZVhsUnRWRDhETzA
    """
    columns = (0, 1, 2)  # 0 - label, 1 - title, 2 - body.
    train_data = _load_and_shuffle_data(
            data_path, 'train.csv', columns, seed, header=None)

    test_data_path = os.path.join(data_path, 'test.csv')
    test_data = pd.read_csv(test_data_path, usecols=columns, header=None)

    # Get train and test labels. Replace label value 5 with value 0.
    train_labels = np.array(train_data.iloc[:, 0])
    train_labels[train_labels == 5] = 0
    test_labels = np.array(test_data.iloc[:, 0])
    test_labels[test_labels == 5] = 0

    # Get train and test texts.
    train_texts = []
    for index, row in train_data.iterrows():
        train_texts.append(_get_amazon_review_text(row))
    test_texts = []
    for index, row in test_data.iterrows():
        test_texts.append(_get_amazon_review_text(row))

    return ((train_texts, train_labels), (test_texts, test_labels))


def _get_amazon_review_text(row):
    """Gets the Amazon review text given row data.

    # Arguments
        row: pandas row data from Amazon review dataset.

    # Returns:
        string, text corresponding to the row.
    """
    title = ''
    if type(row[1]) == str:
        title = row[1].replace('\\n', '\n').replace('\\"', '"')
    body = ''
    if type(row[2]) == str:
        body = row[2].replace('\\n', '\n').replace('\\"', '"')
    return title + ', ' + body


def _load_and_shuffle_data(data_path,
                           file_name,
                           cols,
                           seed,
                           separator=',',
                           header=0):
    """Loads and shuffles the dataset using pandas.

    # Arguments
        data_path: string, path to the data directory.
        file_name: string, name of the data file.
        cols: list, columns to load from the data file.
        seed: int, seed for randomizer.
        separator: string, separator to use for splitting data.
        header: int, row to use as data header.
    """
    np.random.seed(seed)
    data_path = os.path.join(data_path, file_name)
    data = pd.read_csv(data_path, usecols=cols, sep=separator, header=header)
    return data.reindex(np.random.permutation(data.index))


def _split_training_and_validation_sets(texts, labels, validation_split):
    """Splits the texts and labels into training and validation sets.

    # Arguments
        texts: list, text data.
        labels: list, label data.
        validation_split: float, percentage of data to use for validation.

    # Returns
        A tuple of training and validation data.
    """
    num_training_samples = int((1 - validation_split) * len(texts))
    return ((texts[:num_training_samples], labels[:num_training_samples]),
            (texts[num_training_samples:], labels[num_training_samples:]))
