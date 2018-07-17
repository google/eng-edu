""" Module to test training accuracy.

The following tests exercise all modules by going through the complete ML
training process. We measure the accuracies at the end and check that they are
within +/- 2% of an expected number.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import batch_train_sequence_model
import load_data
import pytest
import train_fine_tuned_sequence_model
import train_ngram_model
import train_sequence_model


def test_train_ngram_model():
    data_dir = './data/'
    data = load_data.load_imdb_sentiment_analysis_dataset(data_dir)
    acc, loss = train_ngram_model.train_ngram_model(data)
    assert acc == pytest.approx(0.91, 0.02)
    assert loss == pytest.approx(0.24, 0.02)


def test_train_sequence_model():
    data_dir = './data/'
    data = load_data.load_rotten_tomatoes_sentiment_analysis_dataset(data_dir)
    acc, loss = train_sequence_model.train_sequence_model(data)
    assert acc == pytest.approx(0.68, 0.02)
    assert loss == pytest.approx(0.82, 0.02)


def test_train_fine_tuned_sequence_model():
    data_dir = './data/'
    embedding_data_dir = '~/data/'
    data = load_data.load_tweet_weather_topic_classification_dataset(data_dir)
    acc, loss = \
        train_fine_tuned_sequence_model.train_fine_tuned_sequence_model(
            data, embedding_data_dir)
    assert acc == pytest.approx(0.84, 0.02)
    assert loss == pytest.approx(0.55, 0.02)


def test_batch_train_sequence_model():
    data_dir = './data/'
    data = load_data.load_amazon_reviews_sentiment_analysis_dataset(data_dir)
    acc, loss = batch_train_sequence_model.batch_train_sequence_model(data)
    assert acc == pytest.approx(0.61, 0.02)
    assert loss == pytest.approx(0.89, 0.02)
