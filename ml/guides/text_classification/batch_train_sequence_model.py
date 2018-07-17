"""Module to train sequence model with batches of data.

Vectorizes training and validation texts into sequences and uses that for
training a sequence model - a sepCNN model. We use sequence model for text
classification when the ratio of number of samples to number of words per
sample for the given dataset is very large (>~15K). This module is identical to
the `train_sequence_model` module except that we pass the data in batches for
training. This is required when you have a very large dataset that does not fit
into memory.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time

import tensorflow as tf
import numpy as np

import build_model
import load_data
import vectorize_data
import explore_data


FLAGS = None

# Limit on the number of features. We use the top 20K features.
TOP_K = 20000


def _data_generator(x, y, num_features, batch_size):
    """Generates batches of vectorized texts for training/validation.

    # Arguments
        x: np.matrix, feature matrix.
        y: np.ndarray, labels.
        num_features: int, number of features.
        batch_size: int, number of samples per batch.

    # Returns
        Yields feature and label data in batches.
    """
    num_samples = x.shape[0]
    num_batches = num_samples // batch_size
    if num_samples % batch_size:
        num_batches += 1

    while 1:
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            if end_idx > num_samples:
                end_idx = num_samples
            x_batch = x[start_idx:end_idx]
            y_batch = y[start_idx:end_idx]
            yield x_batch, y_batch


def batch_train_sequence_model(data,
                               learning_rate=1e-3,
                               epochs=1000,
                               batch_size=128,
                               blocks=2,
                               filters=64,
                               dropout_rate=0.2,
                               embedding_dim=200,
                               kernel_size=3,
                               pool_size=3):
    """Trains sequence model on the given dataset.

    # Arguments
        data: tuples of training and test texts and labels.
        learning_rate: float, learning rate for training model.
        epochs: int, number of epochs.
        batch_size: int, number of samples per batch.
        blocks: int, number of pairs of sepCNN and pooling blocks in the model.
        filters: int, output dimension of sepCNN layers in the model.
        dropout_rate: float: percentage of input to drop at Dropout layers.
        embedding_dim: int, dimension of the embedding vectors.
        kernel_size: int, length of the convolution window.
        pool_size: int, factor by which to downscale input at MaxPooling layer.

    # Raises
        ValueError: If validation data has label values which were not seen
            in the training data.
    """
    # Get the data.
    (train_texts, train_labels), (val_texts, val_labels) = data

    # Verify that validation labels are in the same range as training labels.
    num_classes = explore_data.get_num_classes(train_labels)
    unexpected_labels = [v for v in val_labels if v not in range(num_classes)]
    if len(unexpected_labels):
        raise ValueError('Unexpected label values found in the validation set:'
                         ' {unexpected_labels}. Please make sure that the '
                         'labels in the validation set are in the same range '
                         'as training labels.'.format(
                             unexpected_labels=unexpected_labels))

    # Vectorize texts.
    x_train, x_val, word_index = vectorize_data.sequence_vectorize(
            train_texts, val_texts)

    # Number of features will be the embedding input dimension. Add 1 for the
    # reserved index 0.
    num_features = min(len(word_index) + 1, TOP_K)

    # Create model instance.
    model = build_model.sepcnn_model(blocks=blocks,
                                     filters=filters,
                                     kernel_size=kernel_size,
                                     embedding_dim=embedding_dim,
                                     dropout_rate=dropout_rate,
                                     pool_size=pool_size,
                                     input_shape=x_train.shape[1:],
                                     num_classes=num_classes,
                                     num_features=num_features)

    # Compile model with learning parameters.
    if num_classes == 2:
        loss = 'binary_crossentropy'
    else:
        loss = 'sparse_categorical_crossentropy'
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

    # Create callback for early stopping on validation loss. If the loss does
    # not decrease in two consecutive tries, stop training.
    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=2)]

    # Create training and validation generators.
    training_generator = _data_generator(
        x_train, train_labels, num_features, batch_size)
    validation_generator = _data_generator(
        x_val, val_labels, num_features, batch_size)

    # Get number of training steps. This indicated the number of steps it takes
    # to cover all samples in one epoch.
    steps_per_epoch = x_train.shape[0] // batch_size
    if x_train.shape[0] % batch_size:
        steps_per_epoch += 1

    # Get number of validation steps.
    validation_steps = x_val.shape[0] // batch_size
    if x_val.shape[0] % batch_size:
        validation_steps += 1

    # Train and validate model.
    history = model.fit_generator(
            generator=training_generator,
            steps_per_epoch=steps_per_epoch,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            epochs=epochs,
            verbose=2)  # Logs once per epoch.

    # Print results.
    history = history.history
    print('Validation accuracy: {acc}, loss: {loss}'.format(
            acc=history['val_acc'][-1], loss=history['val_loss'][-1]))

    # Save model.
    model.save('amazon_reviews_sepcnn_model.h5')
    return history['val_acc'][-1], history['val_loss'][-1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='input data directory')
    FLAGS, unparsed = parser.parse_known_args()

    # Using the Amazon reviews dataset to demonstrate training of
    # sequence model with batches of data.
    data = load_data.load_amazon_reviews_sentiment_analysis_dataset(
            FLAGS.data_dir)
    batch_train_sequence_model(data)
