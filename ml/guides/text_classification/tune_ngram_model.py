"""Module to demonstrate hyper-parameter tuning.

Trains n-gram model with different combination of hyper-parameters and finds
the one that works best.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

import load_data
import train_ngram_model

FLAGS = None


def tune_ngram_model(data):
    """Tunes n-gram model on the given dataset.

    # Arguments
        data: tuples of training and test texts and labels.
    """
    # Select parameter values to try.
    num_layers = [1, 2, 3]
    num_units = [8, 16, 32, 64, 128]

    # Save parameter combination and results.
    params = {
        'layers': [],
        'units': [],
        'accuracy': [],
    }

    # Iterate over all parameter combinations.
    for layers in num_layers:
        for units in num_units:
                params['layers'].append(layers)
                params['units'].append(units)

                accuracy, _ = train_ngram_model.train_ngram_model(
                    data=data,
                    layers=layers,
                    units=units)
                print(('Accuracy: {accuracy}, Parameters: (layers={layers}, '
                       'units={units})').format(accuracy=accuracy,
                                                layers=layers,
                                                units=units))
                params['accuracy'].append(accuracy)
    _plot_parameters(params)


def _plot_parameters(params):
    """Creates a 3D surface plot of given parameters.

    # Arguments
        params: dict, contains layers, units and accuracy value combinations.
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(params['layers'],
                    params['units'],
                    params['accuracy'],
                    cmap=cm.coolwarm,
                    antialiased=False)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='input data directory')
    FLAGS, unparsed = parser.parse_known_args()

    # Using the IMDb movie reviews dataset to demonstrate training n-gram model
    data = load_data.load_imdb_sentiment_analysis_dataset(FLAGS.data_dir)
    tune_ngram_model(data)
