"""Input and output helpers to load in data."""

import pickle
import numpy as np
from numpy import genfromtxt


def read_dataset(input_file_path):
    """Read input file in csv format from file.

    In this csv, each row is an example, stored in the following format.
    label, pixel1, pixel2, pixel3...

    Args:
        input_file_path(str): Path to the csv file.
    Returns:
        (1) label (np.ndarray): Array of dimension (N,) containing the label.
        (2) feature (np.ndarray): Array of dimension (N, ndims) containing the
        images.
    """
    # Imeplemntation here.
    data = genfromtxt(input_file_path, delimiter=',')
    features = data[:, 1:]
    labels = data[:, 0]
    return labels, features
