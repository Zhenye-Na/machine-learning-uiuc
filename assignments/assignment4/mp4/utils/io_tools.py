"""
Input and output helpers to load in data.

(This file will not be graded.)
"""

import numpy as np
import skimage
import os
from skimage import io


def read_dataset(data_txt_file, image_data_path):
    """Read data into a Python dictionary.

    Args:
        data_txt_file(str): path to the data txt file.
        image_data_path(str): path to the image directory.

    Returns:
        data(dict): A Python dictionary with keys 'image' and 'label'.
            The value of dict['image'] is a numpy array of dimension (N,8,8,3)
            containing the loaded images.

            The value of dict['label'] is a numpy array of dimension (N,1)
            containing the loaded label.

            N is the number of examples in the data split, the examples should
            be stored in the same order as in the txt file.
    """
    data = {}
    data['image'] = []
    data['label'] = []

    # Read txt file
    with open(data_txt_file, 'r') as f:
        for line in f:
            # Split the string to imgdirs & labels
            dirs = line.rstrip().split(',')

            # output of dirs -> ['0000', '-1']
            data['label'].append([int(dirs[1])])

            # Append '.jpg' to filename
            imname = dirs[0] + '.jpg'
            imgdir = os.path.join(image_data_path, imname)
            data['image'].append(io.imread(imgdir))

    # Transform to Numpy array with specific dimension
    data['label'] = np.array(data['label'])
    data['image'] = np.array(data['image'])

    return data
