"""Implements feature extraction and data processing helpers.
"""


import numpy as np
import pylab as pl


def load_dataset(input_file_path, num_samples):
    """
    Generates a dataset by loading an image and creating the specified number
    of noisy samples of it.
    Inputs:
        input_file_path
    Output:
        dataset
    """
    original_img = load_image(input_file_path)
    samples = []
    for i in range(num_samples):
        samples.append(inject_noise(original_img))

    return original_img, samples


def load_image(input_file_path):
    """
    Loads the image and binarizes it by:
    0. Read the image
    1. Consider the first channel in the image
    2. Binarize the pixel values to {-1, 1} by setting the values
    below the binarization_threshold to 0 and above to 1.
    Inputs:
        input_file_path
    Output:
        binarized image
    """
    img = pl.imread(input_file_path)
    img = img[:, :, 0]
    img = np.where(I < 0.1, 0, 1)

    return img


def inject_noise(image):
    """
    Inject noise by flipping the value of some randomly chosen pixels.
    1. Generate a matrix of probabilities of pixels keeping their
    original values.
    2. Flip the pixels if their corresponding probability in the matrix
    is below 0.1.

    Input:
        original image
    Output:
        noisy image
    """
    J = image.copy()

    # Generate the matrix of probabilities of each pixel in the image
    # to keep its value
    N = np.shape(J)[0]
    noise = np.random.rand(N, N)

    # Extract the indices of the pixels to be flipped.
    ind = np.where(noise < 0.1)

    # Flip the pixels
    J[ind] = 1 - J[ind]

    return J


def plot_image(image, title, path):
    pl.figure()
    pl.imshow(image)
    pl.title(title)
    pl.savefig(path)
