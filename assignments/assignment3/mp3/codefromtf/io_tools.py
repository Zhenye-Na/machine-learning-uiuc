"""Input and output helpers to load in data."""
import numpy as np


def read_dataset_tf(path_to_dataset_folder, index_filename):
    """ Read dataset into numpy arrays with preprocessing included
    Args:
        path_to_dataset_folder(str): path to the folder containing samples and indexing.txt
        index_filename(str): indexing.txt
    Returns:
        A(numpy.ndarray): sample feature matrix A = [[1, x1],
                                                     [1, x2],
                                                     [1, x3],
                                                     .......]
                                where xi is the 16-dimensional feature of each sample

        T(numpy.ndarray): class label vector T = [[y1],
                                                  [y2],
                                                  [y3],
                                                   ...]
                             where yi is 1/0, the label of each sample
    """
    ###############################################################
    # Fill your code in this function
    ###############################################################
    # Hint: open(path_to_dataset_folder+'/'+index_filename,'r')
    T = []
    A = []
    with open(path_to_dataset_folder + '/' + index_filename, 'r') as data_file:
        for line in data_file:
            T.append(line[:2])
            T = [int(item) for item in T]
            idx = line.find(' ')
            fdir = line[idx + 1:]
            with open(path_to_dataset_folder + '/' + fdir[:-1], 'r') as sample_file:
                for line_sample in sample_file:
                    temp = line_sample.split(' ')
                    A_prime = []
                    for _ in temp:
                        if _:
                            A_prime.append(_)
                    A_prime = [float(item) for item in A_prime]
                    A.append(A_prime)
    A = np.array(A)
    A = np.concatenate((np.ones((A.shape[0], 1)), A), axis=1)
    T = np.array(T)
    T[T == -1] = 0
    T = np.reshape(T, (T.shape[0], 1))

    return (A, T)
