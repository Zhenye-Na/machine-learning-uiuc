from copy import deepcopy
import numpy as np
import pandas as pd
import sys

'''
In this problem you write your own K-Means
Clustering code.

Your code should return a 2d array containing
the centers.

'''
# Configure the path to data
data_dir = 'data/data/iris.data'

# Import the dataset
df = pd.read_table(data_dir, delimiter=',')
X = df[['V1', 'V2', 'V3', 'V4']].as_matrix()

# Make 3 clusters
k = 3

# Initial Centroids
C = [[2., 0., 3., 4.],
     [1., 2., 1., 3.],
     [0., 2., 1., 0.]]
C = np.array(C)

print("Initial Centers: ")
print(C)


def distance(data_point, center):
    """Compute distance between data point and corresponding center.

    Args:
        data_point(list): single data point.
        center(list): corresponding center.
    Returns:
        distance between data point and corresponding center.
    """
    return np.linalg.norm(data_point - center)


def assign_cluster(data, centers):
    """Assign cluster index to each of data point.

    Args:
        data(np.ndarray): data points
        centers(np.ndarray): centers
    Returns:
        index(list): index of which cluster each of data points belongs to
    """
    index = []
    num_center = centers.shape[0]

    for point in data:
        dist = []
        for i in range(num_center):
            dist.append(distance(point, centers[i]))
        index.append(np.argmin(dist))
    return index


def update_center(X, indx):
    """Update centers for data points.

    Args:
        data(np.ndarray): data points.
        old_index(list): The previous/current index.
    Returns:
        new_centers(np.ndarray): new centers after updating.
    """
    newCenters = []
    for clusterNumber in range(0, k):
        Cluster = X[[i for i, find in enumerate(indx) if find == clusterNumber]]
        centroid = Cluster.mean(axis=0)
        newCenters.append(centroid)
    return np.array(newCenters)


def k_means(C):
    """Perform K-Means Algorithm on dataset.

    Args:
        C(np.ndarray): initial centers.
    """
    # Write your code here!
    # Get the initial cluster of data points
    C = np.array(C)
    indexofcluster = assign_cluster(X, C)
    old_dist = 0
    while (True):
        new_centers = update_center(X, indexofcluster)
        new_indexofcluster = assign_cluster(X, new_centers)

        dist = 0
        for i in range(X.shape[0]):
            dist += distance(X[i], new_centers[new_indexofcluster[i]])
        if abs(dist - old_dist) < 10e-3:
            break
        else:
            old_dist = dist
            indexofcluster = new_indexofcluster
    return new_centers

new_centers = k_means(C)
print("New centers: ")
print(new_centers)
