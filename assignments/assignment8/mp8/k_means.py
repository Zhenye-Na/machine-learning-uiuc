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
data_dir = '/Users/macbookpro/Desktop/cs446/assignments/assignment8/mp8/data/data/iris.data'
# /Users/macbookpro/Desktop/cs446/assignments/assignment8/mp8/data/data

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
        dist(double): distance between data point and corresponding center.
    """
    dimensions = len(data_point)
    dist = 0
    for dim in range(dimensions):
        dist += (data_point[dim] - center[dim]) ** 2
    return (dist ** 0.5)


def assign_cluster(data, centers):
    """Assign cluster index to each of data point.

    Args:
        data(np.ndarray):
        centers(np.ndarray):
    Returns:
        index(list):
    """
    index = []
    num_center = centers.shape[0]

    for point in data:
        dist = []
        for i in range(num_center):
            dist.append(distance(point, centers[i]))
        index.append(dist.index(min(dist)))
    return index


def recompute_center(points):
    """Re-compute center for data points in same cluster.

    Args:
        points(list):
    Returns:
        center(list)
    """
    points = np.array(points)
    points = np.reshape(points, (points.shape[0], X.shape[1]))
    num = points.shape[0]
    dimensions = points.shape[1]
    center = []

    for dim in range(dimensions):
        center.append(sum(points[:, dim]) / num)
    return center


def update_center(data, old_index):
    """Update centers for data points.

    Args:
        data(np.ndarray): data points.
        old_index(list): The previous/current index.
    Returns:
        new_centers(np.ndarray): new centers after updating.
    """
    new_centers = []
    points = []
    for idx_centroid in range(k):
        for row in range(data.shape[0]):
            if old_index[row] == idx_centroid:
                points.append(data[row, :])

        new_centers.append(recompute_center(points))
        points = []

    new_centers = np.array(new_centers)
    return new_centers


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
    i = 0
    while (True):
        new_centers = update_center(X, indexofcluster)

        new_indexofcluster = assign_cluster(X, new_centers)

        dist = 0
        for i in range(X.shape[0]):
            dist += distance(X[i], new_centers[new_indexofcluster[i]])
        if dist - old_dist < 10e-3:
            break
        else:
            old_dist = dist
    return new_centers


new_centers = k_means(C)
print(new_centers)
