from copy import deepcopy
import numpy as np
import pandas as pd
import sys
from collections import defaultdict


'''
In this problem you write your own K-Means
Clustering code.

Your code should return a 2d array containing
the centers.

'''
# Comfigure the path to data
# print(sys.path)
data_dir = 'data/data/iris.data'
# sys.path.insert(0, data_dir)
# print(sys.path)

# Import the dataset
df = pd.read_table(data_dir, delimiter=',')
# print(df)
X = df[['V1', 'V2', 'V3', 'V4']].as_matrix()
# print(X)
# Make 3  clusters
k = 3
# Initial Centroids
C = [[2., 0., 3., 4.], [1., 2., 1., 3.], [0., 2., 1., 0.]]
C = np.array(C)
# print(type(C[0]))
print("Initial Centers: ")
print(C)


def distance(data, centers):
    """
    Compute distance of data points and centers

    Args:
        series of data points.
        data: data points
        centers: center cordinates
    Return:
        dataset
    """
    dimensions = len(data)

    dist = 0
    for dimension in range(dimensions):
        difference_sq = (data[dimension] - centers[dimension]) ** 2
        dist += difference_sq
    return dist ** 0.5


def assign_points(data_points, centers):
    """
    Given a data set and a list of points betweeen other points,
    assign each point to an index that corresponds to the index
    of the center point on it's proximity to that point. 
    Return a an array of indexes of centers that correspond to
    an index in the data set; that is, if there are N points
    in `data_set` the list we return will have N elements. Also
    If there are Y points in `centers` there will be Y unique
    possible values within the returned list.
    """
    assignments = []
    for point in data_points:
        shortest = ()  # positive infinity
        shortest_index = 0
        for i in range(len(centers)):
            val = distance(point, centers[i])
            if val < shortest:
                shortest = val
                shortest_index = i
        assignments.append(shortest_index)
    return assignments

def update_centers(data_set, assignments):
    """
    Accepts a dataset and a list of assignments; the indexes 
    of both lists correspond to each other.
    Compute the center for each of the assigned groups.
    Return `k` centers where `k` is the number of unique assignments.
    """
    new_means = defaultdict(list)
    centers = []
    for assignment, point in zip(assignments, data_set):
        new_means[assignment].append(point)
        
    for points in new_means.itervalues():
        centers.append(point_avg(points))

    return centers


def assign_points(data_points, centers):
    """
    Given a data set and a list of points betweeen other points,
    assign each point to an index that corresponds to the index
    of the center point on it's proximity to that point. 
    Return a an array of indexes of centers that correspond to
    an index in the data set; that is, if there are N points
    in `data_set` the list we return will have N elements. Also
    If there are Y points in `centers` there will be Y unique
    possible values within the returned list.
    """
    assignments = []
    for point in data_points:
        shortest = ()  # positive infinity
        shortest_index = 0
        for i in range(len(centers)):
            val = distance(point, centers[i])
            if val < shortest:
                shortest = val
                shortest_index = i
        assignments.append(shortest_index)
    return assignments

def k_means(X, C):
    # Write your code here!
    assignments = assign_points(X, C)
    old_assignments = np.zeros_like(C[0])
    print(old_assignments)
    print(assignments)
    # for point in assignments:


    while assignments != old_assignments:
        new_centers = update_centers(X, assignments)
        old_assignments = assignments
        assignments = assign_points(X, new_centers)
    return assignments

center = k_means(X, C)
print(center)