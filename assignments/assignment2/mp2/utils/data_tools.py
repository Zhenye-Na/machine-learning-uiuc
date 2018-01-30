"""Implement feature extraction and data processing helpers."""


import numpy as np


def preprocess_data(dataset,
                    feature_columns=[
                        'Id', 'BldgType', 'OverallQual',
                        'GrLivArea', 'GarageArea'
                    ],
                    squared_features=False
                    ):
    """Process the dataset into vector representation.

    When converting the BldgType to a vector, use one-hot encoding, the order
    has been provided in the one_hot_bldg_type helper function. Otherwise,
    the values in the column can be directly used.

    If squared_features is true, then the feature values should be
    element-wise squared.

    Args:
        dataset(dict): Dataset extracted from io_tools.read_dataset
        feature_columns(list): List of feature names.
        squred_features(bool): Whether to square the features.

    Returns:
        processed_datas(list): List of numpy arrays x, y.
            x is a numpy array, of dimension (N,K), N is the number of example
            in the dataset, and K is the length of the feature vector.
            Note: BldgType when converted to one hot vector is of length 5.
            Each row of x contains an example.
            y is a numpy array, of dimension (N,1) containing the SalePrice.
    """
    columns_to_id = {'Id': 0, 'BldgType': 1, 'OverallQual': 6,
                     'GrLivArea': 7, 'GarageArea': 8, 'SalePrice': 9}

    cat_col = []
    for key in feature_columns:
        cat_col.append(columns_to_id[key])
    if 'BldgType' in feature_columns:
        cat_col.extend([2, 3, 4, 5])

    x = []
    y = []

    for keys in dataset:
            temp = list(dataset[keys])
            vector = one_hot_bldg_type(temp[1])

            temp = [temp[0]] + vector + temp[2:]
            temp = list(map(int, temp))

            x.append(temp[0:-1])
            y.append(temp[-1])

    # If squared_features is true, then the feature values should be
    # element-wise squared.
    if squared_features:
        x = np.square(x)

    # Select specific number of features
    x = np.array(x)[:, sorted(cat_col)]
    y = np.reshape(y, (x.shape[0], 1))

    processed_dataset = [x, y]
    return processed_dataset


def one_hot_bldg_type(bldg_type):
    """Build the one-hot encoding vector.

    Args:
        bldg_type(str): String indicating the building type.

    Returns:
        ret(list): A list representing the one-hot encoding vector.
            (e.g. for 1Fam building type, the returned list should be
            [1,0,0,0,0].
    """
    type_to_id = {'1Fam': 0,
                  '2FmCon': 1,
                  'Duplx': 2,
                  'TwnhsE': 3,
                  'TwnhsI': 4,
                  }

    # bldg_type in train.csv first row is '1Fam'
    idx = type_to_id[bldg_type]

    vector = [0, 0, 0, 0, 0]

    # change the vector to one-hot encoding
    vector[idx] = 1

    # vector assignment
    ret = vector[:]
    return ret
