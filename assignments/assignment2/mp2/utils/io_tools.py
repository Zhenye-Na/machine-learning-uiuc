"""Input and output helpers to load in data."""

import csv


def read_dataset(input_csv_file):
    """Read data into a python list.

    Args:
        input_csv_file: Path to the data csv file.

    Returns:
        dataset(dict): A python dictionary with the key value pair of
            (example_id, example_feature).

            example_feature is represented with a tuple
            (Id, BldgType, OverallQual, GrLivArea, GarageArea)

            For example, the first row will be in the train.csv is
            example_id = 1
            example_feature = (1,1Fam,7,1710,548)
    """
    dataset = {}

    # Imeplemntation here.
    with open(input_csv_file) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            dataset[row['Id']] = (row['Id'],
                                  row['BldgType'],
                                  row['OverallQual'],
                                  row['GrLivArea'],
                                  row['GarageArea'],
                                  row['SalePrice'])

    return dataset
