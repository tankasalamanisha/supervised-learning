import os
import tarfile
from six.moves import urllib
import pandas as pd
import numpy as np

def fetch_housing_data(housing_url:str, housing_path:str):
    """Function to automatically fetch the housing data from a url and extracts the data to the given path.
    Args:
    -----------------
    housing_url: URL from where the data needs to be loaded.
    housing_path: path where the extracted files need to be stored.
    """
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    
    tgz_path = os.path.join(housing_path,"housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

def load_housing_data(housing_path:str)->pd.DataFrame:
    """Function to load the csv data
    Args:
    --------------------
    housing_path: path from where the csv file needs to be fetched.
    Returns:
    --------------------
    pd.DataFrame: dataframe of the loaded data."""
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

def split_train_test(data:pd.DataFrame, test_ratio:float):
    """Function to split the dataset in the given ratio
    Args:
    ---------------
    data: Dataframe that needs to be split into the given ratio.
    test_ratio:Floating point to determine the test and train ratio.
    
    Return:
    ------------
    pd.DataFrame, pd.Dataframe: train dataframe, test dataframe."""

    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data)*test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]

    return data.iloc[train_indices], data.iloc[test_indices]
