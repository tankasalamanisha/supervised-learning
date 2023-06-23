import os
import tarfile
from six.moves import urllib
import pandas as pd

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

def load_housing_data(housing_path:str):
    """Function to load the csv data
    Args:
    --------------------
    housing_path: path from where the csv file needs to be fetched.
    Returns:
    --------------------
    pd.DataFrame: dataframe of the loaded data."""
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
