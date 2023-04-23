import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import model_selection

def create_folds(data):
    """Function to implement stratified kfold cross-validation on regression
    Args:
    ----------------------
    data(pandas.DataFrame):Dataframe on which cross-validation sampling needs to be done.
    
    Return:
    ------------------------
    pandas.DataFrame: dataframe with kfold values."""

    data['kfold'] = -1

    # number of bin's bu Surge's rule
    num_bins = int(np.floor(1+np.log2(len(data))))
    print(f"{num_bins}")

    #bin targets
    data.loc[:, "bins"] = pd.cut(data["target"],bins=num_bins, labels=False)
    print(data["bins"])

    #initiating the kfold class from model_selection
    kf = model_selection.StratifiedKFold(n_splits=5)

    #fill the new kfold column
    print(f"kf split:{kf.split(X=data, y=data.bins.values)}")
    for f, (t,v) in enumerate(kf.split(X=data, y=data.bins.values)):
        data.loc[v,'kfold'] = f
    
    #dropping the bins column:
    data.drop(columns='bins',inplace=True)

    return data

if __name__=="__main__":
    # creating sample dataset wityh 15k samples

    X,y = datasets.make_regression(
        n_samples=15000,
        n_features= 100,
        n_targets = 1
    )

    # creating a dataframe out of the numpy arrays we just created
    df = pd.DataFrame(
        X,
        columns=[f"f_{i}" for i in range(X.shape[1])]
    )

    df.loc[:,"target"] = y
    print(df.head())

    # creating folds
    df = create_folds(df)

    print(df.kfold.value_counts())