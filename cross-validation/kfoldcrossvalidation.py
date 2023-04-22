import pandas as pd
from sklearn import model_selection

if __name__=="__main__":
    data_dir = '../data/'
    processed_dir = '../processed/'
    df = pd.read_csv(f"{data_dir}train.csv")

    # creating new column for kfold
    df['kfold'] = -1

    #Randomizing the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)

    #Initiating kfold object
    kf = model_selection.KFold(n_splits=5)

    # fill new kfold values
    for fold,(trn_,val_) in enumerate(kf.split(X=df)):
        df.loc[val_, 'kfold'] = fold

    # saving the file
    df.to_csv(f"{processed_dir}train_folds.csv",index=False)
    print(df.kfold.value_counts(dropna=False))