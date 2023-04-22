import pandas as pd
from sklearn import model_selection

if __name__=="__main__":
    data_dir = '../data/'
    processed_dir = '../processed/'
    df = pd.read_csv(f"{data_dir}train.csv")

    df['kfold'] = -1

    df=df.sample(frac=1).reset_index(drop=True)

    y = df.target.values

    kf = model_selection.StratifiedKFold(n_splits=5)

    for f, (t,v) in enumerate(kf.split(X=df,y=y)):
        df.loc[v,'kfold'] = f

    df.to_csv(f"{processed_dir}train_folds_stratified.csv")
    print(df.head())
    print(df.kfold.value_counts())