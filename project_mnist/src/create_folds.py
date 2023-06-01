import pandas as pd
from sklearn import model_selection

if __name__=="__main__":
    data_dir = '../input/'

    df = pd.read_csv(f"{data_dir}mnist_train.csv")
    

    df['kfold'] = -1

    
    y = df.label.values
    
    kf = model_selection.StratifiedKFold(n_splits=5)

    for f, (t,v) in enumerate(kf.split(X=df,y=y)):
        df.loc[v,'kfold'] = f

    df.to_csv(f"{data_dir}mnist_train_folds.csv")
    print(df.head())
    print(df.kfold.value_counts())