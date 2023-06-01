import joblib
import pandas as pd
from sklearn import metrics
from sklearn import tree

def run(fold:int):
    """Function that extracts, runs and evaluates the Decision Tree model"""
    df = pd.read_csv("../input/mnist_train_folds.csv")

    df_train = df[df.kfold != fold].reset_index(drop=True)

    # print(df[df.kfold == fold].head())
    df_validation = df[df.kfold == fold].reset_index(drop=True)

    x_train = df_train.drop(columns='label').values
    y_train = df_train['label'].values

    x_validation = df_validation.drop(columns='label').values
    y_validation = df_validation['label'].values

    clf = tree.DecisionTreeClassifier()

    clf.fit(X=x_train,y=y_train)

    pred = clf.predict(X=x_validation)

    accuracy = metrics.accuracy_score(y_true=y_validation,y_pred=pred)

    print(f"Fold {fold}:  Accuracy {accuracy}")

    # joblib.dump(clf,f'../models/dt_{fold}.bin')

if __name__ == "__main__":
    # run(fold=0)
    # run(fold=1)
    run(fold=2)
    # run(fold=3)
    # run(fold=4)