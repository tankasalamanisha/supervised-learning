import joblib
import pandas as pd
from sklearn import metrics
from sklearn import tree
import config
import os

import argparse
import model_dispatcher

def run(fold:int, model:object):
    """Function that extracts, runs and evaluates the Decision Tree model"""
    df = pd.read_csv(config.training_file)

    df_train = df[df.kfold != fold].reset_index(drop=True)

    # print(df[df.kfold == fold].head())
    df_validation = df[df.kfold == fold].reset_index(drop=True)

    x_train = df_train.drop(columns='label').values
    y_train = df_train['label'].values

    x_validation = df_validation.drop(columns='label').values
    y_validation = df_validation['label'].values

    clf = model_dispatcher.models[model]

    clf.fit(X=x_train,y=y_train)

    pred = clf.predict(X=x_validation)

    accuracy = metrics.accuracy_score(y_true=y_validation,y_pred=pred)

    print(f"Fold {fold}:  Accuracy {accuracy}")

    # joblib.dump(clf,os.path.join(config.model_output,f"dt_{fold}.bin"))

if __name__ == "__main__":
    # Initializing argument parser
    parser = argparse.ArgumentParser()

    # adding the different arguments we need and their type.
    #currently, we only need fold

    parser.add_argument("--fold",type=int,required=True)
    parser.add_argument("--model",type=str,required=True)

    # reading the arguments from the command line
    args = parser.parse_args()

    # running the fold specified in the argument parser
    run(
        fold=args.fold,
        model=args.model
        )