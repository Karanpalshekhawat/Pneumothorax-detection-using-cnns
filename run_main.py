"""
This script is the main file that calls all the other
scripts to run the ml project.
"""

import os
import joblib
import argparse
import pandas as pd
import src.config as sc

from sklearn import metrics
from src.create_folds import create_folds_using_kfold
from src.model_dispatcher import models


def run_output(fold, df, model):
    """
    Structure, train and save the model
    for given fold number.

    Args:
        fold (int): number for fold
        df (pd.DataFrame): training dataset
        model (str): Model to use (either gini or entropy)

    Returns:

    """
    df_train = df[df['kfold'] != fold].reset_index(drop=True)
    df_valid = df[df['kfold'] == fold].reset_index(drop=True)

    """Convert training dataframe to numpy values to use training modules"""
    x_train = df_train.drop('label', axis=1).values
    y_train = df_train['label'].values

    """Convert validation dataframe to numpy values for evaluation"""
    x_valid = df_valid.drop('label', axis=1).values
    y_valid = df_valid['label'].values

    """import the model required"""
    clf = models[model]

    """fit model on the training data"""
    clf.fit(x_train, y_train)

    """predict on validation dataset"""
    y_pred = clf.predict(x_valid)

    """find accuracy as distribution of all target variables in similar"""
    accuracy = metrics.accuracy_score(y_valid, y_pred)
    print(f"Fold number :{fold}, Accuracy score : {accuracy}")

    """Save Model"""
    joblib.dump(clf, os.path.join(sc.OUTPUT_FILE, f'dt_{fold}.bin'))


if __name__ == '__main__':
    df = create_folds_using_kfold()
    """Create a parser object and add variables that you want to declare"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int)
    parser.add_argument('--model', type=str)
    args = parser.parse_args()
    run_output(args.fold, df, args.model)
