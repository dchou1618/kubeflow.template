from kfp.v2.dsl import Input, Output, component, Dataset, Model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import joblib

@component(
    base_image="python:3.12.10-slim",
    packages_to_install=["scikit-learn", "pandas", "joblib"],
)
def train_model(
    train_dataset: Input[Dataset],
    model: Output[Model],
):
    """
    Trains 
    """
    df = pd.read_csv(train_dataset.path)
    X = df.drop(columns=["y"])
    y = df["y"]
    clf = LogisticRegression()
    clf.fit(X, y)
    joblib.dump(clf, model.path)