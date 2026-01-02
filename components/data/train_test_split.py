from kfp.v2.dsl import component, Input, Output, Dataset

@component(
    base_image="python:3.12.10-slim",
    packages_to_install=["pandas", "numpy", "scikit-learn"],
)
def train_test_data_split(
    dataset: Input[Dataset],
    train_data: Output[Dataset],
    test_data: Output[Dataset],
    test_size: float = 0.2,
    random_state: int = 42,
):
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    """
    Split dataset into train and test
    """
    df = pd.read_csv(dataset.path)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    train_df.to_csv(train_data.path, index=False)
    test_df.to_csv(test_data.path, index=False)