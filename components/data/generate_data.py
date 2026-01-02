from kfp.v2.dsl import component, Dataset, Output
from typing import List

@component(
    base_image="python:3.12.10-slim",
    packages_to_install=["numpy", "pandas"],
)
def generate_data(
    output_data: Output[Dataset],
    weights: List[float],
    n_samples: int = 1000,
    n_features: int = 3,
    seed: int = 42,
):
    """
    Generates a synthetic dataset for logistic regression with set of given weights
    """
    import numpy as np
    import pandas as pd
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features)
    logits = X @ weights
    probs = 1 / (1 + np.exp(-logits))
    y = (probs >= 0.5).astype(int)

    df = pd.DataFrame(X, columns=[f"feature_{i+1}" for i in range(n_features)])
    df["y"] = y
    df.to_csv(output_data.path, index=False)