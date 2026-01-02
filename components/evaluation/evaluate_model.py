from kfp.v2.dsl import component, Input, Output, Dataset, Model, Metrics

@component(
    base_image="python:3.12.10-slim",
    packages_to_install=["pandas", "scikit-learn", "joblib"],
)
def evaluate_model(
    test_dataset: Input[Dataset],
    model: Input[Model],
    metrics: Output[Metrics],
):
    import pandas as pd
    import joblib
    from sklearn.metrics import roc_auc_score
    
    df = pd.read_csv(test_dataset.path)
    X = df.drop(columns=["y"])
    y_true = df["y"]
    clf = joblib.load(model.path)
    y_pred = clf.predict(X)
    auc = roc_auc_score(y_true, y_pred)
    metrics.log_metric("roc_auc", auc)