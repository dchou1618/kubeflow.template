from kfp.v2.dsl import pipeline 
from components.data.generate_data import generate_data
from components.data.train_test_split import train_test_data_split
from components.training.train_model import train_model
from components.evaluation.evaluate_model import evaluate_model

@pipeline(
    name="logistic-regression-pipeline",
    description="A simple logistic regression pipeline template",
)
def logistic_regression_pipeline(
    n_samples: int = 1000,
    test_size: float = 0.2,
):
    data_task = generate_data(n_samples=n_samples)
    split_task = train_test_data_split(
        dataset=data_task.outputs["output_data"],
        test_size=test_size,
    )
    train_task = train_model(
        train_dataset=split_task.outputs["train_data"],
    )
    evaluate_model(
        test_dataset=split_task.outputs["test_data"],
        model=train_task.outputs["model"],
    )