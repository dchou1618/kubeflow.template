from kfp.v2 import compiler 
from logistic_pipeline import logistic_regression_pipeline

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=logistic_regression_pipeline,
        package_path="pipelines/logistic_regression_pipeline.json",
    )