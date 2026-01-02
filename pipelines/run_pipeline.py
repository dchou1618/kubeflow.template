import os

PIPELINE_PACKAGE_PATH = "pipelines/logistic_regression_pipeline.json"

PIPELINE_PARAMS = {
    "n_samples": 1000,
    "test_size": 0.2,
    "weights": [0.5, -0.2, 0.3],
}

EXPERIMENT_NAME = "experiment"
JOB_NAME = "pipeline-run"


def is_local():
    return "KFP_HOST" not in os.environ


if is_local():
    # -------- LOCAL MODE --------
    from kfp import local
    from kfp.local import SubprocessRunner

    print("Running pipeline locally")

    local.init(runner=SubprocessRunner())

    # IMPORT THE PIPELINE FUNCTION
    from pipelines.logistic_pipeline import (
        logistic_regression_pipeline,
    )

    # RUN IT DIRECTLY
    result = logistic_regression_pipeline(**PIPELINE_PARAMS)
    print("Local run completed")

else:
    # -------- REMOTE MODE --------
    import kfp

    print(f"Running pipeline remotely on {os.environ['KFP_HOST']}")

    client = kfp.Client(host=os.environ["KFP_HOST"])

    experiment = client.create_experiment(EXPERIMENT_NAME)

    run = client.run_pipeline(
        experiment_id=experiment.id,
        job_name=JOB_NAME,
        pipeline_package_path=PIPELINE_PACKAGE_PATH,
        params=PIPELINE_PARAMS,
    )

    print(f"Remote run started: {run.id}")