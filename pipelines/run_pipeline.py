import kfp
import os

client = kfp.Client(host=os.environ['KFP_HOST'])

# upload or use existing pipeline
experiment = client.create_experiment('experiment')

run = client.run_pipeline(
    experiment_id=experiment.id,
    job_name='pipeline-run',
    pipeline_package_path='pipelines/logistic_regression_pipeline.json',
    params={
        'n_samples': 1000,
        'test_size': 0.2,
        'weights': [0.5, -0.2, 0.3]
    },
)
print(f"Run started: {run.id}")