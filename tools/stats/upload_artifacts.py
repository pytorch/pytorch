import argparse
import os
from tempfile import TemporaryDirectory

from tools.stats.upload_stats_lib import download_gha_artifacts

ARTIFACTS = [
    "test-jsons",
    "test-reports",
    "usage-log",
]


def get_artifacts(workflow_run_id: int, workflow_run_attempt: int) -> None:
    with TemporaryDirectory() as temp_dir:
        print("Using temporary directory:", temp_dir)
        # os.chdir(temp_dir)
        os.chdir("/tmp/artifacts")

        for artifact in ARTIFACTS:
            artifact_paths = download_gha_artifacts(
                artifact, workflow_run_id, workflow_run_attempt
            )

            for path in artifact_paths:
                print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload test artifacts from GHA to S3")
    parser.add_argument(
        "--workflow-run-id",
        type=int,
        required=True,
        help="id of the workflow to get artifacts from",
    )
    parser.add_argument(
        "--workflow-run-attempt",
        type=int,
        required=True,
        help="which retry of the workflow this is",
    )
    args = parser.parse_args()
    get_artifacts(args.workflow_run_id, args.workflow_run_attempt)
