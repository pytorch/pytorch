import argparse
import os
import re
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.stats.upload_stats_lib import (
    download_gha_artifacts,
    unzip,
    upload_file_to_s3,
)


ARTIFACTS = [
    "sccache-stats",
    "test-jsons",
    "test-reports",
    "torchci-reports",
    "usage-log",
]
BUCKET_NAME = "gha-artifacts"
FILENAME_REGEX = r"-runattempt\d+"


def upload_torchci_reports(
    repo: str, workflow_run_id: int, artifact_path: Path
) -> None:
    # The artifact is named torchci-reports-runattemptN-JOB_ID.zip. Each report goes
    # to the key upload-test-artifacts gives it when it uploads to S3 itself.
    job_id = artifact_path.name.removesuffix(".zip").rpartition("-")[2]
    if not job_id.isdigit():
        print(f"Skipping {artifact_path.name}: no job id in its name")
        return
    for report in unzip(artifact_path).glob("**/*.report.jsonl"):
        upload_file_to_s3(
            file_name=str(report.resolve()),
            bucket=BUCKET_NAME,
            key=f"torchci-reports/{repo}/{workflow_run_id}/{job_id}/{report.name}",
        )


def get_artifacts(repo: str, workflow_run_id: int, workflow_run_attempt: int) -> None:
    with TemporaryDirectory() as temp_dir:
        print("Using temporary directory:", temp_dir)
        os.chdir(temp_dir)

        for artifact in ARTIFACTS:
            artifact_paths = download_gha_artifacts(
                artifact, workflow_run_id, workflow_run_attempt
            )

            for artifact_path in artifact_paths:
                if artifact == "torchci-reports":
                    upload_torchci_reports(repo, workflow_run_id, artifact_path)
                    continue
                # GHA artifact is named as follows: NAME-runattempt${{ github.run_attempt }}-SUFFIX.zip
                # and we want remove the run_attempt to conform with the naming convention on S3, i.e.
                # pytorch/pytorch/WORKFLOW_ID/RUN_ATTEMPT/artifact/NAME-SUFFIX.zip
                s3_filename = re.sub(FILENAME_REGEX, "", artifact_path.name)
                upload_file_to_s3(
                    file_name=str(artifact_path.resolve()),
                    bucket=BUCKET_NAME,
                    key=f"{repo}/{workflow_run_id}/{workflow_run_attempt}/artifact/{s3_filename}",
                )


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
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="which GitHub repo this workflow run belongs to",
    )
    args = parser.parse_args()
    get_artifacts(args.repo, args.workflow_run_id, args.workflow_run_attempt)
