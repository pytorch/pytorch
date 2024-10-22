import glob
import os
import time
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any, List


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LAST_UPDATED = 0.0


@lru_cache(maxsize=1)
def get_s3_resource() -> Any:
    import boto3  # type: ignore[import]

    return boto3.client("s3")


def zip_artifact(file_name: str, paths: List[str]) -> None:
    """Zip the files in the paths listed into file_name. The paths will be used
    in a glob and should be relative to REPO_ROOT."""

    with zipfile.ZipFile(file_name, "w") as f:
        for path in paths:
            for file in glob.glob(f"{REPO_ROOT}/{path}", recursive=True):
                f.write(file, os.path.relpath(file, REPO_ROOT))


def upload_to_s3_artifacts() -> None:
    """Upload the file to S3."""
    workflow_id = os.environ.get("GITHUB_RUN_ID")
    workflow_run_attempt = os.environ.get("GITHUB_RUN_ATTEMPT")
    file_suffix = os.environ.get("ARTIFACTS_FILE_SUFFIX")
    if not workflow_id or not workflow_run_attempt or not file_suffix:
        print(
            "GITHUB_RUN_ID, GITHUB_RUN_ATTEMPT, or ARTIFACTS_FILE_SUFFIX not set, not uploading"
        )
        return

    test_reports_zip_path = f"{REPO_ROOT}/test-reports-{file_suffix}.zip"
    zip_artifact(
        test_reports_zip_path,
        ["test/test-reports/**/*.xml", "test/test-reports/**/*.csv"],
    )
    test_logs_zip_path = f"{REPO_ROOT}/logs-{file_suffix}.zip"
    zip_artifact(test_logs_zip_path, ["test/test-reports/**/*.log"])
    jsons_zip_path = f"{REPO_ROOT}/test-jsons-{file_suffix}.zip"
    zip_artifact(jsons_zip_path, ["test/test-reports/**/*.json"])

    s3_prefix = f"pytorch/pytorch/{workflow_id}/{workflow_run_attempt}/artifact"
    get_s3_resource().upload_file(
        test_reports_zip_path,
        "gha-artifacts",
        f"{s3_prefix}/{Path(test_reports_zip_path).name}",
    )
    get_s3_resource().upload_file(
        test_logs_zip_path,
        "gha-artifacts",
        f"{s3_prefix}/{Path(test_logs_zip_path).name}",
    )
    get_s3_resource().upload_file(
        test_logs_zip_path,
        "gha-artifacts",
        f"{s3_prefix}/{Path(jsons_zip_path).name}",
    )
    get_s3_resource().put_object(
        Body=b"",
        Bucket="gha-artifacts",
        Key=f"workflows_failing_pending_upload/{workflow_id}.txt",
    )


def zip_and_upload_artifacts(failed: bool) -> None:
    # not thread safe but correctness of the LAST_UPDATED var doesn't really
    # matter for this
    # Upload if a test failed or every 20 minutes
    global LAST_UPDATED

    if failed or time.time() - LAST_UPDATED > 20 * 60:
        start = time.time()
        try:
            upload_to_s3_artifacts()
            LAST_UPDATED = time.time()
        except Exception as e:
            print(f"Failed to upload artifacts: {e}")
        print(f"Uploading artifacts took {time.time() - start:.2f} seconds")


def trigger_upload_test_stats_intermediate_workflow() -> None:
    import requests

    # The GITHUB_TOKEN cannot trigger workflow so this isn't used for now
    print("Triggering upload_test_stats_intermediate workflow")
    x = requests.post(
        "https://api.github.com/repos/pytorch/pytorch/actions/workflows/upload_test_stats_intermediate.yml/dispatches",
        headers={
            "Accept": "application/vnd.github.v3+json",
            "Authorization": f"Bearer {os.environ.get('GITHUB_TOKEN')}",
        },
        json={
            "ref": "main",
            "inputs": {
                "workflow_run_id": os.environ.get("GITHUB_RUN_ID"),
                "workflow_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
            },
        },
    )
    print(x.text)
