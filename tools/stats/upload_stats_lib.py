from __future__ import annotations

import gzip
import io
import json
import math
import os
import time
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import boto3  # type: ignore[import]
import requests


PYTORCH_REPO = "https://api.github.com/repos/pytorch/pytorch"


@lru_cache
def get_s3_resource() -> Any:
    return boto3.resource("s3")


# NB: In CI, a flaky test is usually retried 3 times, then the test file would be rerun
# 2 more times
MAX_RETRY_IN_NON_DISABLED_MODE = 3 * 3


def _get_request_headers() -> dict[str, str]:
    return {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": "token " + os.environ["GITHUB_TOKEN"],
    }


def _get_artifact_urls(prefix: str, workflow_run_id: int) -> dict[Path, str]:
    """Get all workflow artifacts with 'test-report' in the name."""
    response = requests.get(
        f"{PYTORCH_REPO}/actions/runs/{workflow_run_id}/artifacts?per_page=100",
        headers=_get_request_headers(),
    )
    artifacts = response.json()["artifacts"]
    while "next" in response.links.keys():
        response = requests.get(
            response.links["next"]["url"], headers=_get_request_headers()
        )
        artifacts.extend(response.json()["artifacts"])

    artifact_urls = {}
    for artifact in artifacts:
        if artifact["name"].startswith(prefix):
            artifact_urls[Path(artifact["name"])] = artifact["archive_download_url"]
    return artifact_urls


def _download_artifact(
    artifact_name: Path, artifact_url: str, workflow_run_attempt: int
) -> Path:
    # [Artifact run attempt]
    # All artifacts on a workflow share a single namespace. However, we can
    # re-run a workflow and produce a new set of artifacts. To avoid name
    # collisions, we add `-runattempt1<run #>-` somewhere in the artifact name.
    #
    # This code parses out the run attempt number from the artifact name. If it
    # doesn't match the one specified on the command line, skip it.
    atoms = str(artifact_name).split("-")
    for atom in atoms:
        if atom.startswith("runattempt"):
            found_run_attempt = int(atom[len("runattempt") :])
            if workflow_run_attempt != found_run_attempt:
                print(
                    f"Skipping {artifact_name} as it is an invalid run attempt. "
                    f"Expected {workflow_run_attempt}, found {found_run_attempt}."
                )

    print(f"Downloading {artifact_name}")

    response = requests.get(artifact_url, headers=_get_request_headers())
    with open(artifact_name, "wb") as f:
        f.write(response.content)
    return artifact_name


def download_s3_artifacts(
    prefix: str, workflow_run_id: int, workflow_run_attempt: int
) -> list[Path]:
    bucket = get_s3_resource().Bucket("gha-artifacts")
    objs = bucket.objects.filter(
        Prefix=f"pytorch/pytorch/{workflow_run_id}/{workflow_run_attempt}/artifact/{prefix}"
    )

    found_one = False
    paths = []
    for obj in objs:
        found_one = True
        p = Path(Path(obj.key).name)
        print(f"Downloading {p}")
        with open(p, "wb") as f:
            f.write(obj.get()["Body"].read())
        paths.append(p)

    if not found_one:
        print(
            "::warning title=s3 artifacts not found::"
            "Didn't find any test reports in s3, there might be a bug!"
        )
    return paths


def download_gha_artifacts(
    prefix: str, workflow_run_id: int, workflow_run_attempt: int
) -> list[Path]:
    artifact_urls = _get_artifact_urls(prefix, workflow_run_id)
    paths = []
    for name, url in artifact_urls.items():
        paths.append(_download_artifact(Path(name), url, workflow_run_attempt))
    return paths


def upload_to_dynamodb(
    dynamodb_table: str,
    repo: str,
    docs: List[Any],
    generate_partition_key: Optional[Callable[[str, Dict[str, Any]], str]],
) -> None:
    print(f"Writing {len(docs)} documents to DynamoDB {dynamodb_table}")
    # https://boto3.amazonaws.com/v1/documentation/api/latest/guide/dynamodb.html#batch-writing
    with boto3.resource("dynamodb").Table(dynamodb_table).batch_writer() as batch:
        for doc in docs:
            if generate_partition_key:
                doc["dynamoKey"] = generate_partition_key(repo, doc)
            # This is to move away the _event_time field from Rockset, which we cannot use when
            # reimport the data
            doc["timestamp"] = int(round(time.time() * 1000))
            batch.put_item(Item=doc)


def upload_to_s3(
    bucket_name: str,
    key: str,
    docs: list[dict[str, Any]],
) -> None:
    print(f"Writing {len(docs)} documents to S3")
    body = io.StringIO()
    for doc in docs:
        json.dump(doc, body)
        body.write("\n")

    get_s3_resource().Object(
        f"{bucket_name}",
        f"{key}",
    ).put(
        Body=gzip.compress(body.getvalue().encode()),
        ContentEncoding="gzip",
        ContentType="application/json",
    )
    print("Done!")


def read_from_s3(
    bucket_name: str,
    key: str,
) -> list[dict[str, Any]]:
    print(f"Reading from s3://{bucket_name}/{key}")
    body = (
        get_s3_resource()
        .Object(
            f"{bucket_name}",
            f"{key}",
        )
        .get()["Body"]
        .read()
    )
    results = gzip.decompress(body).decode().split("\n")
    return [json.loads(result) for result in results if result]


def remove_nan_inf(old: Any) -> Any:
    # Casta NaN, inf, -inf to string from float since json.dumps outputs invalid
    # json with them
    def _helper(o: Any) -> Any:
        if isinstance(o, float) and (math.isinf(o) or math.isnan(o)):
            return str(o)
        if isinstance(o, list):
            return [_helper(v) for v in o]
        if isinstance(o, dict):
            return {_helper(k): _helper(v) for k, v in o.items()}
        if isinstance(o, tuple):
            return tuple(_helper(v) for v in o)
        return o

    return _helper(old)


def upload_workflow_stats_to_s3(
    workflow_run_id: int,
    workflow_run_attempt: int,
    collection: str,
    docs: list[dict[str, Any]],
) -> None:
    bucket_name = "ossci-raw-job-status"
    key = f"{collection}/{workflow_run_id}/{workflow_run_attempt}"
    upload_to_s3(bucket_name, key, docs)


def upload_file_to_s3(
    file_name: str,
    bucket: str,
    key: str,
) -> None:
    """
    Upload a local file to S3
    """
    print(f"Upload {file_name} to s3://{bucket}/{key}")
    boto3.client("s3").upload_file(
        file_name,
        bucket,
        key,
    )


def unzip(p: Path) -> None:
    """Unzip the provided zipfile to a similarly-named directory.

    Returns None if `p` is not a zipfile.

    Looks like: /tmp/test-reports.zip -> /tmp/unzipped-test-reports/
    """
    assert p.is_file()
    unzipped_dir = p.with_name("unzipped-" + p.stem)
    print(f"Extracting {p} to {unzipped_dir}")

    with zipfile.ZipFile(p, "r") as zip:
        zip.extractall(unzipped_dir)


def is_rerun_disabled_tests(tests: dict[str, dict[str, int]]) -> bool:
    """
    Check if the test report is coming from rerun_disabled_tests workflow where
    each test is run multiple times
    """
    return all(
        t.get("num_green", 0) + t.get("num_red", 0) > MAX_RETRY_IN_NON_DISABLED_MODE
        for t in tests.values()
    )


def get_job_id(report: Path) -> int | None:
    # [Job id in artifacts]
    # Retrieve the job id from the report path. In our GHA workflows, we append
    # the job id to the end of the report name, so `report` looks like:
    #     unzipped-test-reports-foo_5596745227/test/test-reports/foo/TEST-foo.xml
    # and we want to get `5596745227` out of it.
    try:
        return int(report.parts[0].rpartition("_")[2])
    except ValueError:
        return None
