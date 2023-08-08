import datetime
import gzip
import inspect
import io
import json
import os
import time
import uuid
import zipfile

from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List
from warnings import warn

import boto3  # type: ignore[import]
import requests
import rockset  # type: ignore[import]

PYTORCH_REPO = "https://api.github.com/repos/pytorch/pytorch"
S3_RESOURCE = boto3.resource("s3")

# NB: In CI, a flaky test is usually retried 3 times, then the test file would be rerun
# 2 more times
MAX_RETRY_IN_NON_DISABLED_MODE = 3 * 3


def _get_request_headers() -> Dict[str, str]:
    return {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": "token " + os.environ["GITHUB_TOKEN"],
    }


def _get_artifact_urls(prefix: str, workflow_run_id: int) -> Dict[Path, str]:
    """Get all workflow artifacts with 'test-report' in the name."""
    response = requests.get(
        f"{PYTORCH_REPO}/actions/runs/{workflow_run_id}/artifacts?per_page=100",
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
) -> List[Path]:
    bucket = S3_RESOURCE.Bucket("gha-artifacts")
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
) -> List[Path]:
    artifact_urls = _get_artifact_urls(prefix, workflow_run_id)
    paths = []
    for name, url in artifact_urls.items():
        paths.append(_download_artifact(Path(name), url, workflow_run_attempt))
    return paths


def upload_to_rockset(
    collection: str, docs: List[Any], workspace: str = "commons"
) -> None:
    print(f"Writing {len(docs)} documents to Rockset")
    client = rockset.RocksetClient(
        host="api.usw2a1.rockset.com", api_key=os.environ["ROCKSET_API_KEY"]
    )
    client.Documents.add_documents(
        collection=collection,
        data=docs,
        workspace=workspace,
    )
    print("Done!")


def upload_to_s3(
    bucket_name: str,
    key: str,
    docs: List[Dict[str, Any]],
) -> None:
    print(f"Writing {len(docs)} documents to S3")
    body = io.StringIO()
    for doc in docs:
        json.dump(doc, body)
        body.write("\n")

    S3_RESOURCE.Object(
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
) -> List[Dict[str, Any]]:
    print(f"Reading from s3://{bucket_name}/{key}")
    body = (
        S3_RESOURCE.Object(
            f"{bucket_name}",
            f"{key}",
        )
        .get()["Body"]
        .read()
    )
    results = gzip.decompress(body).decode().split("\n")
    return [json.loads(result) for result in results if result]


def upload_workflow_stats_to_s3(
    workflow_run_id: int,
    workflow_run_attempt: int,
    collection: str,
    docs: List[Dict[str, Any]],
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


def is_rerun_disabled_tests(tests: Dict[str, Dict[str, int]]) -> bool:
    """
    Check if the test report is coming from rerun_disabled_tests workflow where
    each test is run multiple times
    """
    return all(
        t.get("num_green", 0) + t.get("num_red", 0) > MAX_RETRY_IN_NON_DISABLED_MODE
        for t in tests.values()
    )


def _convert_float_values_to_decimals(data: Dict[str, Any]) -> Dict[str, Any]:
    return {k: Decimal(str(v)) if isinstance(v, float) else v for k, v in data.items()}


class EnvVarMetric:
    name: str
    env_var: str
    required: bool = True
    # Used to cast the value of the env_var to the correct type (defaults to str)
    type_conversion_fn: Any = None

    def __init__(
        self,
        name: str,
        env_var: str,
        required: bool = True,
        type_conversion_fn: Any = None,
    ) -> None:
        self.name = name
        self.env_var = env_var
        self.required = required
        self.type_conversion_fn = type_conversion_fn

    def value(self) -> Any:
        value = os.environ.get(self.env_var)
        if value is None and self.required:
            raise ValueError(
                (
                    f"Missing {self.name}. Please set the {self.env_var}"
                    "environment variable to pass in this value."
                )
            )
        if self.type_conversion_fn:
            return self.type_conversion_fn(value)
        return value


def emit_metric(
    metric_name: str,
    metrics: Dict[str, Any],
) -> None:
    """
    Upload a metric to DynamoDB (and from there, Rockset).

    Parameters:
        metric_name:
            Name of the metric. Every unique metric should have a different name
            and be emitted just once per run attempt.
            Metrics are namespaced by their module and the function that emitted them.
        metrics: The actual data to record.

    Some default values are populated from environment variables, which must be set
    for metrics to be emitted. (If they're not set, this function becomes a noop):
    """

    if metrics is None:
        raise ValueError("You didn't ask to upload any metrics!")

    # We use these env vars that to determine basic info about the workflow run.
    # By using env vars, we don't have to pass this info around to every function.
    # It also helps ensure that we only emit metrics during CI
    env_var_metrics = [
        EnvVarMetric("repo", "GITHUB_REPOSITORY"),
        EnvVarMetric("workflow", "GITHUB_WORKFLOW"),
        EnvVarMetric("build_environment", "BUILD_ENVIRONMENT"),
        EnvVarMetric("job", "GITHUB_JOB"),
        EnvVarMetric("test_config", "TEST_CONFIG", required=False),
        EnvVarMetric("run_id", "GITHUB_RUN_ID", type_conversion_fn=int),
        EnvVarMetric("run_number", "GITHUB_RUN_NUMBER", type_conversion_fn=int),
        EnvVarMetric("run_attempt", "GITHUB_RUN_ATTEMPT", type_conversion_fn=int),
    ]

    # Use info about the function that invoked this one as a namespace and a way to filter metrics.
    calling_frame = inspect.currentframe().f_back  # type: ignore[union-attr]
    calling_frame_info = inspect.getframeinfo(calling_frame)  # type: ignore[arg-type]
    calling_file = os.path.basename(calling_frame_info.filename)
    calling_module = inspect.getmodule(calling_frame).__name__  # type: ignore[union-attr]
    calling_function = calling_frame_info.function

    try:
        reserved_metrics = {
            "metric_name": metric_name,
            "calling_file": calling_file,
            "calling_module": calling_module,
            "calling_function": calling_function,
            "timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f"),
            **{m.name: m.value() for m in env_var_metrics},
        }
    except ValueError as e:
        warn(f"Not emitting metrics. {e}")
        return

    # Prefix key with metric name and timestamp to derisk chance of a uuid1 name collision
    reserved_metrics[
        "dynamo_key"
    ] = f"{metric_name}_{int(time.time())}_{uuid.uuid1().hex}"

    # Ensure the metrics dict doesn't contain any reserved keys
    for key in reserved_metrics.keys():
        used_reserved_keys = [k for k in metrics.keys() if k == key]
        if used_reserved_keys:
            raise ValueError(f"Metrics dict contains reserved keys: [{', '.join(key)}]")

    # boto3 doesn't support uploading float values to DynamoDB, so convert them all to decimals.
    metrics = _convert_float_values_to_decimals(metrics)

    try:
        session = boto3.Session(region_name="us-east-1")
        session.resource("dynamodb").Table("torchci-metrics").put_item(
            Item={
                **reserved_metrics,
                **metrics,
            }
        )
    except Exception as e:
        # We don't want to fail the job if we can't upload the metric.
        # We still raise the ValueErrors outside this try block since those indicate improperly configured metrics
        warn(f"Error uploading metric to DynamoDB: {e}")
        return
