import glob
import gzip
import json
import os
import time
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout

from tools.stats.upload_test_stats import parse_xml_report


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LAST_UPDATED = 0.0
LOG_BUCKET_PREFIX = "temp_logs"


@lru_cache(maxsize=1)
def get_s3_resource() -> Any:
    import boto3  # type: ignore[import]

    return boto3.client("s3")


def zip_artifact(file_name: str, paths: list[str]) -> None:
    """Zip the files in the paths listed into file_name. The paths will be used
    in a glob and should be relative to REPO_ROOT."""

    with zipfile.ZipFile(file_name, "w") as f:
        for path in paths:
            for file in glob.glob(f"{REPO_ROOT}/{path}", recursive=True):
                f.write(file, os.path.relpath(file, REPO_ROOT))


def concated_logs() -> str:
    """Concatenate all the logs in the test-reports directory into a single string."""
    logs = []
    for log_file in glob.glob(
        f"{REPO_ROOT}/test/test-reports/**/*.log", recursive=True
    ):
        logs.append(f"=== {log_file} ===")
        with open(log_file) as f:
            # For every line, prefix with fake timestamp for log classifier
            for line in f:
                line = line.rstrip("\n")  # Remove any trailing newline
                logs.append(f"2020-01-01T00:00:00.0000000Z {line}")
    return "\n".join(logs)


def upload_to_s3_artifacts(failed: bool) -> None:
    """Upload the file to S3."""
    workflow_id = os.environ.get("GITHUB_RUN_ID")
    workflow_run_attempt = os.environ.get("GITHUB_RUN_ATTEMPT")
    file_suffix = os.environ.get("ARTIFACTS_FILE_SUFFIX")
    job_id = os.environ.get("JOB_ID")
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
    if job_id and failed:
        logs = concated_logs()
        # Put logs into bucket so log classifier can access them. We cannot get
        # the actual GH logs so this will have to be a proxy.
        print(f"Uploading logs for {job_id} to S3")
        get_s3_resource().put_object(
            Body=gzip.compress(logs.encode("utf-8")),
            Bucket="gha-artifacts",
            Key=f"{LOG_BUCKET_PREFIX}/{job_id}",
            ContentType="text/plain",
            ContentEncoding="gzip",
        )


def zip_and_upload_artifacts(failed: bool) -> None:
    # not thread safe but correctness of the LAST_UPDATED var doesn't really
    # matter for this
    # Upload if a test failed or every 20 minutes
    global LAST_UPDATED

    if failed or time.time() - LAST_UPDATED > 20 * 60:
        start = time.time()
        try:
            upload_to_s3_artifacts(failed=failed)
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


def parse_xml_and_upload_json() -> None:
    """
    Parse xml test reports that do not yet have a corresponding json report
    uploaded to s3, and upload the json reports to s3. Use filelock to avoid
    uploading the same file from multiple processes.
    """
    try:
        job_id: int | None = int(os.environ.get("JOB_ID", 0))
        if job_id == 0:
            job_id = None
    except (ValueError, TypeError):
        job_id = None

    try:
        for xml_file in glob.glob(
            f"{REPO_ROOT}/test/test-reports/**/*.xml", recursive=True
        ):
            xml_path = Path(xml_file)
            json_file = xml_path.with_suffix(".json")
            lock = FileLock(str(json_file) + ".lock")

            try:
                lock.acquire(timeout=0)  # immediately fails if already locked
                if json_file.exists():
                    continue  # already uploaded
                test_cases = parse_xml_report(
                    "testcase",
                    xml_path,
                    int(os.environ.get("GITHUB_RUN_ID", "0")),
                    int(os.environ.get("GITHUB_RUN_ATTEMPT", "0")),
                    job_id,
                )
                line_by_line_jsons = "\n".join([json.dumps(tc) for tc in test_cases])

                gzipped = gzip.compress(line_by_line_jsons.encode("utf-8"))
                s3_key = (
                    json_file.relative_to(REPO_ROOT / "test/test-reports")
                    .as_posix()
                    .replace("/", "_")
                )

                get_s3_resource().put_object(
                    Body=gzipped,
                    Bucket="gha-artifacts",
                    Key=f"test_jsons_while_running/{os.environ.get('GITHUB_RUN_ID')}/{job_id}/{s3_key}",
                    ContentType="application/json",
                    ContentEncoding="gzip",
                )

                # We don't need to save the json file locally, but doing so lets us
                # track which ones have been uploaded already. We could probably also
                # check S3
                with open(json_file, "w") as f:
                    f.write(line_by_line_jsons)
            except Timeout:
                continue  # another process is working on this file
            finally:
                if lock.is_locked:
                    lock.release()
    except Exception as e:
        print(f"Failed to parse and upload json test reports: {e}")


def parse_pytest_nodeid(nodeid: str) -> tuple[str, str, str, str] | None:
    """Parse a pytest nodeid into (test_file_path, invoking_file, classname, testname).

    Parametrized test ids can legally contain `::` inside `[...]`
    (e.g. `test_foo.py::test_bar[a::b]`), so a plain `split("::")` is wrong.
    This walks the string and splits on top-level `::` only.

    Returns None if `nodeid` doesn't look like a `<path>.py::...` form.

    - `test_file_path`: `dynamo/test_foo.py` — rootdir-relative path, matches
      pytest xunit2's `<testcase file="...">` attribute.
    - `invoking_file`: `dynamo/test_foo` — what `upload_adhoc_failure_json`
      historically expects as its first arg.
    - `classname`: last class segment (`TestFoo`) or `""` for function tests.
      Note: pytest xunit2 uses a dotted *module-qualified* classname; we don't
      try to reconstruct that, so callers must dedup on `(file, testname)`.
    - `testname`: full method name including the parametrize suffix
      (`test_bar[a::b]`).
    """
    sep = ".py::"
    idx = nodeid.find(sep)
    if idx < 0:
        return None
    test_file_path = nodeid[: idx + 3]
    invoking_file = nodeid[:idx]
    tail = nodeid[idx + len(sep) :]
    parts: list[str] = []
    cur: list[str] = []
    depth = 0
    i = 0
    while i < len(tail):
        c = tail[i]
        if c == "[":
            depth += 1
            cur.append(c)
            i += 1
        elif c == "]":
            depth = max(0, depth - 1)
            cur.append(c)
            i += 1
        elif depth == 0 and tail[i : i + 2] == "::":
            parts.append("".join(cur))
            cur = []
            i += 2
        else:
            cur.append(c)
            i += 1
    parts.append("".join(cur))
    if len(parts) >= 2:
        classname = parts[-2]
        testname = parts[-1]
    else:
        classname = ""
        testname = parts[0]
    return test_file_path, invoking_file, classname, testname


def upload_adhoc_failure_json(
    invoking_file: str,
    current_failure: str,
    reason: str | None = None,
    s3_key_suffix: str | None = None,
    classname: str | None = None,
    testname: str | None = None,
    file_attr: str | None = None,
) -> None:
    """
    manually upload a json to s3 indicating that a test failed without pytest
    writing a junit-xml entry for it (e.g. segfault before sessionfinish, or
    shard SIGTERM-on-timeout).

    `reason` overrides the failure message written into the row; defaults to
    the segfault wording for callers that don't pass one. `s3_key_suffix`
    overrides the random per-call suffix used in the S3 key; pass a
    deterministic value to make repeated invocations idempotent.

    `classname` and `testname` override the legacy `current_failure.split("::")`
    parsing — pass these from `parse_pytest_nodeid` so parametrized ids that
    contain `::` inside brackets aren't mis-split.

    `file_attr` overrides the `file` JSON field. Defaults to `f"{invoking_file}.py"`
    for backwards compatibility. Pass the true source file path (from the
    pytest nodeid) when `invoking_file` is the discovery unit and you want
    the row's `file` to point at the real test source — useful when an
    inherited test lives in a different file from the one run_test.py invoked.
    """
    try:
        job_id = int(os.environ["JOB_ID"])
        workflow_id = int(os.environ["GITHUB_RUN_ID"])
    except Exception as e:
        print(f"Failed to get job_id or workflow_id: {e}")
        return

    if classname is None or testname is None:
        split_failure = current_failure.split("::")
        if len(split_failure) >= 2:
            classname = split_failure[-2] if classname is None else classname
            testname = split_failure[-1] if testname is None else testname
        else:
            testname = current_failure if testname is None else testname
            classname = "" if classname is None else classname

    message = reason or (
        "The test file failed but pytest did not generate xml.  "
        "The most likely cause is a segfault"
    )
    j = {
        "invoking_file": invoking_file,
        "file": file_attr if file_attr else f"{invoking_file}.py",
        "name": testname,
        "classname": classname,
        "workflow_id": workflow_id,
        "workflow_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "job_id": job_id,
        "failure": {"message": message, "text": message},
    }
    gzipped = gzip.compress(json.dumps(j).encode("utf-8"))
    suffix = s3_key_suffix if s3_key_suffix is not None else os.urandom(8).hex()
    s3_key = f"{invoking_file.replace('/', '_')}_{suffix}.json"
    get_s3_resource().put_object(
        Body=gzipped,
        Bucket="gha-artifacts",
        Key=f"test_jsons_while_running/{workflow_id}/{job_id}/{s3_key}",
        ContentType="application/json",
        ContentEncoding="gzip",
    )
