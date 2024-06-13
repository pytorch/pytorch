import gzip
import io
import json
import os
import re

import xml.etree.ElementTree as ET
import zipfile
from functools import lru_cache
from multiprocessing import cpu_count, Pool
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Tuple

import boto3  # type: ignore[import]
import requests
import rockset  # type: ignore[import]

PYTORCH_REPO = "https://api.github.com/repos/pytorch/pytorch"
S3_RESOURCE = boto3.resource("s3")

# NB: In CI, a flaky test is usually retried 3 times, then the test file would be rerun
# 2 more times
MAX_RETRY_IN_NON_DISABLED_MODE = 3 * 3
# NB: Rockset has an upper limit of 5000 documents in one request
BATCH_SIZE = 5000


def _get_request_headers() -> Dict[str, str]:
    return {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": "token " + os.environ["GITHUB_TOKEN"],
    }


def _get_artifact_urls(prefix: str, workflow_run_id: int) -> Dict[Path, str]:
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
    collection: str,
    docs: List[Any],
    workspace: str = "commons",
    client: Any = None,
) -> None:
    if not client:
        client = rockset.RocksetClient(
            host="api.usw2a1.rockset.com", api_key=os.environ["ROCKSET_API_KEY"]
        )

    index = 0
    while index < len(docs):
        from_index = index
        to_index = min(from_index + BATCH_SIZE, len(docs))
        print(f"Writing {to_index - from_index} documents to Rockset")

        client.Documents.add_documents(
            collection=collection,
            data=docs[from_index:to_index],
            workspace=workspace,
        )
        index += BATCH_SIZE

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


@lru_cache(maxsize=None)
def get_jobs_for_workflow(
    workflow_id: int, workflow_run_attempt: int
) -> List[Dict[str, Any]]:
    # Returns all workflow jobs for a given workflow + its run attempt
    page = 1
    total_count = None
    all_jobs = []

    while True:
        workflow_info = requests.get(
            f"https://api.github.com/repos/pytorch/pytorch/actions/runs/{workflow_id}/attempts/{workflow_run_attempt}/jobs",
            params={"per_page": 100, "page": page},
        ).json()
        jobs = workflow_info["jobs"]
        if total_count is None:
            total_count = workflow_info["total_count"]
        elif total_count != workflow_info["total_count"]:
            return None
        all_jobs.extend(jobs)
        if len(all_jobs) == total_count:
            break
        page += 1
    return all_jobs


def get_job_ids_for_paths(
    reports: List[Path], workflow_id: int, workflow_run_attempt: int
) -> List[Tuple[Path, Optional[int]]]:
    # Returns a list of tuples of (report, job_id) for each report in `reports`.
    # If a report doesn't have the job id in the name, it attempts to find the
    # job id by matching the report name with the job name.
    existing_ids: List[Tuple[Path, Optional[int]]] = []
    missing = []

    for report in reports:
        job_id = get_job_id(report)
        if job_id is not None:
            existing_ids.append((report, job_id))
        else:
            missing.append(report)
    if not missing:
        return existing_ids

    while True:
        workflow_info = get_jobs_for_workflow(workflow_id, workflow_run_attempt)
        if workflow_info is not None:
            break

    possible_jobs = [
        job
        for job in workflow_info
        if job["id"] not in [job_id for _, job_id in existing_ids]
    ]
    if len(missing) == 1 and len(possible_jobs) == 1:
        return existing_ids + [(missing[0], possible_jobs[0]["id"])]

    regex = r"^unzipped-([\w-]+)-(\w+)-(\d+)-(\d+)-([\w.]+)_"
    for report in missing:
        match = re.match(regex, report.parts[0])
        if not match:
            existing_ids.append((report, None))
            continue

        config = match.group(2)
        shard = match.group(3)
        num_shards = match.group(4)
        runner = match.group(5)
        name = f"{config}, {shard}, {num_shards}, {runner}"
        candidates = [job for job in possible_jobs if name in job["name"]]
        if len(candidates) == 1:
            existing_ids.append((report, candidates[0]["id"]))
        else:
            existing_ids.append((report, job_id))
    return existing_ids


def get_job_id(report: Path) -> Optional[int]:
    # [Job id in artifacts]
    # Retrieve the job id from the report path. In our GHA workflows, we append
    # the job id to the end of the report name, so `report` looks like:
    #     unzipped-test-reports-foo_5596745227/test/test-reports/foo/TEST-foo.xml
    # and we want to get `5596745227` out of it.
    try:
        return int(report.parts[0].rpartition("_")[2])
    except ValueError:
        return None


def parse_xml_report(
    tag: str,
    report: Path,
    workflow_id: int,
    workflow_run_attempt: int,
    job_id: int,
) -> List[Dict[str, Any]]:
    """Convert a test report xml file into a JSON-serializable list of test cases."""
    test_cases: List[Dict[str, Any]] = []

    root = ET.parse(report)
    for test_case in root.iter(tag):
        case = process_xml_element(test_case)
        case["workflow_id"] = workflow_id
        case["workflow_run_attempt"] = workflow_run_attempt
        case["job_id"] = job_id

        # [invoking file]
        # The name of the file that the test is located in is not necessarily
        # the same as the name of the file that invoked the test.
        # For example, `test_jit.py` calls into multiple other test files (e.g.
        # jit/test_dce.py). For sharding/test selection purposes, we want to
        # record the file that invoked the test.
        #
        # To do this, we leverage an implementation detail of how we write out
        # tests (https://bit.ly/3ajEV1M), which is that reports are created
        # under a folder with the same name as the invoking file.
        case["invoking_file"] = report.parent.name
        test_cases.append(case)

    return test_cases


def process_xml_element(element: ET.Element) -> Dict[str, Any]:
    """Convert a test suite element into a JSON-serializable dict."""
    ret: Dict[str, Any] = {}

    # Convert attributes directly into dict elements.
    # e.g.
    #     <testcase name="test_foo" classname="test_bar"></testcase>
    # becomes:
    #     {"name": "test_foo", "classname": "test_bar"}
    ret.update(element.attrib)

    # The XML format encodes all values as strings. Convert to ints/floats if
    # possible to make aggregation possible in Rockset.
    for k, v in ret.items():
        try:
            ret[k] = int(v)
        except ValueError:
            pass
        try:
            ret[k] = float(v)
        except ValueError:
            pass

    # Convert inner and outer text into special dict elements.
    # e.g.
    #     <testcase>my_inner_text</testcase> my_tail
    # becomes:
    #     {"text": "my_inner_text", "tail": " my_tail"}
    if element.text and element.text.strip():
        ret["text"] = element.text
    if element.tail and element.tail.strip():
        ret["tail"] = element.tail

    # Convert child elements recursively, placing them at a key:
    # e.g.
    #     <testcase>
    #       <foo>hello</foo>
    #       <foo>world</foo>
    #       <bar>another</bar>
    #     </testcase>
    # becomes
    #    {
    #       "foo": [{"text": "hello"}, {"text": "world"}],
    #       "bar": {"text": "another"}
    #    }
    for child in element:
        if child.tag not in ret:
            ret[child.tag] = process_xml_element(child)
        else:
            # If there are multiple tags with the same name, they should be
            # coalesced into a list.
            if not isinstance(ret[child.tag], list):
                ret[child.tag] = [ret[child.tag]]
            ret[child.tag].append(process_xml_element(child))
    return ret


def get_tests(workflow_run_id: int, workflow_run_attempt: int) -> List[Dict[str, Any]]:
    with TemporaryDirectory() as temp_dir:
        print("Using temporary directory:", temp_dir)
        current_dir = os.getcwd()
        os.chdir(temp_dir)

        # Download and extract all the reports (both GHA and S3)
        s3_paths = download_s3_artifacts(
            "test-report", workflow_run_id, workflow_run_attempt
        )
        for path in s3_paths:
            unzip(path)

        # Parse the reports and transform them to JSON
        test_cases = []
        mp = Pool(cpu_count())

        for xml_report, job_id in get_job_ids_for_paths(
            list(Path(".").glob("**/*.xml")),
            workflow_run_id,
            workflow_run_attempt,
        ):
            test_cases.append(
                mp.apply_async(
                    parse_xml_report,
                    args=(
                        "testcase",
                        xml_report,
                        workflow_run_id,
                        workflow_run_attempt,
                        job_id,
                    ),
                )
            )
        mp.close()
        mp.join()
        test_cases = [tc.get() for tc in test_cases]
        flattened = [item for sublist in test_cases for item in sublist]
        os.chdir(current_dir)
        return flattened
