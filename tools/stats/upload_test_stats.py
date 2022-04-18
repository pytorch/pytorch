import argparse
import os
import requests
import shutil
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any

import rockset  # type: ignore[import]
import boto3  # type: ignore[import]

PYTORCH_REPO = "https://api.github.com/repos/pytorch/pytorch"
GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
REQUEST_HEADERS = {
    "Accept": "application/vnd.github.v3+json",
    "Authorization": "token " + GITHUB_TOKEN,
}
S3_RESOURCE = boto3.resource("s3")
TEMP_DIR = Path(os.environ["RUNNER_TEMP"]) / "tmp-test-stats"


def parse_xml_report(
    report: Path, workflow_id: int, workflow_run_attempt: int
) -> List[Dict[str, Any]]:
    """Convert a test report xml file into a JSON-serializable list of test cases."""
    # [Job id in artifacts]
    # Retrieve the job id from the report path. In our GHA workflows, we append
    # the job id to the end of the report name, so `report` looks like:
    #     unzipped-test-reports-foo_5596745227/test/test-reports/foo/TEST-foo.xml
    # and we want to get `5596745227` out of it.
    job_id = int(report.parts[0].rpartition("_")[2])

    print(f"Parsing test report: {report}, job id: {job_id}")
    root = ET.parse(report)

    test_cases = []
    for test_case in root.findall("testcase"):
        case = process_xml_element(test_case)
        case["workflow_id"] = workflow_id
        case["workflow_run_attempt"] = workflow_run_attempt
        case["job_id"] = job_id
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

    # By default, all attributes are strings. Apply a few special conversions
    # here for well-known attributes so that they are the right type in Rockset.
    line = ret.get("line")
    if line:
        ret["line"] = int(line)
    time = ret.get("time")
    if time:
        ret["time"] = float(time)

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
    #     </testcase>
    # becomes
    #    {"foo": {"text": "hello"}}
    for child in element:
        ret[child.tag] = process_xml_element(child)
    return ret


def get_artifact_urls(workflow_run_id: int) -> Dict[Path, str]:
    """Get all workflow artifacts with 'test-report' in the name."""
    response = requests.get(
        f"{PYTORCH_REPO}/actions/runs/{workflow_run_id}/artifacts?per_page=100",
    )
    artifacts = response.json()["artifacts"]
    while "next" in response.links.keys():
        response = requests.get(response.links["next"]["url"], headers=REQUEST_HEADERS)
        artifacts.extend(response.json()["artifacts"])

    artifact_urls = {}
    for artifact in artifacts:
        if "test-report" in artifact["name"]:
            artifact_urls[Path(artifact["name"])] = artifact["archive_download_url"]
    return artifact_urls


def unzip(p: Path) -> None:
    """Unzip the provided zipfile to a similarly-named directory.

    Returns None if `p` is not a zipfile.

    Looks like: /tmp/test-reports.zip -> /tmp/unzipped-test-reports/
    """
    assert p.is_file()
    unzipped_dir = p.with_name("unzipped-" + p.stem)

    with zipfile.ZipFile(p, "r") as zip:
        zip.extractall(unzipped_dir)


def download_and_extract_artifact(
    artifact_name: Path, artifact_url: str, workflow_run_attempt: int
) -> None:
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
                print(f"Skipping {artifact_name} as it is an invalid run attempt.")

    print(f"Downloading and extracting {artifact_name}")

    response = requests.get(artifact_url, headers=REQUEST_HEADERS)
    with open(artifact_name, "wb") as f:
        f.write(response.content)
    unzip(artifact_name)


def download_and_extract_s3_reports(
    workflow_run_id: int, workflow_run_attempt: int
) -> None:
    bucket = S3_RESOURCE.Bucket("gha-artifacts")
    objs = bucket.objects.filter(
        Prefix=f"pytorch/pytorch/{workflow_run_id}/{workflow_run_attempt}/artifact/test-reports"
    )

    for obj in objs:
        p = Path(Path(obj.key).name)
        print(f"Downloading and extracting {p}")
        with open(p, "wb") as f:
            f.write(obj.get()["Body"].read())
        unzip(p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload test stats to Rockset")
    parser.add_argument(
        "--workflow-run-id",
        required=True,
        help="id of the workflow to get artifacts from",
    )
    parser.add_argument(
        "--workflow-run-attempt",
        required=True,
        help="which retry of the workflow this is",
    )
    args = parser.parse_args()

    if TEMP_DIR.exists():
        print("rm: ", TEMP_DIR)
        shutil.rmtree(TEMP_DIR)

    print("mkdir: ", TEMP_DIR)
    TEMP_DIR.mkdir()
    print("cd to ", TEMP_DIR)
    os.chdir(TEMP_DIR)

    # Download and extract all the reports (both GHA and S3)
    download_and_extract_s3_reports(args.workflow_run_id, args.workflow_run_attempt)
    artifact_urls = get_artifact_urls(args.workflow_run_id)
    for name, url in artifact_urls.items():
        download_and_extract_artifact(Path(name), url, args.workflow_run_attempt)

    # Parse the reports and transform them to JSON
    test_cases = []
    for xml_report in Path(".").glob("**/*.xml"):
        test_cases.extend(
            parse_xml_report(
                xml_report, int(args.workflow_run_id), int(args.workflow_run_attempt)
            )
        )

    # Write the JSON to rockset
    print(f"Writing {len(test_cases)} test cases to Rockset")
    client = rockset.Client(
        api_server="api.rs2.usw2.rockset.com", api_key=os.environ["ROCKSET_API_KEY"]
    )
    client.Collection.retrieve("test_run").add_docs(test_cases)
    print("Done!")
