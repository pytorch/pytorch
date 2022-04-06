import argparse
import os
import requests
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any
import datetime

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


def parse_xml_report(report: Path, workflow_id: int) -> List[Dict[str, Any]]:
    """Convert a test report xml file into a JSON-serializable list of test cases."""
    # Retrieve the job id from the report path. In our GHA workflows, we append
    # the job id to the end of the report name, so `report` looks like:
    #     unzipped-test-reports-foo_5596745227/test/test-reports/foo/TEST-foo.xml
    # and we want to get `5596745227` out of it.
    job_id = int(report.parts[0].rpartition("_")[2])

    print(f"Parsing test report: {report}, job id: {job_id}")
    root = ET.parse(
        report,
        ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)),  # type: ignore[call-arg]
    )

    test_cases = []
    for test_case in root.findall("testcase"):
        case = process_xml_element(test_case)
        case["workflow_id"] = workflow_id
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
    if line := ret.get("line"):
        ret["line"] = int(line)
    if time := ret.get("time"):
        ret["time"] = float(time)
    if timestamp := ret.get("timestamp"):
        # Timestamps reported are not valid ISO8601 because they have no timezone. Add one.
        # This assumes that
        ret["timestamp"] = (
            datetime.datetime.fromisoformat(timestamp).astimezone().isoformat()
        )

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
        # Special handling for comments.
        if child.tag is ET.Comment:  # type: ignore[comparison-overlap]
            ret["comment"] = child.text
        else:
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


def download_and_extract_artifact(artifact_name: Path, artifact_url: str) -> None:
    response = requests.get(artifact_url, headers=REQUEST_HEADERS)
    print(f"Downloading and extracting {artifact_name}")
    with open(artifact_name, "wb") as f:
        f.write(response.content)
    unzip(artifact_name)


def download_and_extract_s3_reports(workflow_run_id: int) -> None:
    bucket = S3_RESOURCE.Bucket("gha-artifacts")
    objs = bucket.objects.filter(
        Prefix=f"pytorch/pytorch/{workflow_run_id}/artifact/test-reports"
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
    args = parser.parse_args()

    print("mkdir: ", TEMP_DIR)
    TEMP_DIR.mkdir()
    print("cd to ", TEMP_DIR)
    os.chdir(TEMP_DIR)

    # Download and extract all the reports (both GHA and S3)
    download_and_extract_s3_reports(args.workflow_run_id)
    artifact_urls = get_artifact_urls(args.workflow_run_id)
    for name, url in artifact_urls.items():
        download_and_extract_artifact(Path(name), url)

    # Parse the reports and transform them to JSON
    test_cases = []
    for xml_report in Path(".").glob("**/*.xml"):
        test_cases.extend(parse_xml_report(xml_report, int(args.workflow_run_id)))

    # Write the JSON to rockset
    print(f"Writing {len(test_cases)} test cases to Rockset")
    client = rockset.Client(
        api_server="api.rs2.usw2.rockset.com", api_key=os.environ["ROCKSET_API_KEY"]
    )
    client.Collection.retrieve("test_run").add_docs(test_cases)
    print("Done!")
