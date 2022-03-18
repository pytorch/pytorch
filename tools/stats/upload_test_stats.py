import argparse
import os
import requests
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict

import rockset
import boto3  # type: ignore[import]

PYTORCH_REPO = "https://api.github.com/repos/pytorch/pytorch"
GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
REQUEST_HEADERS = {
    "Accept": "application/vnd.github.v3+json",
    "Authorization": "token " + GITHUB_TOKEN,
}
S3_RESOURCE = boto3.resource("s3")


def parse_xml_report(report: Path, workflow_id: int):
    """Convert a test report xml file into a JSON-serializable list of test cases."""
    job_id = report.stem.rpartition("_")[0]
    print(f"Parsing test report: {report}, id: {job_id}")

    # We tack the job name onto the end in upload-test-artifacts
    root = ET.parse(
        report,
        ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)),  # type: ignore [call-arg]
    )

    test_cases = []
    for test_case in root.findall("testcase"):
        case = process_xml_element(test_case)
        case["workflow_id"] = workflow_id
        case["job_id"] = job_id
        test_cases.append(case)

    return test_cases


def process_xml_element(element):
    """Convert a test suite element into a JSON-serializable dict."""
    ret = {}

    # Convert attributes directly into dict elements.
    # e.g.
    #     <testcase name="test_foo" classname="test_bar"></testcase>
    # becomes:
    #     {"name": "test_foo", "classname": "test_bar"}
    ret.update(element.attrib)

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
        if child.tag is ET.Comment:
            ret["comment"] = child.text
        else:
            ret[child.tag] = process_xml_element(child)
    return ret


def get_artifact_urls(workflow_run_id) -> Dict[Path, str]:
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


def extract_job_test_reports(job_report_path):
    with zipfile.ZipFile(job_report_path, "r") as job_artifact_zip:
        p = Path(job_report_path)
        unzipped_report_path = Path(p.stem + "-unzipped")
        job_artifact_zip.extractall(unzipped_report_path)

        return unzipped_report_path.glob("**/*.xml")


def get_unzipped_path(p: Path) -> Path:
    """Return a unique path for unzipped stuff to go, based on the path provided.

    Looks like: /tmp/test-reports.zip -> /tmp/test-reports-unzipped/
    """
    return p.with_name(p.stem + "-unzipped")


def unzip_glob(p: Path):
    """Unzip all zip files, recursively through subdirs"""
    job_reports_paths = p.glob("**/*.zip")

    for job_reports_path in job_reports_paths:
        with zipfile.ZipFile(job_reports_path, "r") as job_artifact_zip:
            unzipped_report_path = get_unzipped_path(job_reports_path)
            job_artifact_zip.extractall(unzipped_report_path)


def download_and_extract_artifact(artifact_name: Path, artifact_url: str) -> Path:
    response = requests.get(artifact_url, headers=REQUEST_HEADERS)
    with open(artifact_name, "wb") as f:
        f.write(response.content)

    # The hierarchy looks like:
    # test-reports-workflow-artifact.zip
    # ├─ test-reports-job1.zip
    # └─ test-reports-job2.zip
    # Extract them all!
    with zipfile.ZipFile(artifact_name, "r") as artifact_zip:
        unzipped_artifact_path = get_unzipped_path(artifact_name)
        artifact_zip.extractall(unzipped_artifact_path)

        unzip_glob(unzipped_artifact_path)

    return unzipped_artifact_path


def download_and_extract_s3_reports(workflow_run_id):
    bucket = S3_RESOURCE.Bucket("gha-artifacts")
    objs = bucket.objects.filter(
        Prefix=f"pytorch/pytorch/{workflow_run_id}/artifact/test-reports"
    )

    for obj in objs:
        p = Path(obj.key).name
        with open(p, "wb") as f:
            f.write(obj.get()["Body"].read())

    unzip_glob(Path("."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload test stats to Rockset")
    parser.add_argument(
        "--workflow-run-id",
        required=True,
        help="id of the workflow to get artifacts from",
    )
    args = parser.parse_args()

    # Download and extract all the reports (both GHA and S3)
    download_and_extract_s3_reports(args.workflow_run_id)
    artifact_urls = get_artifact_urls(args.workflow_run_id)
    for name, url in artifact_urls.items():
        unzipped_path = download_and_extract_artifact(Path(name), url)

    # Parse the reports and transform them to JSON
    test_cases = []
    for xml_report in Path(".").glob("**/*.xml"):
        test_cases.extend(parse_xml_report(xml_report, args.workflow_run_id))

    # Write the JSON to rockset
    print(f"Writing {len(test_cases)} test reports to Rockset")
    client = rockset.Client(
        api_server="api.rs2.usw2.rockset.com", api_key=os.environ["ROCKSET_API_KEY"]
    )
    client.Collection.retrieve("test_run").add_docs(test_cases)
