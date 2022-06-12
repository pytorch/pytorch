import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from tempfile import TemporaryDirectory

from tools.stats.upload_stats_lib import (
    download_gha_artifacts,
    download_s3_artifacts,
    upload_to_rockset,
    unzip,
)


def parse_xml_report(
    tag: str,
    report: Path,
    workflow_id: int,
    workflow_run_attempt: int,
    skip_tag: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Convert a test report xml file into a JSON-serializable list of test cases."""
    print(f"Parsing {tag}s for test report: {report}")
    # [Job id in artifacts]
    # Retrieve the job id from the report path. In our GHA workflows, we append
    # the job id to the end of the report name, so `report` looks like:
    #     unzipped-test-reports-foo_5596745227/test/test-reports/foo/TEST-foo.xml
    # and we want to get `5596745227` out of it.
    job_id = int(report.parts[0].rpartition("_")[2])
    print(f"Found job id: {job_id}")

    root = ET.parse(report)

    test_cases = []
    for test_case in root.iter(tag):
        case = process_xml_element(test_case, skip_tag)
        case["workflow_id"] = workflow_id
        case["workflow_run_attempt"] = workflow_run_attempt
        case["job_id"] = job_id
        test_cases.append(case)

    return test_cases


def process_xml_element(element: ET.Element, skip_tag: Optional[str]) -> Dict[str, Any]:
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
        if child.tag == skip_tag:
            continue

        if child.tag not in ret:
            ret[child.tag] = process_xml_element(child, skip_tag)
        else:
            # If there are multiple tags with the same name, they should be
            # coalesced into a list.
            if not isinstance(ret[child.tag], list):
                ret[child.tag] = [ret[child.tag]]
            ret[child.tag].append(process_xml_element(child, skip_tag))
    return ret


def get_tests(
    workflow_run_id: int, workflow_run_attempt: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    with TemporaryDirectory() as temp_dir:
        print("Using temporary directory:", temp_dir)
        os.chdir(temp_dir)

        # Download and extract all the reports (both GHA and S3)
        s3_paths = download_s3_artifacts(
            "test-report", workflow_run_id, workflow_run_attempt
        )
        for path in s3_paths:
            unzip(path)

        artifact_paths = download_gha_artifacts(
            "test-report", workflow_run_id, workflow_run_attempt
        )
        for path in artifact_paths:
            unzip(path)

        # Parse the reports and transform them to JSON
        test_cases = []
        test_suites = []
        for xml_report in Path(".").glob("**/*.xml"):
            test_cases.extend(
                parse_xml_report(
                    "testcase",
                    xml_report,
                    workflow_run_id,
                    workflow_run_attempt,
                )
            )
            test_suites.extend(
                parse_xml_report(
                    "testsuite",
                    xml_report,
                    workflow_run_id,
                    workflow_run_attempt,
                    skip_tag="testcase",
                )
            )

        return test_cases, test_suites


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload test stats to Rockset")
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
    test_cases, test_suites = get_tests(args.workflow_run_id, args.workflow_run_attempt)
    upload_to_rockset("test_run", test_cases)
    upload_to_rockset("test_suite", test_suites)
