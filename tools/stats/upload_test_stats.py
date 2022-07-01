import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any
from tempfile import TemporaryDirectory

from tools.stats.upload_stats_lib import (
    download_gha_artifacts,
    download_s3_artifacts,
    upload_to_s3,
    unzip,
)


def parse_xml_report(
    tag: str,
    report: Path,
    workflow_id: int,
    workflow_run_attempt: int,
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
        for xml_report in Path(".").glob("**/*.xml"):
            test_cases.extend(
                parse_xml_report(
                    "testcase",
                    xml_report,
                    workflow_run_id,
                    workflow_run_attempt,
                )
            )

        return test_cases


def summarize_test_cases(test_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group test cases by classname, file, and job_id. We perform the aggregation
    manually instead of using the `test-suite` XML tag because xmlrunner does
    not produce reliable output for it.
    """

    def get_key(test_case: Dict[str, Any]) -> Any:
        return (
            test_case.get("file"),
            test_case.get("classname"),
            test_case["job_id"],
            test_case["workflow_id"],
            test_case["workflow_run_attempt"],
        )

    def init_value(test_case: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "file": test_case.get("file"),
            "classname": test_case.get("classname"),
            "job_id": test_case["job_id"],
            "workflow_id": test_case["workflow_id"],
            "workflow_run_attempt": test_case["workflow_run_attempt"],
            "tests": 0,
            "failures": 0,
            "errors": 0,
            "skipped": 0,
            "successes": 0,
            "time": 0.0,
        }

    ret = {}
    for test_case in test_cases:
        key = get_key(test_case)
        if key not in ret:
            ret[key] = init_value(test_case)

        ret[key]["tests"] += 1

        if "failure" in test_case:
            ret[key]["failures"] += 1
        elif "error" in test_case:
            ret[key]["errors"] += 1
        elif "skipped" in test_case:
            ret[key]["skipped"] += 1
        else:
            ret[key]["successes"] += 1

        ret[key]["time"] += test_case["time"]
    return list(ret.values())


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
    parser.add_argument(
        "--head-branch",
        required=True,
        help="Head branch of the workflow",
    )
    args = parser.parse_args()
    test_cases = get_tests(args.workflow_run_id, args.workflow_run_attempt)

    # Flush stdout so that any errors in rockset upload show up last in the logs.
    sys.stdout.flush()

    # For PRs, only upload a summary of test_runs. This helps lower the
    # volume of writes we do to Rockset.
    upload_to_s3(
        args.workflow_run_id,
        args.workflow_run_attempt,
        "test_run_summary",
        summarize_test_cases(test_cases),
    )

    # upload_to_rockset("test_run_summary", summarize_test_cases(test_cases))

    if args.head_branch == "master":
        # For master jobs, upload everytihng.
        upload_to_s3(
            args.workflow_run_id, args.workflow_run_attempt, "test_run", test_cases
        )
