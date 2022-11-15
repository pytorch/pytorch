import argparse
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Generator, Set, Tuple

from tools.stats.upload_stats_lib import (
    download_gha_artifacts,
    download_s3_artifacts,
    unzip,
    upload_to_s3,
)
from tools.stats.upload_test_stats import process_xml_element

TESTCASE_TAG = "testcase"
TARGET_WORKFLOW = "--rerun-disabled-tests"
SEPARATOR = ";"


def is_rerun_disabled_tests(root: ET.ElementTree) -> bool:
    """
    Check if the test report is coming from rerun_disabled_tests workflow
    """
    skipped = root.find(".//*skipped")
    return skipped is not None and TARGET_WORKFLOW in skipped.attrib.get("message", "")


def process_report(report: Path) -> Tuple[Set[str], Dict[str, Dict[str, int]]]:
    """
    Return a list of disabled tests that should be re-enabled and those that are still
    failing
    """
    root = ET.parse(report)

    # A test should be re-enable if it's green after rerunning in all platforms where it
    # is currently disabled
    success_tests: Set[str] = set()
    # Also want to keep num_red and num_green here for additional stats
    failure_tests: Dict[str, Dict[str, int]] = {}

    if not is_rerun_disabled_tests(root):
        return success_tests, failure_tests

    for test_case in root.iter(TESTCASE_TAG):
        parsed_test_case = process_xml_element(test_case)

        # Under --rerun-disabled-tests mode, a test is skipped when:
        # * it's skipped explicitly inside PyToch code
        # * it's skipped because it's a normal enabled test
        # * or it's falky (num_red > 0 and num_green > 0)
        # * or it's failing (num_red > 0 and num_green == 0)
        #
        # We care only about the latter two here
        skipped = parsed_test_case.get("skipped", None)
        if skipped and "num_red" not in skipped.get("message", ""):
            continue

        name = parsed_test_case.get("name", "")
        classname = parsed_test_case.get("classname", "")
        filename = parsed_test_case.get("file", "")

        if not name or not classname or not filename:
            continue

        disabled_test_id = SEPARATOR.join([name, classname, filename])
        # Under --rerun-disabled-tests mode, if a test is not skipped, it's counted
        # as a success. Otherwise, it's still flaky or failing
        if skipped:
            try:
                stats = json.loads(skipped.get("message", ""))
            except json.JSONDecodeError:
                stats = {}

            failure_tests[disabled_test_id] = {
                "num_green": stats.get("num_green", 0),
                "num_red": stats.get("num_red", 0),
            }
        else:
            success_tests.add(disabled_test_id)

    return success_tests, failure_tests


def get_test_reports(
    repo: str, workflow_run_id: int, workflow_run_attempt: int
) -> Generator[Path, None, None]:
    """
    Gather all the test reports from S3 and GHA. It is currently not possible to guess which
    test reports are from rerun_disabled_tests workflow because the name doesn't include the
    test config. So, all reports will need to be downloaded and examined
    """
    with TemporaryDirectory() as temp_dir:
        print("Using temporary directory:", temp_dir)
        os.chdir(temp_dir)

        artifact_paths = download_s3_artifacts(
            "test-reports", workflow_run_id, workflow_run_attempt
        )
        for path in artifact_paths:
            unzip(path)

        artifact_paths = download_gha_artifacts(
            "test-report", workflow_run_id, workflow_run_attempt
        )
        for path in artifact_paths:
            unzip(path)

        for report in Path(".").glob("**/*.xml"):
            yield report


def save_results(
    workflow_id: int,
    workflow_run_attempt: int,
    should_be_enabled_tests: Set[str],
    failure_tests: Dict[str, Dict[str, int]],
) -> None:
    """
    Save the result to S3, so it can go to Rockset
    """
    records = {}

    # Log the results
    print(f"The following {len(should_be_enabled_tests)} tests should be re-enabled:")

    for test_id in should_be_enabled_tests:
        name, classname, filename = test_id.split(SEPARATOR)

        # Follow flaky bot convention here, if that changes, this will also need to be updated
        disabled_test_name = f"{name} (__main__.{classname})"
        print(f"  {disabled_test_name} from {filename}")

        key = (
            workflow_id,
            workflow_run_attempt,
            name,
            classname,
            filename,
        )

        records[key] = {
            "workflow_id": workflow_id,
            "workflow_run_attempt": workflow_run_attempt,
            "name": name,
            "classname": classname,
            "filename": filename,
            "flaky": False,
            "num_red": 0,
            "num_green": 0,
        }

    # Log the results
    print(f"The following {len(failure_tests)} are still flaky:")

    for test_id, stats in failure_tests.items():
        name, classname, filename = test_id.split(SEPARATOR)

        num_red = stats["num_red"]
        num_green = stats["num_green"]

        # Follow flaky bot convention here, if that changes, this will also need to be updated
        disabled_test_name = f"{name} (__main__.{classname})"
        print(
            f"  {disabled_test_name} from {filename}, failing {num_red}/{num_red + num_green}"
        )

        key = (
            workflow_id,
            workflow_run_attempt,
            name,
            classname,
            filename,
        )

        records[key] = {
            "workflow_id": workflow_id,
            "workflow_run_attempt": workflow_run_attempt,
            "name": name,
            "classname": classname,
            "filename": filename,
            "flaky": True,
            "num_red": num_red,
            "num_green": num_green,
        }

    upload_to_s3(
        workflow_id,
        workflow_run_attempt,
        "rerun_disabled_tests",
        list(records.values()),
    )


def main(repo: str, workflow_run_id: int, workflow_run_attempt: int) -> None:
    """
    Find the list of all disabled tests that should be re-enabled
    """
    success_tests: Set[str] = set()
    # Also want to keep num_red and num_green here for additional stats
    failure_tests: Dict[str, Dict[str, int]] = {}

    for report in get_test_reports(
        args.repo, args.workflow_run_id, args.workflow_run_attempt
    ):
        success, failure = process_report(report)

        # A test should be re-enable if it's green after rerunning in all platforms where it
        # is currently disabled. So they all need to be aggregated here
        success_tests.update(success)

        for name, stats in failure.items():
            if name not in failure_tests:
                failure_tests[name] = stats.copy()
            else:
                failure_tests[name]["num_green"] += stats["num_green"]
                failure_tests[name]["num_red"] += stats["num_red"]

    should_be_enabled_tests = success_tests.difference(set(failure_tests.keys()))
    save_results(
        workflow_run_id, workflow_run_attempt, should_be_enabled_tests, failure_tests
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload test artifacts from GHA to S3")
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
        "--repo",
        type=str,
        required=True,
        help="which GitHub repo this workflow run belongs to",
    )

    args = parser.parse_args()
    main(args.repo, args.workflow_run_id, args.workflow_run_attempt)
