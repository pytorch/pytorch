from __future__ import annotations

import argparse
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, TYPE_CHECKING

from tools.stats.upload_stats_lib import (
    download_s3_artifacts,
    is_rerun_disabled_tests,
    unzip,
    upload_workflow_stats_to_s3,
)
from tools.stats.upload_test_stats import process_xml_element


if TYPE_CHECKING:
    from collections.abc import Generator


TESTCASE_TAG = "testcase"
SEPARATOR = ";"


def _merge_code_skip(stats: dict[str, Any], reason: str) -> None:
    if not reason:
        return
    prev = stats.get("code_skip")
    stats["code_skip"] = reason if not prev else f"{prev}\n{reason}"


def _property_values(parsed_test_case: dict[str, Any], prop_name: str) -> list[str]:
    """Values of one JUnit <property name="..."> written by record_property."""
    props = parsed_test_case.get("properties")
    if isinstance(props, dict):
        nodes: list[Any] = [props]
    elif isinstance(props, list):
        nodes = props
    else:
        return []
    values: list[str] = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        raw = node.get("property")
        entries = raw if isinstance(raw, list) else [raw]
        for entry in entries:
            if not isinstance(entry, dict) or entry.get("name") != prop_name:
                continue
            value = entry.get("value")
            if value is None or value == "":
                continue
            values.append(str(value))
    return values


def process_report(
    report: Path,
) -> dict[str, dict[str, Any]]:
    """
    Return a list of disabled tests that should be re-enabled and those that are still
    flaky (failed or skipped)
    """
    root = ET.parse(report)

    # All rerun tests from a report are grouped here:
    #
    # * Success test should be re-enable if it's green after rerunning in all platforms
    #   where it is currently disabled
    # * Failures from pytest because pytest-flakefinder is used to run the same test
    #   multiple times, some could fails
    # * Skipped tests from unittest
    #
    # We want to keep track of how many times the test fails (num_red) or passes (num_green)
    all_tests: dict[str, dict[str, Any]] = {}

    for test_case in root.iter(TESTCASE_TAG):
        # Parse the test case as string values only.
        parsed_test_case = process_xml_element(test_case, output_numbers=False)

        # Under --rerun-disabled-tests mode, a test is skipped when:
        # * it's skipped explicitly inside PyTorch code
        # * it's skipped because it's a normal enabled test
        # * or it's falky (num_red > 0 and num_green > 0)
        # * or it's failing (num_red > 0 and num_green == 0)
        #
        # Skips whose message carries num_red are the enabled-test accounting
        # skip. Any other JUnit skip is a code skip: keep the reason text and
        # do not count it as a pass. A bypassed test that then passes or fails
        # is not a JUnit skip. record_property("code_skip", reason) shows up
        # as a <property> on that case: keep the reason and still count it.
        skipped = parsed_test_case.get("skipped", None)

        # NB: Regular ONNX tests could return a list of subskips here where each item in the
        # list is a skipped message.  In the context of rerunning disabled tests, we could
        # ignore this case as returning a list of subskips only happens when tests are run
        # normally
        if skipped and type(skipped) is list:
            continue

        name = parsed_test_case.get("name", "")
        classname = parsed_test_case.get("classname", "")
        filename = parsed_test_case.get("file", "")

        if not name or not classname or not filename:
            continue

        disabled_test_id = SEPARATOR.join([name, classname, filename])
        if disabled_test_id not in all_tests:
            all_tests[disabled_test_id] = {
                "num_green": 0,
                "num_red": 0,
            }

        skip_message = ""
        if isinstance(skipped, dict):
            skip_message = skipped.get("message", "")
        if skipped and "num_red" not in skip_message:
            _merge_code_skip(all_tests[disabled_test_id], skip_message)
            continue

        # Check if the test is a failure
        failure = parsed_test_case.get("failure", None)

        # Under --rerun-disabled-tests mode, if a test is not skipped or failed, it's
        # counted as a success. Otherwise, it's still flaky or failing
        if skipped:
            try:
                stats = json.loads(skip_message)
            except json.JSONDecodeError:
                stats = {}

            all_tests[disabled_test_id]["num_green"] += stats.get("num_green", 0)
            all_tests[disabled_test_id]["num_red"] += stats.get("num_red", 0)
        elif failure:
            # As a failure, increase the failure count
            all_tests[disabled_test_id]["num_red"] += 1
        else:
            all_tests[disabled_test_id]["num_green"] += 1

        # Accounting skips stay the JSON counts. A pass or fail that still
        # carries the original code-skip reason is counted above and kept.
        if not skipped:
            for reason in _property_values(parsed_test_case, "code_skip"):
                _merge_code_skip(all_tests[disabled_test_id], reason)

    return all_tests


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

        yield from Path(".").glob("**/*.xml")


def get_disabled_test_name(test_id: str) -> tuple[str, str, str, str]:
    """
    Follow flaky bot convention here, if that changes, this will also need to be updated
    """
    name, classname, filename = test_id.split(SEPARATOR)
    return f"{name} (__main__.{classname})", name, classname, filename


def prepare_record(
    workflow_id: int,
    workflow_run_attempt: int,
    name: str,
    classname: str,
    filename: str,
    flaky: bool,
    num_red: int = 0,
    num_green: int = 0,
) -> tuple[Any, dict[str, Any]]:
    """
    Prepare the record to save onto S3
    """
    key = (
        workflow_id,
        workflow_run_attempt,
        name,
        classname,
        filename,
    )

    record = {
        "workflow_id": workflow_id,
        "workflow_run_attempt": workflow_run_attempt,
        "name": name,
        "classname": classname,
        "filename": filename,
        "flaky": flaky,
        "num_green": num_green,
        "num_red": num_red,
    }

    return key, record


def _code_skip_text(stats: dict[str, Any]) -> str:
    reason = stats.get("code_skip")
    if not isinstance(reason, str):
        return ""
    return reason


def prepare_code_skip_record(
    workflow_id: int,
    workflow_run_attempt: int,
    name: str,
    classname: str,
    filename: str,
    num_green: int,
    num_red: int,
    code_skip: str,
) -> dict[str, Any]:
    """Code-skip row. No flaky key; ClickHouse reads that on the other collection."""
    return {
        "workflow_id": workflow_id,
        "workflow_run_attempt": workflow_run_attempt,
        "name": name,
        "classname": classname,
        "filename": filename,
        "num_green": num_green,
        "num_red": num_red,
        "code_skip": code_skip,
    }


def save_results(
    workflow_id: int,
    workflow_run_attempt: int,
    all_tests: dict[str, dict[str, Any]],
) -> None:
    """
    Save the result to S3, which then gets put into the HUD backend database.

    Rows with a code_skip reason go to rerun_disabled_code_skips and are not
    mixed into rerun_disabled_tests. A still-skipped code skip is 0/0; putting
    that in the flaky collection blocks the flaky bot.
    """
    counted_tests = {
        name: stats for name, stats in all_tests.items() if not _code_skip_text(stats)
    }
    code_skip_tests = {
        name: stats for name, stats in all_tests.items() if _code_skip_text(stats)
    }

    should_be_enabled_tests = {
        name: stats
        for name, stats in counted_tests.items()
        if "num_green" in stats
        and stats["num_green"]
        and "num_red" in stats
        and stats["num_red"] == 0
    }
    still_flaky_tests = {
        name: stats
        for name, stats in counted_tests.items()
        if name not in should_be_enabled_tests
    }

    records = {}
    for test_id, stats in counted_tests.items():
        num_green = stats.get("num_green", 0)
        num_red = stats.get("num_red", 0)
        name, classname, filename = get_disabled_test_name(test_id)[1:]

        key, record = prepare_record(
            workflow_id=workflow_id,
            workflow_run_attempt=workflow_run_attempt,
            name=name,
            classname=classname,
            filename=filename,
            flaky=test_id in still_flaky_tests,
            num_green=num_green,
            num_red=num_red,
        )
        records[key] = record

    code_records = []
    n_passed = 0
    n_failed = 0
    n_still_skipped = 0
    for test_id, stats in code_skip_tests.items():
        num_green = stats.get("num_green", 0)
        num_red = stats.get("num_red", 0)
        name, classname, filename = get_disabled_test_name(test_id)[1:]
        code_records.append(
            prepare_code_skip_record(
                workflow_id=workflow_id,
                workflow_run_attempt=workflow_run_attempt,
                name=name,
                classname=classname,
                filename=filename,
                num_green=num_green,
                num_red=num_red,
                code_skip=_code_skip_text(stats),
            )
        )
        if num_green > 0 and num_red == 0:
            n_passed += 1
        elif num_red > 0:
            n_failed += 1
        else:
            n_still_skipped += 1

    # Log the results
    print(f"The following {len(should_be_enabled_tests)} tests should be re-enabled:")
    for test_id in should_be_enabled_tests:
        disabled_test_name, name, classname, filename = get_disabled_test_name(test_id)
        print(f"  {disabled_test_name} from {filename}")

    print(f"The following {len(still_flaky_tests)} are still flaky:")
    for test_id, stats in still_flaky_tests.items():
        num_green = stats.get("num_green", 0)
        num_red = stats.get("num_red", 0)

        disabled_test_name, name, classname, filename = get_disabled_test_name(test_id)
        print(
            f"  {disabled_test_name} from {filename}, failing {num_red}/{num_red + num_green}"
        )

    print(
        "rerun_disabled_code_skips:"
        f" {n_passed} passed, {n_failed} failed, {n_still_skipped} still skipped"
    )

    rerun_docs = list(records.values())
    if rerun_docs:
        upload_workflow_stats_to_s3(
            workflow_id,
            workflow_run_attempt,
            "rerun_disabled_tests",
            rerun_docs,
        )
    if code_records:
        upload_workflow_stats_to_s3(
            workflow_id,
            workflow_run_attempt,
            "rerun_disabled_code_skips",
            code_records,
        )


def main(repo: str, workflow_run_id: int, workflow_run_attempt: int) -> None:
    """
    Find the list of all disabled tests that should be re-enabled
    """
    # Aggregated across all jobs
    all_tests: dict[str, dict[str, Any]] = {}

    for report in get_test_reports(
        args.repo, args.workflow_run_id, args.workflow_run_attempt
    ):
        tests = process_report(report)

        # The scheduled workflow has both rerun disabled tests and memory leak check jobs.
        # We are only interested in the former here
        if not is_rerun_disabled_tests(
            report, workflow_run_id, workflow_run_attempt, tests
        ):
            continue

        for name, stats in tests.items():
            if name not in all_tests:
                all_tests[name] = stats.copy()
            else:
                all_tests[name]["num_green"] += stats.get("num_green", 0)
                all_tests[name]["num_red"] += stats.get("num_red", 0)
                extra = stats.get("code_skip")
                if isinstance(extra, str):
                    _merge_code_skip(all_tests[name], extra)

    save_results(
        workflow_run_id,
        workflow_run_attempt,
        all_tests,
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
