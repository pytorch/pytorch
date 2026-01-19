from __future__ import annotations

import argparse
import os
import re
import sys
import xml.etree.ElementTree as ET
from multiprocessing import cpu_count, Pool
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from tools.stats.test_dashboard import upload_additional_info
from tools.stats.upload_stats_lib import (
    download_s3_artifacts,
    get_job_id,
    remove_nan_inf,
    unzip,
    upload_to_s3,
    upload_workflow_stats_to_s3,
)


def should_upload_full_test_run(head_branch: str | None, head_repository: str) -> bool:
    """Return True if we should upload the full test_run dataset.

    Rules:
    - Only for the main repository (pytorch/pytorch)
    - If head_branch is 'main', or a tag of form 'trunk/{40-hex-sha}'
    """
    is_trunk_tag = bool(re.fullmatch(r"trunk/[0-9a-fA-F]{40}", (head_branch or "")))
    return head_repository == "pytorch/pytorch" and (
        head_branch == "main" or is_trunk_tag
    )


def parse_xml_report(
    tag: str,
    report: Path,
    workflow_id: int,
    workflow_run_attempt: int,
    job_id: int | None = None,
) -> list[dict[str, Any]]:
    """Convert a test report xml file into a JSON-serializable list of test cases."""
    print(f"Parsing {tag}s for test report: {report}")

    if job_id is None:
        job_id = get_job_id(report)
        print(f"Found job id: {job_id}")

    test_cases: list[dict[str, Any]] = []

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


def process_xml_element(
    element: ET.Element, output_numbers: bool = True
) -> dict[str, Any]:
    """Convert a test suite element into a JSON-serializable dict."""
    ret: dict[str, Any] = {}

    # Convert attributes directly into dict elements.
    # e.g.
    #     <testcase name="test_foo" classname="test_bar"></testcase>
    # becomes:
    #     {"name": "test_foo", "classname": "test_bar"}
    ret.update(element.attrib)

    # The XML format encodes all values as strings. Convert to ints/floats if
    # possible to make aggregation possible in SQL.
    if output_numbers:
        for k, v in ret.items():
            try:
                ret[k] = int(v)
            except ValueError:
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


def get_tests(workflow_run_id: int, workflow_run_attempt: int) -> list[dict[str, Any]]:
    with TemporaryDirectory() as temp_dir:
        print("Using temporary directory:", temp_dir)
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
        for xml_report in Path(".").glob("**/*.xml"):
            test_cases.append(
                mp.apply_async(
                    parse_xml_report,
                    args=(
                        "testcase",
                        xml_report,
                        workflow_run_id,
                        workflow_run_attempt,
                    ),
                )
            )
        mp.close()
        mp.join()
        test_cases = [tc.get() for tc in test_cases]
        flattened = [item for sublist in test_cases for item in sublist]
        return flattened


def backfill_test_jsons_while_running(
    workflow_run_id: int, workflow_run_attempt: int
) -> None:
    # The bucket name name is a bit misleading, usually the jsons should be
    # uploaded while the job is running, but that won't happen if the job
    # doesn't have permissions to write to the bucket or if there was an error
    with TemporaryDirectory() as temp_dir:
        print("Using temporary directory:", temp_dir)
        os.chdir(temp_dir)

        # Download and extract all the reports (both GHA and S3)
        s3_xmls = download_s3_artifacts(
            "test-report", workflow_run_id, workflow_run_attempt
        )

        s3_jsons = download_s3_artifacts(
            "test-jsons", workflow_run_id, workflow_run_attempt
        )

        # Unzip artifacts and save their locations
        unzipped_xml_dirs = [unzip(path) for path in s3_xmls]
        unzipped_json_dirs = [unzip(path) for path in s3_jsons]

        all_existing_jsons = []
        for unzipped_dir in unzipped_json_dirs:
            all_existing_jsons.extend(
                [
                    str(Path(json_report).relative_to(unzipped_dir))
                    for json_report in unzipped_dir.glob("**/*.json")
                ]
            )

        for unzipped_dir in unzipped_xml_dirs:
            for xml in unzipped_dir.glob("**/*.xml"):
                corresponding_json = str(
                    xml.with_suffix(".json").relative_to(
                        unzipped_dir / "test" / "test-reports"
                    )
                )
                if corresponding_json in all_existing_jsons:
                    print(f"Skipping upload for existing test json for {xml}")
                    continue
                # print(f"Uploading missing test json for {xml}")
                job_id = get_job_id(xml)
                test_cases = parse_xml_report(
                    "testcase",
                    xml,
                    workflow_run_id,
                    workflow_run_attempt,
                    job_id,
                )
                json_file = xml.with_suffix(".json")
                s3_key = (
                    json_file.relative_to(unzipped_dir / "test" / "test-reports")
                    .as_posix()
                    .replace("/", "_")
                )
                s3_key = f"test_jsons_while_running/{workflow_run_id}/{job_id}/{s3_key}"
                upload_to_s3("gha-artifacts", s3_key, remove_nan_inf(test_cases))


def summarize_test_cases(test_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group test cases by classname, file, and job_id. We perform the aggregation
    manually instead of using the `test-suite` XML tag because xmlrunner does
    not produce reliable output for it.
    """

    def get_key(test_case: dict[str, Any]) -> Any:
        return (
            test_case.get("file"),
            test_case.get("classname"),
            test_case["job_id"],
            test_case["workflow_id"],
            test_case["workflow_run_attempt"],
            # [see: invoking file]
            test_case["invoking_file"],
        )

    def init_value(test_case: dict[str, Any]) -> dict[str, Any]:
        return {
            "file": test_case.get("file"),
            "classname": test_case.get("classname"),
            "job_id": test_case["job_id"],
            "workflow_id": test_case["workflow_id"],
            "workflow_run_attempt": test_case["workflow_run_attempt"],
            # [see: invoking file]
            "invoking_file": test_case["invoking_file"],
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
    parser = argparse.ArgumentParser(description="Upload test stats to s3")
    parser.add_argument(
        "--workflow-run-id",
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
    parser.add_argument(
        "--head-repository",
        required=True,
        help="Head repository of the workflow",
    )
    args = parser.parse_args()

    print(f"Workflow id is: {args.workflow_run_id}")

    test_cases = get_tests(args.workflow_run_id, args.workflow_run_attempt)

    # Flush stdout so that any errors in the upload show up last in the logs.
    sys.stdout.flush()

    # For PRs, only upload a summary of test_runs. This helps lower the
    # volume of writes we do to the HUD backend database.
    test_case_summary = summarize_test_cases(test_cases)

    upload_workflow_stats_to_s3(
        args.workflow_run_id,
        args.workflow_run_attempt,
        "test_run_summary",
        remove_nan_inf(test_case_summary),
    )

    # Separate out the failed test cases.
    # Uploading everything is too data intensive most of the time,
    # but these will be just a tiny fraction.
    failed_tests_cases = []
    for test_case in test_cases:
        if "rerun" in test_case or "failure" in test_case or "error" in test_case:
            failed_tests_cases.append(test_case)

    upload_workflow_stats_to_s3(
        args.workflow_run_id,
        args.workflow_run_attempt,
        "failed_test_runs",
        remove_nan_inf(failed_tests_cases),
    )

    backfill_test_jsons_while_running(args.workflow_run_id, args.workflow_run_attempt)

    # Upload full test_run only for trusted refs (main or trunk/{sha} tags)
    if should_upload_full_test_run(args.head_branch, args.head_repository):
        # For jobs on main branch, upload everything.
        upload_workflow_stats_to_s3(
            args.workflow_run_id,
            args.workflow_run_attempt,
            "test_run",
            remove_nan_inf(test_cases),
        )

    upload_additional_info(args.workflow_run_id, args.workflow_run_attempt)
