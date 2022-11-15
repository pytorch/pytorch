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
)
from tools.stats.upload_test_stats import process_xml_element

TESTCASE_TAG = "testcase"
TARGET_WORKFLOW = "--rerun-disabled-tests"


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

        # Follow flaky bot convention here, if that changes, this will also need to be updated
        name = parsed_test_case.get("name", "")
        classname = parsed_test_case.get("classname", "")

        if not name or not classname:
            continue

        disabled_test_name = f"{name} (__main__.{classname})"
        # Under --rerun-disabled-tests mode, if a test is not skipped, it's counted
        # as a success. Otherwise, it's still flaky or failing
        if skipped:
            try:
                stats = json.loads(skipped.get("message", ""))
            except json.JSONDecodeError:
                stats = {}

            failure_tests[disabled_test_name] = {
                "num_green": stats.get("num_green", 0),
                "num_red": stats.get("num_red", 0),
            }
        else:
            success_tests.add(disabled_test_name)

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
        # os.chdir(temp_dir)
        os.chdir("/tmp/debug")

        # artifact_paths = download_s3_artifacts(
        #    "test-reports", workflow_run_id, workflow_run_attempt
        # )

        artifact_paths = [
            Path("/tmp/debug/test-reports-bazel-build-and-test_9441735758.zip"),
            Path(
                "/tmp/debug/test-reports-test-backwards_compat-1-1-linux.2xlarge_9441892954.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-backwards_compat-1-1-linux.2xlarge_9441893058.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-crossref-1-2-linux.2xlarge_9441896118.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-crossref-1-2-linux.2xlarge_9441896255.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-crossref-2-2-linux.2xlarge_9441896370.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-crossref-2-2-linux.2xlarge_9441896508.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-default-1-1-linux.2xlarge_9441901060.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-default-1-1-linux.2xlarge_9441901149.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-default-1-2-linux.2xlarge_9441891096.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-default-1-2-linux.2xlarge_9441891275.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-default-1-2-linux.2xlarge_9441895683.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-default-1-2-linux.2xlarge_9441895819.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-default-1-2-linux.2xlarge_9441903236.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-default-1-2-linux.2xlarge_9441903376.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-default-1-2-windows.4xlarge_9442208211.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-default-1-2-windows.4xlarge_9442208355.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-default-1-4-linux.4xlarge.nvidia.gpu_9442047237.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-default-1-4-linux.4xlarge.nvidia.gpu_9442047397.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-default-1-5-linux.2xlarge_9441949612.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-default-1-5-linux.2xlarge_9441949809.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-default-2-2-linux.2xlarge_9441891413.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-default-2-2-linux.2xlarge_9441891536.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-default-2-2-linux.2xlarge_9441895929.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-default-2-2-linux.2xlarge_9441896020.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-default-2-2-linux.2xlarge_9441903536.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-default-2-2-linux.2xlarge_9441903655.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-default-2-2-windows.4xlarge_9442208442.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-default-2-2-windows.4xlarge_9442208547.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-default-2-4-linux.4xlarge.nvidia.gpu_9442047506.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-default-2-4-linux.4xlarge.nvidia.gpu_9442047605.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-default-2-5-linux.2xlarge_9441949923.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-default-2-5-linux.2xlarge_9441950040.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-default-3-4-linux.4xlarge.nvidia.gpu_9442047722.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-default-3-4-linux.4xlarge.nvidia.gpu_9442047806.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-default-3-5-linux.2xlarge_9441950164.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-default-3-5-linux.2xlarge_9441950268.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-default-4-4-linux.4xlarge.nvidia.gpu_9442047923.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-default-4-4-linux.4xlarge.nvidia.gpu_9442048033.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-default-4-5-linux.4xlarge_9441950431.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-default-4-5-linux.4xlarge_9441950580.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-default-5-5-linux.2xlarge_9441950713.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-default-5-5-linux.2xlarge_9441950903.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-deploy-1-1-linux.4xlarge.nvidia.gpu_9442049116.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-deploy-1-1-linux.4xlarge.nvidia.gpu_9442049384.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-distributed-1-2-linux.2xlarge_9441891653.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-distributed-1-2-linux.2xlarge_9441891798.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-distributed-1-3-linux.8xlarge.nvidia.gpu_9442048133.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-distributed-1-3-linux.8xlarge.nvidia.gpu_9442048240.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-distributed-2-2-linux.2xlarge_9441891930.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-distributed-2-2-linux.2xlarge_9441892035.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-distributed-2-3-linux.8xlarge.nvidia.gpu_9442048319.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-distributed-2-3-linux.8xlarge.nvidia.gpu_9442048398.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-distributed-3-3-linux.8xlarge.nvidia.gpu_9442048489.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-distributed-3-3-linux.8xlarge.nvidia.gpu_9442048626.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-docs_test-1-1-linux.2xlarge_9441892415.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-docs_test-1-1-linux.2xlarge_9441892548.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-dynamo-1-2-linux.2xlarge_9441896601.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-dynamo-1-2-linux.2xlarge_9441896728.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-dynamo-2-2-linux.2xlarge_9441896852.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-dynamo-2-2-linux.2xlarge_9441897050.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-functorch-1-1-linux.2xlarge_9441892175.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-functorch-1-1-linux.2xlarge_9441892304.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-functorch-1-1-linux.2xlarge_9441897204.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-functorch-1-1-linux.2xlarge_9441897312.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-functorch-1-1-linux.2xlarge_9441951080.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-functorch-1-1-linux.2xlarge_9441951227.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-functorch-1-1-linux.4xlarge.nvidia.gpu_9442048777.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-functorch-1-1-linux.4xlarge.nvidia.gpu_9442048991.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-functorch-1-1-windows.4xlarge_9442208676.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-functorch-1-1-windows.4xlarge_9442208799.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-jit_legacy-1-1-linux.2xlarge_9441892696.zip"
            ),
            Path(
                "/tmp/debug/test-reports-test-jit_legacy-1-1-linux.2xlarge_9441892816.zip"
            ),
            Path("/tmp/debug/test-reports-test-xla-1-1-linux.2xlarge_9441962243.zip"),
            Path("/tmp/debug/test-reports-test-xla-1-1-linux.2xlarge_9441962400.zip"),
        ]

        for path in artifact_paths:
            unzip(path)

        # artifact_paths = download_gha_artifacts(
        #     "test-report", workflow_run_id, workflow_run_attempt
        # )
        # for path in artifact_paths:
        #    unzip(path)

        for report in Path(".").glob("**/*.xml"):
            yield report


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

        if failure:
            print(f"FOUND {len(failure)} FLAKY TESTS on {report}")

        for name, stats in failure.items():
            if name not in failure_tests:
                failure_tests[name] = stats.copy()
            else:
                failure_tests[name]["num_green"] += stats["num_green"]
                failure_tests[name]["num_red"] += stats["num_red"]

    should_be_enabled_tests = success_tests.difference(set(failure_tests.keys()))

    # Log the result
    print(f"The following {len(should_be_enabled_tests)} tests should be re-enabled:")
    for name in should_be_enabled_tests:
        print(f"  {name}")

    print(f"The following {len(failure_tests)} are still flaky:")
    for name, stats in failure_tests.items():
        num_red = stats["num_red"]
        num_green = stats["num_green"]
        total = num_red + num_green

        percent = "N/A"
        if total:
            percent = str(int(num_red * 100 / total))

        print(f"  {name}: failing {num_red}/{total} ({percent}%)")

    # upload_to_s3(
    #    args.workflow_run_id,
    #    args.workflow_run_attempt,
    #    "test_run_summary",
    #    test_case_summary,
    # )


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
