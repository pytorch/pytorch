import json
import os
import re
import time
from collections import defaultdict
import contextlib
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast, Dict, List

import requests

from tools.stats.upload_stats_lib import (
    _get_request_headers,
    download_gha_artifacts,
    download_s3_artifacts,
    get_job_id,
    unzip,
    upload_workflow_stats_to_s3,
)

REGEX_JOB_INFO = r"(.*) \/ .*test \(([^,]*), .*\)"


@lru_cache(maxsize=1000)
def get_job_name(job_id: int) -> str:
    try:
        return cast(
            str,
            requests.get(
                f"https://api.github.com/repos/pytorch/pytorch/actions/jobs/{job_id}",
                headers=_get_request_headers(),
            ).json()["name"],
        )
    except Exception as e:
        print(f"Failed to get job name for job id {job_id}: {e}")
        return "NoJobName"


@lru_cache(maxsize=1000)
def get_build_name(job_name: str) -> str:
    try:
        return re.match(REGEX_JOB_INFO, job_name).group(1)  # type: ignore[union-attr]
    except AttributeError:
        print(f"Failed to match job name: {job_name}")
        return "NoBuildEnv"


@lru_cache(maxsize=1000)
def get_test_config(job_name: str) -> str:
    try:
        return re.match(REGEX_JOB_INFO, job_name).group(2)  # type: ignore[union-attr]
    except AttributeError:
        print(f"Failed to match job name: {job_name}")
        return "NoTestConfig"


def get_tests(
    workflow_run_id: int, workflow_run_attempt: int
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    start = time.time()
    temp_dir = f"/Users/csl/zzzzzzzz/tmp/{workflow_run_id}"
    current_dir = os.getcwd()
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        print("Using temporary directory:", temp_dir)
        os.chdir(temp_dir)

        # Download and extract all the reports (both GHA and S3)
        s3_paths = download_s3_artifacts(
            "test-jsons", workflow_run_id, workflow_run_attempt
        )
        for path in s3_paths:
            unzip(path)

        artifact_paths = download_gha_artifacts(
            "test-jsons", workflow_run_id, workflow_run_attempt
        )
        for path in artifact_paths:
            unzip(path)
    if True:
        os.chdir(temp_dir)

        # Parse the reports and transform them to JSON
        test_cases = []
        mp = Pool(20)
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
        exclusions = get_td_exclusions()
        os.chdir(current_dir)
        print(f"Time taken to get tests: {time.time() - start}")

        return flattened, exclusions


def get_td_exclusions() -> Dict[str, Any]:
    grouped: Dict[str, Any] = defaultdict(lambda: defaultdict(set))
    for td_exclusions in Path(".").glob("**/td_exclusions.json"):
        with open(td_exclusions) as f:
            exclusions = json.load(f)
            for exclusion in exclusions["excluded"]:
                job_id = get_job_id(td_exclusions)
                job_name = get_job_name(job_id)
                build_name = get_build_name(job_name)
                test_config = get_test_config(job_name)
                grouped[build_name][test_config].add(exclusion["test_file"])

        for build_name, build in grouped.items():
            for test_config, test_files in build.items():
                grouped[build_name][test_config] = sorted(test_files)
        return grouped


def group_test_cases(test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    start = time.time()
    grouped: Dict[str, Any] = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        )
    )
    for test_case in test_cases:
        job_name = get_job_name(test_case["job_id"])
        build_name = get_build_name(job_name)
        if "bazel" in build_name:
            continue
        test_config = get_test_config(job_name)
        class_name = test_case.pop("classname", "NoClass")
        name = test_case.pop("name", "NoName")
        invoking_file = test_case.pop("invoking_file", "NoFile")
        invoking_file = invoking_file.replace(".", "/")
        test_case.pop("workflow_id")
        test_case.pop("workflow_run_attempt")
        grouped[build_name][test_config][invoking_file][class_name][name].append(
            test_case
        )

    print(f"Time taken to group tests: {time.time() - start}")
    return grouped


def get_reruns(grouped: Dict[str, Any]) -> Dict[str, Any]:
    reruns: Dict[str, Any] = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        )
    )
    for build_name, build in grouped.items():
        for test_config, test_config_data in build.items():
            for invoking_file, invoking_file_data in test_config_data.items():
                for class_name, class_data in invoking_file_data.items():
                    for test_name, test_data in class_data.items():
                        if len(test_data) > 1:
                            if invoking_file in (
                                "distributed/test_distributed_spawn",
                                "onnx/test_fx_to_onnx_with_onnxruntime",
                                "distributed/algorithms/quantization/test_quantization",
                            ):
                                continue
                            reruns[build_name][test_config][invoking_file][class_name][
                                test_name
                            ] = test_data
    return reruns


def get_builds_summary(grouped):
    builds_summary = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: {"count": 0, "time": 0.0}))
    )
    for build_name, build in grouped.items():
        for test_config, test_config_data in build.items():
            for invoking_file, invoking_file_data in test_config_data.items():
                for class_name, class_data in invoking_file_data.items():
                    for test_name, test_data in class_data.items():
                        builds_summary[build_name][test_config][invoking_file][
                            "count"
                        ] += 1
                        for i in test_data:
                            builds_summary[build_name][test_config][invoking_file][
                                "time"
                            ] += i["time"]

    return builds_summary


def compare_build(job_summary, base_job_summary, build_name):
    # Compare the two summaries for a single build

    build_summary = job_summary[build_name]
    base_build_summary = base_job_summary[build_name]

    return compare(build_summary, base_build_summary)


def compare_tests(tests, base_tests):
    diff = {}

    return diff


def get_new_removed_tests(grouped, base_grouped):
    def get_a_minus_b(a, b):
        if isinstance(a, list):
            if len(b) == 0:
                return {
                    "count": 1,
                }
            return {"count": 0}

        class Summary:
            def __init__(self):
                self.count = 0
                self.nodes = defaultdict(Summary)

            def toJSON(self):
                return {
                    "count": self.count,
                    "nodes": {key: value.toJSON() for key, value in self.nodes.items()},
                }

        diff = Summary()

        def count(obj):
            if isinstance(obj, dict):
                return sum(count(value) for value in obj.values())
            if isinstance(obj, list):
                return 1
            return 1

        for key in a:
            print(key)
            if key not in b:
                diff.nodes[key].count = count(a[key])
            else:
                small_diff = get_a_minus_b(a[key], b[key])
                print(small_diff)
                if small_diff["count"] > 0:
                    diff.nodes[key] = small_diff
        diff.count = sum(diff.nodes[key]["count"] for key in diff.nodes)
        return diff.toJSON()

    return get_a_minus_b(grouped, base_grouped), get_a_minus_b(base_grouped, grouped)


# def get_time_comparisons(grouped, base_grouped):
#     def get_all_keys(a, b):
#         all_keys = set(a.keys()) & set(b.keys())
#         for key in all_keys:
#             yield key, a.get(key, {}), b.get(key, {})
#     class Summary:
#         def __init__(self):
#             self.time = 0
#             self.count = 0

#     for build_name, build, base_build in get_all_keys(grouped, base_grouped):
#         for test_config_name, test_config, base_test_config in get_all_keys(build, base_build):
#             for invoking_file_name, invoking_file, base_invoking_file in get_all_keys(test_config, base_test_config):
#                 for class_name, class_data, base_class_data in get_all_keys(invoking_file, base_invoking_file):
#                     for test_name, test_data, base_test_data in get_all_keys(class_data, base_class_data):


def compare(job_summary, base_job_summary):
    # Compare the two summaries
    start = time.time()
    new, removed = get_new_removed_tests(job_summary, base_job_summary)
    print(f"Time taken to compare tests: {time.time() - start}")
    return {
        "new": new,
        "removed": removed,
    }


def get_parser():
    parser = argparse.ArgumentParser(description="Upload test stats to Rockset")
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
        "--base-workflow-run-id",
        type=int,
        help="id of the base workflow to get artifacts from",
    )

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()

    print(f"Workflow id is: {args.workflow_run_id}")

    test_cases, exclusions = get_tests(args.workflow_run_id, args.workflow_run_attempt)
    # upload_workflow_stats_to_s3(
    #     args.workflow_run_id,
    #     args.workflow_run_attempt,
    #     "additional_info/reruns",
    #     get_reruns(group_test_cases(test_cases)),
    # )
    # upload_workflow_stats_to_s3(
    #     args.workflow_run_id,
    #     args.workflow_run_attempt,
    #     "additional_info/td_exclusions",
    #     exclusions,
    # )
    with open("test/test-reports/a.json", "w") as f:
        print(json.dumps(exclusions, indent=2, sort_keys=True), file=f)
    with open("test/test-reports/b.json", "w") as f:
        print(json.dumps(get_reruns(group_test_cases(test_cases)), indent=2, sort_keys=True), file=f)

    # base_test_cases, exclusions = get_tests(args.base_workflow_run_id, 1)
    # job_summary = get_per_job_summary(test_cases)
    # base_job_summary = get_per_job_summary(base_test_cases)
    # builds_summary = get_builds_summary(job_summary)
    # print(len(test_cases))
    # # diff = compare(job_summary, base_job_summary)
    # with open("test/test-reports/a.json", "w") as f:
    #     print(json.dumps(builds_summary, indent=2), file=f)

    # with open("test/test-reports/b.json", "w") as f:
    #     print(json.dumps(get_builds_summary(base_job_summary), indent=2), file=f)

    # # Flush stdout so that any errors in Rockset upload show up last in the logs.
    # sys.stdout.flush()

# python -m tools.stats.upload_test_stats --workflow-run-id 8697202370 --workflow-run-attempt 1 --head-branch main --head-repo pytorch/pytorch
