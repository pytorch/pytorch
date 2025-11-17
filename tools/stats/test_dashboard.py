from __future__ import annotations

import json
import os
import re
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import requests

from tools.stats.upload_stats_lib import (
    _get_request_headers,
    download_s3_artifacts,
    get_job_id,
    get_s3_resource,
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


def get_td_exclusions(
    workflow_run_id: int, workflow_run_attempt: int
) -> dict[str, Any]:
    with TemporaryDirectory() as temp_dir:
        print("Using temporary directory:", temp_dir)
        os.chdir(temp_dir)

        # Download and extract all the reports (both GHA and S3)
        s3_paths = download_s3_artifacts(
            "test-jsons", workflow_run_id, workflow_run_attempt
        )
        for path in s3_paths:
            unzip(path)

        grouped_tests: dict[str, Any] = defaultdict(lambda: defaultdict(set))
        for td_exclusions in Path(".").glob("**/td_exclusions*.json"):
            with open(td_exclusions) as f:
                exclusions = json.load(f)
                for exclusion in exclusions["excluded"]:
                    job_id = get_job_id(td_exclusions)
                    job_name = get_job_name(job_id)
                    build_name = get_build_name(job_name)
                    test_config = get_test_config(job_name)
                    grouped_tests[build_name][test_config].add(exclusion["test_file"])

        for build_name, build in grouped_tests.items():
            for test_config, test_files in build.items():
                grouped_tests[build_name][test_config] = sorted(test_files)
        return grouped_tests


def group_test_cases(test_cases: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    # Returns a list of lists. Each inner list contains test cases with the same
    # build name, test config, file name, class name, and test name (ex if it was run multiple times)
    start = time.time()
    test_case_with_job_info = defaultdict(list)

    for test_case in test_cases:
        job_name = get_job_name(test_case["job_id"])
        build_name = get_build_name(job_name)
        if "bazel" in build_name:
            continue
        test_config = get_test_config(job_name)

        test_case["job_name"] = job_name
        test_case["build_name"] = build_name
        test_case["test_config"] = test_config

        key = (
            build_name,
            test_config,
            test_case.get("file", "NoFile"),
            test_case.get("classname", "NoClass"),
            test_case.get("name", "NoName"),
        )
        test_case_with_job_info[key].append(test_case)

    print(f"Time taken to group tests: {time.time() - start}")
    return list(test_case_with_job_info.values())


def get_reruns(grouped_tests: list[list[dict[str, Any]]]) -> dict[str, Any]:
    reruns: dict[str, Any] = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        )
    )

    for tests in grouped_tests:
        if len(tests) > 1:
            build_name = tests[0]["build_name"]
            test_config = tests[0]["test_config"]
            file = tests[0].get("file", "NoFile")
            class_name = tests[0].get("classname", "NoClass")
            test_name = tests[0].get("name", "NoName")
            if file in (
                "distributed/test_distributed_spawn.py",
                "onnx/test_fx_to_onnx_with_onnxruntime.py",
                "distributed/algorithms/quantization/test_quantization.py",
            ):
                continue
            reruns[build_name][test_config][file][class_name][test_name] = tests

    return reruns


def get_invoking_file_summary(
    grouped_tests: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    summary_flat = {}
    for tests in grouped_tests:
        build_name = tests[0]["build_name"]
        test_config = tests[0]["test_config"]
        short_job_name = f"{build_name} / test ({test_config})"
        file = tests[0].get("file", "NoFile")

        key = (build_name, test_config, file)
        if key not in summary_flat:
            summary_flat[key] = {
                "count": 0,
                "time": 0.0,
                "skipped": 0,
                "failures": 0,
                "errors": 0,
                "successes": 0,
                "short_job_name": short_job_name,
                "file": file,
            }
        summary_flat[key]["count"] += 1
        status = "successes"
        for test in tests:
            summary_flat[key]["time"] += test["time"]
            if "skipped" in test:
                status = "skipped"
            elif "failure" in test:
                status = "failures"
            elif "error" in test:
                status = "errors"
        summary_flat[key][status] += 1

    invoking_file_summary: dict[str, Any] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: {"count": 0, "time": 0.0}))
    )

    for (build_name, test_config, file), data in summary_flat.items():
        invoking_file_summary[build_name][test_config][file] = data
    return invoking_file_summary


def get_all_run_attempts(workflow_run_id: int) -> list[int]:
    # Returns all run attempts for a given workflow run id that have test
    # artifacts
    bucket = get_s3_resource().Bucket("gha-artifacts")
    prefix = f"pytorch/pytorch/{workflow_run_id}/"
    objs = bucket.objects.filter(Prefix=prefix)
    run_attempts = set()
    for obj in objs:
        no_prefix = obj.key[len(prefix) :]
        try:
            run_attempt = int(no_prefix.split("/")[0])
            run_attempts.add(run_attempt)
        except ValueError:
            continue
    return sorted(run_attempts)


def get_test_status(test_cases: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    # Returns a list of dicts with test status info (flaky, success, failure,
    # skipped)
    only_status_info = []
    for tests in test_cases:
        build_name = tests[0]["build_name"]
        test_config = tests[0]["test_config"]
        short_job_name = f"{build_name} / test ({test_config})"
        file = tests[0].get("file", "NoFile")

        statuses = []
        for test in tests:
            if "skipped" in test:
                statuses.append("skipped")
            elif "failure" in test or "error" in test:
                statuses.append("failure")
            else:
                statuses.append("success")
        if "failure" in statuses and "success" in statuses:
            status = "flaky"
        else:
            status = statuses[0]

        only_status_info.append(
            {
                "short_job_name": short_job_name,
                "file": file,
                "name": test["name"],
                "status": status,
            }
        )

    return only_status_info


def upload_additional_info(
    workflow_run_id: int, workflow_run_attempt: int, test_cases: list[dict[str, Any]]
) -> None:
    grouped_tests = group_test_cases(test_cases)
    reruns = get_reruns(grouped_tests)
    exclusions = get_td_exclusions(workflow_run_id, workflow_run_attempt)
    invoking_file_summary = get_invoking_file_summary(grouped_tests)

    upload_workflow_stats_to_s3(
        workflow_run_id,
        workflow_run_attempt,
        "additional_info/reruns",
        [reruns],
    )
    upload_workflow_stats_to_s3(
        workflow_run_id,
        workflow_run_attempt,
        "additional_info/td_exclusions",
        [exclusions],
    )
    upload_workflow_stats_to_s3(
        workflow_run_id,
        workflow_run_attempt,
        "additional_info/invoking_file_summary",
        [invoking_file_summary],
    )
    upload_workflow_stats_to_s3(
        workflow_run_id,
        workflow_run_attempt,
        "additional_info/test_status",
        get_test_status(grouped_tests),
    )
