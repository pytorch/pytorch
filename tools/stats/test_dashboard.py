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


def group_test_cases(test_cases: list[dict[str, Any]]) -> dict[str, Any]:
    start = time.time()
    grouped_tests: dict[str, Any] = defaultdict(
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
        grouped_tests[build_name][test_config][invoking_file][class_name][name].append(
            test_case
        )

    print(f"Time taken to group tests: {time.time() - start}")
    return grouped_tests


def get_reruns(grouped_tests: dict[str, Any]) -> dict[str, Any]:
    reruns: dict[str, Any] = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        )
    )
    for build_name, build in grouped_tests.items():
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


def get_invoking_file_summary(grouped_tests: dict[str, Any]) -> dict[str, Any]:
    invoking_file_summary: dict[str, Any] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: {"count": 0, "time": 0.0}))
    )
    for build_name, build in grouped_tests.items():
        for test_config, test_config_data in build.items():
            for invoking_file, invoking_file_data in test_config_data.items():
                for class_data in invoking_file_data.values():
                    for test_data in class_data.values():
                        invoking_file_summary[build_name][test_config][invoking_file][
                            "count"
                        ] += 1
                        for i in test_data:
                            invoking_file_summary[build_name][test_config][
                                invoking_file
                            ]["time"] += i["time"]

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
