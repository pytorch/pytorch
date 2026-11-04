from __future__ import annotations

import json
import os
import re
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
from tools.testing.target_determination.gen_artifact import (
    LLM_SELECTION_SOURCE,
    TD_SOURCE,
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
) -> dict[str, dict[str, Any]]:
    """The test files TD's split excluded (TD_SOURCE) and, for every config
    with a call where the LLM selection made the split, the files that some
    call of that config excluded and no call of it ran whole
    (LLM_SELECTION_SOURCE), per build and test config. Such a config is listed
    even if it excluded no file."""
    with TemporaryDirectory() as temp_dir:
        print("Using temporary directory:", temp_dir)
        os.chdir(temp_dir)

        # Download and extract all the reports (both GHA and S3)
        s3_paths = download_s3_artifacts(
            "test-jsons", workflow_run_id, workflow_run_attempt
        )
        for path in s3_paths:
            unzip(path)

        grouped_tests: dict[str, Any] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(set))
        )
        excluded_by_config: dict[tuple[str, str], set[str]] = defaultdict(set)
        ran: dict[tuple[str, str], set[str]] = defaultdict(set)
        enforced: set[tuple[str, str]] = set()
        # Calls that excluded nothing matter only for what they ran, so their
        # jobs are looked up only if they ran a file an enforced config excluded.
        ran_only: list[tuple[Path, list[str]]] = []
        for td_exclusions in Path(".").glob("**/td_exclusions*.json"):
            with open(td_exclusions) as f:
                exclusions = json.load(f)
            excluded = exclusions["excluded"]
            included = [x["test_file"] for x in exclusions["included"]]
            # Older artifacts hold only TD's split.
            td_excluded = exclusions.get("td_excluded", excluded)
            is_llm = exclusions.get("source") == LLM_SELECTION_SOURCE
            if not (excluded or td_excluded or is_llm):
                ran_only.append((td_exclusions, included))
                continue
            job_name = get_job_name(get_job_id(td_exclusions))
            build_name = get_build_name(job_name)
            test_config = get_test_config(job_name)
            if td_excluded:
                tests = grouped_tests[TD_SOURCE][build_name][test_config]
                tests.update(x["test_file"] for x in td_excluded)
            config = build_name, test_config
            excluded_files = {x["test_file"] for x in excluded}
            excluded_by_config[config].update(excluded_files)
            # A file TD split by class ran only if none of its runs were excluded.
            ran[config].update(f for f in included if f not in excluded_files)
            if is_llm:
                enforced.add(config)
        watched = set().union(*(excluded_by_config[c] for c in enforced))
        for path, included in ran_only:
            if watched.isdisjoint(included):
                continue
            job_name = get_job_name(get_job_id(path))
            ran[get_build_name(job_name), get_test_config(job_name)].update(included)

        for build_name, test_config in enforced:
            llm = grouped_tests[LLM_SELECTION_SOURCE][build_name]
            config = build_name, test_config
            llm[test_config] = excluded_by_config[config] - ran[config]
        for by_build in grouped_tests.values():
            for build in by_build.values():
                for test_config, test_files in build.items():
                    build[test_config] = sorted(test_files)
        return grouped_tests


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


def upload_additional_info(workflow_run_id: int, workflow_run_attempt: int) -> None:
    exclusions = get_td_exclusions(workflow_run_id, workflow_run_attempt)

    upload_workflow_stats_to_s3(
        workflow_run_id,
        workflow_run_attempt,
        "additional_info/td_exclusions",
        [exclusions.get(TD_SOURCE, {})],
    )
    if LLM_SELECTION_SOURCE in exclusions:
        upload_workflow_stats_to_s3(
            workflow_run_id,
            workflow_run_attempt,
            "additional_info/llm_selection_exclusions",
            [exclusions[LLM_SELECTION_SOURCE]],
        )
