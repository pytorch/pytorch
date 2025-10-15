#!/usr/bin/env python3

from __future__ import annotations

import datetime
import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable, cast
from urllib.request import urlopen


REPO_ROOT = Path(__file__).resolve().parents[2]


def get_disabled_issues() -> list[str]:
    reenabled_issues = os.getenv("REENABLED_ISSUES", "")
    issue_numbers = reenabled_issues.split(",")
    print("Ignoring disabled issues: ", issue_numbers)
    return issue_numbers


DISABLED_TESTS_FILE = ".pytorch-disabled-tests.json"
ADDITIONAL_CI_FILES_FOLDER = Path(".additional_ci_files")
TEST_TIMES_FILE = "test-times.json"
TEST_CLASS_TIMES_FILE = "test-class-times.json"
TEST_FILE_RATINGS_FILE = "test-file-ratings.json"
TEST_CLASS_RATINGS_FILE = "test-class-ratings.json"
TD_HEURISTIC_PROFILING_FILE = "td_heuristic_profiling.json"
TD_HEURISTIC_HISTORICAL_EDITED_FILES = "td_heuristic_historical_edited_files.json"
TD_HEURISTIC_PREVIOUSLY_FAILED = "previous_failures.json"
TD_HEURISTIC_PREVIOUSLY_FAILED_ADDITIONAL = "previous_failures_additional.json"

FILE_CACHE_LIFESPAN_SECONDS = datetime.timedelta(hours=3).seconds


def fetch_and_cache(
    dirpath: str | Path,
    name: str,
    url: str,
    process_fn: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    """
    This fetch and cache utils allows sharing between different process.
    """
    Path(dirpath).mkdir(exist_ok=True)

    path = os.path.join(dirpath, name)
    print(f"Downloading {url} to {path}")

    def is_cached_file_valid() -> bool:
        # Check if the file is new enough (see: FILE_CACHE_LIFESPAN_SECONDS). A real check
        # could make a HEAD request and check/store the file's ETag
        fname = Path(path)
        now = datetime.datetime.now()
        mtime = datetime.datetime.fromtimestamp(fname.stat().st_mtime)
        diff = now - mtime
        return diff.total_seconds() < FILE_CACHE_LIFESPAN_SECONDS

    if os.path.exists(path) and is_cached_file_valid():
        # Another test process already download the file, so don't re-do it
        with open(path) as f:
            return cast(dict[str, Any], json.load(f))

    for _ in range(3):
        try:
            contents = urlopen(url, timeout=5).read().decode("utf-8")
            processed_contents = process_fn(json.loads(contents))
            with open(path, "w") as f:
                f.write(json.dumps(processed_contents))
            return processed_contents
        except Exception as e:
            print(f"Could not download {url} because: {e}.")
    print(f"All retries exhausted, downloading {url} failed.")
    return {}


def get_test_times() -> dict[str, dict[str, float]]:
    return get_from_test_infra_generated_stats(
        "test-times.json",
        TEST_TIMES_FILE,
        "Couldn't download test times...",
    )


def get_test_class_times() -> dict[str, dict[str, float]]:
    return get_from_test_infra_generated_stats(
        "test-class-times.json",
        TEST_CLASS_TIMES_FILE,
        "Couldn't download test times...",
    )


def get_disabled_tests(
    dirpath: str, filename: str = DISABLED_TESTS_FILE
) -> dict[str, Any] | None:
    def process_disabled_test(the_response: dict[str, Any]) -> dict[str, Any]:
        # remove re-enabled tests and condense even further by getting rid of pr_num
        disabled_issues = get_disabled_issues()
        disabled_test_from_issues = {}
        for test_name, (pr_num, link, platforms) in the_response.items():
            if pr_num not in disabled_issues:
                disabled_test_from_issues[test_name] = (
                    link,
                    platforms,
                )
        return disabled_test_from_issues

    try:
        url = "https://ossci-metrics.s3.amazonaws.com/disabled-tests-condensed.json"
        return fetch_and_cache(dirpath, filename, url, process_disabled_test)
    except Exception:
        print("Couldn't download test skip set, leaving all tests enabled...")
        return {}


def get_test_file_ratings() -> dict[str, Any]:
    return get_from_test_infra_generated_stats(
        "file_test_rating.json",
        TEST_FILE_RATINGS_FILE,
        "Couldn't download test file ratings file, not reordering...",
    )


def get_test_class_ratings() -> dict[str, Any]:
    return get_from_test_infra_generated_stats(
        "file_test_class_rating.json",
        TEST_CLASS_RATINGS_FILE,
        "Couldn't download test class ratings file, not reordering...",
    )


def get_td_heuristic_historial_edited_files_json() -> dict[str, Any]:
    return get_from_test_infra_generated_stats(
        "td_heuristic_historical_edited_files.json",
        TD_HEURISTIC_HISTORICAL_EDITED_FILES,
        "Couldn't download td_heuristic_historical_edited_files.json, not reordering...",
    )


def get_td_heuristic_profiling_json() -> dict[str, Any]:
    return get_from_test_infra_generated_stats(
        "td_heuristic_profiling.json",
        TD_HEURISTIC_PROFILING_FILE,
        "Couldn't download td_heuristic_profiling.json not reordering...",
    )


def copy_pytest_cache() -> None:
    original_path = REPO_ROOT / ".pytest_cache/v/cache/lastfailed"
    if not original_path.exists():
        return
    shutil.copyfile(
        original_path,
        REPO_ROOT / ADDITIONAL_CI_FILES_FOLDER / TD_HEURISTIC_PREVIOUSLY_FAILED,
    )


def copy_additional_previous_failures() -> None:
    original_path = (
        REPO_ROOT / ".pytest_cache" / TD_HEURISTIC_PREVIOUSLY_FAILED_ADDITIONAL
    )
    if not original_path.exists():
        return
    shutil.copyfile(
        original_path,
        REPO_ROOT
        / ADDITIONAL_CI_FILES_FOLDER
        / TD_HEURISTIC_PREVIOUSLY_FAILED_ADDITIONAL,
    )


def get_from_test_infra_generated_stats(
    from_file: str, to_file: str, failure_explanation: str
) -> dict[str, Any]:
    url = f"https://raw.githubusercontent.com/pytorch/test-infra/generated-stats/stats/{from_file}"
    try:
        return fetch_and_cache(
            REPO_ROOT / ADDITIONAL_CI_FILES_FOLDER, to_file, url, lambda x: x
        )
    except Exception:
        print(failure_explanation)
        return {}
