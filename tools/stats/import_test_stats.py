#!/usr/bin/env python3

import datetime
import json
import os
import pathlib
from typing import Any, Callable, cast, Dict, List, Optional
from urllib.request import urlopen


def get_disabled_issues() -> List[str]:
    reenabled_issues = os.getenv("REENABLED_ISSUES", "")
    issue_numbers = reenabled_issues.split(",")
    print("Ignoring disabled issues: ", issue_numbers)
    return issue_numbers


SLOW_TESTS_FILE = ".pytorch-slow-tests.json"
DISABLED_TESTS_FILE = ".pytorch-disabled-tests.json"


FILE_CACHE_LIFESPAN_SECONDS = datetime.timedelta(hours=3).seconds


def fetch_and_cache(
    dirpath: str,
    name: str,
    url: str,
    process_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> Dict[str, Any]:
    """
    This fetch and cache utils allows sharing between different process.
    """
    path = os.path.join(dirpath, name)
    print(f"Downloading {url} to {path}")

    def is_cached_file_valid() -> bool:
        # Check if the file is new enough (see: FILE_CACHE_LIFESPAN_SECONDS). A real check
        # could make a HEAD request and check/store the file's ETag
        fname = pathlib.Path(path)
        now = datetime.datetime.now()
        mtime = datetime.datetime.fromtimestamp(fname.stat().st_mtime)
        diff = now - mtime
        return diff.total_seconds() < FILE_CACHE_LIFESPAN_SECONDS

    if os.path.exists(path) and is_cached_file_valid():
        # Another test process already download the file, so don't re-do it
        with open(path) as f:
            return cast(Dict[str, Any], json.load(f))

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


def get_slow_tests(
    dirpath: str, filename: str = SLOW_TESTS_FILE
) -> Optional[Dict[str, float]]:
    url = "https://ossci-metrics.s3.amazonaws.com/slow-tests.json"
    try:
        return fetch_and_cache(dirpath, filename, url, lambda x: x)
    except Exception:
        print("Couldn't download slow test set, leaving all tests enabled...")
        return {}


def get_test_times(dirpath: str, filename: str) -> Dict[str, Dict[str, float]]:
    url = "https://raw.githubusercontent.com/pytorch/test-infra/generated-stats/stats/test-times.json"
    try:
        return fetch_and_cache(dirpath, filename, url, lambda x: x)
    except Exception:
        print("Couldn't download test times...")
        return {}


def get_disabled_tests(
    dirpath: str, filename: str = DISABLED_TESTS_FILE
) -> Optional[Dict[str, Any]]:
    def process_disabled_test(the_response: Dict[str, Any]) -> Dict[str, Any]:
        # remove re-enabled tests and condense even further by getting rid of pr_num
        disabled_issues = get_disabled_issues()
        disabled_test_from_issues = dict()
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


def get_test_file_ratings(dirpath: str, filename: str) -> Optional[Dict[str, Any]]:
    url = "https://raw.githubusercontent.com/pytorch/test-infra/generated-stats/stats/file_test_rating.json"
    try:
        return fetch_and_cache(dirpath, filename, url, lambda x: x)
    except Exception:
        print("Couldn't download test file ratings file, not reordering...")
        return {}
