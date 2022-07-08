#!/usr/bin/env python3

import datetime
import json
import os
import pathlib
import re
from typing import Any, Callable, Dict, List, Optional, cast
from urllib.request import urlopen


def get_disabled_issues() -> List[str]:
    pr_body = os.getenv("PR_BODY", "")
    commit_messages = os.getenv("COMMIT_MESSAGES", "")
    # The below regex is meant to match all *case-insensitive* keywords that
    # GitHub has delineated would link PRs to issues, more details here:
    # https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue.
    # E.g., "Close #62851", "fixES #62851" and "RESOLVED #62851" would all match, but not
    # "closes  #62851" --> extra space, "fixing #62851" --> not a keyword, nor "fix 62851" --> no #
    regex = "(?i)(Close(d|s)?|Resolve(d|s)?|Fix(ed|es)?) (#|https://github.com/pytorch/pytorch/issues/)([0-9]+)"
    issue_numbers = [x[5] for x in re.findall(regex, pr_body + commit_messages)]
    print("Ignoring disabled issues: ", issue_numbers)
    return issue_numbers


IGNORE_DISABLED_ISSUES: List[str] = get_disabled_issues()

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
        with open(path, "r") as f:
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
    url = "https://raw.githubusercontent.com/pytorch/test-infra/generated-stats/stats/slow-tests.json"
    try:
        return fetch_and_cache(dirpath, filename, url, lambda x: x)
    except Exception:
        print("Couldn't download slow test set, leaving all tests enabled...")
        return {}


def get_test_times(dirpath: str, filename: str) -> Optional[Dict[str, float]]:
    url = "https://raw.githubusercontent.com/pytorch/test-infra/generated-stats/stats/test-times.json"

    def process_response(the_response: Dict[str, Any]) -> Any:
        build_environment = os.environ["BUILD_ENVIRONMENT"]
        test_config = os.environ["TEST_CONFIG"]
        return the_response[build_environment][test_config]

    try:
        return fetch_and_cache(dirpath, filename, url, process_response)
    except Exception:
        print("Couldn't download test times...")
        return {}


def get_disabled_tests(
    dirpath: str, filename: str = DISABLED_TESTS_FILE
) -> Optional[Dict[str, Any]]:
    def process_disabled_test(the_response: Dict[str, Any]) -> Dict[str, Any]:
        disabled_test_from_issues = dict()
        for item in the_response["items"]:
            title = item["title"]
            key = "DISABLED "
            issue_url = item["html_url"]
            issue_number = issue_url.split("/")[-1]
            if title.startswith(key) and issue_number not in IGNORE_DISABLED_ISSUES:
                test_name = title[len(key) :].strip()
                body = item["body"]
                platforms_to_skip = []
                key = "platforms:"
                # When the issue has no body, it is assumed that all platforms should skip the test
                if body is not None:
                    for line in body.splitlines():
                        line = line.lower()
                        if line.startswith(key):
                            pattern = re.compile(r"^\s+|\s*,\s*|\s+$")
                            platforms_to_skip.extend(
                                [x for x in pattern.split(line[len(key) :]) if x]
                            )
                disabled_test_from_issues[test_name] = (
                    item["html_url"],
                    platforms_to_skip,
                )
        return disabled_test_from_issues

    try:
        url = "https://raw.githubusercontent.com/pytorch/test-infra/generated-stats/stats/disabled-tests.json"
        return fetch_and_cache(dirpath, filename, url, process_disabled_test)
    except Exception:
        print("Couldn't download test skip set, leaving all tests enabled...")
        return {}
