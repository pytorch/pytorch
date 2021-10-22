#!/usr/bin/env python3

import datetime
import json
import os
import pathlib
import re
from typing import Any, Callable, Dict, List, Optional, cast
from urllib.request import urlopen

# PYTORCH_IGNORE_DISABLED_ISSUES should only be set during CI (along with IN_CI) as a
# comma-separated list of issue numbers. The intent is to re-enable any disabled tests
# associated with the issues in this list.
#
# There is normally no reason to use this locally as the disabled tests list should not
# affect your local development and every test should be enabled. If for whatever reason
# you would like to use this during local development, please note the following caveat:
#
# Whenever you set OR reset PYTORCH_IGNORE_DISABLED_ISSUES, you should delete the existing
# .pytorch-disabled-tests.json and redownload/parse the file for your change to apply, as
# PYTORCH_IGNORE_DISABLED_ISSUES is used during the parsing stage. To download the files,
# run test/run_test.py with IN_CI=1.
IGNORE_DISABLED_ISSUES: List[str] = os.getenv('PYTORCH_IGNORE_DISABLED_ISSUES', '').split(',')

SLOW_TESTS_FILE = '.pytorch-slow-tests.json'
DISABLED_TESTS_FILE = '.pytorch-disabled-tests.json'

FILE_CACHE_LIFESPAN_SECONDS = datetime.timedelta(hours=3).seconds

def fetch_and_cache(
    dirpath: str,
    name: str,
    url: str,
    process_fn: Callable[[Dict[str, Any]], Dict[str, Any]]
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
        # Another test process already downloaded the file, so don't re-do it
        with open(path, "r") as f:
            return cast(Dict[str, Any], json.load(f))
    try:
        contents = urlopen(url, timeout=1).read().decode('utf-8')
        processed_contents = process_fn(json.loads(contents))
        with open(path, "w") as f:
            f.write(json.dumps(processed_contents))
        return processed_contents
    except Exception as e:
        print(f'Could not download {url} because of error {e}.')
        return {}


def get_slow_tests(dirpath: str, filename: str = SLOW_TESTS_FILE) -> Optional[Dict[str, float]]:
    url = "https://raw.githubusercontent.com/pytorch/test-infra/main/stats/slow-tests.json"
    try:
        return fetch_and_cache(dirpath, filename, url, lambda x: x)
    except Exception:
        print("Couldn't download slow test set, leaving all tests enabled...")
        return {}


def get_disabled_tests(dirpath: str, filename: str = DISABLED_TESTS_FILE) -> Optional[Dict[str, Any]]:
    def process_disabled_test(the_response: Dict[str, Any]) -> Dict[str, Any]:
        disabled_test_from_issues = dict()
        for item in the_response['items']:
            title = item['title']
            key = 'DISABLED '
            issue_url = item['html_url']
            issue_number = issue_url.split('/')[-1]
            if title.startswith(key) and issue_number not in IGNORE_DISABLED_ISSUES:
                test_name = title[len(key):].strip()
                body = item['body']
                platforms_to_skip = []
                key = 'platforms:'
                for line in body.splitlines():
                    line = line.lower()
                    if line.startswith(key):
                        pattern = re.compile(r"^\s+|\s*,\s*|\s+$")
                        platforms_to_skip.extend([x for x in pattern.split(line[len(key):]) if x])
                disabled_test_from_issues[test_name] = (item['html_url'], platforms_to_skip)
        return disabled_test_from_issues
    try:
        url = 'https://raw.githubusercontent.com/pytorch/test-infra/main/stats/disabled-tests.json'
        return fetch_and_cache(dirpath, filename, url, process_disabled_test)
    except Exception:
        print("Couldn't download test skip set, leaving all tests enabled...")
        return {}
