from __future__ import annotations

import json
import os
import re
import subprocess
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import cast, Dict, TYPE_CHECKING
from urllib.request import Request, urlopen
from warnings import warn


if TYPE_CHECKING:
    from tools.testing.test_run import TestRun

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent


def python_test_file_to_test_name(tests: set[str]) -> set[str]:
    prefix = f"test{os.path.sep}"
    valid_tests = {f for f in tests if f.startswith(prefix) and f.endswith(".py")}
    valid_tests = {f[len(prefix) : -len(".py")] for f in valid_tests}

    return valid_tests


@lru_cache(maxsize=None)
def get_pr_number() -> int | None:
    pr_number = os.environ.get("PR_NUMBER", "")
    if pr_number == "":
        re_match = re.match(r"^refs/tags/.*/(\d+)$", os.environ.get("GITHUB_REF", ""))
        if re_match is not None:
            pr_number = re_match.group(1)
    if pr_number != "":
        return int(pr_number)
    return None


@lru_cache(maxsize=None)
def get_merge_base() -> str:
    pr_number = get_pr_number()
    if pr_number is not None:
        github_token = os.environ.get("GITHUB_TOKEN")
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "Authorization": f"token {github_token}",
        }
        url = f"https://api.github.com/repos/pytorch/pytorch/pulls/{pr_number}"
        with urlopen(Request(url, headers=headers)) as conn:
            pr_info = json.loads(conn.read().decode())
            base = f"origin/{pr_info['base']['ref']}"
        merge_base = (
            subprocess.check_output(["git", "merge-base", base, "HEAD"])
            .decode()
            .strip()
        )
        return merge_base
    default_branch = f"origin/{os.environ.get('GIT_DEFAULT_BRANCH', 'main')}"
    merge_base = (
        subprocess.check_output(["git", "merge-base", default_branch, "HEAD"])
        .decode()
        .strip()
    )

    head = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()

    if merge_base == head:
        # We are on the default branch, so check for changes since the last commit
        merge_base = "HEAD^"
    return merge_base


def query_changed_files() -> list[str]:
    base_commit = get_merge_base()

    proc = subprocess.run(
        ["git", "diff", "--name-only", base_commit, "HEAD"],
        capture_output=True,
        check=False,
    )
    print(f"base_commit: {base_commit}")

    if proc.returncode != 0:
        raise RuntimeError("Unable to get changed files")

    lines = proc.stdout.decode().strip().split("\n")
    lines = [line.strip() for line in lines]
    print(f"Changed files: {lines}")
    return lines


@lru_cache(maxsize=None)
def get_git_commit_info() -> str:
    """Gets the commit info since the last commit on the default branch."""
    base_commit = get_merge_base()

    return (
        subprocess.check_output(
            ["git", "log", f"{base_commit}..HEAD"],
        )
        .decode()
        .strip()
    )


@lru_cache(maxsize=None)
def get_issue_or_pr_body(number: int) -> str:
    """Gets the body of an issue or PR"""
    github_token = os.environ.get("GITHUB_TOKEN")
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {github_token}",
    }
    # Despite the 'issues' in the link, this also works for PRs
    url = f"https://api.github.com/repos/pytorch/pytorch/issues/{number}"
    with urlopen(Request(url, headers=headers)) as conn:
        body: str = json.loads(conn.read().decode())["body"] or ""
        return body


def normalize_ratings(
    ratings: dict[TestRun, float], max_value: float, min_value: float = 0
) -> dict[TestRun, float]:
    # Takse the ratings, makes the max value into max_value, and proportionally
    # distributes the rest of the ratings.
    # Ex [1,2,3,4] and max_value 8 gets converted to [2,4,6,8]
    # Assumes all rankings are >= 0
    # min_value is what 0 gets mapped to and shifts the values accordingly.  Ex
    # [1,2,3,4], min_value 1, max_value 5 gets converted to [2,3,4,5]
    # Don't modify in place
    if len(ratings) == 0:
        return ratings
    min_rating = min(ratings.values())
    assert min_rating > 0
    max_rating = max(ratings.values())
    assert max_rating > 0
    normalized_ratings = {}
    for tf, rank in ratings.items():
        normalized_ratings[tf] = rank / max_rating * (max_value - min_value) + min_value
    return normalized_ratings


def get_ratings_for_tests(file: str | Path) -> dict[str, float]:
    path = REPO_ROOT / file
    if not os.path.exists(path):
        print(f"could not find path {path}")
        return {}
    with open(path) as f:
        test_file_ratings = cast(Dict[str, Dict[str, float]], json.load(f))
    try:
        changed_files = query_changed_files()
    except Exception as e:
        warn(f"Can't query changed test files due to {e}")
        return {}
    ratings: dict[str, float] = defaultdict(float)
    for file in changed_files:
        for test_file, score in test_file_ratings.get(file, {}).items():
            ratings[test_file] += score
    return ratings


def get_correlated_tests(file: str | Path) -> list[str]:
    ratings = get_ratings_for_tests(file)
    prioritize = sorted(ratings, key=lambda x: -ratings[x])
    return prioritize
