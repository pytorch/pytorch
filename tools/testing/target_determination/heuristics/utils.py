import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import cast, Dict, List, Set, Union
from urllib.request import Request, urlopen
from warnings import warn

from tools.testing.test_run import TestRun

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent


def python_test_file_to_test_name(tests: Set[str]) -> Set[str]:
    prefix = f"test{os.path.sep}"
    valid_tests = {f for f in tests if f.startswith(prefix) and f.endswith(".py")}
    valid_tests = {f[len(prefix) : -len(".py")] for f in valid_tests}

    return valid_tests


def query_changed_files() -> List[str]:
    default_branch = f"origin/{os.environ.get('GIT_DEFAULT_BRANCH', 'main')}"
    merge_base = (
        subprocess.check_output(["git", "merge-base", default_branch, "HEAD"])
        .decode()
        .strip()
    )

    head = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()

    base_commit = merge_base
    if base_commit == head:
        # We are on the default branch, so check for changes since the last commit
        base_commit = "HEAD^"

    proc = subprocess.run(
        ["git", "diff", "--name-only", base_commit, "HEAD"],
        capture_output=True,
        check=False,
    )

    if proc.returncode != 0:
        raise RuntimeError("Unable to get changed files")

    lines = proc.stdout.decode().strip().split("\n")
    lines = [line.strip() for line in lines]
    return lines


def get_git_commit_info() -> str:
    """Gets the commit info since the last commit on the default branch."""
    default_branch = f"origin/{os.environ.get('GIT_DEFAULT_BRANCH', 'main')}"

    merge_base = (
        subprocess.check_output(["git", "merge-base", default_branch, "HEAD"])
        .decode()
        .strip()
    )

    head = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()

    base_commit = merge_base
    if base_commit == head:
        # We are on the default branch, so check for changes since the last commit
        base_commit = "HEAD^"

    return (
        subprocess.check_output(
            ["git", "log", f"{base_commit}..HEAD"],
        )
        .decode()
        .strip()
    )


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
        body: str = json.loads(conn.read().decode())["body"]
        return body


def normalize_ratings(
    ratings: Dict[TestRun, float], max_value: float
) -> Dict[TestRun, float]:
    # Takse the ratings, makes the max value into max_value, and proportionally
    # distributes the rest of the ratings.
    # Ex [1,2,3,4] and max_value 8 gets converted to [2,4,6,8]
    # Assumes all rankings are >= 0
    # Don't modify in place
    if len(ratings) == 0:
        return ratings
    min_rating = min(ratings.values())
    assert min_rating > 0
    max_rating = max(ratings.values())
    assert max_rating > 0
    normalized_ratings = {}
    for tf, rank in ratings.items():
        normalized_ratings[tf] = rank / max_rating * max_value
    return normalized_ratings


def get_ratings_for_tests(file: Union[str, Path]) -> Dict[str, float]:
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
    ratings: Dict[str, float] = defaultdict(float)
    for file in changed_files:
        for test_file, score in test_file_ratings.get(file, {}).items():
            ratings[test_file] += score
    return ratings


def get_correlated_tests(file: Union[str, Path]) -> List[str]:
    ratings = get_ratings_for_tests(file)
    prioritize = sorted(ratings, key=lambda x: -ratings[x])
    return prioritize
