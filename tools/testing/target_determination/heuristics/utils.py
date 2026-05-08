from __future__ import annotations

import json
import os
import re
import subprocess
from collections import defaultdict
from functools import cache
from pathlib import Path
from typing import cast, TYPE_CHECKING
from urllib.parse import quote
from urllib.request import Request, urlopen


if TYPE_CHECKING:
    from tools.testing.test_run import TestRun


REPO_ROOT = Path(__file__).resolve().parents[4]


def _github_api_json(path: str) -> object:
    github_repository = os.environ.get("GITHUB_REPOSITORY", "pytorch/pytorch")
    github_token = os.environ.get("GITHUB_TOKEN")
    headers = {
        "Accept": "application/vnd.github.v3+json",
    }
    if github_token:
        headers["Authorization"] = f"token {github_token}"
    url = f"https://api.github.com/repos/{github_repository}/{path}"
    with urlopen(Request(url, headers=headers)) as conn:
        return json.loads(conn.read().decode())


def _git_merge_base(base: str) -> str:
    return subprocess.check_output(["git", "merge-base", base, "HEAD"]).decode().strip()


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()


def _get_pr_info(pr_number: int) -> dict[str, object]:
    return cast(dict[str, object], _github_api_json(f"pulls/{pr_number}"))


def _get_pr_merge_base(pr_info: dict[str, object]) -> str:
    base_info = cast(dict[str, object], pr_info["base"])
    base_ref = cast(str, base_info["ref"])
    base_sha = cast(str | None, base_info.get("sha"))
    last_error: Exception | None = None

    for base in (f"origin/{base_ref}", base_sha):
        if not base:
            continue
        try:
            return _git_merge_base(base)
        except subprocess.CalledProcessError as e:
            last_error = e

    head = _git_head()
    compare = cast(
        dict[str, object],
        _github_api_json(f"compare/{quote(base_ref, safe='')}...{head}"),
    )
    merge_base_commit = cast(dict[str, object], compare.get("merge_base_commit", {}))
    merge_base = cast(str | None, merge_base_commit.get("sha"))
    if merge_base:
        return merge_base
    if base_sha:
        return base_sha
    if last_error:
        raise last_error
    raise RuntimeError(f"Unable to determine merge base for PR base {base_ref}")


def _query_changed_files_from_github(pr_number: int) -> list[str]:
    changed_files: list[str] = []
    page = 1
    while True:
        files = cast(
            list[dict[str, object]],
            _github_api_json(f"pulls/{pr_number}/files?per_page=100&page={page}"),
        )
        changed_files.extend(cast(str, file["filename"]) for file in files)
        if len(files) < 100:
            return changed_files
        page += 1


def python_test_file_to_test_name(tests: set[str]) -> set[str]:
    prefix = f"test{os.path.sep}"
    valid_tests = {f for f in tests if f.startswith(prefix) and f.endswith(".py")}
    valid_tests = {f[len(prefix) : -len(".py")] for f in valid_tests}

    return valid_tests


@cache
def get_pr_number() -> int | None:
    pr_number = os.environ.get("PR_NUMBER", "")
    if pr_number == "":
        re_match = re.match(r"^refs/tags/.*/(\d+)$", os.environ.get("GITHUB_REF", ""))
        if re_match is not None:
            pr_number = re_match.group(1)
    if pr_number != "":
        return int(pr_number)
    return None


@cache
def get_merge_base() -> str:
    pr_number = get_pr_number()
    if pr_number is not None:
        return _get_pr_merge_base(_get_pr_info(pr_number))
    default_branch = f"origin/{os.environ.get('GIT_DEFAULT_BRANCH', 'main')}"
    merge_base = _git_merge_base(default_branch)

    head = _git_head()

    if merge_base == head:
        # We are on the default branch, so check for changes since the last commit
        merge_base = "HEAD^"
    return merge_base


def query_changed_files() -> list[str]:
    pr_number = get_pr_number()
    try:
        base_commit = get_merge_base()

        proc = subprocess.run(
            ["git", "diff", "--name-only", base_commit, "HEAD"],
            capture_output=True,
            check=False,
        )
        print(f"base_commit: {base_commit}")

        if proc.returncode != 0:
            raise RuntimeError("Unable to get changed files")
    except (subprocess.CalledProcessError, RuntimeError):
        if pr_number is None:
            raise
        lines = _query_changed_files_from_github(pr_number)
        print(f"Changed files from GitHub: {lines}")
        return lines

    lines = proc.stdout.decode().strip().split("\n")
    lines = [line.strip() for line in lines]
    print(f"Changed files: {lines}")
    return lines


# File extensions that are documentation-only and don't require running tests
DOCS_ONLY_EXTENSIONS = frozenset({".rst", ".md"})


def is_docs_only_change(changed_files: list[str]) -> bool:
    """
    Returns True if all changed files are documentation-only files
    (e.g., .rst, .md files) that don't require running tests.
    """
    if not changed_files:
        return False

    for f in changed_files:
        # Skip empty strings that might come from git diff output
        if not f:
            continue
        # Check if the file extension is in the docs-only set
        ext = os.path.splitext(f)[1].lower()
        if ext not in DOCS_ONLY_EXTENSIONS:
            return False

    return True


@cache
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


@cache
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
    if min_rating <= 0:
        raise AssertionError(f"min_rating must be > 0, got {min_rating}")
    max_rating = max(ratings.values())
    if max_rating <= 0:
        raise AssertionError(f"max_rating must be > 0, got {max_rating}")
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
        test_file_ratings = cast(dict[str, dict[str, float]], json.load(f))
    changed_files = query_changed_files()
    ratings: dict[str, float] = defaultdict(float)
    for file in changed_files:
        for test_file, score in test_file_ratings.get(file, {}).items():
            ratings[test_file] += score
    return ratings


def get_correlated_tests(file: str | Path) -> list[str]:
    ratings = get_ratings_for_tests(file)
    prioritize = sorted(ratings, key=lambda x: -ratings[x])
    return prioritize
