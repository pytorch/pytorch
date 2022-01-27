#!/usr/bin/env python3
"""
This script finds the merger responsible for labeling a PR by a commit SHA. It is used by the workflow in
'.github/workflows/pr-labels.yml'. If there exists no PR associated with the commit or the PR is properly labeled,
this script is a no-op.

Note: we ping the merger only, not the reviewers, as the reviewers can sometimes be external to pytorch
with no labeling responsibility, so we don't want to bother them.
This script is based on: https://github.com/pytorch/vision/blob/main/.github/process_commit.py
"""

import sys
from typing import Any, Set, Tuple

import requests

# For a PR to be properly labeled it should have release notes label and one topic label
PRIMARY_LEBEL_FILTER = "release notes:"
SECONDARY_LEBELS = {
    "topic: bc_breaking",
    "topic: deprecation",
    "topic: new feature",
    "topic: improvements",
    "topic: bug fixes",
    "topic: performance",
    "topic: documentation",
    "topic: developer feature",
    "topic: non-user visible",
}

def query_pytroch(cmd: str, *, accept: str) -> Any:
    response = requests.get(f"https://api.github.com/repos/pytorch/pytorch/{cmd}", headers=dict(Accept=accept))
    return response.json()


def get_pr_number(commit_hash: str) -> Any:
    # See https://docs.github.com/en/rest/reference/repos#list-pull-requests-associated-with-a-commit
    data = query_pytroch(f"commits/{commit_hash}/pulls", accept="application/vnd.github.groot-preview+json")
    if not data:
        return None
    return data[0]["number"]


def get_pr_merger_and_labels(pr_number: int) -> Tuple[str, Set[str]]:
    # See https://docs.github.com/en/rest/reference/pulls#get-a-pull-request
    data = query_pytroch(f"pulls/{pr_number}", accept="application/vnd.github.v3+json")
    merger = data["merged_by"]["login"]
    labels = {label["name"] for label in data["labels"]}
    return merger, labels


if __name__ == "__main__":
    commit_hash = sys.argv[1]
    pr_number = get_pr_number(commit_hash)

    if not pr_number:
        sys.exit(0)

    merger, labels = get_pr_merger_and_labels(pr_number)
    response = query_pytroch("labels", accept="application/json")
    response_labels = list(map(lambda x: str(x["name"]), response.json()))
    primary_labels = set(filter(lambda x: x.startswith(PRIMARY_LEBEL_FILTER), response_labels))
    is_properly_labeled = bool(primary_labels.intersection(labels) and SECONDARY_LEBELS.intersection(labels))

    if not is_properly_labeled:
        print(f"@{merger}")
