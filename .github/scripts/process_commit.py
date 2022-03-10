#!/usr/bin/env python3
"""
This script finds the user/pr creator responsible for labeling a PR by a commit SHA. It is used by the workflow in
'.github/workflows/pr-labels.yml'. If there exists no PR associated with the commit or the PR is properly labeled,
this script is a no-op.

Note: we ping the user only, not the reviewers, as the reviewers can sometimes be external to pytorch
with no labeling responsibility, so we don't want to bother them.
This script is based on: https://github.com/pytorch/vision/blob/main/.github/process_commit.py
"""

import sys
from typing import Any, Set, Tuple, List
import re
import os
import json
import requests

# For a PR to be properly labeled it should have release notes label and one topic label
PULL_REQUEST_EXP = "Pull Request resolved:.*pull/(.*)"
PRIMARY_LABEL_FILTER = "release notes:"
SECONDARY_LABELS = {
    "topic: bc_breaking",
    "topic: deprecation",
    "topic: new feature",
    "topic: improvements",
    "topic: bug fixes",
    "topic: performance",
    "topic: documentation",
    "topic: developer feature",
    "topic: not user facing",
}
# This secondary does not require a primary
ALLOWED_ONLY_SECONDARY = {"topic: not user facing"}
PYTORCH_REPO = "https://api.github.com/repos/pytorch/pytorch"
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN')
REQUEST_HEADERS = {'Accept': 'application/vnd.github.v3+json', 'Authorization': f'token {GITHUB_TOKEN}'}


def query_pytorch(cmd: str) -> Any:
    response = requests.get(f"{PYTORCH_REPO}/{cmd}", headers=REQUEST_HEADERS)
    return response.json()


def get_pr_number(commit_hash: str) -> Any:
    data = query_pytorch(f"commits/{commit_hash}")
    if not data or (not data["commit"]["message"]):
        return None
    message = data["commit"]["message"]
    p = re.compile(PULL_REQUEST_EXP)
    result = p.search(message)
    if not result:
        return None
    return result.group(1)


def get_pr_author_and_labels(pr_number: int) -> Tuple[str, Set[str]]:
    # See https://docs.github.com/en/rest/reference/pulls#get-a-pull-request
    data = query_pytorch(f"pulls/{pr_number}")
    user = data["user"]["login"]
    labels = {label["name"] for label in data["labels"]}
    return user, labels

def get_repo_labels() -> List[str]:
    collected_labels: List[str] = list()
    for page in range(0, 10):
        response = query_pytorch(f"labels?per_page=100&page={page}")
        page_labels = list(map(lambda x: str(x["name"]), response))
        if not page_labels:
            break
        collected_labels += page_labels
    return collected_labels

def post_pytorch_comment(pr_number: int, merger: str) -> Any:
    message = {'body' : f"Hey @{merger}." + """
You've committed this PR, but it does not have both a 'release notes: ...' and 'topics: ...' label. \
Please add one of each to the PR. The 'release notes: ...' label should represent the part of \
PyTorch that this PR changes (fx, autograd, distributed, etc) and the 'topics: ...' label should \
represent the kind of PR it is (not user facing, new feature, bug fix, perf improvement, etc). \
The list of valid labels can be found [here](https://github.com/pytorch/pytorch/labels?q=release+notes) \
for the 'release notes: ...' and [here](https://github.com/pytorch/pytorch/labels?q=topic) for the \
'topics: ...'.
For changes that are 'topic: not user facing' there is no need for a release notes label."""}

    response = requests.post(
        f"{PYTORCH_REPO}/issues/{pr_number}/comments",
        json.dumps(message),
        headers=REQUEST_HEADERS)
    return response.json()

if __name__ == "__main__":
    commit_hash = sys.argv[1]
    pr_number = get_pr_number(commit_hash)

    if not pr_number:
        sys.exit(0)

    user, labels = get_pr_author_and_labels(pr_number)
    repo_labels = get_repo_labels()

    primary_labels = set(filter(lambda x: x.startswith(PRIMARY_LABEL_FILTER), repo_labels))
    has_both_labels = bool(primary_labels.intersection(labels) and SECONDARY_LABELS.intersection(labels))
    is_properly_labeled = has_both_labels or bool(ALLOWED_ONLY_SECONDARY.intersection(labels))

    if not is_properly_labeled:
        post_pytorch_comment(pr_number, user)
