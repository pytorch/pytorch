#!/usr/bin/env python3
"""Add the "triaged" label to an "open source" PR when a maintainer comments or
reviews it. A maintainer is anyone who could approve the PR per the files it
touches, as defined by .github/merge_rules.yaml.

Invoked by .github/workflows/auto-triage.yml. All inputs come from environment
variables and are treated as untrusted data: the PR state and merge rules are
re-fetched/re-parsed from trusted server-side sources before any decision.
"""

from __future__ import annotations

import os

from label_utils import BOT_AUTHORS, gh_add_labels
from trymerge import GitHubPR, is_maintainer_for_pr


OPEN_SOURCE_LABEL = "open source"
TRIAGED_LABEL = "triaged"
# Explicitly excluded actors, plus the bots whose automated chatter must never
# self-trigger triage.
EXCLUDED_USERS = {"jansel", *BOT_AUTHORS}


def main() -> None:
    actor = os.environ["ACTOR"]
    pr_num = int(os.environ["PR_NUM"])
    dry_run = os.environ.get("DRY_RUN", "0") == "1"

    if actor in EXCLUDED_USERS:
        print(f"Skipping: {actor} is excluded")
        return

    pr = GitHubPR("pytorch", "pytorch", pr_num)
    labels = pr.get_labels()
    if OPEN_SOURCE_LABEL not in labels:
        print(f"Skipping: PR #{pr_num} does not have the '{OPEN_SOURCE_LABEL}' label")
        return
    if TRIAGED_LABEL in labels:
        print(f"Skipping: PR #{pr_num} already has the '{TRIAGED_LABEL}' label")
        return

    if not is_maintainer_for_pr(pr, actor):
        print(f"Skipping: {actor} is not a maintainer for PR #{pr_num}")
        return

    print(f"Adding '{TRIAGED_LABEL}' to PR #{pr_num} (triggered by {actor})")
    gh_add_labels("pytorch", "pytorch", pr_num, [TRIAGED_LABEL], dry_run)


if __name__ == "__main__":
    main()
