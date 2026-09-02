#!/usr/bin/env python3

from __future__ import annotations

import base64
import json
import os
import re
import time
import traceback
import urllib.parse
from collections import defaultdict
from dataclasses import dataclass
from functools import cache, partial
from pathlib import Path
from re import Pattern
from typing import Any, cast, NamedTuple, TYPE_CHECKING
from warnings import warn

import yaml
from github_utils import (
    gh_close_pr,
    gh_fetch_json_dict,
    gh_fetch_json_list,
    gh_fetch_merge_base,
    gh_fetch_url,
    gh_graphql,
    gh_merge_pr,
    gh_post_commit_comment,
    gh_post_pr_comment,
    gh_update_pr_state,
    GITHUB_API_URL,
    GitHubComment,
)
from gitutils import (
    are_ghstack_branches_in_sync,
    get_git_remote_name,
    get_git_repo_dir,
    GitRepo,
    patterns_to_regex,
    retries_decorator,
)
from greenlight_guard import (
    evaluate_greenlight_guard,
    GreenlightWaitWindow,
    GuardVerdict,
    PRUnderMerge,
)
from greenlight_identity import is_greenlight, normalize_login
from label_utils import (
    gh_add_labels,
    gh_remove_label,
    has_required_labels,
    LABEL_ERR_MSG,
    NOT_USER_FACING_LABEL,
)
from trymerge_explainer import get_revert_message, TryMergeExplainer


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


# labels
MERGE_IN_PROGRESS_LABEL = "merging"
MERGE_COMPLETE_LABEL = "merged"


class JobCheckState(NamedTuple):
    name: str
    url: str
    status: str | None
    classification: str | None
    job_id: int | None
    title: str | None
    summary: str | None


JobNameToStateDict = dict[str, JobCheckState]


class WorkflowCheckState:
    def __init__(self, name: str, url: str, run_id: int, status: str | None):
        self.name: str = name
        self.url: str = url
        self.run_id: int = run_id
        self.status: str | None = status
        self.jobs: JobNameToStateDict = {}


GH_PR_REVIEWS_FRAGMENT = """
fragment PRReviews on PullRequestReviewConnection {
  nodes {
    author {
      login
    }
    bodyText
    createdAt
    authorAssociation
    editor {
      login
    }
    databaseId
    url
    state
  }
  pageInfo {
    startCursor
    hasPreviousPage
  }
}
"""

GH_CHECKSUITES_FRAGMENT = """
fragment PRCheckSuites on CheckSuiteConnection {
  edges {
    node {
      workflowRun {
        workflow {
          name
          databaseId
        }
        databaseId
        url
      }
      checkRuns(first: 50) {
        nodes {
          name
          conclusion
          detailsUrl
          databaseId
          title
          summary
        }
        pageInfo {
          endCursor
          hasNextPage
        }
      }
      conclusion
    }
    cursor
  }
  pageInfo {
    hasNextPage
  }
}
"""

GH_COMMIT_AUTHORS_FRAGMENT = """
fragment CommitAuthors on PullRequestCommitConnection {
  nodes {
    commit {
      authors(first: 2) {
        nodes {
          user {
            login
          }
          email
          name
        }
      }
      oid
    }
  }
  pageInfo {
    endCursor
    hasNextPage
  }
}
"""

GH_GET_PR_INFO_QUERY = (
    GH_PR_REVIEWS_FRAGMENT
    + GH_CHECKSUITES_FRAGMENT
    + GH_COMMIT_AUTHORS_FRAGMENT
    + """
query ($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      closed
      isCrossRepository
      author {
        login
      }
      title
      body
      headRefName
      headRepository {
        nameWithOwner
      }
      baseRefName
      baseRefOid
      baseRepository {
        nameWithOwner
        isPrivate
        defaultBranchRef {
          name
        }
      }
      mergeCommit {
        oid
      }
      commits_with_authors: commits(first: 100) {
        ...CommitAuthors
        totalCount
      }
      commits(last: 1) {
        nodes {
          commit {
            checkSuites(first: 10) {
              ...PRCheckSuites
            }
            status {
              contexts {
                context
                state
                targetUrl
              }
            }
            oid
          }
        }
      }
      changedFiles
      files(first: 100) {
        nodes {
          path
        }
        pageInfo {
          endCursor
          hasNextPage
        }
      }
      reviews(last: 100) {
        ...PRReviews
      }
      comments(last: 5) {
        nodes {
          bodyText
          createdAt
          author {
            login
            url
          }
          authorAssociation
          editor {
            login
          }
          databaseId
          url
        }
        pageInfo {
          startCursor
          hasPreviousPage
        }
      }
      labels(first: 100) {
        edges {
          node {
            name
          }
        }
      }
    }
  }
}
"""
)

GH_GET_PR_NEXT_FILES_QUERY = """
query ($owner: String!, $name: String!, $number: Int!, $cursor: String!) {
  repository(name: $name, owner: $owner) {
    pullRequest(number: $number) {
      files(first: 100, after: $cursor) {
        nodes {
          path
        }
        pageInfo {
          endCursor
          hasNextPage
        }
      }
    }
  }
}
"""

GH_GET_PR_NEXT_CHECKSUITES = (
    GH_CHECKSUITES_FRAGMENT
    + """
query ($owner: String!, $name: String!, $number: Int!, $cursor: String!) {
  repository(name: $name, owner: $owner) {
    pullRequest(number: $number) {
      commits(last: 1) {
        nodes {
          commit {
            oid
            checkSuites(first: 10, after: $cursor) {
              ...PRCheckSuites
            }
          }
        }
      }
    }
  }
}
"""
)

GH_GET_PR_NEXT_CHECK_RUNS = """
query ($owner: String!, $name: String!, $number: Int!, $cs_cursor: String, $cr_cursor: String!) {
  repository(name: $name, owner: $owner) {
    pullRequest(number: $number) {
      commits(last: 1) {
        nodes {
          commit {
            oid
            checkSuites(first: 1, after: $cs_cursor) {
              nodes {
                checkRuns(first: 100, after: $cr_cursor) {
                  nodes {
                    name
                    conclusion
                    detailsUrl
                    databaseId
                    title
                    summary
                  }
                  pageInfo {
                    endCursor
                    hasNextPage
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
"""

GH_GET_PR_PREV_COMMENTS = """
query ($owner: String!, $name: String!, $number: Int!, $cursor: String!) {
  repository(name: $name, owner: $owner) {
    pullRequest(number: $number) {
      comments(last: 100, before: $cursor) {
        nodes {
          bodyText
          createdAt
          author {
            login
          }
          authorAssociation
          editor {
            login
          }
          databaseId
          url
        }
        pageInfo {
          startCursor
          hasPreviousPage
        }
      }
    }
  }
}
"""

# This query needs read-org permission
GH_GET_TEAM_MEMBERS_QUERY = """
query($org: String!, $name: String!, $cursor: String) {
  organization(login: $org) {
    team(slug: $name) {
      members(first: 100, after: $cursor) {
        nodes {
          login
        }
        pageInfo {
          hasNextPage
          endCursor
        }
      }
    }
  }
}
"""

GH_GET_PR_NEXT_AUTHORS_QUERY = (
    GH_COMMIT_AUTHORS_FRAGMENT
    + """
query ($owner: String!, $name: String!, $number: Int!, $cursor: String) {
  repository(name: $name, owner: $owner) {
    pullRequest(number: $number) {
      commits_with_authors: commits(first: 100, after: $cursor) {
        ...CommitAuthors
      }
    }
  }
}
"""
)

GH_GET_PR_PREV_REVIEWS_QUERY = (
    GH_PR_REVIEWS_FRAGMENT
    + """
query ($owner: String!, $name: String!, $number: Int!, $cursor: String!) {
  repository(name: $name, owner: $owner) {
    pullRequest(number: $number) {
      reviews(last: 100, before: $cursor) {
        ...PRReviews
      }
    }
  }
}
"""
)

GH_GET_REPO_SUBMODULES = """
query ($owner: String!, $name: String!) {
  repository(owner: $owner, name: $name) {
    submodules(first: 100) {
      nodes {
        path
      }
      pageInfo {
        endCursor
        hasNextPage
      }
    }
  }
}
"""

RE_GHSTACK_HEAD_REF = re.compile(r"^(gh/[^/]+/[0-9]+/)head$")
RE_GHSTACK_DESC = re.compile(r"Stack.*:\r?\n(\* [^\r\n]+\r?\n)+", re.MULTILINE)
RE_PULL_REQUEST_RESOLVED = re.compile(
    r"(Pull Request resolved|Pull-Request-resolved|Pull-Request): "
    r"https://github.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/pull/(?P<number>[0-9]+)",
    re.MULTILINE,
)
RE_PR_CC_LINE = re.compile(r"^cc:? @\w+.*\r?\n?$", re.MULTILINE)
RE_DIFF_REV = re.compile(r"^Differential Revision:.+?(D[0-9]+)", re.MULTILINE)
CIFLOW_LABEL = re.compile(r"^ciflow/.+")
CIFLOW_TRUNK_LABEL = re.compile(r"^ciflow/trunk")
MERGE_RULE_PATH = Path(".github") / "merge_rules.yaml"
REMOTE_MAIN_BRANCH = "origin/main"
DRCI_CHECKRUN_NAME = "Dr.CI"
INTERNAL_CHANGES_CHECKRUN_NAME = "Meta Internal-Only Changes Check"
HAS_NO_CONNECTED_DIFF_TITLE = (
    "There is no internal Diff connected, this can be merged now"
)
# This could be set to -1 to ignore all flaky and broken trunk failures. On the
# other hand, using a large value like 10 here might be useful in sev situation
IGNORABLE_FAILED_CHECKS_THESHOLD = 10

REVIEWS_PER_PAGE = 100
REVIEW_PAGE_LIMIT = 10

# CI docker images are keyed by the git tree hash of the .ci/docker directory
# (see .github/workflows/docker-builds.yml, which tags the images it builds with
# `git rev-parse HEAD:.ci/docker`).  A PR that touches these files must have its
# images pre-built and pushed to ECR by the docker-builds (ciflow/docker)
# workflow, and the .ci/docker tree hash of the merge commit must match the tree
# hash that was actually built.  If another docker-affecting PR lands in the
# skew window, the merge commit would reference an image tag that was never
# built, leaving every downstream trunk job unable to find its image (see the
# #190927 / #186302 land race).
DOCKER_CI_PATH = ".ci/docker"
DOCKER_BUILDS_WORKFLOW_NAME = "docker-builds"


def iter_issue_timeline_until_comment(
    org: str, repo: str, issue_number: int, target_comment_id: int, max_pages: int = 200
) -> Any:
    """
    Yield timeline entries in order until (and including) the entry whose id == target_comment_id
    for a 'commented' event. Stops once the target comment is encountered.
    """
    page = 1

    while page <= max_pages:
        url = (
            f"https://api.github.com/repos/{org}/{repo}/issues/{issue_number}/timeline"
        )
        params = {"per_page": 100, "page": page}

        batch = gh_fetch_json_list(url, params)

        if not batch:
            return
        for ev in batch:
            # The target is the issue comment row with event == "commented" and id == issue_comment_id
            if ev.get("event") == "commented" and ev.get("id") == target_comment_id:
                yield ev  # nothing in the timeline after this matters, so stop early
                return
            yield ev
        if len(batch) < 100:
            return
        page += 1

    # If we got here without finding the comment, then we either hit a bug or some github PR
    # has a _really_ long timeline.
    # The max # of pages found on any pytorch/pytorch PR at the time of this change was 41
    raise RuntimeError(
        f"Could not find a merge commit in the first {max_pages} pages of the timeline at url {url}."
        f"This is most likely a bug, please report it to the @pytorch/pytorch-dev-infra team."
    )


def sha_from_committed_event(ev: dict[str, Any]) -> str | None:
    """Extract SHA from committed event in timeline"""
    return ev.get("sha")


def sha_from_force_push_after(ev: dict[str, Any]) -> str | None:
    """Extract SHA from force push event in timeline"""
    # The current GitHub API format
    commit_id = ev.get("commit_id")
    if commit_id:
        return str(commit_id)

    # Legacy format
    after = ev.get("after") or ev.get("after_commit") or {}
    if isinstance(after, dict):
        return after.get("sha") or after.get("oid")
    return ev.get("after_sha") or ev.get("head_sha")


def gh_get_pr_info(org: str, proj: str, pr_no: int) -> Any:
    rc = gh_graphql(GH_GET_PR_INFO_QUERY, name=proj, owner=org, number=pr_no)
    return rc["data"]["repository"]["pullRequest"]


@cache
def gh_get_team_members(org: str, name: str) -> list[str]:
    rc: list[str] = []
    team_members: dict[str, Any] = {
        "pageInfo": {"hasNextPage": "true", "endCursor": None}
    }
    while bool(team_members["pageInfo"]["hasNextPage"]):
        query = gh_graphql(
            GH_GET_TEAM_MEMBERS_QUERY,
            org=org,
            name=name,
            cursor=team_members["pageInfo"]["endCursor"],
        )
        team = query["data"]["organization"]["team"]
        if team is None:
            warn(f"Requested non-existing team {org}/{name}")
            return []
        team_members = team["members"]
        rc += [member["login"] for member in team_members["nodes"]]
    return rc


def get_check_run_name_prefix(workflow_run: Any) -> str:
    if workflow_run is None:
        return ""
    else:
        return f"{workflow_run['workflow']['name']} / "


def is_passing_status(status: str | None) -> bool:
    return status is not None and status.upper() in ["SUCCESS", "SKIPPED", "NEUTRAL"]


def is_docker_affecting_files(files: Iterable[str]) -> bool:
    """Whether any of the given files change the CI docker image tree hash.

    The docker image tag is derived purely from the .ci/docker tree, so only
    changes under that directory matter (changes to docker-builds.yml or
    .lintrunner.toml re-trigger the build workflow but do not change the tag).
    """
    return any(f == DOCKER_CI_PATH or f.startswith(f"{DOCKER_CI_PATH}/") for f in files)


def get_docker_build_checks(checks: JobNameToStateDict) -> JobNameToStateDict:
    """Return the subset of checks that belong to the docker-builds workflow."""
    return {
        name: check
        for name, check in checks.items()
        if name == DOCKER_BUILDS_WORKFLOW_NAME
        or name.startswith(f"{DOCKER_BUILDS_WORKFLOW_NAME} / ")
    }


def add_workflow_conclusions(
    checksuites: Any,
    get_next_checkruns_page: Callable[[list[dict[str, dict[str, Any]]], int, Any], Any],
    get_next_checksuites: Callable[[Any], Any],
) -> JobNameToStateDict:
    # graphql seems to favor the most recent workflow run, so in theory we
    # shouldn't need to account for reruns, but do it just in case

    # workflow -> job -> job info
    workflows: dict[str, WorkflowCheckState] = {}

    # for the jobs that don't have a workflow
    no_workflow_obj: WorkflowCheckState = WorkflowCheckState("", "", 0, None)

    def add_conclusions(edges: Any) -> None:
        for edge_idx, edge in enumerate(edges):
            node = edge["node"]
            workflow_run = node["workflowRun"]
            checkruns = node["checkRuns"]

            workflow_obj: WorkflowCheckState = no_workflow_obj

            if workflow_run is not None:
                # This is the usual workflow run ID we see on GitHub
                workflow_run_id = workflow_run["databaseId"]
                # While this is the metadata name and ID of the workflow itself
                workflow_name = workflow_run["workflow"]["name"]
                workflow_id = workflow_run["workflow"]["databaseId"]

                workflow_conclusion = node["conclusion"]
                # Do not override existing status with cancelled
                if workflow_conclusion == "CANCELLED" and workflow_name in workflows:
                    continue

                # Only keep the latest workflow run for each workflow, heuristically,
                # it's the run with largest run ID
                if (
                    workflow_id not in workflows
                    or workflows[workflow_id].run_id < workflow_run_id
                ):
                    workflows[workflow_id] = WorkflowCheckState(
                        name=workflow_name,
                        status=workflow_conclusion,
                        url=workflow_run["url"],
                        run_id=workflow_run_id,
                    )
                workflow_obj = workflows[workflow_id]

            while checkruns is not None:
                for checkrun_node in checkruns["nodes"]:
                    if not isinstance(checkrun_node, dict):
                        warn(f"Expected dictionary, but got {type(checkrun_node)}")
                        continue
                    checkrun_name = f"{get_check_run_name_prefix(workflow_run)}{checkrun_node['name']}"
                    existing_checkrun = workflow_obj.jobs.get(checkrun_name)
                    if existing_checkrun is None or not is_passing_status(
                        existing_checkrun.status
                    ):
                        workflow_obj.jobs[checkrun_name] = JobCheckState(
                            checkrun_name,
                            checkrun_node["detailsUrl"],
                            checkrun_node["conclusion"],
                            classification=None,
                            job_id=checkrun_node["databaseId"],
                            title=checkrun_node["title"],
                            summary=checkrun_node["summary"],
                        )

                if bool(checkruns["pageInfo"]["hasNextPage"]):
                    checkruns = get_next_checkruns_page(edges, edge_idx, checkruns)
                else:
                    checkruns = None

    all_edges = checksuites["edges"].copy()
    while bool(checksuites["pageInfo"]["hasNextPage"]):
        checksuites = get_next_checksuites(checksuites)
        all_edges.extend(checksuites["edges"])

    add_conclusions(all_edges)

    # Flatten the dictionaries.  If there exists jobs in the workflow run, put
    # the jobs in but don't put the workflow in.  We care more about the jobs in
    # the workflow that ran than the container workflow.
    res: JobNameToStateDict = {}
    for workflow in workflows.values():
        if len(workflow.jobs) > 0:
            for job_name, job in workflow.jobs.items():
                res[job_name] = job
        else:
            res[workflow.name] = JobCheckState(
                workflow.name,
                workflow.url,
                workflow.status,
                classification=None,
                job_id=None,
                title=None,
                summary=None,
            )
    for job_name, job in no_workflow_obj.jobs.items():
        res[job_name] = job
    return res


def parse_args() -> Any:
    from argparse import ArgumentParser

    parser = ArgumentParser("Merge PR into default branch")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--revert", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--ignore-current", action="store_true")
    parser.add_argument("--check-mergeability", action="store_true")
    parser.add_argument("--comment-id", type=int)
    parser.add_argument("--reason", type=str)
    parser.add_argument("pr_num", type=int)
    return parser.parse_args()


def _unedited_comment_author(
    pr: GitHubPR, comment_id: int | None
) -> tuple[str, str | None] | None:
    """(author_login, author_url) of the triggering comment.

    None when there is no comment or it has been edited, so a caller can never
    act on an identity someone may have rewritten.
    """
    if comment_id is None:
        return None
    comment = pr.get_comment_by_id(comment_id)
    if comment.editor_login is not None:
        return None
    return comment.author_login, comment.author_url


def can_skip_internal_checks(pr: GitHubPR, comment_id: int | None = None) -> bool:
    author = _unedited_comment_author(pr, comment_id)
    if author is None:
        return False
    author_login, author_url = author
    if author_login == "facebook-github-bot":
        return True
    # facebook-github-tools is a GitHub App; identify by its app URL.
    return author_url == "https://github.com/apps/facebook-github-tools"


# Bot identities that auto-merge a co-dev PR once its internal Phabricator diff
# has landed. `meta-codesync` took this over from `facebook-github-tools`, so
# both are listed: the older one still appears on long-lived PRs.
#
# Matched on login rather than on the App URL that can_skip_internal_checks
# uses. Only `comments(last: 5)` selects `author { url }` — GH_GET_PR_PREV_COMMENTS
# and the reviews fragment select `login` alone, so author_url is silently None
# for any comment older than the prefetched window (see _comment_from_node's
# `.get("url", None)`). `login` is selected everywhere, and a `[bot]` suffix is
# unforgeable because GitHub usernames cannot contain brackets.
#
# Deliberately NOT reusing can_skip_internal_checks' allowlist. That predicate
# also waives the "must be landed via Phabricator" guard, so adding an identity
# to it is a trust decision; this list only decides whether a missing
# release-notes label is auto-filled.
CODEV_MERGE_BOT_LOGINS = frozenset(
    {
        "facebook-github-bot",
        "facebook-github-tools[bot]",
        "meta-codesync[bot]",
    }
)
# Same identities by App URL, for the queries that do select it.
CODEV_MERGE_BOT_APP_URLS = frozenset(
    {
        "https://github.com/apps/facebook-github-tools",
        "https://github.com/apps/meta-codesync",
    }
)


def is_bot_initiated_codev_merge(pr: GitHubPR, comment_id: int | None) -> bool:
    # A co-dev merge is initiated by one of the Meta export bots once the
    # internal Phabricator diff has landed. Such a PR always carries a
    # "Differential Revision:" line (get_diff_revision); the local check runs
    # first so a human merge does not pay a comment fetch just to be rejected.
    if pr.get_diff_revision() is None:
        return False
    author = _unedited_comment_author(pr, comment_id)
    if author is None:
        return False
    author_login, author_url = author
    return (
        author_login in CODEV_MERGE_BOT_LOGINS or author_url in CODEV_MERGE_BOT_APP_URLS
    )


def ensure_mergeable_labels(
    pr: GitHubPR, comment_id: int | None, dry_run: bool
) -> None:
    if has_required_labels(pr):
        return
    # A co-dev merge runs after the internal diff already landed, so the author
    # can no longer act on a missing release-notes label. Auto-apply
    # "topic: not user facing" to unblock (as maintainers do manually today) and
    # leave a comment so the choice is auditable and can be corrected if the
    # change is in fact user facing. Human-initiated merges still fail.
    if is_bot_initiated_codev_merge(pr, comment_id):
        # Comment BEFORE labeling. If the label went on first and the comment
        # then failed, the merge would abort with the PR already satisfying
        # has_required_labels — so the retry would take the early return above
        # and the promised audit trail would never be written.
        gh_post_pr_comment(
            pr.org,
            pr.project,
            pr.pr_num,
            f"Adding the `{NOT_USER_FACING_LABEL}` label automatically: this co-dev "
            "PR is being merged after its internal diff landed, without a "
            "release-notes label. If this change is user facing, please replace "
            "the label with the appropriate `release notes: ...` one.",
            dry_run,
        )
        gh_add_labels(pr.org, pr.project, pr.pr_num, [NOT_USER_FACING_LABEL], dry_run)
    else:
        raise RuntimeError(LABEL_ERR_MSG.lstrip(" #"))


def _revlist_to_prs(
    repo: GitRepo,
    pr: GitHubPR,
    rev_list: Iterable[str],
    should_skip: Callable[[int, GitHubPR], bool] | None = None,
) -> list[tuple[GitHubPR, str]]:
    rc: list[tuple[GitHubPR, str]] = []
    for idx, rev in enumerate(rev_list):
        msg = repo.commit_message(rev)
        # findall doesn't return named captures, so we need to use finditer
        all_matches = list(RE_PULL_REQUEST_RESOLVED.finditer(msg))
        if len(all_matches) != 1:
            raise RuntimeError(
                f"Found an unexpected number of PRs mentioned in commit {rev}: "
                f"{len(all_matches)}.  This is probably because you are using an "
                "old version of ghstack.  Please update ghstack and resubmit "
                "your PRs"
            )

        m = all_matches[0]
        if pr.org != m.group("owner") or pr.project != m.group("repo"):
            raise RuntimeError(
                f"PR {m.group('number')} resolved to wrong owner/repo pair"
            )
        pr_num = int(m.group("number"))
        candidate = GitHubPR(pr.org, pr.project, pr_num) if pr_num != pr.pr_num else pr
        if should_skip is not None and should_skip(idx, candidate):
            continue
        rc.append((candidate, rev))
    return rc


def get_ghstack_prs(
    repo: GitRepo, pr: GitHubPR, open_only: bool = True
) -> list[tuple[GitHubPR, str]]:
    """
    Get the PRs in the stack that are below this PR (inclusive).  Throws error if any of the open PRs are out of sync.
    @:param open_only: Only return open PRs
    """
    # For ghstack, cherry-pick commits based from origin
    orig_ref = f"{repo.remote}/{pr.get_ghstack_orig_ref()}"
    rev_list = repo.revlist(f"{pr.default_branch()}..{orig_ref}")

    def skip_func(idx: int, candidate: GitHubPR) -> bool:
        if not open_only or not candidate.is_closed():
            return False
        print(
            f"Skipping {idx + 1} of {len(rev_list)} PR (#{candidate.pr_num}) as its already been merged"
        )
        return True

    if not pr.is_ghstack_pr():
        raise AssertionError(
            f"Expected PR #{pr.pr_num} to be a ghstack PR, but head_ref is {pr.head_ref()}"
        )
    entire_stack = _revlist_to_prs(repo, pr, reversed(rev_list), skip_func)
    print(
        f"Found {len(entire_stack)} PRs in the stack for {pr.pr_num}: {[x[0].pr_num for x in entire_stack]}"
    )

    for stacked_pr, rev in entire_stack:
        if stacked_pr.is_closed():
            continue
        base_ref = stacked_pr.base_ref()
        if base_ref == pr.default_branch():
            base_ref = repo.get_merge_base(
                f"{repo.remote}/{base_ref}", f"{repo.remote}/{stacked_pr.head_ref()}"
            )
        if not are_ghstack_branches_in_sync(repo, stacked_pr.head_ref(), base_ref):
            raise RuntimeError(
                f"PR {stacked_pr.pr_num} is out of sync with the corresponding revision {rev} on "
                + f"branch {stacked_pr.get_ghstack_orig_ref()} that would be merged into {stacked_pr.default_branch()}.  "
                + "This usually happens because there is a non ghstack change in the PR.  "
                + f"Please sync them and try again (ex. make the changes on {orig_ref} and run ghstack)."
            )
    return entire_stack


class GitHubPR:
    def __init__(self, org: str, project: str, pr_num: int) -> None:
        if not isinstance(pr_num, int):
            raise AssertionError(
                f"pr_num must be an int, got {type(pr_num).__name__}: {pr_num}"
            )
        self.org = org
        self.project = project
        self.pr_num = pr_num
        self.info = gh_get_pr_info(org, project, pr_num)
        self.changed_files: list[str] | None = None
        self.labels: list[str] | None = None
        self.conclusions: JobNameToStateDict | None = None
        self.comments: list[GitHubComment] | None = None
        self._authors: list[tuple[str, str]] | None = None
        self._reviews: list[tuple[str, str]] | None = None
        self.merge_base: str | None = None
        self.submodules: list[str] | None = None

    def is_closed(self) -> bool:
        return bool(self.info["closed"])

    def is_cross_repo(self) -> bool:
        return bool(self.info["isCrossRepository"])

    def base_ref(self) -> str:
        return cast(str, self.info["baseRefName"])

    def default_branch(self) -> str:
        return cast(str, self.info["baseRepository"]["defaultBranchRef"]["name"])

    def head_ref(self) -> str:
        return cast(str, self.info["headRefName"])

    def is_ghstack_pr(self) -> bool:
        return RE_GHSTACK_HEAD_REF.match(self.head_ref()) is not None

    def get_ghstack_orig_ref(self) -> str:
        if not self.is_ghstack_pr():
            raise AssertionError(
                f"get_ghstack_orig_ref called on non-ghstack PR #{self.pr_num}"
            )
        return re.sub(r"/head$", "/orig", self.head_ref())

    def is_base_repo_private(self) -> bool:
        return bool(self.info["baseRepository"]["isPrivate"])

    def get_changed_files_count(self) -> int:
        return int(self.info["changedFiles"])

    def last_commit(self) -> Any:
        return self.info["commits"]["nodes"][-1]["commit"]

    def last_commit_sha(self, default: str | None = None) -> str:
        # for commits, the oid is the sha

        if default is None:
            return str(self.last_commit()["oid"])

        return str(self.last_commit().get("oid", default))

    def get_merge_base(self) -> str:
        if self.merge_base:
            return self.merge_base

        last_commit_sha = self.last_commit_sha()
        # NB: We could use self.base_ref() here for regular PR, however, that doesn't
        # work for ghstack where the base is the custom branch, i.e. gh/USER/ID/base,
        # so let's just use main instead
        self.merge_base = gh_fetch_merge_base(
            self.org, self.project, last_commit_sha, self.default_branch()
        )

        # Fallback to baseRefOid if the API call fails, i.e. rate limit. Note that baseRefOid
        # points to the base ref associated with the PR or, in other words, the head of main
        # when the PR is created or rebased. This is not necessarily the merge base commit,
        # but it could serve as a fallback in most cases and it's readily available as part
        # of the PR info
        if not self.merge_base:
            self.merge_base = cast(str, self.info["baseRefOid"])

        return self.merge_base

    def get_changed_files(self) -> list[str]:
        if self.changed_files is None:
            info = self.info
            unique_changed_files = set()
            # Do not try to fetch more than 10K files
            for _ in range(100):
                unique_changed_files.update([x["path"] for x in info["files"]["nodes"]])
                if not info["files"]["pageInfo"]["hasNextPage"]:
                    break
                rc = gh_graphql(
                    GH_GET_PR_NEXT_FILES_QUERY,
                    name=self.project,
                    owner=self.org,
                    number=self.pr_num,
                    cursor=info["files"]["pageInfo"]["endCursor"],
                )
                info = rc["data"]["repository"]["pullRequest"]
            self.changed_files = list(unique_changed_files)

        if len(self.changed_files) != self.get_changed_files_count():
            raise RuntimeError("Changed file count mismatch")
        return self.changed_files

    def get_submodules(self) -> list[str]:
        if self.submodules is None:
            rc = gh_graphql(GH_GET_REPO_SUBMODULES, name=self.project, owner=self.org)
            info = rc["data"]["repository"]["submodules"]
            self.submodules = [s["path"] for s in info["nodes"]]
        return self.submodules

    def get_changed_submodules(self) -> list[str]:
        submodules = self.get_submodules()
        return [f for f in self.get_changed_files() if f in submodules]

    def is_docker_affecting(self) -> bool:
        """Whether this PR modifies files that change the CI docker image tag."""
        return is_docker_affecting_files(self.get_changed_files())

    def has_invalid_submodule_updates(self) -> bool:
        """Submodule updates in PR are invalid if submodule keyword
        is not mentioned in neither the title nor body/description
        nor in any of the labels.
        """
        return (
            len(self.get_changed_submodules()) > 0
            and "submodule" not in self.get_title().lower()
            and "submodule" not in self.get_body().lower()
            and all("submodule" not in label for label in self.get_labels())
        )

    def _get_reviews(self) -> list[tuple[str, str]]:
        if self._reviews is None:
            self._reviews = []
            info = self.info
            for _ in range(100):
                nodes = info["reviews"]["nodes"]
                self._reviews = [
                    (node["author"]["login"], node["state"]) for node in nodes
                ] + self._reviews
                if not info["reviews"]["pageInfo"]["hasPreviousPage"]:
                    break
                rc = gh_graphql(
                    GH_GET_PR_PREV_REVIEWS_QUERY,
                    name=self.project,
                    owner=self.org,
                    number=self.pr_num,
                    cursor=info["reviews"]["pageInfo"]["startCursor"],
                )
                info = rc["data"]["repository"]["pullRequest"]
        reviews = {
            author: state for author, state in self._reviews if state != "COMMENTED"
        }
        return list(reviews.items())

    def get_approved_by(self) -> list[str]:
        return [login for (login, state) in self._get_reviews() if state == "APPROVED"]

    def get_changes_requested_by(self) -> list[str]:
        return [
            login
            for (login, state) in self._get_reviews()
            if state == "CHANGES_REQUESTED"
        ]

    def get_updated_at(self) -> str | None:
        """The PR's `updated_at` timestamp, or None if it could not be read.

        Read over REST rather than added to GH_GET_PR_INFO_QUERY because that query's
        text is the cache key for test_trymerge.py's recorded GraphQL fixtures, so
        adding a field to it invalidates every one of them.
        """
        try:
            info = gh_fetch_json_dict(
                f"{GITHUB_API_URL}/repos/{self.org}/{self.project}/pulls/{self.pr_num}"
            )
            updated_at = info.get("updated_at")
            return str(updated_at) if updated_at else None
        except Exception as e:
            print(f"Warning: failed to read updated_at for PR #{self.pr_num}: {e}")
            traceback.print_exc()
            return None

    def get_bot_reviewers(self) -> frozenset[str] | None:
        """Logins of this PR's reviewers that GitHub reports as Bot accounts, or None.

        GraphQL renders an App's login bare, so `_get_reviews` cannot tell a GitHub App
        apart from a human; only the REST reviews payload carries `user.type`. GitHub
        returns reviews oldest first, so every page has to be walked to see the recent
        ones; a PR with more than `REVIEW_PAGE_LIMIT` pages of reviews leaves the tail
        unclassified.

        None means GitHub could not be asked, which callers must not read as "this PR
        has no App reviewers": that would turn a transient GitHub failure into a
        permanent merge refusal.
        """
        bots = set()
        try:
            for page in range(1, REVIEW_PAGE_LIMIT + 1):
                reviews = gh_fetch_json_list(
                    f"{GITHUB_API_URL}/repos/{self.org}/{self.project}/pulls/{self.pr_num}/reviews",
                    params={"per_page": REVIEWS_PER_PAGE, "page": page},
                )
                for review in reviews:
                    user = review.get("user")
                    if not isinstance(user, dict):
                        continue
                    if str(user.get("type", "")).lower() == "bot" and user.get("login"):
                        bots.add(str(user["login"]))
                if len(reviews) < REVIEWS_PER_PAGE:
                    break
        except Exception as e:
            print(f"Warning: failed to read reviewer types for PR #{self.pr_num}: {e}")
            traceback.print_exc()
            return None
        return frozenset(bots)

    def get_commit_count(self) -> int:
        return int(self.info["commits_with_authors"]["totalCount"])

    def get_commit_sha_at_comment(self, comment_id: int) -> str | None:
        """
        Get the PR head commit SHA that was present when a specific comment was posted.
        This ensures we only merge the state of the PR at the time the merge command was issued,
        not any subsequent commits that may have been pushed after.

        Returns None if no head-changing events found before the comment or if the comment was not found.
        """
        head = None

        try:
            for event in iter_issue_timeline_until_comment(
                self.org, self.project, self.pr_num, comment_id
            ):
                etype = event.get("event")
                if etype == "committed":
                    sha = sha_from_committed_event(event)
                    if sha:
                        head = sha
                        print(f"Timeline: Found commit event for SHA {sha}")
                elif etype == "head_ref_force_pushed":
                    sha = sha_from_force_push_after(event)
                    if sha:
                        head = sha
                        print(f"Timeline: Found force push event for SHA {sha}")
                elif etype == "commented":
                    if event.get("id") == comment_id:
                        print(f"Timeline: Found final comment with sha {sha}")
                        return head
        except Exception as e:
            print(
                f"Warning: Failed to reconstruct timeline for comment {comment_id}: {e}"
            )
            return None

        print(f"Did not find comment with id {comment_id} in the PR timeline")
        return None

    def get_pr_creator_login(self) -> str:
        return cast(str, self.info["author"]["login"])

    def is_dependabot_pr(self) -> bool:
        return self.get_pr_creator_login() == "dependabot[bot]"

    def _fetch_authors(self) -> list[tuple[str, str]]:
        if self._authors is not None:
            return self._authors
        authors: list[tuple[str, str]] = []

        def add_authors(info: dict[str, Any]) -> None:
            for node in info["commits_with_authors"]["nodes"]:
                for author_node in node["commit"]["authors"]["nodes"]:
                    user_node = author_node["user"]
                    author = f"{author_node['name']} <{author_node['email']}>"
                    if user_node is None:
                        # If author is not github user, user node will be null
                        authors.append(("", author))
                    else:
                        authors.append((cast(str, user_node["login"]), author))

        info = self.info
        for _ in range(100):
            add_authors(info)
            if not info["commits_with_authors"]["pageInfo"]["hasNextPage"]:
                break
            rc = gh_graphql(
                GH_GET_PR_NEXT_AUTHORS_QUERY,
                name=self.project,
                owner=self.org,
                number=self.pr_num,
                cursor=info["commits_with_authors"]["pageInfo"]["endCursor"],
            )
            info = rc["data"]["repository"]["pullRequest"]
        self._authors = authors
        return authors

    def get_committer_login(self, num: int = 0) -> str:
        return self._fetch_authors()[num][0]

    def get_committer_author(self, num: int = 0) -> str:
        return self._fetch_authors()[num][1]

    def get_labels(self) -> list[str]:
        if self.labels is not None:
            return self.labels
        labels = (
            [node["node"]["name"] for node in self.info["labels"]["edges"]]
            if "labels" in self.info
            else []
        )
        self.labels = labels
        return self.labels

    def get_checkrun_conclusions(self) -> JobNameToStateDict:
        """Returns dict of checkrun -> [conclusion, url]"""
        if self.conclusions is not None:
            return self.conclusions
        orig_last_commit = self.last_commit()

        def get_pr_next_check_runs(
            edges: list[dict[str, dict[str, Any]]], edge_idx: int, checkruns: Any
        ) -> Any:
            rc = gh_graphql(
                GH_GET_PR_NEXT_CHECK_RUNS,
                name=self.project,
                owner=self.org,
                number=self.pr_num,
                cs_cursor=edges[edge_idx - 1]["cursor"] if edge_idx > 0 else None,
                cr_cursor=checkruns["pageInfo"]["endCursor"],
            )
            last_commit = rc["data"]["repository"]["pullRequest"]["commits"]["nodes"][
                -1
            ]["commit"]
            checkruns = last_commit["checkSuites"]["nodes"][-1]["checkRuns"]
            return checkruns

        def get_pr_next_checksuites(checksuites: Any) -> Any:
            rc = gh_graphql(
                GH_GET_PR_NEXT_CHECKSUITES,
                name=self.project,
                owner=self.org,
                number=self.pr_num,
                cursor=checksuites["edges"][-1]["cursor"],
            )
            info = rc["data"]["repository"]["pullRequest"]
            last_commit = info["commits"]["nodes"][-1]["commit"]
            if last_commit["oid"] != orig_last_commit["oid"]:
                raise RuntimeError("Last commit changed on PR")
            return last_commit["checkSuites"]

        checksuites = orig_last_commit["checkSuites"]

        self.conclusions = add_workflow_conclusions(
            checksuites, get_pr_next_check_runs, get_pr_next_checksuites
        )

        # Append old style statuses(like ones populated by CircleCI or EasyCLA) to conclusions
        if orig_last_commit["status"] and orig_last_commit["status"]["contexts"]:
            for status in orig_last_commit["status"]["contexts"]:
                name = status["context"]
                self.conclusions[name] = JobCheckState(
                    name,
                    status["targetUrl"],
                    status["state"],
                    classification=None,
                    job_id=None,
                    title=None,
                    summary=None,
                )

        # Same issue for Claude Code: triggered by comment events, uses a
        # protected environment (bedrock), resulting in ACTION_REQUIRED
        # check suite conclusions that block merges
        self.conclusions.pop("Claude Code", None)

        return self.conclusions

    def get_authors(self) -> dict[str, str]:
        rc = {}
        for idx in range(len(self._fetch_authors())):
            rc[self.get_committer_login(idx)] = self.get_committer_author(idx)

        return rc

    def get_author(self) -> str:
        authors = self.get_authors()
        if len(authors) == 1:
            return next(iter(authors.values()))
        creator = self.get_pr_creator_login()
        # If PR creator is not among authors
        # Assume it was authored by first commit author
        if creator not in authors:
            return self.get_committer_author(0)
        return authors[creator]

    def get_title(self) -> str:
        return cast(str, self.info["title"])

    def get_body(self) -> str:
        return cast(str, self.info["body"])

    def get_merge_commit(self) -> str | None:
        mc = self.info["mergeCommit"]
        return mc["oid"] if mc is not None else None

    def get_pr_url(self) -> str:
        return f"https://github.com/{self.org}/{self.project}/pull/{self.pr_num}"

    @staticmethod
    def _comment_from_node(node: Any) -> GitHubComment:
        editor = node["editor"]
        return GitHubComment(
            body_text=node["bodyText"],
            created_at=node.get("createdAt", ""),
            author_login=node["author"]["login"],
            author_url=node["author"].get("url", None),
            author_association=node["authorAssociation"],
            editor_login=editor["login"] if editor else None,
            database_id=node["databaseId"],
            url=node["url"],
        )

    def get_comments(self) -> list[GitHubComment]:
        if self.comments is not None:
            return self.comments
        self.comments = []
        info = self.info["comments"]
        # Do not try to fetch more than 10K comments
        for _ in range(100):
            self.comments = [
                self._comment_from_node(node) for node in info["nodes"]
            ] + self.comments
            if not info["pageInfo"]["hasPreviousPage"]:
                break
            rc = gh_graphql(
                GH_GET_PR_PREV_COMMENTS,
                name=self.project,
                owner=self.org,
                number=self.pr_num,
                cursor=info["pageInfo"]["startCursor"],
            )
            info = rc["data"]["repository"]["pullRequest"]["comments"]
        return self.comments

    def get_last_comment(self) -> GitHubComment:
        return self._comment_from_node(self.info["comments"]["nodes"][-1])

    def get_comment_by_id(self, database_id: int) -> GitHubComment:
        if self.comments is None:
            # Fastpath - try searching in partial prefetched comments
            for node in self.info["comments"]["nodes"]:
                comment = self._comment_from_node(node)
                if comment.database_id == database_id:
                    return comment

        for comment in self.get_comments():
            if comment.database_id == database_id:
                return comment

        # The comment could have actually been a review left on the PR (the message written alongside the review).
        # (This is generally done to trigger the merge right when a comment is left)
        # Check those review comments to see if one of those was the comment in question.
        for node in self.info["reviews"]["nodes"]:
            # These review comments contain all the fields regular comments need
            comment = self._comment_from_node(node)
            if comment.database_id == database_id:
                return comment

        raise RuntimeError(f"Comment with id {database_id} not found")

    def get_diff_revision(self) -> str | None:
        rc = RE_DIFF_REV.search(self.get_body())
        return rc.group(1) if rc is not None else None

    def has_internal_changes(self) -> bool:
        checkrun_name = INTERNAL_CHANGES_CHECKRUN_NAME
        if self.get_diff_revision() is None:
            return False
        checks = self.get_checkrun_conclusions()
        if checks is None or checkrun_name not in checks:
            return False
        return checks[checkrun_name].status != "SUCCESS"

    def has_no_connected_diff(self) -> bool:
        checkrun_name = INTERNAL_CHANGES_CHECKRUN_NAME
        checks = self.get_checkrun_conclusions()
        if checks is None or checkrun_name not in checks:
            return False
        return checks[checkrun_name].title == HAS_NO_CONNECTED_DIFF_TITLE

    def merge_ghstack_into(
        self,
        repo: GitRepo,
        skip_mandatory_checks: bool,
        comment_id: int | None = None,
        skip_all_rule_checks: bool = False,
        ghstack_prs: list[tuple[GitHubPR, str]] | None = None,
    ) -> list[GitHubPR]:
        if not self.is_ghstack_pr():
            raise AssertionError(
                f"merge_ghstack_into called on non-ghstack PR #{self.pr_num}"
            )
        if ghstack_prs is None:
            ghstack_prs = get_ghstack_prs(
                repo, self, open_only=False
            )  # raises error if out of sync
        pr_dependencies = []
        for pr, rev in ghstack_prs:
            if pr.is_closed():
                pr_dependencies.append(pr)
                continue

            commit_msg = pr.gen_commit_message(
                filter_ghstack=True, ghstack_deps=pr_dependencies
            )
            if pr.pr_num != self.pr_num and not skip_all_rule_checks:
                try:
                    find_matching_merge_rule(
                        pr,
                        repo,
                        skip_mandatory_checks=skip_mandatory_checks,
                        skip_internal_checks=can_skip_internal_checks(self, comment_id),
                    )
                except MergeRuleFailedError as ex:
                    raise type(ex)(
                        f"Merge rule check failed for stacked PR #{pr.pr_num}:\n\n{ex}",
                        ex.rule,
                    ) from ex
            repo.cherry_pick(rev)
            repo.amend_commit_message(commit_msg)
            pr_dependencies.append(pr)
        return [x for x, _ in ghstack_prs if not x.is_closed()]

    def gen_commit_message(
        self,
        filter_ghstack: bool = False,
        ghstack_deps: list[GitHubPR] | None = None,
    ) -> str:
        """Fetches title and body from PR description
        adds reviewed by, pull request resolved and optionally
        filters out ghstack info"""
        # Adding the url here makes it clickable within the Github UI
        approved_by_urls = ", ".join(
            prefix_with_github_url(login) for login in self.get_approved_by()
        )
        # Remove "cc: " line from the message body
        msg_body = re.sub(RE_PR_CC_LINE, "", self.get_body())
        if filter_ghstack:
            msg_body = re.sub(RE_GHSTACK_DESC, "", msg_body)
        msg = self.get_title() + f" (#{self.pr_num})\n\n"
        msg += msg_body

        msg += f"\nPull Request resolved: {self.get_pr_url()}\n"
        msg += f"Approved by: {approved_by_urls}\n"
        if ghstack_deps:
            msg += f"ghstack dependencies: {', '.join([f'#{pr.pr_num}' for pr in ghstack_deps])}\n"

        # Mention PR co-authors, which should be at the end of the message
        # And separated from the body by two newlines
        first_coauthor = True
        for author_login, author_name in self.get_authors().items():
            if author_login != self.get_pr_creator_login():
                if first_coauthor:
                    msg, first_coauthor = (msg + "\n", False)
                msg += f"\nCo-authored-by: {author_name}"

        return msg

    def add_numbered_label(self, label_base: str, dry_run: bool) -> None:
        labels = self.get_labels() if self.labels is not None else []
        full_label = label_base
        count = 0
        for label in labels:
            if label_base in label:
                count += 1
                full_label = f"{label_base}X{count}"
        self.add_label(full_label, dry_run)

    def add_label(self, label: str, dry_run: bool) -> None:
        gh_add_labels(self.org, self.project, self.pr_num, [label], dry_run)

    def merge_into(
        self,
        repo: GitRepo,
        *,
        skip_mandatory_checks: bool = False,
        dry_run: bool = False,
        comment_id: int,
        ignore_current_checks: list[str] | None = None,
        greenlight_wait: GreenlightWaitWindow | None = None,
    ) -> None:
        skip_internal_checks = can_skip_internal_checks(self, comment_id)
        # Raises exception if matching rule is not found
        (
            merge_rule,
            pending_checks,
            failed_checks,
            ignorable_checks,
        ) = find_matching_merge_rule(
            self,
            repo,
            skip_mandatory_checks=skip_mandatory_checks,
            skip_internal_checks=skip_internal_checks,
            ignore_current_checks=ignore_current_checks,
        )
        ghstack_prs: list[tuple[GitHubPR, str]] | None = None
        prs_to_merge = [self]
        if self.is_ghstack_pr():
            ghstack_prs = get_ghstack_prs(repo, self, open_only=False)
            prs_to_merge = [pr for pr, _ in ghstack_prs if not pr.is_closed()]

        check_greenlight_reviewed_head_sha(
            self,
            repo,
            prs_to_merge,
            wait_window=greenlight_wait,
            dry_run=dry_run,
            skip_mandatory_checks=skip_mandatory_checks,
            skip_internal_checks=skip_internal_checks,
            ignore_current_checks=ignore_current_checks,
        )

        # A ghstack merge lands all open PRs below this one. Use the topmost
        # docker-affecting PR because its cumulative head has the final docker
        # tree for which images must have been built. Enforced even on force
        # merges.
        docker_pr = get_topmost_docker_pr(prs_to_merge)
        if docker_pr is not None:
            check_docker_builds_ready(docker_pr)

        # Dependabot commits are authored/signed by the bot; merge them through
        # GitHub's squash+merge API so that signature is preserved and dependabot
        # can track the merge, instead of re-authoring a squash commit locally.
        if self.is_dependabot_pr():
            additional_merged_prs: list[GitHubPR] = []
            merge_commit_sha = self.merge_via_github_api(dry_run)
        else:
            additional_merged_prs = self.merge_changes_locally(
                repo,
                skip_mandatory_checks,
                comment_id,
                ghstack_prs=ghstack_prs,
            )

            # Log, but do not block on, a docker land race.
            if docker_pr is not None:
                warn_on_docker_merge_skew(repo, docker_pr)

            repo.push(self.default_branch(), dry_run)
            # When the merge process reaches this part, we can assume that the
            # commit has been successfully pushed to trunk
            merge_commit_sha = repo.rev_parse(name=self.default_branch())
        if not dry_run:
            self.add_numbered_label(MERGE_COMPLETE_LABEL, dry_run)
            for pr in additional_merged_prs:
                pr.add_numbered_label(MERGE_COMPLETE_LABEL, dry_run)

        if comment_id and self.pr_num:
            # Finally, upload the record to s3. The list of pending and failed
            # checks are at the time of the merge
            save_merge_record(
                comment_id=comment_id,
                pr_num=self.pr_num,
                owner=self.org,
                project=self.project,
                author=self.get_author(),
                pending_checks=pending_checks,
                failed_checks=failed_checks,
                ignore_current_checks=ignorable_checks.get("IGNORE_CURRENT_CHECK", []),
                broken_trunk_checks=ignorable_checks.get("BROKEN_TRUNK", []),
                flaky_checks=ignorable_checks.get("FLAKY", []),
                unstable_checks=ignorable_checks.get("UNSTABLE", []),
                last_commit_sha=self.last_commit_sha(default=""),
                merge_base_sha=self.get_merge_base(),
                merge_commit_sha=merge_commit_sha,
                is_failed=False,
                skip_mandatory_checks=skip_mandatory_checks,
                ignore_current=bool(ignore_current_checks),
            )
        else:
            print("Missing comment ID or PR number, couldn't upload to s3")

        # Usually Github will see that the commit has "resolves <pr_num>" in the
        # commit message and close the PR, but sometimes it doesn't, leading to
        # confusion.  When it doesn't, we close it manually.
        time.sleep(60)  # Give Github some time to close the PR
        manually_close_merged_pr(
            pr=self,
            additional_merged_prs=additional_merged_prs,
            merge_commit_sha=merge_commit_sha,
            dry_run=dry_run,
        )

    def merge_via_github_api(self, dry_run: bool = False) -> str:
        """Squash-and-merge this PR through GitHub's merge API, pinned to the
        commit that was reviewed, and return the resulting merge commit sha."""
        msg = self.gen_commit_message()
        title, _, body = msg.partition("\n\n")
        return gh_merge_pr(
            self.org,
            self.project,
            self.pr_num,
            merge_method="squash",
            commit_title=title,
            commit_message=body,
            sha=self.last_commit_sha(),
            dry_run=dry_run,
        )

    def merge_changes_locally(
        self,
        repo: GitRepo,
        skip_mandatory_checks: bool = False,
        comment_id: int | None = None,
        branch: str | None = None,
        skip_all_rule_checks: bool = False,
        ghstack_prs: list[tuple[GitHubPR, str]] | None = None,
    ) -> list[GitHubPR]:
        """
        :param skip_all_rule_checks: If true, skips all rule checks on ghstack PRs, useful for dry-running merge locally
        """
        branch_to_merge_into = self.default_branch() if branch is None else branch
        if repo.current_branch() != branch_to_merge_into:
            repo.checkout(branch_to_merge_into)

        # It's okay to skip the commit SHA check for ghstack PRs since
        # authoring requires write access to the repo.
        if self.is_ghstack_pr():
            return self.merge_ghstack_into(
                repo,
                skip_mandatory_checks,
                comment_id=comment_id,
                skip_all_rule_checks=skip_all_rule_checks,
                ghstack_prs=ghstack_prs,
            )

        msg = self.gen_commit_message()
        pr_branch_name = f"__pull-request-{self.pr_num}__init__"

        # Determine which commit SHA to merge
        commit_to_merge = None
        if not comment_id:
            raise ValueError("Must provide --comment-id when merging regular PRs")

        # Get the commit SHA that was present when the comment was made
        commit_to_merge = self.get_commit_sha_at_comment(comment_id)
        if not commit_to_merge:
            raise RuntimeError(
                f"Could not find commit that was pushed before comment {comment_id}"
            )

        # Validate that this commit is the latest commit on the PR
        latest_commit = self.last_commit_sha()
        if commit_to_merge != latest_commit:
            raise RuntimeError(
                f"Commit {commit_to_merge} was HEAD when comment {comment_id} was posted "
                f"but now the latest commit on the PR is {latest_commit}. "
                f"Please re-issue the merge command to merge the latest commit."
            )

        print(f"Merging commit {commit_to_merge} locally")

        repo.fetch(commit_to_merge, pr_branch_name)
        repo._run_git("merge", "--squash", pr_branch_name)
        repo._run_git("commit", f'--author="{self.get_author()}"', "-m", msg)

        # Did the PR change since we started the merge?
        pulled_sha = repo.show_ref(pr_branch_name)
        latest_pr_status = GitHubPR(self.org, self.project, self.pr_num)
        if (
            pulled_sha != latest_pr_status.last_commit_sha()
            or pulled_sha != commit_to_merge
        ):
            raise RuntimeError(
                "PR has been updated since CI checks last passed. Please rerun the merge command."
            )
        return []


class MergeRuleFailedError(RuntimeError):
    def __init__(self, message: str, rule: MergeRule | None = None) -> None:
        super().__init__(message)
        self.rule = rule


class MandatoryChecksMissingError(MergeRuleFailedError):
    pass


class PostCommentError(Exception):
    pass


@dataclass
class MergeRule:
    name: str
    patterns: list[str]
    approved_by: list[str]
    mandatory_checks_name: list[str] | None
    ignore_flaky_failures: bool = True


def gen_new_issue_link(
    org: str, project: str, labels: list[str], template: str = "bug-report.yml"
) -> str:
    labels_str = ",".join(labels)
    return (
        f"https://github.com/{org}/{project}/issues/new?"
        f"labels={urllib.parse.quote(labels_str)}&"
        f"template={urllib.parse.quote(template)}"
    )


def read_merge_rules(repo: GitRepo | None, org: str, project: str) -> list[MergeRule]:
    """Returns the list of all merge rules for the repo or project.

    NB: this function is used in Meta-internal workflows, see the comment
    at the top of this file for details.
    """
    repo_relative_rules_path = MERGE_RULE_PATH
    if repo is None:
        json_data = gh_fetch_url(
            f"https://api.github.com/repos/{org}/{project}/contents/{repo_relative_rules_path}",
            headers={"Accept": "application/vnd.github.v3+json"},
            reader=json.load,
        )
        content = base64.b64decode(json_data["content"])
        return [MergeRule(**x) for x in yaml.safe_load(content)]
    else:
        rules_path = Path(repo.repo_dir) / repo_relative_rules_path
        if not rules_path.exists():
            print(f"{rules_path} does not exist, returning empty rules")
            return []
        with open(rules_path) as fp:
            rc = yaml.safe_load(fp)
        return [MergeRule(**x) for x in rc]


def rule_approvers(rule: MergeRule) -> set[str]:
    """The logins one rule accepts, with its `org/team-slug` refs expanded to members."""
    approvers: set[str] = set()
    for approver in rule.approved_by:
        if "/" in approver:
            team_org, _, team_name = approver.partition("/")
            if "/" in team_name:
                # gh_get_team_members warns and returns [] for a team that does not
                # exist, and find_matching_merge_rule skips its approver check when a
                # rule resolves to no approvers: a mis-typed ref would match anything.
                raise ValueError(
                    f"approved_by team ref must be 'org/team-slug', got {approver!r}"
                )
            approvers.update(gh_get_team_members(team_org, team_name))
        else:
            approvers.add(approver)
    return approvers


def merge_authorized_logins(
    repo: GitRepo | None, org: str, project: str
) -> frozenset[str]:
    """Every login that authorizes a merge under some rule, lowercased, patterns ignored.

    This is deliberately coarser than `find_matching_merge_rule`, which is case-sensitive
    and only consults the rules whose file patterns cover a PR. It mirrors greenlight's
    `merge_authz.resolve_authorized_logins` (pytorch/test-infra, under
    `greenlight/src/greenlight/`), which is the set greenlight itself tests an approver
    against before deciding a human has already handled a PR. Answering that question
    with a different set would have the guard wait for a review that never comes.

    An unusable rules file yields the empty set rather than an error: the guard would
    otherwise invent a refusal out of a failure the real merge gate hits on the same
    pass, and would blame the wrong thing when it did.
    """
    try:
        rules = read_merge_rules(repo, org, project)
        # frozenset() forces the generator, so rule_approvers' GraphQL calls run here.
        return frozenset(
            normalize_login(login) for rule in rules for login in rule_approvers(rule)
        )
    except Exception as e:
        print(f"Warning: failed to resolve {MERGE_RULE_PATH} for {org}/{project}: {e}")
        traceback.print_exc()
        return frozenset()


def _find_non_matching_files(patterns: list[str], files: list[str]) -> list[str]:
    """Return files that do not match the given patterns.

    Patterns prefixed with '-' are exclusions: a file matches the rule if it
    matches at least one positive pattern AND does not match any negative one.
    """
    positive = [p for p in patterns if not p.startswith("-")]
    negative = [p[1:] for p in patterns if p.startswith("-")]
    patterns_re = patterns_to_regex(positive)
    exclude_re = patterns_to_regex(negative) if negative else None
    return [
        f
        for f in files
        if not patterns_re.match(f) or (exclude_re is not None and exclude_re.match(f))
    ]


def find_matching_merge_rule(
    pr: GitHubPR,
    repo: GitRepo | None = None,
    skip_mandatory_checks: bool = False,
    skip_internal_checks: bool = False,
    ignore_current_checks: list[str] | None = None,
    approved_by_override: set[str] | None = None,
) -> tuple[
    MergeRule,
    list[tuple[str, str | None, int | None]],
    list[tuple[str, str | None, int | None]],
    dict[str, list[Any]],
]:
    """
    Returns merge rule matching to this pr together with the list of associated pending
    and failing jobs OR raises an exception.

    approved_by_override replaces the PR's real approver set, so a caller can ask what
    would happen if a given approval were absent. None keeps the PR's real approvers.

    NB: this function is used in Meta-internal workflows, see the comment at the top of
    this file for details.
    """
    changed_files = pr.get_changed_files()
    approved_by = (
        set(pr.get_approved_by())
        if approved_by_override is None
        else set(approved_by_override)
    )

    issue_link = gen_new_issue_link(
        org=pr.org,
        project=pr.project,
        labels=["module: ci"],
    )
    reject_reason = f"No rule found to match PR. Please [report]{issue_link} this issue to DevX team."

    rules = read_merge_rules(repo, pr.org, pr.project)
    if not rules:
        reject_reason = f"Rejecting the merge as no rules are defined for the repository in {MERGE_RULE_PATH}"
        raise RuntimeError(reject_reason)

    checks = pr.get_checkrun_conclusions()
    checks = get_classifications(
        pr.pr_num,
        pr.project,
        checks,
        ignore_current_checks=ignore_current_checks,
    )

    # This keeps the list of all approvers that could stamp the change
    all_rule_approvers = {}

    # PRs can fail multiple merge rules, but it only needs to pass one rule to be approved.
    # If it fails all rules, we need to find the rule that it came closest to passing and report
    # that to the dev.
    #
    # reject_reason_score ranks rules by relevancy. The higher the score, the more relevant the
    # rule & rejection reason, and we only care about the most relevant rule/reason
    #
    # reject_reason_score intrepretation:
    # Score 0 to 10K - how many files rule matched
    # Score 10K - matched all files, but no overlapping approvers
    # Score 20K - matched all files and approvers, but mandatory checks are pending
    # Score 30k - Matched all files and approvers, but mandatory checks failed
    reject_reason_score = 0
    for rule in rules:
        rule_name = rule.name
        non_matching_files = _find_non_matching_files(rule.patterns, changed_files)
        if len(non_matching_files) > 0:
            num_matching_files = len(changed_files) - len(non_matching_files)
            if num_matching_files > reject_reason_score:
                reject_reason_score = num_matching_files
                reject_reason = "\n".join(
                    (
                        f"Not all files match rule `{rule_name}`.",
                        f"{num_matching_files} files matched, but there are still non-matching files:",
                        f"{','.join(non_matching_files[:5])}{', ...' if len(non_matching_files) > 5 else ''}",
                    )
                )
            continue

        # If rule needs approvers but PR has not been reviewed, skip it
        if len(rule.approved_by) > 0 and len(approved_by) == 0:
            if reject_reason_score < 10000:
                reject_reason_score = 10000
                reject_reason = f"PR #{pr.pr_num} has not been reviewed yet"
            continue

        # Does the PR have the required approvals for this rule?
        this_rule_approvers = rule_approvers(rule)
        approvers_intersection = approved_by.intersection(this_rule_approvers)
        # If rule requires approvers but they aren't the ones that reviewed PR
        if len(approvers_intersection) == 0 and len(this_rule_approvers) > 0:
            # Less than or equal is intentionally used here to gather all potential
            # approvers
            if reject_reason_score <= 10000:
                reject_reason_score = 10000

                all_rule_approvers[rule.name] = rule.approved_by
                # Prepare the reject reason
                all_rule_approvers_msg = [
                    f"- {name} ({', '.join(approved_by[:5])}{', ...' if len(approved_by) > 5 else ''})"
                    for name, approved_by in all_rule_approvers.items()
                ]

                reject_reason = "Approvers from one of the following sets are needed:\n"
                reject_reason += "\n".join(all_rule_approvers_msg)

            continue

        # Does the PR pass the checks required by this rule?
        mandatory_checks = (
            rule.mandatory_checks_name if rule.mandatory_checks_name is not None else []
        )
        required_checks = list(
            filter(
                lambda x: ("EasyCLA" in x)
                or ("Facebook CLA Check" in x)
                or not skip_mandatory_checks,
                mandatory_checks,
            )
        )
        pending_checks, failed_checks, _ = categorize_checks(
            checks,
            required_checks,
            ok_failed_checks_threshold=IGNORABLE_FAILED_CHECKS_THESHOLD
            if rule.ignore_flaky_failures
            else 0,
        )

        # categorize_checks assumes all tests are required if required_checks is empty.
        # this is a workaround as we want to keep that behavior for categorize_checks
        # generally.
        if not required_checks:
            pending_checks = []
            failed_checks = []

        hud_link = f"https://hud.pytorch.org/{pr.org}/{pr.project}/commit/{pr.last_commit_sha()}"
        if len(failed_checks) > 0:
            if reject_reason_score < 30000:
                reject_reason_score = 30000
                reject_reason = "\n".join(
                    (
                        f"{len(failed_checks)} mandatory check(s) failed.  The first few are:",
                        *checks_to_markdown_bullets(failed_checks),
                        "",
                        f"Dig deeper by [viewing the failures on hud]({hud_link})",
                    )
                )
            continue
        elif len(pending_checks) > 0:
            if reject_reason_score < 20000:
                reject_reason_score = 20000
                reject_reason = "\n".join(
                    (
                        f"{len(pending_checks)} mandatory check(s) are pending/not yet run.  The first few are:",
                        *checks_to_markdown_bullets(pending_checks),
                        "",
                        f"Dig deeper by [viewing the pending checks on hud]({hud_link})",
                    )
                )
            continue

        if not skip_internal_checks and pr.has_internal_changes():
            raise RuntimeError(
                "This PR has internal changes and must be landed via Phabricator! Please try reimporting/rexporting the PR!"
            )

        # Categorize all checks when skip_mandatory_checks (force merge) is set. Do it here
        # where the list of checks is readily available. These records will be saved into
        # s3 merge records
        (
            pending_mandatory_checks,
            failed_mandatory_checks,
            ignorable_checks,
        ) = categorize_checks(
            checks,
            [],
            ok_failed_checks_threshold=IGNORABLE_FAILED_CHECKS_THESHOLD,
        )
        return (
            rule,
            pending_mandatory_checks,
            failed_mandatory_checks,
            ignorable_checks,
        )

    if reject_reason_score == 20000:
        raise MandatoryChecksMissingError(reject_reason, rule)
    raise MergeRuleFailedError(reject_reason, rule)


# One merge command runs as one process, and its retry loop re-asks this question every
# five minutes with the same answer. The repeated call is not free: find_matching_merge_rule
# posts to Dr.CI, which rewrites the PR's Dr.CI comment as a side effect.
_AUTHORIZED_WITHOUT_GREENLIGHT: dict[tuple[Any, ...], bool] = {}


def is_authorized_without_greenlight(
    pr: GitHubPR,
    repo: GitRepo | None,
    *,
    skip_mandatory_checks: bool = False,
    skip_internal_checks: bool = False,
    ignore_current_checks: list[str] | None = None,
) -> bool:
    """Whether some merge rule still matches once greenlight's approval is dropped.

    Asking "is greenlight the only approver" instead would be unsound: pytorch/pytorch
    is public, so any stranger can approve a PR, and that would switch the land-time
    greenlight check off without contributing anything to the merge's authorization.

    The caller passes the same gate arguments `merge_into` gives the real
    `find_matching_merge_rule`. Relaxing any of them here would let a rule that the real
    gate rejects answer this question, and the real gate would then fall through to the
    Greenlight rule -- leaving greenlight the sole authority with the guard switched off.

    An answer is cached per approver set and head commit. Both are re-read from GitHub on
    every retry, so an approval added mid-merge still takes effect; what the cache drops
    is the identical re-run, and with it a redundant Dr.CI comment rewrite.
    """
    approvers = pr.get_approved_by()
    key = (
        pr.org,
        pr.project,
        pr.pr_num,
        pr.last_commit_sha(default=""),
        frozenset(approvers),
        skip_mandatory_checks,
        skip_internal_checks,
        tuple(ignore_current_checks or ()),
    )
    if key in _AUTHORIZED_WITHOUT_GREENLIGHT:
        return _AUTHORIZED_WITHOUT_GREENLIGHT[key]
    remaining = {login for login in approvers if not is_greenlight(login)}
    try:
        find_matching_merge_rule(
            pr,
            repo,
            skip_mandatory_checks=skip_mandatory_checks,
            skip_internal_checks=skip_internal_checks,
            ignore_current_checks=ignore_current_checks,
            approved_by_override=remaining,
        )
        authorized = True
    except MergeRuleFailedError as e:
        print(
            f"PR #{pr.pr_num} has no merge rule without greenlight's approval, so the "
            f"greenlight land-time check applies: {e}"
        )
        authorized = False
    except RuntimeError as e:
        # An unusable rule set or an internal-changes PR: the real gate would refuse the
        # merge outright, so this hypothetical proves nothing about greenlight's role.
        print(
            f"PR #{pr.pr_num} could not be evaluated without greenlight's approval, so "
            f"the greenlight land-time check applies: {e}"
        )
        traceback.print_exc()
        authorized = False
    _AUTHORIZED_WITHOUT_GREENLIGHT[key] = authorized
    return authorized


def check_greenlight_reviewed_head_sha(
    pr: GitHubPR,
    repo: GitRepo | None,
    prs_to_merge: list[GitHubPR],
    *,
    wait_window: GreenlightWaitWindow | None,
    dry_run: bool = False,
    skip_mandatory_checks: bool = False,
    skip_internal_checks: bool = False,
    ignore_current_checks: list[str] | None = None,
) -> None:
    """Block a merge that only greenlight authorizes and that greenlight has not
    approved at the head commit being landed.

    Raises MandatoryChecksMissingError to wait, because merge()'s retry loop catches
    exactly that type and re-enters merge_into five minutes later. wait_window is what
    bounds that waiting across those re-entries; None means the caller cannot retry.

    The gate arguments are the ones `merge_into` gave the real `find_matching_merge_rule`,
    so that asking whether greenlight's approval is load-bearing asks it of the same gate.

    The ghstack path lands a cherry-pick of each PR's head rather than the head itself,
    but get_ghstack_prs has already proven the two carry identical content, and the head
    is what greenlight records, so the head is what gets compared.
    """
    # One rules file governs the whole stack, so every PR asks the same question of it.
    authorized_logins = partial(merge_authorized_logins, repo, pr.org, pr.project)
    result = evaluate_greenlight_guard(
        f"{pr.org}/{pr.project}",
        [
            PRUnderMerge(
                pr_num=stacked.pr_num,
                head_sha=stacked.last_commit_sha(default=""),
                approved_by=stacked.get_approved_by(),
                changes_requested_by=stacked.get_changes_requested_by(),
                labels=stacked.get_labels(),
                get_updated_at=stacked.get_updated_at,
                get_bot_reviewers=stacked.get_bot_reviewers,
                get_merge_authorized_logins=authorized_logins,
                is_authorized_without_greenlight=partial(
                    is_authorized_without_greenlight,
                    stacked,
                    repo,
                    skip_mandatory_checks=skip_mandatory_checks,
                    skip_internal_checks=skip_internal_checks,
                    ignore_current_checks=ignore_current_checks,
                ),
            )
            for stacked in prs_to_merge
        ],
        wait_window=wait_window,
    )
    if result.comment:
        try:
            gh_post_pr_comment(pr.org, pr.project, pr.pr_num, result.comment, dry_run)
        except Exception as e:
            # The comment is a courtesy; failing to post it must not end a merge that
            # the guard itself is willing to keep waiting on.
            print(
                f"Warning: failed to comment the greenlight wait on PR #{pr.pr_num}: {e}"
            )
            traceback.print_exc()
    if result.verdict is GuardVerdict.WAIT:
        raise MandatoryChecksMissingError(result.message)
    if result.verdict is GuardVerdict.DENY:
        raise MergeRuleFailedError(result.message)


def get_topmost_docker_pr(prs: list[GitHubPR]) -> GitHubPR | None:
    """Find the highest docker-affecting PR in a bottom-to-top stack."""
    return next((pr for pr in reversed(prs) if pr.is_docker_affecting()), None)


def check_docker_builds_ready(pr: GitHubPR) -> None:
    """Block merge of a docker-affecting PR unless its docker images have been
    pre-built.

    PRs that change .ci/docker must run the docker-builds (ciflow/docker)
    workflow so the images are built and pushed to ECR before landing.  If they
    aren't, every trunk job that needs one of those images fails because it
    can't find the image (see the #190927 / #186302 land race).  This gate is
    enforced even for force merges, since -f is exactly what bypassed it before.
    """
    if not pr.is_docker_affecting():
        return

    docker_checks = get_docker_build_checks(pr.get_checkrun_conclusions())

    if not docker_checks:
        raise MergeRuleFailedError(
            f"This PR changes files under {DOCKER_CI_PATH}/, but the "
            f"`{DOCKER_BUILDS_WORKFLOW_NAME}` workflow has not run on it. The CI "
            "docker images must be pre-built and pushed to ECR before this lands, "
            "otherwise trunk jobs will not be able to find the image they need. "
            "Please add the `ciflow/docker` label to this PR, wait for the docker "
            "builds to finish, and then re-issue the merge command."
        )

    pending = sorted(name for name, c in docker_checks.items() if c.status is None)
    failed = sorted(
        name
        for name, c in docker_checks.items()
        if c.status is not None and not is_passing_status(c.status)
    )

    if pending:
        # Raise MandatoryChecksMissingError so that a normal (non-force) merge
        # keeps retrying until the docker builds finish, mirroring how other
        # mandatory checks are waited on.
        raise MandatoryChecksMissingError(
            f"This PR changes files under {DOCKER_CI_PATH}/, so the "
            f"`{DOCKER_BUILDS_WORKFLOW_NAME}` builds must finish before merging. "
            f"Still waiting for {len(pending)} docker build job(s), the first few "
            f"are: {', '.join(pending[:5])}"
        )

    if failed:
        raise MergeRuleFailedError(
            f"This PR changes files under {DOCKER_CI_PATH}/, so the "
            f"`{DOCKER_BUILDS_WORKFLOW_NAME}` builds must all pass before merging, "
            f"but {len(failed)} of them failed, the first few are: "
            f"{', '.join(failed[:5])}. The docker images could not be built, so "
            "trunk jobs would be unable to find them. Please fix the docker build "
            "and re-run `ciflow/docker` before merging."
        )


def warn_on_docker_merge_skew(repo: GitRepo, pr: GitHubPR) -> None:
    """Log when a docker-affecting PR raced with another docker change.

    The CI docker images are tagged by the git tree hash of .ci/docker, so a
    docker change landing on the base after this PR's images were built leaves
    the merge commit asking for an untested tree. This used to refuse the merge
    (#191508); it now only warns, since docker-builds.yml publishes the merged
    tree's images on push and jobs wait for them.

    Must be called after the merge commit has been created locally.
    """
    if not pr.is_docker_affecting():
        return

    # HEAD is the freshly created (squash/cherry-picked) merge commit.
    # A wholesale deletion of the directory intentionally fails this lookup.
    merge_commit_tree = repo.rev_parse(f"HEAD:{DOCKER_CI_PATH}")

    # Compare against the tree that CI actually built and tested on the PR head.
    pr_head_sha = pr.last_commit_sha()
    # The PR head commit may not be present locally (for example, for a fork
    # PR), so make sure we have the object before reading its tree.
    repo.fetch(pr_head_sha)
    pr_head_tree = repo.rev_parse(f"{pr_head_sha}:{DOCKER_CI_PATH}")

    if merge_commit_tree != pr_head_tree:
        print(
            f"WARNING: docker land race on PR #{pr.pr_num}: the {DOCKER_CI_PATH} "
            f"tree of the merge commit ({merge_commit_tree}) does not match the "
            f"tree that ciflow/docker built and tested on the PR head "
            f"({pr_head_tree}). Another docker-affecting change landed on the base "
            "branch after these images were built, so the first trunk jobs after "
            "this merge may wait for docker-builds to publish the merged tree's "
            "images, or rebuild them locally. Merging anyway."
        )


def checks_to_str(checks: list[tuple[str, str | None]]) -> str:
    return ", ".join(f"[{c[0]}]({c[1]})" if c[1] is not None else c[0] for c in checks)


def checks_to_markdown_bullets(
    checks: list[tuple[str, str | None, int | None]],
) -> list[str]:
    return [
        f"- [{c[0]}]({c[1]})" if c[1] is not None else f"- {c[0]}" for c in checks[:5]
    ]


def post_starting_merge_comment(
    repo: GitRepo,
    pr: GitHubPR,
    explainer: TryMergeExplainer,
    dry_run: bool,
    ignore_current_checks_info: list[tuple[str, str | None, int | None]] | None = None,
) -> None:
    """Post the initial merge starting message on the PR. Also post a short
    message on all PRs in the stack."""
    gh_post_pr_comment(
        pr.org,
        pr.project,
        pr.pr_num,
        explainer.get_merge_message(ignore_current_checks_info),
        dry_run=dry_run,
    )
    if pr.is_ghstack_pr():
        for additional_prs, _ in get_ghstack_prs(repo, pr):
            if additional_prs.pr_num != pr.pr_num:
                gh_post_pr_comment(
                    additional_prs.org,
                    additional_prs.project,
                    additional_prs.pr_num,
                    f"Starting merge as part of PR stack under #{pr.pr_num}",
                    dry_run=dry_run,
                )


def manually_close_merged_pr(
    pr: GitHubPR,
    additional_merged_prs: list[GitHubPR],
    merge_commit_sha: str,
    dry_run: bool,
) -> None:
    def _comment_and_close(pr: GitHubPR, comment: str) -> None:
        pr = GitHubPR(pr.org, pr.project, pr.pr_num)  # Refresh the PR
        if not pr.is_closed():
            gh_post_pr_comment(pr.org, pr.project, pr.pr_num, comment, dry_run)
            gh_close_pr(pr.org, pr.project, pr.pr_num, dry_run)

    message = (
        f"This PR (#{pr.pr_num}) was merged in {merge_commit_sha} but it is still open, likely due to a Github bug, "
        "so mergebot is closing it manually.  If you think this is a mistake, please feel free to reopen and contact Dev Infra."
    )
    _comment_and_close(pr, message)
    for additional_pr in additional_merged_prs:
        message = (
            f"This PR (#{additional_pr.pr_num}) was merged as part of PR #{pr.pr_num} in the stack under {merge_commit_sha} "
            "but it is still open, likely due to a Github bug, so mergebot is closing it manually. "
            "If you think this is a mistake, please feel free to reopen and contact Dev Infra."
        )
        _comment_and_close(additional_pr, message)

    print(f"PR {pr.pr_num} and all additional PRs in the stack have been closed.")


@retries_decorator()
def save_merge_record(
    comment_id: int,
    pr_num: int,
    owner: str,
    project: str,
    author: str,
    pending_checks: list[tuple[str, str | None, int | None]],
    failed_checks: list[tuple[str, str | None, int | None]],
    ignore_current_checks: list[tuple[str, str | None, int | None]],
    broken_trunk_checks: list[tuple[str, str | None, int | None]],
    flaky_checks: list[tuple[str, str | None, int | None]],
    unstable_checks: list[tuple[str, str | None, int | None]],
    last_commit_sha: str,
    merge_base_sha: str,
    merge_commit_sha: str = "",
    is_failed: bool = False,
    skip_mandatory_checks: bool = False,
    ignore_current: bool = False,
    error: str = "",
) -> None:
    """
    This saves the merge records as a json, which can later be uploaded to s3
    """

    # Prepare the record to be written into s3
    data = [
        {
            "comment_id": comment_id,
            "pr_num": pr_num,
            "owner": owner,
            "project": project,
            "author": author,
            "pending_checks": pending_checks,
            "failed_checks": failed_checks,
            "ignore_current_checks": ignore_current_checks,
            "broken_trunk_checks": broken_trunk_checks,
            "flaky_checks": flaky_checks,
            "unstable_checks": unstable_checks,
            "last_commit_sha": last_commit_sha,
            "merge_base_sha": merge_base_sha,
            "merge_commit_sha": merge_commit_sha,
            "is_failed": is_failed,
            "skip_mandatory_checks": skip_mandatory_checks,
            "ignore_current": ignore_current,
            "error": error,
            # This is a unique identifier for the record for deduping purposes
            # in Rockset.  Any unique string would work.  This will not be used
            # after we migrate off Rockset
            "_id": f"{project}-{pr_num}-{comment_id}-{os.environ.get('GITHUB_RUN_ID')}",
        }
    ]
    repo_root = Path(__file__).resolve().parent.parent.parent

    with open(repo_root / "merge_record.json", "w") as f:
        json.dump(data, f)


@retries_decorator()
def get_drci_classifications(pr_num: int, project: str = "pytorch") -> Any:
    """
    Query HUD API to find similar failures to decide if they are flaky
    """
    # NB: This doesn't work internally atm because this requires making an
    # external API call to HUD
    failures = gh_fetch_url(
        f"https://hud.pytorch.org/api/drci/drci?prNumber={pr_num}",
        data=f"repo={project}",
        headers={
            "Authorization": os.getenv("DRCI_BOT_KEY", ""),
            "Accept": "application/vnd.github.v3+json",
            "x-hud-internal-bot": os.getenv("HUD_API_TOKEN", ""),
        },
        method="POST",
        reader=json.load,
    )

    return failures.get(str(pr_num), {}) if failures else {}


REMOVE_JOB_NAME_SUFFIX_REGEX = re.compile(r", [0-9]+, [0-9]+, .+\)$")


def remove_job_name_suffix(name: str, replacement: str = ")") -> str:
    return re.sub(REMOVE_JOB_NAME_SUFFIX_REGEX, replacement, name)


def is_broken_trunk(
    check: JobCheckState,
    drci_classifications: Any,
) -> bool:
    if not check or not drci_classifications:
        return False

    name = check.name
    job_id = check.job_id

    # Consult the list of broken trunk failures from Dr.CI
    return any(
        (name == broken_trunk["name"]) or (job_id and job_id == broken_trunk["id"])
        for broken_trunk in drci_classifications.get("BROKEN_TRUNK", [])
    )


def is_unstable(
    check: JobCheckState,
    drci_classifications: Any,
) -> bool:
    if not check or not drci_classifications:
        return False

    name = check.name
    job_id = check.job_id

    # The job name has the unstable keyword. This is the original way to mark a job
    # as unstable on HUD, Dr.CI, and trymerge
    if "unstable" in name:
        return True

    # Consult the list of unstable failures from Dr.CI
    return any(
        (name == unstable["name"] or (job_id and job_id == unstable["id"]))
        for unstable in drci_classifications.get("UNSTABLE", [])
    )


def is_flaky(
    check: JobCheckState,
    drci_classifications: Any,
) -> bool:
    if not check or not drci_classifications:
        return False

    name = check.name
    job_id = check.job_id

    # Consult the list of flaky failures from Dr.CI
    return any(
        (name == flaky["name"] or (job_id and job_id == flaky["id"]))
        for flaky in drci_classifications.get("FLAKY", [])
    )


def is_invalid_cancel(
    name: str,
    conclusion: str | None,
    drci_classifications: Any,
) -> bool:
    """
    After https://github.com/pytorch/test-infra/pull/4579, invalid cancelled
    signals have been removed from HUD and Dr.CI. The same needs to be done
    here for consistency
    """
    if (
        not name
        or not drci_classifications
        or not conclusion
        or conclusion.upper() != "CANCELLED"
    ):
        return False

    # If a job is cancelled and not listed as a failure or unclassified failure
    # by Dr.CI, it's an invalid signal and can be ignored. UNKNOWN covers jobs
    # whose workflow did not run on the merge base, so Dr.CI has no signal to
    # tell whether the failure is pre-existing -- those should still block merge.
    return all(
        name != failure["name"]
        for failure in drci_classifications.get("FAILED", [])
        + drci_classifications.get("UNKNOWN", [])
    )


def is_crcr_l3(check: JobCheckState, drci_classifications: Any) -> bool:
    """Return True if this check is a non-blocking CRCR L3 failure.

    Dr.CI is the classification authority. A check is L3 non-blocking when
    Dr.CI returns it under the ``CRCR_L3`` category. CRCR (cross-repo CI
    relay) L3 check runs are named ``crcr/<owner>/<repo>/<workflow>`` and will be
    classified as CRCR_L3 by Dr.CI when their downstream_repo_level is L3.
    """
    if not check or not drci_classifications:
        return False

    name = check.name
    job_id = check.job_id

    return any(
        name == crcr["name"] or (job_id and job_id == crcr["id"])
        for crcr in drci_classifications.get("CRCR_L3", [])
    )


def get_classifications(
    pr_num: int,
    project: str,
    checks: dict[str, JobCheckState],
    ignore_current_checks: list[str] | None,
) -> dict[str, JobCheckState]:
    # Get the failure classification from Dr.CI, which is the source of truth
    # going forward. It's preferable to try calling Dr.CI API directly first
    # to get the latest results as well as update Dr.CI PR comment
    drci_classifications = get_drci_classifications(pr_num=pr_num, project=project)

    def get_readable_drci_results(drci_classifications: Any) -> str:
        try:
            s = f"From Dr.CI API ({pr_num}):\n"
            for classification, jobs in drci_classifications.items():
                s += f"  {classification}: \n"
                for job in jobs:
                    s += f"    {job['id']} {job['name']}\n"
            return s
        except Exception:
            return f"From Dr.CI API: {json.dumps(drci_classifications)}"

    print(get_readable_drci_results(drci_classifications))

    # NB: if the latest results from Dr.CI is not available, i.e. when calling from
    # SandCastle, we fallback to any results we can find on Dr.CI check run summary
    if (
        not drci_classifications
        and DRCI_CHECKRUN_NAME in checks
        and checks[DRCI_CHECKRUN_NAME]
        and checks[DRCI_CHECKRUN_NAME].summary
    ):
        drci_summary = checks[DRCI_CHECKRUN_NAME].summary
        try:
            print(f"From Dr.CI checkrun summary: {drci_summary}")
            drci_classifications = json.loads(str(drci_summary))
        except json.JSONDecodeError:
            warn("Invalid Dr.CI checkrun summary")
            drci_classifications = {}

    checks_with_classifications = checks.copy()
    for name, check in checks.items():
        if check.status == "SUCCESS" or check.status == "NEUTRAL":
            continue

        if is_unstable(check, drci_classifications):
            checks_with_classifications[name] = JobCheckState(
                check.name,
                check.url,
                check.status,
                "UNSTABLE",
                check.job_id,
                check.title,
                check.summary,
            )
            continue

        # NB: It's important to note that when it comes to ghstack and broken trunk classification,
        # Dr.CI uses the base of the whole stack
        if is_broken_trunk(check, drci_classifications):
            checks_with_classifications[name] = JobCheckState(
                check.name,
                check.url,
                check.status,
                "BROKEN_TRUNK",
                check.job_id,
                check.title,
                check.summary,
            )
            continue

        elif is_flaky(check, drci_classifications):
            checks_with_classifications[name] = JobCheckState(
                check.name,
                check.url,
                check.status,
                "FLAKY",
                check.job_id,
                check.title,
                check.summary,
            )
            continue

        elif is_invalid_cancel(name, check.status, drci_classifications):
            # NB: Create a new category here for invalid cancelled signals because
            # there are usually many of them when they happen. So, they shouldn't
            # be counted toward ignorable failures threshold
            checks_with_classifications[name] = JobCheckState(
                check.name,
                check.url,
                check.status,
                "INVALID_CANCEL",
                check.job_id,
                check.title,
                check.summary,
            )
            continue

        elif is_crcr_l3(check, drci_classifications):
            checks_with_classifications[name] = JobCheckState(
                check.name,
                check.url,
                check.status,
                "CRCR_L3",
                check.job_id,
                check.title,
                check.summary,
            )
            continue

        if ignore_current_checks is not None and name in ignore_current_checks:
            checks_with_classifications[name] = JobCheckState(
                check.name,
                check.url,
                check.status,
                "IGNORE_CURRENT_CHECK",
                check.job_id,
                check.title,
                check.summary,
            )

    return checks_with_classifications


def filter_checks_with_lambda(
    checks: JobNameToStateDict, status_filter: Callable[[str | None], bool]
) -> list[JobCheckState]:
    return [check for check in checks.values() if status_filter(check.status)]


def get_pr_commit_sha(repo: GitRepo, pr: GitHubPR) -> str:
    commit_sha = pr.get_merge_commit()
    if commit_sha is not None:
        return commit_sha
    commits = repo.commits_resolving_gh_pr(pr.pr_num)
    if len(commits) == 0:
        raise PostCommentError("Can't find any commits resolving PR")
    return commits[0]


def validate_revert(
    repo: GitRepo, pr: GitHubPR, *, comment_id: int | None = None
) -> tuple[str, str]:
    comment = (
        pr.get_last_comment()
        if comment_id is None
        else pr.get_comment_by_id(comment_id)
    )
    if comment.editor_login is not None:
        raise PostCommentError(
            "Halting the revert as the revert comment has been edited."
        )
    author_association = comment.author_association
    author_login = comment.author_login
    allowed_reverters = ["COLLABORATOR", "MEMBER", "OWNER"]
    # For some reason, one can not be a member of private repo, only CONTRIBUTOR
    if pr.is_base_repo_private():
        allowed_reverters.append("CONTRIBUTOR")
    # Special case GitHub Apps that don't have a repo association
    # but should be able to issue revert commands
    allowed_apps = {
        "https://github.com/apps/pytorch-auto-revert",
        "https://github.com/apps/facebook-github-tools",
        "https://github.com/apps/meta-codesync",
    }
    if comment.author_url in allowed_apps:
        allowed_reverters.append("NONE")

    if author_association not in allowed_reverters:
        raise PostCommentError(
            f"Will not revert as @{author_login} is not one of "
            f"[{', '.join(allowed_reverters)}], but instead is {author_association}."
        )

    commit_sha = get_pr_commit_sha(repo, pr)
    return (author_login, commit_sha)


def get_ghstack_dependent_prs(
    repo: GitRepo, pr: GitHubPR, only_closed: bool = True
) -> list[tuple[str, GitHubPR]]:
    """
    Get the PRs in the stack that are above this PR (inclusive).
    Throws error if stack have branched or original branches are gone
    """
    if not pr.is_ghstack_pr():
        raise AssertionError(
            f"get_ghstack_dependent_prs called on non-ghstack PR #{pr.pr_num}"
        )
    orig_ref = f"{repo.remote}/{pr.get_ghstack_orig_ref()}"
    rev_list = repo.revlist(f"{pr.default_branch()}..{orig_ref}")
    if len(rev_list) == 0:
        raise RuntimeError(
            f"PR {pr.pr_num} does not have any revisions associated with it"
        )
    skip_len = len(rev_list) - 1
    for branch in repo.branches_containing_ref(orig_ref):
        candidate = repo.revlist(f"{pr.default_branch()}..{branch}")
        # Pick longest candidate
        if len(candidate) > len(rev_list):
            candidate, rev_list = rev_list, candidate
        # Validate that candidate always ends rev-list
        if rev_list[-len(candidate) :] != candidate:
            raise RuntimeError(
                f"Branch {branch} revlist {', '.join(candidate)} is not a subset of {', '.join(rev_list)}"
            )
    # Remove commits original PR depends on
    if skip_len > 0:
        rev_list = rev_list[:-skip_len]
    rc: list[tuple[str, GitHubPR]] = []
    for pr_, _sha in _revlist_to_prs(repo, pr, rev_list):
        if not pr_.is_closed():
            if not only_closed:
                rc.append(("", pr_))
            continue
        commit_sha = get_pr_commit_sha(repo, pr_)
        rc.append((commit_sha, pr_))
    return rc


def do_revert_prs(
    repo: GitRepo,
    original_pr: GitHubPR,
    shas_and_prs: list[tuple[str, GitHubPR]],
    *,
    author_login: str,
    extra_msg: str = "",
    skip_internal_checks: bool = False,
    dry_run: bool = False,
) -> None:
    # Prepare and push revert commits
    for commit_sha, pr in shas_and_prs:
        revert_msg = f"\nReverted {pr.get_pr_url()} on behalf of {prefix_with_github_url(author_login)}"
        revert_msg += extra_msg
        repo.checkout(pr.default_branch())
        repo.revert(commit_sha)
        msg = repo.commit_message("HEAD")
        msg = re.sub(RE_PULL_REQUEST_RESOLVED, "", msg)
        msg += revert_msg
        repo.amend_commit_message(msg)
    repo.push(shas_and_prs[0][1].default_branch(), dry_run)

    # Comment/reopen PRs
    for commit_sha, pr in shas_and_prs:
        revert_message = ""
        if pr.pr_num == original_pr.pr_num:
            revert_message += (
                f"@{pr.get_pr_creator_login()} your PR has been successfully reverted."
            )
        else:
            revert_message += (
                f"@{pr.get_pr_creator_login()} your PR has been reverted as part of the stack under "
                f"#{original_pr.pr_num}.\n"
            )
        if (
            pr.has_internal_changes()
            and not pr.has_no_connected_diff()
            and not skip_internal_checks
        ):
            revert_message += "\n:warning: This PR might contain internal changes"
            revert_message += "\ncc: @pytorch/pytorch-dev-infra"
        gh_post_pr_comment(
            pr.org, pr.project, pr.pr_num, revert_message, dry_run=dry_run
        )

        pr.add_numbered_label("reverted", dry_run)
        pr.add_label("ci-no-td", dry_run)
        if not dry_run:
            gh_post_commit_comment(pr.org, pr.project, commit_sha, revert_msg)
            gh_update_pr_state(pr.org, pr.project, pr.pr_num)


def try_revert(
    repo: GitRepo,
    pr: GitHubPR,
    *,
    dry_run: bool = False,
    comment_id: int | None = None,
    reason: str | None = None,
) -> None:
    try:
        author_login, commit_sha = validate_revert(repo, pr, comment_id=comment_id)
    except PostCommentError as e:
        gh_post_pr_comment(pr.org, pr.project, pr.pr_num, str(e), dry_run=dry_run)
        return

    extra_msg = f" due to {reason}" if reason is not None else ""
    extra_msg += (
        f" ([comment]({pr.get_comment_by_id(comment_id).url}))\n"
        if comment_id is not None
        else "\n"
    )
    shas_and_prs = [(commit_sha, pr)]
    if pr.is_ghstack_pr():
        try:
            shas_and_prs = get_ghstack_dependent_prs(repo, pr)
            prs_to_revert = " ".join([t[1].get_pr_url() for t in shas_and_prs])
            print(f"About to stack of PRs: {prs_to_revert}")
        except Exception as e:
            print(
                f"Failed to fetch dependent PRs: {str(e)}, fall over to single revert"
            )

    if not shas_and_prs:
        raise RuntimeError(
            f"No revertable PRs found in ghstack for #{pr.pr_num}. "
            f"This typically means the PR is still open (not merged) or "
            f"its GitHub state is inconsistent. Only closed/merged PRs can be reverted."
        )

    do_revert_prs(
        repo,
        pr,
        shas_and_prs,
        author_login=author_login,
        extra_msg=extra_msg,
        dry_run=dry_run,
        skip_internal_checks=can_skip_internal_checks(pr, comment_id),
    )


def prefix_with_github_url(suffix_str: str) -> str:
    return f"https://github.com/{suffix_str}"


def check_for_sev(org: str, project: str, skip_mandatory_checks: bool) -> None:
    if skip_mandatory_checks:
        return
    response = cast(
        dict[str, Any],
        gh_fetch_json_list(
            "https://api.github.com/search/issues",  # @lint-ignore
            # Having two label: queries is an AND operation
            params={
                "q": f'repo:{org}/{project} is:open is:issue label:"ci: sev" label:"merge blocking"'
            },
        ),
    )
    if response["total_count"] != 0:
        raise RuntimeError(
            "Not merging any PRs at the moment because there is a "
            + "merge blocking https://github.com/pytorch/pytorch/labels/ci:%20sev issue open at: \n"
            + f"{response['items'][0]['html_url']}"
        )
    return


def has_label(labels: list[str], pattern: Pattern[str] = CIFLOW_LABEL) -> bool:
    return len(list(filter(pattern.match, labels))) > 0


def categorize_checks(
    check_runs: JobNameToStateDict,
    required_checks: list[str],
    ok_failed_checks_threshold: int | None = None,
) -> tuple[
    list[tuple[str, str | None, int | None]],
    list[tuple[str, str | None, int | None]],
    dict[str, list[Any]],
]:
    """
    Categories all jobs into the list of pending and failing jobs. All known flaky
    failures and broken trunk are ignored by defaults when ok_failed_checks_threshold
    is not set (unlimited)
    """
    pending_checks: list[tuple[str, str | None, int | None]] = []
    failed_checks: list[tuple[str, str | None, int | None]] = []

    # failed_checks_categorization is used to keep track of all ignorable failures when saving the merge record on s3
    failed_checks_categorization: dict[str, list[Any]] = defaultdict(list)

    # If required_checks is not set or empty, consider all names are relevant
    relevant_checknames = [
        name
        for name in check_runs
        if not required_checks or any(x in name for x in required_checks)
    ]

    for checkname in required_checks:
        if all(checkname not in x for x in check_runs):
            pending_checks.append((checkname, None, None))

    for checkname in relevant_checknames:
        status = check_runs[checkname].status
        url = check_runs[checkname].url
        classification = check_runs[checkname].classification
        job_id = check_runs[checkname].job_id

        if status is None and classification != "UNSTABLE":
            # NB: No need to wait if the job classification is unstable as it would be
            # ignored anyway. This is useful to not need to wait for scarce resources
            # like ROCm, which is also frequently in unstable mode
            pending_checks.append((checkname, url, job_id))
        elif classification == "INVALID_CANCEL":
            continue
        elif not is_passing_status(check_runs[checkname].status):
            target = (
                failed_checks_categorization[classification]
                if classification
                in (
                    "IGNORE_CURRENT_CHECK",
                    "BROKEN_TRUNK",
                    "FLAKY",
                    "UNSTABLE",
                    "CRCR_L3",
                )
                else failed_checks
            )
            target.append((checkname, url, job_id))

    flaky_or_broken_trunk = (
        failed_checks_categorization["BROKEN_TRUNK"]
        + failed_checks_categorization["FLAKY"]
    )

    if flaky_or_broken_trunk:
        warn(
            f"The following {len(flaky_or_broken_trunk)} checks failed but were likely due flakiness or broken trunk: "
            + ", ".join([x[0] for x in flaky_or_broken_trunk])
            + (
                f" but this is greater than the threshold of {ok_failed_checks_threshold} so merge will fail"
                if ok_failed_checks_threshold is not None
                and len(flaky_or_broken_trunk) > ok_failed_checks_threshold
                else ""
            )
        )

    if (
        ok_failed_checks_threshold is not None
        and len(flaky_or_broken_trunk) > ok_failed_checks_threshold
    ):
        failed_checks = failed_checks + flaky_or_broken_trunk

    # The list of failed_checks_categorization is returned so that it can be saved into the s3 merge record
    return (pending_checks, failed_checks, failed_checks_categorization)


def merge(
    pr: GitHubPR,
    repo: GitRepo,
    comment_id: int,
    dry_run: bool = False,
    skip_mandatory_checks: bool = False,
    timeout_minutes: int = 400,
    stale_pr_days: int = 3,
    ignore_current: bool = False,
) -> None:
    initial_commit_sha = pr.last_commit_sha()
    pr_link = f"https://github.com/{pr.org}/{pr.project}/pull/{pr.pr_num}"
    print(f"Attempting merge of {initial_commit_sha} ({pr_link})")

    if MERGE_IN_PROGRESS_LABEL not in pr.get_labels():
        gh_add_labels(pr.org, pr.project, pr.pr_num, [MERGE_IN_PROGRESS_LABEL], dry_run)

    explainer = TryMergeExplainer(
        skip_mandatory_checks,
        pr.get_labels(),
        pr.pr_num,
        pr.org,
        pr.project,
        ignore_current,
    )

    # probably a bad name, but this is a list of current checks that should be
    # ignored and is toggled by the --ignore-current flag
    ignore_current_checks_info = []

    if pr.is_ghstack_pr():
        get_ghstack_prs(repo, pr)  # raises error if out of sync

    check_for_sev(pr.org, pr.project, skip_mandatory_checks)

    if skip_mandatory_checks:
        post_starting_merge_comment(repo, pr, explainer, dry_run)
        # This return is outside the retry loop below, so there is no iteration for
        # the greenlight check to wait in: it has to decide now or refuse.
        return pr.merge_into(
            repo,
            dry_run=dry_run,
            skip_mandatory_checks=skip_mandatory_checks,
            comment_id=comment_id,
            greenlight_wait=None,
        )

    # Check for approvals
    find_matching_merge_rule(pr, repo, skip_mandatory_checks=True)

    ensure_mergeable_labels(pr, comment_id, dry_run)

    if ignore_current:
        checks = pr.get_checkrun_conclusions()
        _, failing, _ = categorize_checks(
            checks,
            list(checks.keys()),
            ok_failed_checks_threshold=IGNORABLE_FAILED_CHECKS_THESHOLD,
        )
        ignore_current_checks_info = failing

    post_starting_merge_comment(
        repo,
        pr,
        explainer,
        dry_run,
        ignore_current_checks_info=ignore_current_checks_info,
    )

    start_time = time.time()
    last_exception = ""
    elapsed_time = 0.0
    # Owned out here so the greenlight wait budget spans every iteration below rather
    # than restarting each time merge_into is re-entered.
    greenlight_wait = GreenlightWaitWindow()
    ignore_current_checks = [
        x[0] for x in ignore_current_checks_info
    ]  # convert to List[str] for convenience
    while elapsed_time < timeout_minutes * 60:
        check_for_sev(pr.org, pr.project, skip_mandatory_checks)
        current_time = time.time()
        elapsed_time = current_time - start_time
        print(
            f"Attempting merge of https://github.com/{pr.org}/{pr.project}/pull/{pr.pr_num} ({elapsed_time / 60} minutes elapsed)"
        )
        pr = GitHubPR(pr.org, pr.project, pr.pr_num)
        if initial_commit_sha != pr.last_commit_sha():
            raise RuntimeError(
                "New commits were pushed while merging. Please rerun the merge command."
            )
        try:
            required_checks = []
            failed_rule_message = None
            ignore_flaky_failures = True
            try:
                find_matching_merge_rule(
                    pr, repo, ignore_current_checks=ignore_current_checks
                )
            except MandatoryChecksMissingError as ex:
                if ex.rule is not None:
                    ignore_flaky_failures = ex.rule.ignore_flaky_failures
                    if ex.rule.mandatory_checks_name is not None:
                        required_checks = ex.rule.mandatory_checks_name
                failed_rule_message = ex

            checks = pr.get_checkrun_conclusions()
            checks = get_classifications(
                pr.pr_num,
                pr.project,
                checks,
                ignore_current_checks=ignore_current_checks,
            )
            pending, failing, _ = categorize_checks(
                checks,
                required_checks + [x for x in checks if x not in required_checks],
                ok_failed_checks_threshold=IGNORABLE_FAILED_CHECKS_THESHOLD
                if ignore_flaky_failures
                else 0,
            )
            # HACK until GitHub will be better about surfacing those
            startup_failures = filter_checks_with_lambda(
                checks, lambda status: status == "STARTUP_FAILURE"
            )
            if len(startup_failures) > 0:
                raise RuntimeError(
                    f"{len(startup_failures)} STARTUP failures reported, please check workflows syntax! "
                    + ", ".join(f"[{x.name}]({x.url})" for x in startup_failures[:5])
                )
            # END of HACK

            if len(failing) > 0:
                raise RuntimeError(
                    f"{len(failing)} jobs have failed, first few of them are: "
                    + ", ".join(f"[{x[0]}]({x[1]})" for x in failing[:5])
                )
            if len(pending) > 0:
                if failed_rule_message is not None:
                    raise failed_rule_message
                else:
                    raise MandatoryChecksMissingError(
                        f"Still waiting for {len(pending)} jobs to finish, "
                        + f"first few of them are: {', '.join(x[0] for x in pending[:5])}"
                    )

            return pr.merge_into(
                repo,
                dry_run=dry_run,
                skip_mandatory_checks=skip_mandatory_checks,
                comment_id=comment_id,
                ignore_current_checks=ignore_current_checks,
                greenlight_wait=greenlight_wait,
            )
        except MandatoryChecksMissingError as ex:
            last_exception = str(ex)
            print(
                f"Merge of https://github.com/{pr.org}/{pr.project}/pull/{pr.pr_num} failed due to: {ex}. Retrying in 5 min",
                flush=True,
            )
            time.sleep(5 * 60)
    # Finally report timeout back
    msg = f"Merged timed out after {timeout_minutes} minutes. Please contact the pytorch_dev_infra team."
    msg += f"The last exception was: {last_exception}"
    gh_add_labels(pr.org, pr.project, pr.pr_num, ["land-failed"], dry_run)
    raise RuntimeError(msg)


def main() -> None:
    args = parse_args()
    repo = GitRepo(get_git_repo_dir(), get_git_remote_name())
    org, project = repo.gh_owner_and_name()
    pr = GitHubPR(org, project, args.pr_num)

    def handle_exception(e: Exception, title: str = "Merge failed") -> None:
        exception = f"**Reason**: {e}"

        failing_rule = None
        if isinstance(e, MergeRuleFailedError):
            failing_rule = e.rule.name if e.rule else None

        internal_debugging = ""
        run_url = os.getenv("GH_RUN_URL")
        if run_url is not None:
            # Hide this behind a collapsed bullet since it's not helpful to most devs
            internal_debugging = "\n".join(
                line
                for line in (
                    "<details><summary>Details for Dev Infra team</summary>",
                    f'Raised by <a href="{run_url}">workflow job</a>\n',
                    f"Failing merge rule: {failing_rule}" if failing_rule else "",
                    "</details>",
                )
                if line
            )  # ignore empty lines during the join

        msg = "\n".join((f"## {title}", f"{exception}", "", f"{internal_debugging}"))

        gh_post_pr_comment(org, project, args.pr_num, msg, dry_run=args.dry_run)
        traceback.print_exc()

    if args.revert:
        try:
            gh_post_pr_comment(
                org,
                project,
                args.pr_num,
                get_revert_message(org, project, pr.pr_num),
                args.dry_run,
            )
            try_revert(
                repo,
                pr,
                dry_run=args.dry_run,
                comment_id=args.comment_id,
                reason=args.reason,
            )
        except Exception as e:
            handle_exception(e, f"Reverting PR {args.pr_num} failed")
        return

    if pr.is_closed():
        gh_post_pr_comment(
            org,
            project,
            args.pr_num,
            f"Can't merge closed PR #{args.pr_num}",
            dry_run=args.dry_run,
        )
        return

    if pr.is_cross_repo() and pr.is_ghstack_pr():
        gh_post_pr_comment(
            org,
            project,
            args.pr_num,
            "Cross-repo ghstack merges are not supported",
            dry_run=args.dry_run,
        )
        return
    if not pr.is_ghstack_pr() and pr.base_ref() != pr.default_branch():
        gh_post_pr_comment(
            org,
            project,
            args.pr_num,
            f"PR targets {pr.base_ref()} rather than {pr.default_branch()}, refusing merge request",
            dry_run=args.dry_run,
        )
        return

    if args.check_mergeability:
        if pr.is_ghstack_pr():
            get_ghstack_prs(repo, pr)  # raises error if out of sync
        pr.merge_changes_locally(
            repo,
            skip_mandatory_checks=True,
            skip_all_rule_checks=True,
        )
        return

    if not args.force and pr.has_invalid_submodule_updates():
        message = (
            f"This PR updates submodules {', '.join(pr.get_changed_submodules())}\n"
        )
        message += '\nIf those updates are intentional, please add "submodule" keyword to PR title/description.'
        gh_post_pr_comment(org, project, args.pr_num, message, dry_run=args.dry_run)
        return
    try:
        # Ensure comment id is set, else fail
        if not args.comment_id:
            raise ValueError(
                "Comment ID is required for merging PRs, please provide it using --comment-id"
            )

        merge(
            pr,
            repo,
            comment_id=args.comment_id,
            dry_run=args.dry_run,
            skip_mandatory_checks=args.force,
            ignore_current=args.ignore_current,
        )
    except Exception as e:
        handle_exception(e)

        if args.comment_id and args.pr_num:
            # Finally, upload the record to s3, we don't have access to the
            # list of pending and failed checks here, but they are not really
            # needed at the moment
            save_merge_record(
                comment_id=args.comment_id,
                pr_num=args.pr_num,
                owner=org,
                project=project,
                author=pr.get_author(),
                pending_checks=[],
                failed_checks=[],
                ignore_current_checks=[],
                broken_trunk_checks=[],
                flaky_checks=[],
                unstable_checks=[],
                last_commit_sha=pr.last_commit_sha(default=""),
                merge_base_sha=pr.get_merge_base(),
                is_failed=True,
                skip_mandatory_checks=args.force,
                ignore_current=args.ignore_current,
                error=str(e),
            )
        else:
            print("Missing comment ID or PR number, couldn't upload to s3")
    finally:
        if not args.check_mergeability:
            gh_remove_label(
                org, project, args.pr_num, MERGE_IN_PROGRESS_LABEL, args.dry_run
            )


if __name__ == "__main__":
    main()
