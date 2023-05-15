#!/usr/bin/env python3

# NB: the following functions are used in Meta-internal workflows
# (github_first_try_merge/my_handler.py) and thus have functionality limitations
# (no `git` command access, no network access besides the strict allow list):
#
# find_matching_merge_rule
# read_merge_rules
#
# Also any signature changes of these functions, as well as changes to the `GitHubPR`
# class, will likely require corresponding changes for the internal workflows.

import base64
import json
import os
import re
import time
import urllib.parse
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, cast, Dict, List, NamedTuple, Optional, Pattern, Tuple
from warnings import warn

import yaml
from github_utils import (
    gh_fetch_json_list,
    gh_fetch_url,
    gh_post_commit_comment,
    gh_post_pr_comment,
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
from label_utils import (
    gh_add_labels,
    gh_remove_label,
    has_required_labels,
    LABEL_ERR_MSG,
)
from trymerge_explainer import get_revert_message, TryMergeExplainer

# labels
MERGE_IN_PROGRESS_LABEL = "merging"
MERGE_COMPLETE_LABEL = "merged"


class JobCheckState(NamedTuple):
    name: str
    url: str
    status: Optional[str]
    classification: Optional[str]
    job_id: Optional[int]


JobNameToStateDict = Dict[str, JobCheckState]


class WorkflowCheckState:
    def __init__(self, name: str, url: str, status: Optional[str]):
        self.name: str = name
        self.url: str = url
        self.status: Optional[str] = status
        self.jobs: JobNameToStateDict = {}


class FlakyRule:
    def __init__(self, name: str, captures: List[str]):
        self.name = name
        self.captures = captures

    def matches(self, job: Optional[Dict[str, Any]]) -> bool:
        return (
            job is not None
            and self.name in job.get("name", "")
            and job.get("failure_captures") is not None
            and all(
                capture in job.get("failure_captures", []) for capture in self.captures
            )
        )

    def __repr__(self) -> str:
        return f"FlakyRule[name='{self.name}', captures={self.captures}]"


GH_PR_REVIEWS_FRAGMENT = """
fragment PRReviews on PullRequestReviewConnection {
  nodes {
    author {
      login
    }
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
      app {
        name
        databaseId
      }
      workflowRun {
        workflow {
          name
        }
        url
      }
      checkRuns(first: 50) {
        nodes {
          name
          conclusion
          detailsUrl
          databaseId
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
      author {
        user {
          login
        }
        email
        name
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
            pushedDate
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
    r"Pull Request resolved: "
    r"https://github.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/pull/(?P<number>[0-9]+)",
    re.MULTILINE,
)
RE_PR_CC_LINE = re.compile(r"^cc:? @\w+.*\r?\n?$", re.MULTILINE)
RE_DIFF_REV = re.compile(r"^Differential Revision:.+?(D[0-9]+)", re.MULTILINE)
CIFLOW_LABEL = re.compile(r"^ciflow/.+")
CIFLOW_TRUNK_LABEL = re.compile(r"^ciflow/trunk")
MERGE_RULE_PATH = Path(".github") / "merge_rules.yaml"
ROCKSET_MERGES_COLLECTION = "merges"
ROCKSET_MERGES_WORKSPACE = "commons"
REMOTE_MAIN_BRANCH = "origin/main"


def gh_graphql(query: str, **kwargs: Any) -> Dict[str, Any]:
    rc = gh_fetch_url(
        "https://api.github.com/graphql",
        data={"query": query, "variables": kwargs},
        reader=json.load,
    )
    if "errors" in rc:
        raise RuntimeError(
            f"GraphQL query {query}, args {kwargs} failed: {rc['errors']}"
        )
    return cast(Dict[str, Any], rc)


def gh_get_pr_info(org: str, proj: str, pr_no: int) -> Any:
    rc = gh_graphql(GH_GET_PR_INFO_QUERY, name=proj, owner=org, number=pr_no)
    return rc["data"]["repository"]["pullRequest"]


@lru_cache(maxsize=None)
def gh_get_team_members(org: str, name: str) -> List[str]:
    rc: List[str] = []
    team_members: Dict[str, Any] = {
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
        return f'{workflow_run["workflow"]["name"]} / '


def is_passing_status(status: Optional[str]) -> bool:
    return status is not None and status.upper() in ["SUCCESS", "SKIPPED", "NEUTRAL"]


def add_workflow_conclusions(
    checksuites: Any,
    get_next_checkruns_page: Callable[[List[Dict[str, Dict[str, Any]]], int, Any], Any],
    get_next_checksuites: Callable[[Any], Any],
) -> JobNameToStateDict:
    # graphql seems to favor the most recent workflow run, so in theory we
    # shouldn't need to account for reruns, but do it just in case

    # workflow -> job -> job info
    workflows: Dict[str, WorkflowCheckState] = {}

    # for the jobs that don't have a workflow
    no_workflow_obj: WorkflowCheckState = WorkflowCheckState("", "", None)

    def add_conclusions(edges: Any) -> None:
        for edge_idx, edge in enumerate(edges):
            node = edge["node"]
            workflow_run = node["workflowRun"]
            checkruns = node["checkRuns"]

            workflow_obj: WorkflowCheckState = no_workflow_obj

            if workflow_run is not None:
                workflow_name = workflow_run["workflow"]["name"]
                workflow_conclusion = node["conclusion"]
                # Do not override existing status with cancelled
                if workflow_conclusion == "CANCELLED" and workflow_name in workflows:
                    continue
                if workflow_name not in workflows:
                    workflows[workflow_name] = WorkflowCheckState(
                        name=workflow_name,
                        status=workflow_conclusion,
                        url=workflow_run["url"],
                    )
                workflow_obj = workflows[workflow_name]

            while checkruns is not None:
                for checkrun_node in checkruns["nodes"]:
                    if not isinstance(checkrun_node, dict):
                        warn(f"Expected dictionary, but got {type(checkrun_node)}")
                        continue
                    checkrun_name = f'{get_check_run_name_prefix(workflow_run)}{checkrun_node["name"]}'
                    existing_checkrun = workflow_obj.jobs.get(checkrun_name)
                    if existing_checkrun is None or not is_passing_status(
                        existing_checkrun.status
                    ):
                        workflow_obj.jobs[checkrun_name] = JobCheckState(
                            checkrun_name,
                            checkrun_node["detailsUrl"],
                            checkrun_node["conclusion"],
                            None,
                            checkrun_node["databaseId"],
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
    for workflow_name, workflow in workflows.items():
        if len(workflow.jobs) > 0:
            for job_name, job in workflow.jobs.items():
                res[job_name] = job
        else:
            res[workflow_name] = JobCheckState(
                workflow.name,
                workflow.url,
                workflow.status,
                None,
                None,
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
    parser.add_argument("--comment-id", type=int)
    parser.add_argument("--reason", type=str)
    parser.add_argument("pr_num", type=int)
    return parser.parse_args()


def can_skip_internal_checks(pr: "GitHubPR", comment_id: Optional[int] = None) -> bool:
    if comment_id is None:
        return False
    comment = pr.get_comment_by_id(comment_id)
    if comment.editor_login is not None:
        return False
    return comment.author_login == "facebook-github-bot"


def get_ghstack_prs(repo: GitRepo, pr: "GitHubPR") -> List[Tuple["GitHubPR", str]]:
    """
    Get the open PRs in the stack that are below this PR.  Throws error if any of the PRs are out of sync.
    """
    assert pr.is_ghstack_pr()
    entire_stack: List[Tuple["GitHubPR", str]] = []
    # For ghstack, cherry-pick commits based from origin
    orig_ref = f"{repo.remote}/{re.sub(r'/head$', '/orig', pr.head_ref())}"
    rev_list = repo.revlist(f"{pr.default_branch()}..{orig_ref}")
    for idx, rev in enumerate(reversed(rev_list)):
        msg = repo.commit_message(rev)
        m = RE_PULL_REQUEST_RESOLVED.search(msg)
        if m is None:
            raise RuntimeError(
                f"Could not find PR-resolved string in {msg} of ghstacked PR {pr.pr_num}"
            )
        if pr.org != m.group("owner") or pr.project != m.group("repo"):
            raise RuntimeError(
                f"PR {m.group('number')} resolved to wrong owner/repo pair"
            )
        stacked_pr_num = int(m.group("number"))
        if stacked_pr_num != pr.pr_num:
            stacked_pr = GitHubPR(pr.org, pr.project, stacked_pr_num)
            if stacked_pr.is_closed():
                print(
                    f"Skipping {idx+1} of {len(rev_list)} PR (#{stacked_pr_num}) as its already been merged"
                )
                continue
            entire_stack.append((stacked_pr, rev))
        else:
            entire_stack.append((pr, rev))

    for stacked_pr, rev in entire_stack:
        if not are_ghstack_branches_in_sync(repo, stacked_pr.head_ref()):
            raise RuntimeError(
                f"PR {stacked_pr.pr_num} is out of sync with the corresponding revision {rev} on "
                + f"branch {orig_ref} that would be merged into main.  "
                + "This usually happens because there is a non ghstack change in the PR.  "
                + f"Please sync them and try again (ex. make the changes on {orig_ref} and run ghstack)."
            )
    return entire_stack


class GitHubPR:
    def __init__(self, org: str, project: str, pr_num: int) -> None:
        assert isinstance(pr_num, int)
        self.org = org
        self.project = project
        self.pr_num = pr_num
        self.info = gh_get_pr_info(org, project, pr_num)
        self.changed_files: Optional[List[str]] = None
        self.labels: Optional[List[str]] = None
        self.conclusions: Optional[JobNameToStateDict] = None
        self.comments: Optional[List[GitHubComment]] = None
        self._authors: Optional[List[Tuple[str, str]]] = None
        self._reviews: Optional[List[Tuple[str, str]]] = None
        self.merge_base: Optional[str] = None
        self.submodules: Optional[List[str]] = None

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

    def is_base_repo_private(self) -> bool:
        return bool(self.info["baseRepository"]["isPrivate"])

    def get_changed_files_count(self) -> int:
        return int(self.info["changedFiles"])

    def last_pushed_at(self) -> Optional[datetime]:
        pushed_date = self.last_commit()["pushedDate"]
        if pushed_date is None:
            return None
        return datetime.fromisoformat(pushed_date[:-1])

    def last_commit(self) -> Any:
        return self.info["commits"]["nodes"][-1]["commit"]

    def get_merge_base(self) -> str:
        return cast(str, self.info["baseRefOid"])

    def get_changed_files(self) -> List[str]:
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

    def get_submodules(self) -> List[str]:
        if self.submodules is None:
            rc = gh_graphql(GH_GET_REPO_SUBMODULES, name=self.project, owner=self.org)
            info = rc["data"]["repository"]["submodules"]
            self.submodules = [s["path"] for s in info["nodes"]]
        return self.submodules

    def get_changed_submodules(self) -> List[str]:
        submodules = self.get_submodules()
        return [f for f in self.get_changed_files() if f in submodules]

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

    def _get_reviews(self) -> List[Tuple[str, str]]:
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
        reviews = {}
        for author, state in self._reviews:
            if state != "COMMENTED":
                reviews[author] = state
        return list(reviews.items())

    def get_approved_by(self) -> List[str]:
        return [login for (login, state) in self._get_reviews() if state == "APPROVED"]

    def get_commit_count(self) -> int:
        return int(self.info["commits_with_authors"]["totalCount"])

    def get_pr_creator_login(self) -> str:
        return cast(str, self.info["author"]["login"])

    def _fetch_authors(self) -> List[Tuple[str, str]]:
        if self._authors is not None:
            return self._authors
        authors: List[Tuple[str, str]] = []

        def add_authors(info: Dict[str, Any]) -> None:
            for node in info["commits_with_authors"]["nodes"]:
                author_node = node["commit"]["author"]
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

    def get_labels(self) -> List[str]:
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
            edges: List[Dict[str, Dict[str, Any]]], edge_idx: int, checkruns: Any
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
                    None,
                    None,
                )

        return self.conclusions

    def get_authors(self) -> Dict[str, str]:
        rc = {}
        # TODO: replace with  `self.get_commit_count()` when GraphQL pagination can be used
        # to fetch all commits, see https://gist.github.com/malfet/4f35321b0c9315bcd7116c7b54d83372
        # and https://support.github.com/ticket/enterprise/1642/1659119
        if self.get_commit_count() <= 250:
            assert len(self._fetch_authors()) == self.get_commit_count()
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

    def get_merge_commit(self) -> Optional[str]:
        mc = self.info["mergeCommit"]
        return mc["oid"] if mc is not None else None

    def get_pr_url(self) -> str:
        return f"https://github.com/{self.org}/{self.project}/pull/{self.pr_num}"

    @staticmethod
    def _comment_from_node(node: Any) -> GitHubComment:
        editor = node["editor"]
        return GitHubComment(
            body_text=node["bodyText"],
            created_at=node["createdAt"] if "createdAt" in node else "",
            author_login=node["author"]["login"],
            author_association=node["authorAssociation"],
            editor_login=editor["login"] if editor else None,
            database_id=node["databaseId"],
            url=node["url"],
        )

    def get_comments(self) -> List[GitHubComment]:
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
        raise RuntimeError(f"Comment with id {database_id} not found")

    def get_diff_revision(self) -> Optional[str]:
        rc = RE_DIFF_REV.search(self.get_body())
        return rc.group(1) if rc is not None else None

    def has_internal_changes(self) -> bool:
        checkrun_name = "Meta Internal-Only Changes Check"
        if self.get_diff_revision() is None:
            return False
        checks = self.get_checkrun_conclusions()
        if checks is None or checkrun_name not in checks:
            return False
        return checks[checkrun_name].status != "SUCCESS"

    def merge_ghstack_into(
        self,
        repo: GitRepo,
        skip_mandatory_checks: bool,
        comment_id: Optional[int] = None,
    ) -> List["GitHubPR"]:
        assert self.is_ghstack_pr()
        ghstack_prs = get_ghstack_prs(repo, self)  # raises error if out of sync
        for pr, rev in ghstack_prs:
            commit_msg = pr.gen_commit_message(filter_ghstack=True)
            if pr.pr_num != self.pr_num:
                # Raises exception if matching rule is not found
                find_matching_merge_rule(
                    pr,
                    repo,
                    skip_mandatory_checks=skip_mandatory_checks,
                    skip_internal_checks=can_skip_internal_checks(self, comment_id),
                )
            repo.cherry_pick(rev)
            repo.amend_commit_message(commit_msg)
        return [x for x, _ in ghstack_prs]

    def gen_commit_message(self, filter_ghstack: bool = False) -> str:
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
        return msg

    def add_numbered_label(self, label_base: str) -> None:
        labels = self.get_labels() if self.labels is not None else []
        full_label = label_base
        count = 0
        for label in labels:
            if label_base in label:
                count += 1
                full_label = f"{label_base}X{count}"
        gh_add_labels(self.org, self.project, self.pr_num, [full_label])

    def merge_into(
        self,
        repo: GitRepo,
        *,
        skip_mandatory_checks: bool = False,
        dry_run: bool = False,
        comment_id: Optional[int] = None,
        ignore_current_checks: Optional[List[str]] = None,
    ) -> None:
        # Raises exception if matching rule is not found
        merge_rule, pending_checks, failed_checks = find_matching_merge_rule(
            self,
            repo,
            skip_mandatory_checks=skip_mandatory_checks,
            skip_internal_checks=can_skip_internal_checks(self, comment_id),
            ignore_current_checks=ignore_current_checks,
        )
        additional_merged_prs = self.merge_changes(
            repo, skip_mandatory_checks, comment_id
        )

        repo.push(self.default_branch(), dry_run)
        if not dry_run:
            self.add_numbered_label(MERGE_COMPLETE_LABEL)
            for pr in additional_merged_prs:
                pr.add_numbered_label(MERGE_COMPLETE_LABEL)

        if comment_id and self.pr_num:
            # When the merge process reaches this part, we can assume that the commit
            # has been successfully pushed to trunk
            merge_commit_sha = repo.rev_parse(name=REMOTE_MAIN_BRANCH)

            # Finally, upload the record to Rockset. The list of pending and failed
            # checks are at the time of the merge
            save_merge_record(
                collection=ROCKSET_MERGES_COLLECTION,
                comment_id=comment_id,
                pr_num=self.pr_num,
                owner=self.org,
                project=self.project,
                author=self.get_author(),
                pending_checks=pending_checks,
                failed_checks=failed_checks,
                last_commit_sha=self.last_commit().get("oid", ""),
                merge_base_sha=self.get_merge_base(),
                merge_commit_sha=merge_commit_sha,
                is_failed=False,
                dry_run=dry_run,
                skip_mandatory_checks=skip_mandatory_checks,
                ignore_current=bool(ignore_current_checks),
                workspace=ROCKSET_MERGES_WORKSPACE,
            )
        else:
            print("Missing comment ID or PR number, couldn't upload to Rockset")

    def merge_changes(
        self,
        repo: GitRepo,
        skip_mandatory_checks: bool = False,
        comment_id: Optional[int] = None,
        branch: Optional[str] = None,
    ) -> List["GitHubPR"]:
        branch_to_merge_into = self.default_branch() if branch is None else branch
        if repo.current_branch() != branch_to_merge_into:
            repo.checkout(branch_to_merge_into)
        if not self.is_ghstack_pr():
            msg = self.gen_commit_message()
            pr_branch_name = f"__pull-request-{self.pr_num}__init__"
            repo.fetch(f"pull/{self.pr_num}/head", pr_branch_name)
            repo._run_git("merge", "--squash", pr_branch_name)
            repo._run_git("commit", f'--author="{self.get_author()}"', "-m", msg)
            return []
        else:
            return self.merge_ghstack_into(
                repo,
                skip_mandatory_checks,
                comment_id=comment_id,
            )


class MergeRuleFailedError(RuntimeError):
    def __init__(self, message: str, rule: Optional["MergeRule"] = None) -> None:
        super().__init__(message)
        self.rule = rule


class MandatoryChecksMissingError(MergeRuleFailedError):
    pass


class PostCommentError(Exception):
    pass


@dataclass
class MergeRule:
    name: str
    patterns: List[str]
    approved_by: List[str]
    mandatory_checks_name: Optional[List[str]]
    ignore_flaky_failures: bool = True


def gen_new_issue_link(
    org: str, project: str, labels: List[str], template: str = "bug-report.yml"
) -> str:
    labels_str = ",".join(labels)
    return (
        f"https://github.com/{org}/{project}/issues/new?"
        f"labels={urllib.parse.quote(labels_str)}&"
        f"template={urllib.parse.quote(template)}"
    )


def read_merge_rules(
    repo: Optional[GitRepo], org: str, project: str
) -> List[MergeRule]:
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


@lru_cache(maxsize=None)
def read_flaky_rules() -> List[FlakyRule]:
    # NOTE: This is currently hardcoded, can be extended to do per repo rules
    FLAKY_RULES_URL = "https://raw.githubusercontent.com/pytorch/test-infra/generated-stats/stats/flaky-rules.json"
    return _get_flaky_rules(FLAKY_RULES_URL)


def find_matching_merge_rule(
    pr: GitHubPR,
    repo: Optional[GitRepo] = None,
    skip_mandatory_checks: bool = False,
    skip_internal_checks: bool = False,
    ignore_current_checks: Optional[List[str]] = None,
) -> Tuple[
    MergeRule,
    List[Tuple[str, Optional[str], Optional[int]]],
    List[Tuple[str, Optional[str], Optional[int]]],
]:
    """
    Returns merge rule matching to this pr together with the list of associated pending
    and failing jobs OR raises an exception.

    NB: this function is used in Meta-internal workflows, see the comment at the top of
    this file for details.
    """
    changed_files = pr.get_changed_files()
    approved_by = set(pr.get_approved_by())

    issue_link = gen_new_issue_link(
        org=pr.org,
        project=pr.project,
        labels=["module: ci"],
    )
    reject_reason = f"No rule found to match PR. Please [report]{issue_link} this issue to DevX team."

    rules = read_merge_rules(repo, pr.org, pr.project)
    flaky_rules = read_flaky_rules()
    if not rules:
        reject_reason = f"Rejecting the merge as no rules are defined for the repository in {MERGE_RULE_PATH}"
        raise RuntimeError(reject_reason)
    checks = pr.get_checkrun_conclusions()
    base_rev = None
    try:
        # is allowed to fail if git is not available
        base_rev = pr.get_merge_base()
    except Exception as e:
        print(
            f"Failed fetching base git revision for {pr.pr_num}. Skipping additional classifications.\n"
            f"{type(e)}\n{e}"
        )
    checks = get_classifications(
        checks,
        pr.last_commit()["oid"],
        base_rev,
        flaky_rules,
        ignore_current_checks=ignore_current_checks,
    )

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
        patterns_re = patterns_to_regex(rule.patterns)
        non_matching_files = []

        # Does this rule apply to all the files?
        for fname in changed_files:
            if not patterns_re.match(fname):
                non_matching_files.append(fname)
        if len(non_matching_files) > 0:
            num_matching_files = len(changed_files) - len(non_matching_files)
            if num_matching_files > reject_reason_score:
                reject_reason_score = num_matching_files
                reject_reason = "\n".join(
                    (
                        f"Not all files match rule `{rule_name}`."
                        f"{num_matching_files} files matched, but there are still non-matching files:"
                        f"{','.join(non_matching_files[:5])}{', ...' if len(non_matching_files) > 5 else ''}"
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
        rule_approvers_set = set()
        for approver in rule.approved_by:
            if "/" in approver:
                org, name = approver.split("/")
                rule_approvers_set.update(gh_get_team_members(org, name))
            else:
                rule_approvers_set.add(approver)
        approvers_intersection = approved_by.intersection(rule_approvers_set)
        # If rule requires approvers but they aren't the ones that reviewed PR
        if len(approvers_intersection) == 0 and len(rule_approvers_set) > 0:
            if reject_reason_score < 10000:
                reject_reason_score = 10000
                reject_reason = "\n".join(
                    (
                        "Approval needed from one of the following:",
                        f"{', '.join(list(rule_approvers_set)[:5])}{', ...' if len(rule_approvers_set) > 5 else ''}",
                    )
                )
            continue

        # Does the PR pass the checks required by this rule?
        mandatory_checks = (
            rule.mandatory_checks_name if rule.mandatory_checks_name is not None else []
        )
        required_checks = list(
            filter(
                lambda x: "EasyCLA" in x or not skip_mandatory_checks, mandatory_checks
            )
        )
        pending_checks, failed_checks = categorize_checks(
            checks,
            required_checks,
            ok_failed_checks_threshold=3 if rule.ignore_flaky_failures else 0,
        )

        hud_link = f"https://hud.pytorch.org/{pr.org}/{pr.project}/commit/{pr.last_commit()['oid']}"
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
                "This PR has internal changes and must be landed via Phabricator"
            )

        # Categorize all checks when skip_mandatory_checks (force merge) is set. Do it here
        # where the list of checks is readily available
        pending_mandatory_checks, failed_mandatory_checks = categorize_checks(
            checks,
            [],
            ok_failed_checks_threshold=0,
        )
        return (rule, pending_mandatory_checks, failed_mandatory_checks)

    if reject_reason_score == 20000:
        raise MandatoryChecksMissingError(reject_reason, rule)
    raise MergeRuleFailedError(reject_reason, rule)


def checks_to_str(checks: List[Tuple[str, Optional[str]]]) -> str:
    return ", ".join(f"[{c[0]}]({c[1]})" if c[1] is not None else c[0] for c in checks)


def checks_to_markdown_bullets(
    checks: List[Tuple[str, Optional[str], Optional[int]]]
) -> List[str]:
    return [
        f"- [{c[0]}]({c[1]})" if c[1] is not None else f"- {c[0]}" for c in checks[:5]
    ]


@retries_decorator(rc=[])
def _get_flaky_rules(url: str) -> List[FlakyRule]:
    return [FlakyRule(**rule) for rule in gh_fetch_json_list(url)]


@retries_decorator()
def save_merge_record(
    collection: str,
    comment_id: int,
    pr_num: int,
    owner: str,
    project: str,
    author: str,
    pending_checks: List[Tuple[str, Optional[str], Optional[int]]],
    failed_checks: List[Tuple[str, Optional[str], Optional[int]]],
    last_commit_sha: str,
    merge_base_sha: str,
    merge_commit_sha: str = "",
    is_failed: bool = False,
    dry_run: bool = False,
    skip_mandatory_checks: bool = False,
    ignore_current: bool = False,
    error: str = "",
    workspace: str = "commons",
) -> None:
    """
    This saves the merge records into Rockset, so we can query them (for fun and profit)
    """
    if dry_run:
        # Decide not to save the record to Rockset if dry-run is set to not pollute
        # the collection
        return

    try:
        import rockset  # type: ignore[import]

        # Prepare the record to be written into Rockset
        data = [
            {
                "comment_id": comment_id,
                "pr_num": pr_num,
                "owner": owner,
                "project": project,
                "author": author,
                "pending_checks": pending_checks,
                "failed_checks": failed_checks,
                "last_commit_sha": last_commit_sha,
                "merge_base_sha": merge_base_sha,
                "merge_commit_sha": merge_commit_sha,
                "is_failed": is_failed,
                "skip_mandatory_checks": skip_mandatory_checks,
                "ignore_current": ignore_current,
                "error": error,
            }
        ]

        client = rockset.RocksetClient(
            host="api.usw2a1.rockset.com", api_key=os.environ["ROCKSET_API_KEY"]
        )
        client.Documents.add_documents(
            collection=collection,
            data=data,
            workspace=workspace,
        )

    except ModuleNotFoundError:
        print("Rockset is missing, no record will be saved")
        return


@retries_decorator(rc=[])
def get_rockset_results(head_sha: str, merge_base: str) -> List[Dict[str, Any]]:
    query = f"""
SELECT
    w.name as workflow_name,
    j.id,
    j.name,
    j.conclusion,
    j.completed_at,
    j.html_url,
    j.head_sha,
    j.torchci_classification.captures as failure_captures,
    LENGTH(j.steps) as steps,
FROM
    commons.workflow_job j join commons.workflow_run w on w.id = j.run_id
where
    j.head_sha in ('{head_sha}','{merge_base}')
"""
    try:
        import rockset  # type: ignore[import]

        res = rockset.RocksetClient(
            host="api.usw2a1.rockset.com", api_key=os.environ["ROCKSET_API_KEY"]
        ).sql(query)
        return cast(List[Dict[str, Any]], res.results)
    except ModuleNotFoundError:
        print("Could not use RockSet as rocket dependency is missing")
        return []


def get_classifications(
    checks: Dict[str, JobCheckState],
    head_sha: str,
    merge_base: Optional[str],
    flaky_rules: List[FlakyRule],
    ignore_current_checks: Optional[List[str]],
) -> Dict[str, JobCheckState]:
    head_sha_jobs: Dict[str, Dict[str, Any]] = {}
    merge_base_jobs: Dict[str, Dict[str, Any]] = {}

    if merge_base is not None:

        def insert(d: Dict[str, Dict[str, Any]], key: str, val: Dict[str, Any]) -> None:
            if key not in d:
                d[key] = val
                return
            if d[key]["id"] < val["id"]:
                d[key] = val

        rockset_results = get_rockset_results(head_sha, merge_base)
        for rockset_result in rockset_results:
            name = f"{rockset_result['workflow_name']} / {rockset_result['name']}"
            if rockset_result["head_sha"] == head_sha:
                insert(head_sha_jobs, name, rockset_result)
            else:
                insert(merge_base_jobs, name, rockset_result)

    checks_with_classifications = checks.copy()
    for name, check in checks.items():
        if check.status == "SUCCESS":
            continue
        if ignore_current_checks is not None and name in ignore_current_checks:
            checks_with_classifications[name] = JobCheckState(
                check.name,
                check.url,
                check.status,
                "IGNORE_CURRENT_CHECK",
                check.job_id,
            )
            continue
        head_sha_job = head_sha_jobs.get(name)
        merge_base_job = merge_base_jobs.get(name)
        if (
            head_sha_job is not None
            and merge_base_job is not None
            and head_sha_job["conclusion"] == merge_base_job["conclusion"]
            and head_sha_job["failure_captures"] == merge_base_job["failure_captures"]
        ):
            checks_with_classifications[name] = JobCheckState(
                check.name, check.url, check.status, "BROKEN_TRUNK", check.job_id
            )
        elif any(rule.matches(head_sha_job) for rule in flaky_rules):
            checks_with_classifications[name] = JobCheckState(
                check.name, check.url, check.status, "FLAKY", check.job_id
            )
    return checks_with_classifications


def filter_checks_with_lambda(
    checks: JobNameToStateDict, status_filter: Callable[[Optional[str]], bool]
) -> List[JobCheckState]:
    return [check for check in checks.values() if status_filter(check.status)]


def validate_revert(
    repo: GitRepo, pr: GitHubPR, *, comment_id: Optional[int] = None
) -> Tuple[str, str]:
    comment = (
        pr.get_last_comment()
        if comment_id is None
        else pr.get_comment_by_id(comment_id)
    )
    if comment.editor_login is not None:
        raise PostCommentError("Don't want to revert based on edited command")
    author_association = comment.author_association
    author_login = comment.author_login
    allowed_reverters = ["COLLABORATOR", "MEMBER", "OWNER"]
    # For some reason, one can not be a member of private repo, only CONTRIBUTOR
    if pr.is_base_repo_private():
        allowed_reverters.append("CONTRIBUTOR")
    if author_association not in allowed_reverters:
        raise PostCommentError(
            (
                f"Will not revert as @{author_login} is not one of "
                f"[{', '.join(allowed_reverters)}], but instead is {author_association}."
            )
        )
    skip_internal_checks = can_skip_internal_checks(pr, comment_id)

    # Raises exception if matching rule is not found, but ignores all status checks
    find_matching_merge_rule(
        pr, repo, skip_mandatory_checks=True, skip_internal_checks=skip_internal_checks
    )
    commit_sha = pr.get_merge_commit()
    if commit_sha is None:
        commits = repo.commits_resolving_gh_pr(pr.pr_num)
        if len(commits) == 0:
            raise PostCommentError("Can't find any commits resolving PR")
        commit_sha = commits[0]
    msg = repo.commit_message(commit_sha)
    rc = RE_DIFF_REV.search(msg)
    if rc is not None and not skip_internal_checks:
        raise PostCommentError(
            f"Can't revert PR that was landed via phabricator as {rc.group(1)}.  "
            + "Please revert by going to the internal diff and clicking Unland."
        )
    return (author_login, commit_sha)


def try_revert(
    repo: GitRepo,
    pr: GitHubPR,
    *,
    dry_run: bool = False,
    comment_id: Optional[int] = None,
    reason: Optional[str] = None,
) -> None:
    def post_comment(msg: str) -> None:
        gh_post_pr_comment(pr.org, pr.project, pr.pr_num, msg, dry_run=dry_run)

    try:
        author_login, commit_sha = validate_revert(repo, pr, comment_id=comment_id)
    except PostCommentError as e:
        return post_comment(str(e))
    revert_msg = f"\nReverted {pr.get_pr_url()} on behalf of {prefix_with_github_url(author_login)}"
    revert_msg += f" due to {reason}" if reason is not None else ""
    revert_msg += (
        f" ([comment]({pr.get_comment_by_id(comment_id).url}))\n"
        if comment_id is not None
        else "\n"
    )
    repo.checkout(pr.default_branch())
    repo.revert(commit_sha)
    msg = repo.commit_message("HEAD")
    msg = re.sub(RE_PULL_REQUEST_RESOLVED, "", msg)
    msg += revert_msg
    repo.amend_commit_message(msg)
    repo.push(pr.default_branch(), dry_run)
    post_comment(
        f"@{pr.get_pr_creator_login()} your PR has been successfully reverted."
    )
    if not dry_run:
        pr.add_numbered_label("reverted")
        gh_post_commit_comment(pr.org, pr.project, commit_sha, revert_msg)


def prefix_with_github_url(suffix_str: str) -> str:
    return f"https://github.com/{suffix_str}"


def check_for_sev(org: str, project: str, skip_mandatory_checks: bool) -> None:
    if skip_mandatory_checks:
        return
    response = cast(
        Dict[str, Any],
        gh_fetch_json_list(
            "https://api.github.com/search/issues",
            params={"q": f'repo:{org}/{project} is:open is:issue label:"ci: sev"'},
        ),
    )
    if response["total_count"] != 0:
        for item in response["items"]:
            if "MERGE BLOCKING" in item["body"]:
                raise RuntimeError(
                    "Not merging any PRs at the moment because there is a "
                    + "merge blocking https://github.com/pytorch/pytorch/labels/ci:%20sev issue open at: \n"
                    + f"{item['html_url']}"
                )
    return


def has_label(labels: List[str], pattern: Pattern[str] = CIFLOW_LABEL) -> bool:
    return len(list(filter(pattern.match, labels))) > 0


def categorize_checks(
    check_runs: JobNameToStateDict,
    required_checks: List[str],
    ok_failed_checks_threshold: int = 3,
) -> Tuple[
    List[Tuple[str, Optional[str], Optional[int]]],
    List[Tuple[str, Optional[str], Optional[int]]],
]:
    pending_checks: List[Tuple[str, Optional[str], Optional[int]]] = []
    failed_checks: List[Tuple[str, Optional[str], Optional[int]]] = []
    ok_failed_checks: List[Tuple[str, Optional[str], Optional[int]]] = []

    # If required_checks is not set or empty, consider all names are relevant
    relevant_checknames = [
        name
        for name in check_runs.keys()
        if not required_checks or any(x in name for x in required_checks)
    ]

    for checkname in required_checks:
        if all(checkname not in x for x in check_runs.keys()):
            pending_checks.append((checkname, None, None))

    for checkname in relevant_checknames:
        status = check_runs[checkname].status
        url = check_runs[checkname].url
        classification = check_runs[checkname].classification
        job_id = check_runs[checkname].job_id

        if status is None:
            pending_checks.append((checkname, url, job_id))
        elif not is_passing_status(check_runs[checkname].status):
            if classification == "IGNORE_CURRENT_CHECK":
                pass
            elif classification in ("BROKEN_TRUNK", "FLAKY"):
                ok_failed_checks.append((checkname, url, job_id))
            else:
                failed_checks.append((checkname, url, job_id))

    if ok_failed_checks:
        print(
            f"The following {len(ok_failed_checks)} checks failed but were likely due flakiness or broken trunk: "
            + ", ".join([x[0] for x in ok_failed_checks])
            + (
                f" but this is greater than the threshold of {ok_failed_checks_threshold} so merge will fail"
                if len(ok_failed_checks) > ok_failed_checks_threshold
                else ""
            )
        )

    if len(ok_failed_checks) > ok_failed_checks_threshold:
        failed_checks = failed_checks + ok_failed_checks

    return (pending_checks, failed_checks)


def merge(
    pr: GitHubPR,
    repo: GitRepo,
    dry_run: bool = False,
    skip_mandatory_checks: bool = False,
    comment_id: Optional[int] = None,
    timeout_minutes: int = 400,
    stale_pr_days: int = 3,
    ignore_current: bool = False,
) -> None:
    initial_commit_sha = pr.last_commit()["oid"]
    print(f"Attempting merge of {initial_commit_sha}")

    if MERGE_IN_PROGRESS_LABEL not in pr.get_labels():
        gh_add_labels(pr.org, pr.project, pr.pr_num, [MERGE_IN_PROGRESS_LABEL])

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

    if skip_mandatory_checks or can_skip_internal_checks(pr, comment_id):
        # do not wait for any pending signals if PR is closed as part of co-development process
        gh_post_pr_comment(
            pr.org,
            pr.project,
            pr.pr_num,
            explainer.get_merge_message(),
            dry_run=dry_run,
        )
        return pr.merge_into(
            repo,
            dry_run=dry_run,
            skip_mandatory_checks=skip_mandatory_checks,
            comment_id=comment_id,
        )

    # Check for approvals
    find_matching_merge_rule(pr, repo, skip_mandatory_checks=True)

    if not has_required_labels(pr):
        raise RuntimeError(LABEL_ERR_MSG.lstrip(" #"))

    if ignore_current:
        checks = pr.get_checkrun_conclusions()
        _, failing = categorize_checks(checks, list(checks.keys()))
        ignore_current_checks_info = failing

    gh_post_pr_comment(
        pr.org,
        pr.project,
        pr.pr_num,
        explainer.get_merge_message(ignore_current_checks_info),
        dry_run=dry_run,
    )

    if pr.last_pushed_at() is None:
        print(
            f"Can't get commit {pr.last_commit()['oid']} pushed date. Is it merge commit by chance?"
        )
    elif (datetime.utcnow() - cast(datetime, pr.last_pushed_at())).days > stale_pr_days:
        raise RuntimeError(
            f"This PR is too stale; the last push date was more than {stale_pr_days} days ago. "
            "Please rebase and try again. You can rebase and merge by leaving the following comment on this PR:\n"
            "`@pytorchbot merge -r`\n"
            "Or just rebase by leaving `@pytorchbot rebase` comment"
        )

    start_time = time.time()
    last_exception = ""
    elapsed_time = 0.0
    flaky_rules = read_flaky_rules()
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
        if initial_commit_sha != pr.last_commit()["oid"]:
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
                checks,
                pr.last_commit()["oid"],
                pr.get_merge_base(),
                flaky_rules,
                ignore_current_checks=ignore_current_checks,
            )
            pending, failing = categorize_checks(
                checks,
                required_checks
                + [x for x in checks.keys() if x not in required_checks],
                ok_failed_checks_threshold=3 if ignore_flaky_failures else 0,
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
            )
        except MandatoryChecksMissingError as ex:
            last_exception = str(ex)
            print(
                f"Merge of https://github.com/{pr.org}/{pr.project}/pull/{pr.pr_num} failed due to: {ex}. Retrying in 5 min"
            )
            time.sleep(5 * 60)
    # Finally report timeout back
    msg = f"Merged timed out after {timeout_minutes} minutes. Please contact the pytorch_dev_infra team."
    msg += f"The last exception was: {last_exception}"
    if not dry_run:
        gh_add_labels(pr.org, pr.project, pr.pr_num, ["land-failed"])
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
        import traceback

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

    if not args.force and pr.has_invalid_submodule_updates():
        message = (
            f"This PR updates submodules {', '.join(pr.get_changed_submodules())}\n"
        )
        message += '\nIf those updates are intentional, please add "submodule" keyword to PR title/description.'
        gh_post_pr_comment(org, project, args.pr_num, message, dry_run=args.dry_run)
        return
    try:
        merge(
            pr,
            repo,
            dry_run=args.dry_run,
            skip_mandatory_checks=args.force,
            comment_id=args.comment_id,
            ignore_current=args.ignore_current,
        )
    except Exception as e:
        handle_exception(e)

        if args.comment_id and args.pr_num:
            # Finally, upload the record to Rockset, we don't have access to the
            # list of pending and failed checks here, but they are not really
            # needed at the moment
            save_merge_record(
                collection=ROCKSET_MERGES_COLLECTION,
                comment_id=args.comment_id,
                pr_num=args.pr_num,
                owner=org,
                project=project,
                author=pr.get_author(),
                pending_checks=[],
                failed_checks=[],
                last_commit_sha=pr.last_commit().get("oid", ""),
                merge_base_sha=pr.get_merge_base(),
                is_failed=True,
                dry_run=args.dry_run,
                skip_mandatory_checks=args.force,
                ignore_current=args.ignore_current,
                error=str(e),
                workspace=ROCKSET_MERGES_WORKSPACE,
            )
        else:
            print("Missing comment ID or PR number, couldn't upload to Rockset")
    finally:
        gh_remove_label(org, project, args.pr_num, MERGE_IN_PROGRESS_LABEL)


if __name__ == "__main__":
    main()
