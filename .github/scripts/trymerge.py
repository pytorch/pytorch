#!/usr/bin/env python3

import base64
import json
import os
import re
import time
import urllib.parse
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
import yaml
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Pattern,
    Tuple,
    Union,
    cast,
    NamedTuple
)
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from warnings import warn
from pathlib import Path

from gitutils import (
    GitRepo,
    get_git_remote_name,
    get_git_repo_dir,
    patterns_to_regex,
)
from trymerge_explainer import (
    TryMergeExplainer,
    get_revert_message,
)

class WorkflowCheckState(NamedTuple):
    status: Optional[str]
    url: str
    name: str

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
      }
      checkRuns(first: 50) {
        nodes {
          name
          conclusion
          detailsUrl
        }
        pageInfo {
          endCursor
          hasNextPage
        }
      }
      conclusion
      url
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

GH_GET_PR_INFO_QUERY = GH_PR_REVIEWS_FRAGMENT + GH_CHECKSUITES_FRAGMENT + GH_COMMIT_AUTHORS_FRAGMENT + """
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

GH_GET_PR_NEXT_CHECKSUITES = GH_CHECKSUITES_FRAGMENT + """
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

GH_GET_COMMIT_CHECKSUITES = GH_CHECKSUITES_FRAGMENT + """
query ($owner: String!, $name: String!, $commit: String) {
  repository(name: $name, owner: $owner) {
    object(expression: $commit) {
      ... on Commit {
        checkSuites {
          ...PRCheckSuites
        }
      }
    }
  }
}
"""

GH_GET_COMMIT_NEXT_CHECKSUITES = GH_CHECKSUITES_FRAGMENT + """
query ($owner: String!, $name: String!, $commit: String, $cursor: String!) {
  repository(name: $name, owner: $owner) {
    object(expression: $commit) {
      ... on Commit {
        oid
        checkSuites(first: 10, after: $cursor) {
          ...PRCheckSuites
        }
      }
    }
  }
}
"""

GH_GET_COMMIT_NEXT_CHECK_RUNS = """
query ($owner: String!, $name: String!, $cs_cursor: String, $cr_cursor: String!, $commit: String) {
  repository(name: $name, owner: $owner) {
    object(expression: $commit) {
      ... on Commit {
        oid
        checkSuites(first: 1, after: $cs_cursor) {
          nodes {
            checkRuns(first: 100, after: $cr_cursor) {
              nodes {
                name
                conclusion
                detailsUrl
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
"""

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

GH_GET_PR_NEXT_AUTHORS_QUERY = GH_COMMIT_AUTHORS_FRAGMENT + """
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

GH_GET_PR_PREV_REVIEWS_QUERY = GH_PR_REVIEWS_FRAGMENT + """
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

RE_GHSTACK_HEAD_REF = re.compile(r"^(gh/[^/]+/[0-9]+/)head$")
RE_GHSTACK_DESC = re.compile(r'Stack.*:\r?\n(\* [^\r\n]+\r?\n)+', re.MULTILINE)
RE_PULL_REQUEST_RESOLVED = re.compile(
    r'Pull Request resolved: '
    r'https://github.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/pull/(?P<number>[0-9]+)',
    re.MULTILINE
)
RE_PR_CC_LINE = re.compile(r'^cc:? @\w+.*\r?\n?$', re.MULTILINE)
RE_DIFF_REV = re.compile(r'^Differential Revision:.+?(D[0-9]+)', re.MULTILINE)
CIFLOW_LABEL = re.compile(r"^ciflow/.+")
CIFLOW_TRUNK_LABEL = re.compile(r"^ciflow/trunk")
MERGE_RULE_PATH = Path(".github") / "merge_rules.yaml"


def _fetch_url(url: str, *,
               headers: Optional[Dict[str, str]] = None,
               data: Optional[Dict[str, Any]] = None,
               method: Optional[str] = None,
               reader: Callable[[Any], Any] = lambda x: x.read()) -> Any:
    if headers is None:
        headers = {}
    token = os.environ.get("GITHUB_TOKEN")
    if token is not None and url.startswith('https://api.github.com/'):
        headers['Authorization'] = f'token {token}'
    data_ = json.dumps(data).encode() if data is not None else None
    try:
        with urlopen(Request(url, headers=headers, data=data_, method=method)) as conn:
            return reader(conn)
    except HTTPError as err:
        if err.code == 403 and all(key in err.headers for key in ['X-RateLimit-Limit', 'X-RateLimit-Used']):
            print(f"Rate limit exceeded: {err.headers['X-RateLimit-Used']}/{err.headers['X-RateLimit-Limit']}")
        raise

def fetch_json(url: str,
               params: Optional[Dict[str, Any]] = None,
               data: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    headers = {'Accept': 'application/vnd.github.v3+json'}
    if params is not None and len(params) > 0:
        url += '?' + '&'.join(f"{name}={urllib.parse.quote(str(val))}" for name, val in params.items())
    return cast(List[Dict[str, Any]], _fetch_url(url, headers=headers, data=data, reader=json.load))

def fetch_json_dict(url: str,
                    params: Optional[Dict[str, Any]] = None,
                    data: Optional[Dict[str, Any]] = None) -> Dict[str, Any] :
    headers = {'Accept': 'application/vnd.github.v3+json'}
    if params is not None and len(params) > 0:
        url += '?' + '&'.join(f"{name}={urllib.parse.quote(str(val))}" for name, val in params.items())
    return cast(Dict[str, Any], _fetch_url(url, headers=headers, data=data, reader=json.load))

def _gh_post_comment(url: str, comment: str, dry_run: bool = False) -> List[Dict[str, Any]]:
    if dry_run:
        print(comment)
        return []
    return fetch_json(url, data={"body": comment})


def gh_post_pr_comment(org: str, project: str, pr_num: int, comment: str, dry_run: bool = False) -> List[Dict[str, Any]]:
    return _gh_post_comment(f'https://api.github.com/repos/{org}/{project}/issues/{pr_num}/comments', comment, dry_run)


def gh_post_commit_comment(org: str, project: str, sha: str, comment: str, dry_run: bool = False) -> List[Dict[str, Any]]:
    return _gh_post_comment(f'https://api.github.com/repos/{org}/{project}/commits/{sha}/comments', comment, dry_run)


def gh_add_labels(org: str, project: str, pr_num: int, labels: Union[str, List[str]]) -> None:
    fetch_json(f'https://api.github.com/repos/{org}/{project}/issues/{pr_num}/labels',
               data={"labels": labels})


def gh_graphql(query: str, **kwargs: Any) -> Dict[str, Any]:
    rc = _fetch_url("https://api.github.com/graphql", data={"query": query, "variables": kwargs}, reader=json.load)
    if "errors" in rc:
        raise RuntimeError(f"GraphQL query {query}, args {kwargs} failed: {rc['errors']}")
    return cast(Dict[str, Any], rc)


def gh_get_pr_info(org: str, proj: str, pr_no: int) -> Any:
    rc = gh_graphql(GH_GET_PR_INFO_QUERY, name=proj, owner=org, number=pr_no)
    return rc["data"]["repository"]["pullRequest"]

def gh_get_land_check_info(org: str, proj: str, commit: str) -> Any:
    rc = gh_graphql(GH_GET_COMMIT_CHECKSUITES, name=proj, owner=org, commit=commit)
    return rc["data"]["repository"]["object"]

@lru_cache(maxsize=None)
def gh_get_team_members(org: str, name: str) -> List[str]:
    rc: List[str] = []
    team_members: Dict[str, Any] = {"pageInfo": {"hasNextPage": "true", "endCursor": None}}
    while bool(team_members["pageInfo"]["hasNextPage"]):
        query = gh_graphql(GH_GET_TEAM_MEMBERS_QUERY, org=org, name=name, cursor=team_members["pageInfo"]["endCursor"])
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

def add_workflow_conclusions(
    checksuites: Any,
    get_next_checkruns_page: Callable[[List[Dict[str, Dict[str, Any]]], int, Any], Any],
    get_next_checksuites: Callable[[Any], Any]
) -> Dict[str, WorkflowCheckState]:
    conclusions = {}

    def add_conclusions(edges: Any) -> None:
        for edge_idx, edge in enumerate(edges):
            node = edge["node"]
            workflow_run = node["workflowRun"]
            checkruns = node["checkRuns"]
            if workflow_run is not None:
                workflow_name = workflow_run["workflow"]["name"]
                workflow_conclusion = node["conclusion"]
                # Do not override existing status with cancelled
                if workflow_conclusion == "CANCELLED" and workflow_name in conclusions:
                    continue
                conclusions[workflow_name] = WorkflowCheckState(
                    name=workflow_name,
                    status=workflow_conclusion,
                    url=node["url"])
            has_failing_check = False
            while checkruns is not None:
                for checkrun_node in checkruns["nodes"]:
                    if not isinstance(checkrun_node, dict):
                        warn(f"Expected dictionary, but got {type(checkrun_node)}")
                        continue
                    if checkrun_node["conclusion"] == 'FAILURE':
                        has_failing_check = True
                    checkrun_name = f'{get_check_run_name_prefix(workflow_run)}{checkrun_node["name"]}'
                    conclusions[checkrun_name] = WorkflowCheckState(
                        name=checkrun_name,
                        status=checkrun_node["conclusion"],
                        url=checkrun_node["detailsUrl"]
                    )
                if bool(checkruns["pageInfo"]["hasNextPage"]):
                    checkruns = get_next_checkruns_page(edges, edge_idx, checkruns)
                else:
                    checkruns = None
            # Github doesn't set conclusion to failure if a job is still pending
            if workflow_run is not None and has_failing_check:
                workflow_name = workflow_run["workflow"]["name"]
                conclusions[workflow_name] = WorkflowCheckState(
                    name=workflow_name,
                    status="FAILURE",
                    url=node["url"])

    add_conclusions(checksuites["edges"])
    while bool(checksuites["pageInfo"]["hasNextPage"]):
        checksuites = get_next_checksuites(checksuites)
        add_conclusions(checksuites["edges"])

    return conclusions


def parse_args() -> Any:
    from argparse import ArgumentParser
    parser = ArgumentParser("Merge PR into default branch")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--on-green", action="store_true")
    parser.add_argument("--on-mandatory", action="store_true")
    parser.add_argument("--land-checks", action="store_true")
    parser.add_argument("--revert", action="store_true")
    parser.add_argument("--force", action="store_true")
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


@dataclass
class GitHubComment:
    body_text: str
    created_at: str
    author_login: str
    author_association: str
    editor_login: Optional[str]
    database_id: int


class GitHubPR:
    def __init__(self, org: str, project: str, pr_num: int) -> None:
        assert isinstance(pr_num, int)
        self.org = org
        self.project = project
        self.pr_num = pr_num
        self.info = gh_get_pr_info(org, project, pr_num)
        self.changed_files: Optional[List[str]] = None
        self.labels: Optional[List[str]] = None
        self.conclusions: Optional[Dict[str, WorkflowCheckState]] = None
        self.comments: Optional[List[GitHubComment]] = None
        self._authors: Optional[List[Tuple[str, str]]] = None
        self._reviews: Optional[List[Tuple[str, str]]] = None

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

    def last_pushed_at(self) -> datetime:
        return datetime.fromisoformat(self.last_commit()['pushedDate'][:-1])

    def last_commit(self) -> Any:
        return self.info["commits"]["nodes"][-1]["commit"]

    def get_changed_files(self) -> List[str]:
        if self.changed_files is None:
            info = self.info
            self.changed_files = []
            # Do not try to fetch more than 10K files
            for _ in range(100):
                self.changed_files += [x["path"] for x in info["files"]["nodes"]]
                if not info["files"]["pageInfo"]["hasNextPage"]:
                    break
                rc = gh_graphql(GH_GET_PR_NEXT_FILES_QUERY,
                                name=self.project,
                                owner=self.org,
                                number=self.pr_num,
                                cursor=info["files"]["pageInfo"]["endCursor"])
                info = rc["data"]["repository"]["pullRequest"]

        if len(self.changed_files) != self.get_changed_files_count():
            raise RuntimeError("Changed file count mismatch")
        return self.changed_files

    def _get_reviews(self) -> List[Tuple[str, str]]:
        if self._reviews is None:
            self._reviews = []
            info = self.info
            for _ in range(100):
                nodes = info["reviews"]["nodes"]
                self._reviews = [(node["author"]["login"], node["state"]) for node in nodes] + self._reviews
                if not info["reviews"]["pageInfo"]["hasPreviousPage"]:
                    break
                rc = gh_graphql(GH_GET_PR_PREV_REVIEWS_QUERY,
                                name=self.project,
                                owner=self.org,
                                number=self.pr_num,
                                cursor=info["reviews"]["pageInfo"]["startCursor"])
                info = rc["data"]["repository"]["pullRequest"]
        reviews = {}
        for (author, state) in self._reviews:
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
            rc = gh_graphql(GH_GET_PR_NEXT_AUTHORS_QUERY,
                            name=self.project,
                            owner=self.org,
                            number=self.pr_num,
                            cursor=info["commits_with_authors"]["pageInfo"]["endCursor"])
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
        labels = [node['node']['name'] for node in self.info["labels"]["edges"]] if "labels" in self.info else []
        self.labels = labels
        return self.labels

    def get_checkrun_conclusions(self) -> Dict[str, WorkflowCheckState]:
        """ Returns dict of checkrun -> [conclusion, url] """
        if self.conclusions is not None:
            return self.conclusions
        orig_last_commit = self.info["commits"]["nodes"][-1]["commit"]

        def get_pr_next_check_runs(edges: List[Dict[str, Dict[str, Any]]], edge_idx: int, checkruns: Any) -> Any:
            rc = gh_graphql(GH_GET_PR_NEXT_CHECK_RUNS,
                            name=self.project,
                            owner=self.org,
                            number=self.pr_num,
                            cs_cursor=edges[edge_idx - 1]["cursor"] if edge_idx > 0 else None,
                            cr_cursor=checkruns["pageInfo"]["endCursor"])
            last_commit = rc["data"]["repository"]["pullRequest"]["commits"]["nodes"][-1]["commit"]
            checkruns = last_commit["checkSuites"]["nodes"][-1]["checkRuns"]
            return checkruns

        def get_pr_next_checksuites(checksuites: Any) -> Any:
            rc = gh_graphql(GH_GET_PR_NEXT_CHECKSUITES,
                            name=self.project,
                            owner=self.org,
                            number=self.pr_num,
                            cursor=checksuites["edges"][-1]["cursor"])
            info = rc["data"]["repository"]["pullRequest"]
            last_commit = info["commits"]["nodes"][-1]["commit"]
            if last_commit["oid"] != orig_last_commit["oid"]:
                raise RuntimeError("Last commit changed on PR")
            return last_commit["checkSuites"]

        checksuites = orig_last_commit["checkSuites"]

        self.conclusions = add_workflow_conclusions(checksuites, get_pr_next_check_runs, get_pr_next_checksuites)

        # Append old style statuses(like ones populated by CircleCI or EasyCLA) to conclusions
        if orig_last_commit["status"] and orig_last_commit["status"]["contexts"]:
            for status in orig_last_commit["status"]["contexts"]:
                name = status["context"]
                self.conclusions[name] = WorkflowCheckState(name=name, status=status["state"], url=status["targetUrl"])

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
        return GitHubComment(body_text=node["bodyText"],
                             created_at=node["createdAt"] if "createdAt" in node else "",
                             author_login=node["author"]["login"],
                             author_association=node["authorAssociation"],
                             editor_login=editor["login"] if editor else None,
                             database_id=node["databaseId"]
                             )

    def get_comments(self) -> List[GitHubComment]:
        if self.comments is not None:
            return self.comments
        self.comments = []
        info = self.info["comments"]
        # Do not try to fetch more than 10K comments
        for _ in range(100):
            self.comments = [self._comment_from_node(node) for node in info["nodes"]] + self.comments
            if not info["pageInfo"]["hasPreviousPage"]:
                break
            rc = gh_graphql(GH_GET_PR_PREV_COMMENTS,
                            name=self.project,
                            owner=self.org,
                            number=self.pr_num,
                            cursor=info["pageInfo"]["startCursor"])
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
        land_check_commit: Optional[str] = None
    ) -> None:
        assert self.is_ghstack_pr()
        # For ghstack, cherry-pick commits based from origin
        orig_ref = f"{repo.remote}/{re.sub(r'/head$', '/orig', self.head_ref())}"
        rev_list = repo.revlist(f"{self.default_branch()}..{orig_ref}")
        for idx, rev in enumerate(reversed(rev_list)):
            msg = repo.commit_message(rev)
            m = RE_PULL_REQUEST_RESOLVED.search(msg)
            if m is None:
                raise RuntimeError(f"Could not find PR-resolved string in {msg} of ghstacked PR {self.pr_num}")
            if self.org != m.group('owner') or self.project != m.group('repo'):
                raise RuntimeError(f"PR {m.group('number')} resolved to wrong owner/repo pair")
            pr_num = int(m.group('number'))
            commit_msg = self.gen_commit_message(filter_ghstack=True)
            if pr_num != self.pr_num:
                pr = GitHubPR(self.org, self.project, pr_num)
                if pr.is_closed():
                    print(f"Skipping {idx+1} of {len(rev_list)} PR (#{pr_num}) as its already been merged")
                    continue
                commit_msg = pr.gen_commit_message(filter_ghstack=True)
                # Raises exception if matching rule is not found
                find_matching_merge_rule(
                    pr,
                    repo,
                    skip_mandatory_checks=skip_mandatory_checks,
                    skip_internal_checks=can_skip_internal_checks(self, comment_id),
                    land_check_commit=land_check_commit)

            repo.cherry_pick(rev)
            repo.amend_commit_message(commit_msg)

    def gen_commit_message(self, filter_ghstack: bool = False) -> str:
        """ Fetches title and body from PR description
            adds reviewed by, pull request resolved and optionally
            filters out ghstack info """
        # Adding the url here makes it clickable within the Github UI
        approved_by_urls = ', '.join(prefix_with_github_url(login) for login in self.get_approved_by())
        # Remove "cc: " line from the message body
        msg_body = re.sub(RE_PR_CC_LINE, "", self.get_body())
        if filter_ghstack:
            msg_body = re.sub(RE_GHSTACK_DESC, "", msg_body)
        msg = self.get_title() + f" (#{self.pr_num})\n\n"
        msg += msg_body
        msg += f"\nPull Request resolved: {self.get_pr_url()}\n"
        msg += f"Approved by: {approved_by_urls}\n"
        return msg

    def merge_into(self, repo: GitRepo, *,
                   skip_mandatory_checks: bool = False,
                   dry_run: bool = False,
                   comment_id: Optional[int] = None,
                   land_check_commit: Optional[str] = None) -> None:
        # Raises exception if matching rule is not found
        find_matching_merge_rule(
            self,
            repo,
            skip_mandatory_checks=skip_mandatory_checks,
            skip_internal_checks=can_skip_internal_checks(self, comment_id),
            land_check_commit=land_check_commit)
        self.merge_changes(repo, skip_mandatory_checks, comment_id, land_check_commit=land_check_commit)

        repo.push(self.default_branch(), dry_run)
        if not dry_run:
            if land_check_commit:
                self.delete_land_time_check_branch(repo)
            gh_add_labels(self.org, self.project, self.pr_num, ["merged"])

    def merge_changes(self,
                      repo: GitRepo,
                      skip_mandatory_checks: bool = False,
                      comment_id: Optional[int] = None,
                      land_check_commit: Optional[str] = None,
                      branch: Optional[str] = None) -> None:
        branch_to_merge_into = self.default_branch() if branch is None else branch
        if repo.current_branch() != branch_to_merge_into:
            repo.checkout(branch_to_merge_into)
        if not self.is_ghstack_pr():
            msg = self.gen_commit_message()
            pr_branch_name = f"__pull-request-{self.pr_num}__init__"
            repo.fetch(f"pull/{self.pr_num}/head", pr_branch_name)
            repo._run_git("merge", "--squash", pr_branch_name)
            repo._run_git("commit", f"--author=\"{self.get_author()}\"", "-m", msg)
        else:
            self.merge_ghstack_into(
                repo,
                skip_mandatory_checks,
                comment_id=comment_id,
                land_check_commit=land_check_commit
            )

    def create_land_time_check_branch(self,
                                      repo: GitRepo,
                                      branch: str,
                                      skip_mandatory_checks: bool = False,
                                      comment_id: Optional[int] = None,) -> str:
        orig_branch = repo.current_branch()
        self.merge_changes(
            repo,
            branch=branch,
            skip_mandatory_checks=skip_mandatory_checks,
            comment_id=comment_id
        )
        land_check_branch = f'landchecks/{self.pr_num}'
        try:
            repo._run_git('branch', "-D", land_check_branch)
        except Exception:
            pass
        repo._run_git('checkout', "-b", land_check_branch)
        repo._run_git('push', '-u', 'origin', land_check_branch, '--force')
        commit = repo.get_commit('HEAD').commit_hash
        # Important, return to original branch
        if repo.current_branch() != orig_branch:
            repo.checkout(orig_branch)
        return commit

    def delete_land_time_check_branch(self,
                                      repo: GitRepo) -> None:
        land_check_branch = f'landchecks/{self.pr_num}'
        repo._run_git('push', 'origin', '-d', land_check_branch)


class MandatoryChecksMissingError(Exception):
    pass

class PostCommentError(Exception):
    pass


@dataclass
class MergeRule:
    name: str
    patterns: List[str]
    approved_by: List[str]
    mandatory_checks_name: Optional[List[str]]


def gen_new_issue_link(
    org: str,
    project: str,
    labels: List[str],
    template: str = "bug-report.yml"
) -> str:
    labels_str = ",". join(labels)
    return (f"https://github.com/{org}/{project}/issues/new?"
            f"labels={urllib.parse.quote(labels_str)}&"
            f"template={urllib.parse.quote(template)}")


def read_merge_rules(repo: Optional[GitRepo], org: str, project: str) -> List[MergeRule]:
    repo_relative_rules_path = MERGE_RULE_PATH
    if repo is None:
        json_data = _fetch_url(
            f"https://api.github.com/repos/{org}/{project}/contents/{repo_relative_rules_path}",
            headers={'Accept': 'application/vnd.github.v3+json'},
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


def find_matching_merge_rule(pr: GitHubPR,
                             repo: Optional[GitRepo] = None,
                             skip_mandatory_checks: bool = False,
                             skip_internal_checks: bool = False,
                             land_check_commit: Optional[str] = None,
                             ) -> MergeRule:
    """Returns merge rule matching to this pr or raises an exception"""
    changed_files = pr.get_changed_files()
    approved_by = set(pr.get_approved_by())
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
                reject_reason = "\n".join((
                    f"Not all files match rule `{rule_name}`."
                    f"{num_matching_files} files matched, but there are still non-matching files:"
                    f"{','.join(non_matching_files[:5])}{', ...' if len(non_matching_files) > 5 else ''}"
                ))
            continue

        # If rule needs approvers but PR has not been reviewed, skip it
        if len(rule.approved_by) > 0 and len(approved_by) == 0:
            if reject_reason_score < 10000:
                reject_reason_score = 10000
                reject_reason = f"PR #{pr.pr_num} has not been reviewed yet (Rule {rule_name})"
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
                reject_reason = "\n".join((
                    f"Approval needed from one of the following (Rule '{rule_name}'):",
                    f"{', '.join(list(rule_approvers_set)[:5])}{', ...' if len(rule_approvers_set) > 5 else ''}"
                ))
            continue

        # Does the PR pass the checks required by this rule?
        mandatory_checks = rule.mandatory_checks_name if rule.mandatory_checks_name is not None else []
        checks = get_combined_checks_from_pr_and_land_validation(pr, land_check_commit)
        required_checks = filter(lambda x: skip_mandatory_checks is False or "EasyCLA" in x, mandatory_checks)
        [pending_checks, failed_checks] = categorize_checks(checks, required_checks)

        hud_link = f"https://hud.pytorch.org/{pr.org}/{pr.project}/commit/{pr.last_commit()['oid']}"
        if len(failed_checks) > 0:
            if reject_reason_score < 30000:
                reject_reason_score = 30000
                reject_reason = "\n".join((
                    f"The following mandatory check(s) failed (Rule `{rule_name}`):",
                    *checks_to_markdown_bullets(failed_checks),
                    "",
                    f"Dig deeper by [viewing the failures on hud]({hud_link})"
                ))
            continue
        elif len(pending_checks) > 0:
            if reject_reason_score < 20000:
                reject_reason_score = 20000
                reject_reason = "\n".join((
                    f"The following mandatory check(s) are pending/not yet run (Rule `{rule_name}`):",
                    *checks_to_markdown_bullets(pending_checks),
                    "",
                    f"Dig deeper by [viewing the pending checks on hud]({hud_link})"
                ))
            continue

        if not skip_internal_checks and pr.has_internal_changes():
            raise RuntimeError("This PR has internal changes and must be landed via Phabricator")

        return rule

    if reject_reason_score == 20000:
        raise MandatoryChecksMissingError(reject_reason)
    raise RuntimeError(reject_reason)


def get_land_checkrun_conclusions(org: str, project: str, commit: str) -> Dict[str, WorkflowCheckState]:

    def get_commit_next_check_runs(edges: List[Dict[str, Dict[str, Any]]], edge_idx: int, checkruns: Any) -> Any:
        rc = gh_graphql(GH_GET_COMMIT_NEXT_CHECK_RUNS,
                        name=project,
                        owner=org,
                        cs_cursor=edges[edge_idx - 1]["cursor"] if edge_idx > 0 else None,
                        cr_cursor=checkruns["pageInfo"]["endCursor"],
                        commit=commit)
        return rc["data"]["repository"]["object"]["checkSuites"]["nodes"][-1]["checkRuns"]

    def get_commit_next_checksuites(checksuites: Any) -> Any:
        rc = gh_graphql(GH_GET_COMMIT_NEXT_CHECKSUITES,
                        name=project,
                        owner=org,
                        commit=commit,
                        cursor=checksuites["edges"][-1]["cursor"])
        info = rc["data"]["repository"]["object"]
        return info["checkSuites"]

    land_check_info = gh_get_land_check_info(org, project, commit)
    checksuites = land_check_info["checkSuites"]

    return add_workflow_conclusions(checksuites, get_commit_next_check_runs, get_commit_next_checksuites)


def checks_to_str(checks: List[Tuple[str, Optional[str]]]) -> str:
    return ", ".join(f"[{c[0]}]({c[1]})" if c[1] is not None else c[0] for c in checks)


def checks_to_markdown_bullets(checks: List[Tuple[str, Optional[str]]]) -> List[str]:
    return [f"- [{c[0]}]({c[1]})" if c[1] is not None else f"- {c[0]}" for c in checks]

def get_combined_checks_from_pr_and_land_validation(
    pr: GitHubPR,
    land_check_commit: Optional[str]
) -> Dict[str, WorkflowCheckState]:
    """
    Combines checks from both the PR and land validation to get a holistic view
    of all checks.

    This helps us cover the corner case where certain workflows may have been
    requested on the PR but are not part of land validation (e.g. nightly
    builds) or are implicitly run on PRs but not on land validation branches
    (like CLA Checks).

    At the same time, we prioritize the signal workflows which do run on land
    validation.

    E.g. if a workflow fails on the PR but passes on land validation then we'd
    use the successful result from the land validation.
    """

    pr_checks = pr.get_checkrun_conclusions()
    land_validation_checks = get_land_checkrun_conclusions(pr.org, pr.project, land_check_commit) if land_check_commit else {}

    # Merge the two checks together. Land validation check results (if any) overwrite pr check results
    merged_checks = {**pr_checks, **land_validation_checks}  # explanation: https://stackoverflow.com/a/26853961/21539
    return merged_checks

def filter_checks_with_lambda(
    checks: Dict[str, WorkflowCheckState],
    status_filter: Callable[[Optional[str]], bool]
) -> List[WorkflowCheckState]:
    return [check for check in checks.values() if status_filter(check.status)]

def filter_pending_checks(checks: Dict[str, WorkflowCheckState]) -> List[WorkflowCheckState]:
    return filter_checks_with_lambda(checks, lambda x: x is None)

def filter_failed_checks(checks: Dict[str, WorkflowCheckState]) -> List[WorkflowCheckState]:
    return filter_checks_with_lambda(checks, lambda x: x in ["FAILURE", "STARTUP_FAILURE"])

def validate_revert(repo: GitRepo, pr: GitHubPR, *,
                    comment_id: Optional[int] = None) -> Tuple[str, str]:
    comment = pr.get_last_comment() if comment_id is None else pr.get_comment_by_id(comment_id)
    if comment.editor_login is not None:
        raise PostCommentError("Don't want to revert based on edited command")
    author_association = comment.author_association
    author_login = comment.author_login
    allowed_reverters = ["COLLABORATOR", "MEMBER", "OWNER"]
    # For some reason, one can not be a member of private repo, only CONTRIBUTOR
    if pr.is_base_repo_private():
        allowed_reverters.append("CONTRIBUTOR")
    if author_association not in allowed_reverters:
        raise PostCommentError((
            f"Will not revert as @{author_login} is not one of "
            f"[{', '.join(allowed_reverters)}], but instead is {author_association}."
        ))
    skip_internal_checks = can_skip_internal_checks(pr, comment_id)

    # Raises exception if matching rule is not found, but ignores all status checks
    find_matching_merge_rule(pr, repo, skip_mandatory_checks=True, skip_internal_checks=skip_internal_checks)
    commit_sha = pr.get_merge_commit()
    if commit_sha is None:
        commits = repo.commits_resolving_gh_pr(pr.pr_num)
        if len(commits) == 0:
            raise PostCommentError("Can't find any commits resolving PR")
        commit_sha = commits[0]
    msg = repo.commit_message(commit_sha)
    rc = RE_DIFF_REV.search(msg)
    if rc is not None and not can_skip_internal_checks:
        raise PostCommentError(f"Can't revert PR that was landed via phabricator as {rc.group(1)}")
    return (author_login, commit_sha)


def try_revert(repo: GitRepo, pr: GitHubPR, *,
               dry_run: bool = False,
               comment_id: Optional[int] = None,
               reason: Optional[str] = None) -> None:
    def post_comment(msg: str) -> None:
        gh_post_pr_comment(pr.org, pr.project, pr.pr_num, msg, dry_run=dry_run)
    try:
        author_login, commit_sha = validate_revert(repo, pr, comment_id=comment_id)
    except PostCommentError as e:
        return post_comment(str(e))
    revert_msg = f"\nReverted {pr.get_pr_url()} on behalf of {prefix_with_github_url(author_login)}"
    revert_msg += f" due to {reason}\n" if reason is not None else "\n"
    repo.checkout(pr.default_branch())
    repo.revert(commit_sha)
    msg = repo.commit_message("HEAD")
    msg = re.sub(RE_PULL_REQUEST_RESOLVED, "", msg)
    msg += revert_msg
    repo.amend_commit_message(msg)
    repo.push(pr.default_branch(), dry_run)
    post_comment(f"@{pr.get_pr_creator_login()} your PR has been successfully reverted.")
    if not dry_run:
        gh_add_labels(pr.org, pr.project, pr.pr_num, ["reverted"])
        gh_post_commit_comment(pr.org, pr.project, commit_sha, revert_msg)


def prefix_with_github_url(suffix_str: str) -> str:
    return f"https://github.com/{suffix_str}"

def check_for_sev(org: str, project: str, skip_mandatory_checks: bool) -> None:
    if skip_mandatory_checks:
        return
    response = cast(
        Dict[str, Any],
        fetch_json(
            "https://api.github.com/search/issues",
            params={"q": f'repo:{org}/{project} is:open is:issue label:"ci: sev"'},
        ),
    )
    if response["total_count"] != 0:
        for item in response["items"]:
            if "merge blocking" in item["body"].lower():
                raise RuntimeError(
                    "Not merging any PRs at the moment because there is a "
                    + "merge blocking https://github.com/pytorch/pytorch/labels/ci:%20sev issue open at: \n"
                    + f"{item['html_url']}"
                )
    return

def validate_land_time_checks(org: str, project: str, commit: str) -> None:
    checks = get_land_checkrun_conclusions(org, project, commit)
    if len(checks) == 0:
        raise MandatoryChecksMissingError("Refusing to merge as land check(s) are not yet run")

    [pending_checks, failed_checks] = categorize_checks(checks, checks)

    if len(failed_checks) > 0:
        raise RuntimeError(f"Failed to merge; some land checks failed: {checks_to_str(failed_checks)}")
    if len(pending_checks) > 0:
        raise MandatoryChecksMissingError(f"Refusing to merge as land check(s) {checks_to_str(pending_checks)} are not yet run")

def has_label(labels: List[str], pattern: Pattern[str] = CIFLOW_LABEL) -> bool:
    return len(list(filter(pattern.match, labels))) > 0

def categorize_checks(check_runs: Dict[str, WorkflowCheckState],
                      required_checks: Iterable[str]) -> Tuple[List[Tuple[str, Optional[str]]], List[Tuple[str, Optional[str]]]]:
    pending_checks: List[Tuple[str, Optional[str]]] = []
    failed_checks: List[Tuple[str, Optional[str]]] = []
    for checkname in required_checks:
        if checkname not in check_runs:
            pending_checks.append((checkname, None))
        elif check_runs[checkname].status is None:
            pending_checks.append((checkname, check_runs[checkname].url))
        elif (str(check_runs[checkname].status).upper() not in ['SUCCESS', 'SKIPPED', 'NEUTRAL']):
            failed_checks.append((checkname, check_runs[checkname].url))
    return (pending_checks, failed_checks)

def merge(pr_num: int, repo: GitRepo,
          dry_run: bool = False,
          skip_mandatory_checks: bool = False,
          comment_id: Optional[int] = None,
          mandatory_only: bool = False,
          on_green: bool = False,
          land_checks: bool = False,
          timeout_minutes: int = 400,
          stale_pr_days: int = 3) -> None:
    repo = GitRepo(get_git_repo_dir(), get_git_remote_name())
    org, project = repo.gh_owner_and_name()
    pr = GitHubPR(org, project, pr_num)
    initial_commit_sha = pr.last_commit()['oid']
    explainer = TryMergeExplainer(skip_mandatory_checks, on_green, land_checks, pr.get_labels(), pr.pr_num, org, project)
    on_green, land_checks = explainer.get_flags()
    land_check_commit = None

    check_for_sev(org, project, skip_mandatory_checks)

    if skip_mandatory_checks or can_skip_internal_checks(pr, comment_id):
        # do not wait for any pending signals if PR is closed as part of co-development process
        gh_post_pr_comment(org, project, pr.pr_num, explainer.get_merge_message())
        return pr.merge_into(
            repo,
            dry_run=dry_run,
            skip_mandatory_checks=skip_mandatory_checks,
            comment_id=comment_id
        )

    # Important: check for merge rule once before starting land checks
    # because we want to make sure that only approved PRs can start CI
    # jobs. If there's missing approval, a RuntimeError will be raised
    # here to stop the merge process right away
    find_matching_merge_rule(pr, repo, skip_mandatory_checks=True)

    if land_checks and not dry_run:
        land_check_commit = pr.create_land_time_check_branch(
            repo,
            'viable/strict',
            skip_mandatory_checks=skip_mandatory_checks,
            comment_id=comment_id
        )

    gh_post_pr_comment(org, project, pr.pr_num, explainer.get_merge_message(land_check_commit))
    if (datetime.utcnow() - pr.last_pushed_at()).days > stale_pr_days:
        if land_checks and not dry_run:
            pr.delete_land_time_check_branch(repo)
        raise RuntimeError(f"This PR is too stale; the last push date was more than {stale_pr_days} days ago. "
                           "Please rebase and try again. You can rebase by leaving the following comment on this PR:\n"
                           "`@pytorchbot rebase`")

    start_time = time.time()
    last_exception = ''
    elapsed_time = 0.0
    while elapsed_time < timeout_minutes * 60:
        check_for_sev(org, project, skip_mandatory_checks)
        current_time = time.time()
        elapsed_time = current_time - start_time
        print(f"Attempting merge of https://github.com/{org}/{project}/pull/{pr_num} ({elapsed_time / 60} minutes elapsed)")
        pr = GitHubPR(org, project, pr_num)
        if initial_commit_sha != pr.last_commit()['oid']:
            if land_checks and not dry_run:
                pr.delete_land_time_check_branch(repo)
            raise RuntimeError("New commits were pushed while merging. Please rerun the merge command.")
        try:
            find_matching_merge_rule(pr, repo)
            checks = get_combined_checks_from_pr_and_land_validation(pr, land_check_commit)
            pending = filter_pending_checks(checks)
            failing = filter_failed_checks(checks)

            # HACK until GitHub will be better about surfacing those
            startup_failures = filter_checks_with_lambda(checks, lambda status: status == "STARTUP_FAILURE")
            if len(startup_failures) > 0:
                raise RuntimeError(f"{len(failing)} STARTUP failures reported, please check workflows syntax! " +
                                   ' ,'.join(f"[{x.name}]({x.url})" for x in startup_failures[:5]))
            # END of HACK

            if (not mandatory_only and on_green) and len(failing) > 0:
                raise RuntimeError(f"{len(failing)} additional jobs have failed, first few of them are: " +
                                   ' ,'.join(f"[{x.name}]({x.url})" for x in failing[:5]))
            if (not mandatory_only and on_green) and len(pending) > 0:
                raise MandatoryChecksMissingError(f"Still waiting for {len(pending)} additional jobs to finish, " +
                                                  f"first few of them are: {' ,'.join(x.name for x in pending[:5])}")
            if land_checks and land_check_commit is not None:
                validate_land_time_checks(org, project, land_check_commit)

            return pr.merge_into(
                repo,
                dry_run=dry_run,
                skip_mandatory_checks=skip_mandatory_checks,
                comment_id=comment_id,
                land_check_commit=land_check_commit
            )
        except MandatoryChecksMissingError as ex:
            last_exception = str(ex)
            print(f"Merge of https://github.com/{org}/{project}/pull/{pr_num} failed due to: {ex}. Retrying in 5 min")
            time.sleep(5 * 60)
        except RuntimeError:
            if land_checks and not dry_run:
                pr.delete_land_time_check_branch(repo)
            raise
    # Finally report timeout back
    msg = f"Merged timed out after {timeout_minutes} minutes. Please contact the pytorch_dev_infra team."
    msg += f"The last exception was: {last_exception}"
    if not dry_run:
        if land_checks:
            pr.delete_land_time_check_branch(repo)
        gh_add_labels(org, project, pr_num, ["land-failed"])
    raise RuntimeError(msg)

def main() -> None:
    args = parse_args()
    repo = GitRepo(get_git_repo_dir(), get_git_remote_name())
    org, project = repo.gh_owner_and_name()
    pr = GitHubPR(org, project, args.pr_num)

    def handle_exception(e: Exception, title: str = "Merge failed") -> None:
        exception = f"**Reason**: {e}"

        internal_debugging = ""
        run_url = os.getenv("GH_RUN_URL")
        if run_url is not None:
            # Hide this behind a collapsed bullet since it's not helpful to most devs
            internal_debugging = "\n".join((
                "<details><summary>Details for Dev Infra team</summary>",
                f"Raised by <a href=\"{run_url}\">workflow job</a>",
                "</details>"
            ))

        msg = "\n".join((
            f"## {title}",
            f"{exception}",
            "",
            f"{internal_debugging}"
        ))

        gh_post_pr_comment(org, project, args.pr_num, msg, dry_run=args.dry_run)
        import traceback
        traceback.print_exc()

    if args.revert:
        try:
            gh_post_pr_comment(org, project, args.pr_num, get_revert_message(org, project, pr.pr_num), args.dry_run)
            try_revert(repo, pr, dry_run=args.dry_run, comment_id=args.comment_id, reason=args.reason)
        except Exception as e:
            handle_exception(e, f"Reverting PR {args.pr_num} failed")
        return

    if pr.is_closed():
        gh_post_pr_comment(org, project, args.pr_num, f"Can't merge closed PR #{args.pr_num}", dry_run=args.dry_run)
        return

    if pr.is_cross_repo() and pr.is_ghstack_pr():
        gh_post_pr_comment(org, project, args.pr_num, "Cross-repo ghstack merges are not supported", dry_run=args.dry_run)
        return

    try:
        merge(args.pr_num, repo,
              dry_run=args.dry_run,
              skip_mandatory_checks=args.force,
              comment_id=args.comment_id,
              on_green=args.on_green,
              mandatory_only=args.on_mandatory,
              land_checks=args.land_checks)
    except Exception as e:
        handle_exception(e)


if __name__ == "__main__":
    main()
