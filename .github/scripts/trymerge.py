#!/usr/bin/env python3

import base64
import json
import os
import re
from dataclasses import dataclass
from urllib.request import urlopen, Request
from urllib.error import HTTPError
from typing import cast, Any, Callable, Dict, List, Optional, Tuple, Union
from gitutils import get_git_remote_name, get_git_repo_dir, patterns_to_regex, GitRepo
from functools import lru_cache
from warnings import warn


GH_GET_PR_INFO_QUERY = """
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
      commits_with_authors:commits(first: 100) {
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
        totalCount
      }
      commits(last: 1) {
        nodes {
          commit {
            checkSuites(first: 10) {
              nodes {
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
                  }
                  pageInfo {
                    endCursor
                    hasNextPage
                  }
                }
                conclusion
              }
              pageInfo {
                endCursor
                hasNextPage
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
        nodes {
          author {
            login
          }
          state
        }
        totalCount
      }
      comments(last: 5) {
        nodes {
          bodyText
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

GH_GET_PR_NEXT_CHECK_RUNS = """
query ($owner: String!, $name: String!, $number: Int!, $cursor: String!) {
  repository(name: $name, owner: $owner) {
    pullRequest(number: $number) {
      commits(last: 1) {
        nodes {
          commit {
            oid
            checkSuites(first: 10, after: $cursor) {
              nodes {
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
                  }
                  pageInfo {
                    endCursor
                    hasNextPage
                  }
                }
                conclusion
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

GH_GET_PR_PREV_COMMENTS = """
query ($owner: String!, $name: String!, $number: Int!, $cursor: String!) {
  repository(name: $name, owner: $owner) {
    pullRequest(number: $number) {
      comments(last: 100, before: $cursor) {
        nodes {
          bodyText
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

GH_GET_PR_NEXT_AUTHORS_QUERY = """
query ($owner: String!, $name: String!, $number: Int!, $cursor: String) {
  repository(name: $name, owner: $owner) {
    pullRequest(number: $number) {
      commits_with_authors: commits(first: 100, after: $cursor) {
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
    }
  }
}
"""

RE_GHSTACK_HEAD_REF = re.compile(r"^(gh/[^/]+/[0-9]+/)head$")
RE_GHSTACK_SOURCE_ID = re.compile(r'^ghstack-source-id: (.+)\n?', re.MULTILINE)
RE_PULL_REQUEST_RESOLVED = re.compile(
    r'Pull Request resolved: '
    r'https://github.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/pull/(?P<number>[0-9]+)',
    re.MULTILINE
)
RE_REVERT_CMD = re.compile(r"@pytorch(merge|)bot\s+revert\s+this")
RE_DIFF_REV = re.compile(r'^Differential Revision:.+?(D[0-9]+)', re.MULTILINE)


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
        url += '?' + '&'.join(f"{name}={val}" for name, val in params.items())
    return cast(List[Dict[str, Any]], _fetch_url(url, headers=headers, data=data, reader=json.load))


def gh_post_comment(org: str, project: str, pr_num: int, comment: str, dry_run: bool = False) -> List[Dict[str, Any]]:
    if dry_run:
        print(comment)
        return []
    return fetch_json(f'https://api.github.com/repos/{org}/{project}/issues/{pr_num}/comments',
                      data={"body": comment})


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


def parse_args() -> Any:
    from argparse import ArgumentParser
    parser = ArgumentParser("Merge PR into default branch")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--revert", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--comment-id", type=int)
    parser.add_argument("pr_num", type=int)
    return parser.parse_args()


@dataclass
class GitHubComment:
    body_text: str
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
        self.conclusions: Optional[Dict[str, str]] = None
        self.comments: Optional[List[GitHubComment]] = None
        self._authors: Optional[List[Tuple[str, str]]] = None

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

    def _get_reviewers(self) -> List[Tuple[str, str]]:
        reviews_count = int(self.info["reviews"]["totalCount"])
        nodes = self.info["reviews"]["nodes"]
        if len(nodes) != reviews_count:
            raise RuntimeError("Can't fetch all PR reviews")
        reviews = {}
        for node in nodes:
            author = node["author"]["login"]
            state = node["state"]
            if state != "COMMENTED":
                reviews[author] = state
        return list(reviews.items())

    def get_approved_by(self) -> List[str]:
        return [login for (login, state) in self._get_reviewers() if state == "APPROVED"]

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

    def get_checkrun_conclusions(self) -> Dict[str, str]:
        """ Returns list of checkrun / conclusions """
        if self.conclusions is not None:
            return self.conclusions
        orig_last_commit = self.info["commits"]["nodes"][-1]["commit"]
        checksuites = orig_last_commit["checkSuites"]
        conclusions = {}

        def add_conclusions(nodes: List[Dict[str, Any]]) -> None:
            for node in nodes:
                workflow_run = node["workflowRun"]
                checkruns = node["checkRuns"]
                if workflow_run is not None:
                    conclusions[workflow_run["workflow"]["name"]] = node["conclusion"]
                    continue
                if checkruns is not None:
                    for checkrun_node in checkruns["nodes"]:
                        conclusions[checkrun_node["name"]] = checkrun_node["conclusion"]

        add_conclusions(checksuites["nodes"])
        while bool(checksuites["pageInfo"]["hasNextPage"]):
            rc = gh_graphql(GH_GET_PR_NEXT_CHECK_RUNS,
                            name=self.project,
                            owner=self.org,
                            number=self.pr_num,
                            cursor=checksuites["pageInfo"]["endCursor"])
            info = rc["data"]["repository"]["pullRequest"]
            last_commit = info["commits"]["nodes"][-1]["commit"]
            if last_commit["oid"] != orig_last_commit["oid"]:
                raise RuntimeError("Last commit changed on PR")
            checksuites = last_commit["checkSuites"]
            add_conclusions(checksuites["nodes"])
        self.conclusions = conclusions
        return conclusions

    def get_authors(self) -> Dict[str, str]:
        rc = {}
        for idx in range(self.get_commit_count()):
            rc[self.get_committer_login(idx)] = self.get_committer_author(idx)

        return rc

    def get_author(self) -> str:
        authors = self.get_authors()
        if len(authors) == 1:
            return next(iter(authors.values()))
        return self.get_authors()[self.get_pr_creator_login()]

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
        return checks[checkrun_name] != "SUCCESS"

    def merge_ghstack_into(self, repo: GitRepo, force: bool) -> None:
        assert self.is_ghstack_pr()
        approved_by = self.get_approved_by()
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
            if pr_num != self.pr_num:
                pr = GitHubPR(self.org, self.project, pr_num)
                if pr.is_closed():
                    print(f"Skipping {idx+1} of {len(rev_list)} PR (#{pr_num}) as its already been merged")
                    continue
                approved_by = pr.get_approved_by()
                # Raises exception if matching rule is not found
                find_matching_merge_rule(pr, repo, force=force)

            # Adding the url here makes it clickable within the Github UI
            approved_by_urls = ', '.join(prefix_with_github_url(login) for login in approved_by)
            repo.cherry_pick(rev)
            msg = re.sub(RE_GHSTACK_SOURCE_ID, "", msg)
            msg += f"\nApproved by: {approved_by_urls}\n"
            repo.amend_commit_message(msg)

    def merge_into(self, repo: GitRepo, *, force: bool = False, dry_run: bool = False) -> None:
        # Raises exception if matching rule is not found
        find_matching_merge_rule(self, repo, force=force)
        if self.has_internal_changes():
            raise RuntimeError("This PR must be landed via phabricator")
        if repo.current_branch() != self.default_branch():
            repo.checkout(self.default_branch())
        if not self.is_ghstack_pr():
            # Adding the url here makes it clickable within the Github UI
            approved_by_urls = ', '.join(prefix_with_github_url(login) for login in self.get_approved_by())
            msg = self.get_title() + "\n\n" + self.get_body()
            msg += f"\nPull Request resolved: {self.get_pr_url()}\n"
            msg += f"Approved by: {approved_by_urls}\n"
            pr_branch_name = f"__pull-request-{self.pr_num}__init__"
            repo.fetch(f"pull/{self.pr_num}/head", pr_branch_name)
            repo._run_git("merge", "--squash", pr_branch_name)
            repo._run_git("commit", f"--author=\"{self.get_author()}\"", "-m", msg)
        else:
            self.merge_ghstack_into(repo, force)

        repo.push(self.default_branch(), dry_run)


@dataclass
class MergeRule:
    name: str
    patterns: List[str]
    approved_by: List[str]
    mandatory_checks_name: Optional[List[str]]


def read_merge_rules(repo: Optional[GitRepo], org: str, project: str) -> List[MergeRule]:
    from pathlib import Path

    repo_relative_rules_path = Path(".github") / "merge_rules.json"
    if repo is None:
        json_data = _fetch_url(
            f"https://api.github.com/repos/{org}/{project}/contents/{repo_relative_rules_path}",
            headers={'Accept': 'application/vnd.github.v3+json'},
            reader=json.load,
        )
        content = base64.b64decode(json_data["content"])
        return cast(List[MergeRule], json.loads(content, object_hook=lambda x: MergeRule(**x)))
    else:
        rules_path = Path(repo.repo_dir) / repo_relative_rules_path
        if not rules_path.exists():
            print(f"{rules_path} does not exist, returning empty rules")
            return []
        with open(rules_path) as fp:
            rc = json.load(fp, object_hook=lambda x: MergeRule(**x))
        return cast(List[MergeRule], rc)


def find_matching_merge_rule(pr: GitHubPR,
                             repo: Optional[GitRepo] = None,
                             force: bool = False,
                             skip_internal_checks: bool = False
                             ) -> MergeRule:
    """Returns merge rule matching to this pr or raises an exception"""
    changed_files = pr.get_changed_files()
    approved_by = set(pr.get_approved_by())
    rules = read_merge_rules(repo, pr.org, pr.project)
    reject_reason = f"PR {pr.pr_num} does not match merge rules"
    #  Used to determine best rejection reason
    # Score 0 to 10K - how many files rule matched
    # Score 10K - matched all files, but no overlapping approvers
    # Score 20K - matched all files and approvers, but lacks mandatory checks
    reject_reason_score = 0
    for rule in rules:
        rule_name = rule.name
        patterns_re = patterns_to_regex(rule.patterns)
        non_matching_files = []
        for fname in changed_files:
            if not patterns_re.match(fname):
                non_matching_files.append(fname)
        if len(non_matching_files) > 0:
            num_matching_files = len(changed_files) - len(non_matching_files)
            if num_matching_files > reject_reason_score:
                reject_reason_score = num_matching_files
                reject_reason = (f"{num_matching_files} files matched rule {rule_name}, but there are still non-matching files: " +
                                 f"{','.join(non_matching_files[:5])}{', ...' if len(non_matching_files) > 5 else ''}")
            continue
        # If rule needs approvers but PR has not been reviewed, skip it
        if len(rule.approved_by) > 0 and len(approved_by) == 0:
            if reject_reason_score < 10000:
                reject_reason_score = 10000
                reject_reason = f"Matched rule {rule_name}, but PR has not been reviewed yet"
            continue

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
                reject_reason = (f"Matched rule {rule_name}, but it was not reviewed yet by any of:" +
                                 f"{','.join(list(rule_approvers_set)[:5])}{', ...' if len(rule_approvers_set) > 5 else ''}")
            continue
        if rule.mandatory_checks_name is not None:
            pass_checks = True
            checks = pr.get_checkrun_conclusions()
            # HACK: We don't want to skip CLA check, even when forced
            for checkname in filter(lambda x: force is False or "CLA Check" in x, rule.mandatory_checks_name):
                if checkname not in checks or checks[checkname] != "SUCCESS":
                    if reject_reason_score < 20000:
                        reject_reason_score = 20000
                        reject_reason = f"Refusing to merge as mandatory check {checkname} "
                        reject_reason += "has not been run" if checkname not in checks else "failed"
                        reject_reason += f" for rule {rule_name}"
                    pass_checks = False
            if not pass_checks:
                continue
        if not skip_internal_checks and pr.has_internal_changes():
            raise RuntimeError("This PR has internal changes and must be landed via Phabricator")
        return rule
    raise RuntimeError(reject_reason)


def try_revert(repo: GitRepo, pr: GitHubPR, *, dry_run: bool = False, comment_id: Optional[int] = None) -> None:
    def post_comment(msg: str) -> None:
        gh_post_comment(pr.org, pr.project, pr.pr_num, msg, dry_run=dry_run)
    if not pr.is_closed():
        return post_comment(f"Can't revert open PR #{pr.pr_num}")
    comment = pr.get_last_comment() if comment_id is None else pr.get_comment_by_id(comment_id)
    if not RE_REVERT_CMD.match(comment.body_text):
        raise RuntimeError(f"Comment {comment.body_text} does not seem to be a valid revert command")
    if comment.editor_login is not None:
        return post_comment("Don't want to revert based on edited command")
    author_association = comment.author_association
    author_login = comment.author_login
    # For some reason, one can not be a member of private repo, only CONTRIBUTOR
    expected_association = "CONTRIBUTOR" if pr.is_base_repo_private() else "MEMBER"
    if author_association != expected_association and author_association != "OWNER":
        return post_comment(f"Will not revert as @{author_login} is not a {expected_association}, but {author_association}")

    # Raises exception if matching rule is not found, but ignores all status checks
    find_matching_merge_rule(pr, repo, force=True)
    commit_sha = pr.get_merge_commit()
    if commit_sha is None:
        commits = repo.commits_resolving_gh_pr(pr.pr_num)
        if len(commits) == 0:
            raise RuntimeError("Can't find any commits resolving PR")
        commit_sha = commits[0]
    msg = repo.commit_message(commit_sha)
    rc = RE_DIFF_REV.search(msg)
    if rc is not None:
        raise RuntimeError(f"Can't revert PR that was landed via phabricator as {rc.group(1)}")
    repo.checkout(pr.default_branch())
    repo.revert(commit_sha)
    msg = repo.commit_message("HEAD")
    msg = re.sub(RE_PULL_REQUEST_RESOLVED, "", msg)
    msg += f"\nReverted {pr.get_pr_url()} on behalf of {prefix_with_github_url(author_login)}\n"
    repo.amend_commit_message(msg)
    repo.push(pr.default_branch(), dry_run)
    if not dry_run:
        gh_add_labels(pr.org, pr.project, pr.pr_num, ["reverted"])


def prefix_with_github_url(suffix_str: str) -> str:
    return f"https://github.com/{suffix_str}"


def main() -> None:
    args = parse_args()
    repo = GitRepo(get_git_repo_dir(), get_git_remote_name())
    org, project = repo.gh_owner_and_name()

    pr = GitHubPR(org, project, args.pr_num)
    if args.revert:
        try:
            try_revert(repo, pr, dry_run=args.dry_run, comment_id=args.comment_id)
        except Exception as e:
            msg = f"Reverting PR {args.pr_num} failed due to {e}"
            run_url = os.getenv("GH_RUN_URL")
            if run_url is not None:
                msg += f"\nRaised by {run_url}"
            gh_post_comment(org, project, args.pr_num, msg, dry_run=args.dry_run)
        return

    if pr.is_closed():
        gh_post_comment(org, project, args.pr_num, f"Can't merge closed PR #{args.pr_num}", dry_run=args.dry_run)
        return

    if pr.is_cross_repo() and pr.is_ghstack_pr():
        gh_post_comment(org, project, args.pr_num, "Cross-repo ghstack merges are not supported", dry_run=args.dry_run)
        return

    try:
        pr.merge_into(repo, dry_run=args.dry_run, force=args.force)
    except Exception as e:
        msg = f"Merge failed due to {e}"
        run_url = os.getenv("GH_RUN_URL")
        if run_url is not None:
            msg += f"\nRaised by {run_url}"
        gh_post_comment(org, project, args.pr_num, msg, dry_run=args.dry_run)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
