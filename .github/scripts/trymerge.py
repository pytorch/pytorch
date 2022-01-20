#!/usr/bin/env python3

import json
import os
import re
from dataclasses import dataclass
from urllib.request import urlopen, Request
from urllib.error import HTTPError
from typing import cast, Any, Callable, Dict, List, Optional, Tuple
from gitutils import get_git_remote_name, get_git_repo_dir, patterns_to_regex, GitRepo


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
        defaultBranchRef {
          name
        }
      }
      commits(first: 100) {
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
            checkSuites(filterBy: {appId: 12274}, first: 1) {
              nodes {
                app {
                  databaseId
                }
                conclusion
              }
            }
          }
        }
        totalCount
      }
      changedFiles,
      files(last: 100) {
        nodes {
          path
        }
      }
      latestReviews(last: 100) {
        nodes {
          author {
            login
          },
          state
        },
        totalCount
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


def gh_graphql(query: str, **kwargs: Any) -> Dict[str, Any]:
    rc = _fetch_url("https://api.github.com/graphql", data={"query": query, "variables": kwargs}, reader=json.load)
    if "errors" in rc:
        raise RuntimeError(f"GraphQL query {query} failed: {rc['errors']}")
    return cast(Dict[str, Any], rc)


def gh_get_pr_info(org: str, proj: str, pr_no: int) -> Any:
    rc = gh_graphql(GH_GET_PR_INFO_QUERY, name=proj, owner=org, number=pr_no)
    return rc["data"]["repository"]["pullRequest"]


def parse_args() -> Any:
    from argparse import ArgumentParser
    parser = ArgumentParser("Merge PR into default branch")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("pr_num", type=int)
    return parser.parse_args()


class GitHubPR:
    def __init__(self, org: str, project: str, pr_num: int) -> None:
        assert isinstance(pr_num, int)
        self.org = org
        self.project = project
        self.pr_num = pr_num
        self.info = gh_get_pr_info(org, project, pr_num)

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

    def get_changed_files_count(self) -> int:
        return int(self.info["changedFiles"])

    def get_changed_files(self) -> List[str]:
        rc = [x["path"] for x in self.info["files"]["nodes"]]
        if len(rc) != self.get_changed_files_count():
            raise RuntimeError("Changed file count mismatch")
        return rc

    def _get_reviewers(self) -> List[Tuple[str, str]]:
        reviews_count = int(self.info["latestReviews"]["totalCount"])
        if len(self.info["latestReviews"]["nodes"]) != reviews_count:
            raise RuntimeError("Can't fetch all PR reviews")
        return [(x["author"]["login"], x["state"]) for x in self.info["latestReviews"]["nodes"]]

    def get_approved_by(self) -> List[str]:
        return [login for (login, state) in self._get_reviewers() if state == "APPROVED"]

    def get_commit_count(self) -> int:
        return int(self.info["commits"]["totalCount"])

    def get_pr_creator_login(self) -> str:
        return cast(str, self.info["author"]["login"])

    def get_committer_login(self, num: int = 0) -> str:
        return cast(str, self.info["commits"]["nodes"][num]["commit"]["author"]["user"]["login"])

    def get_committer_author(self, num: int = 0) -> str:
        node = self.info["commits"]["nodes"][num]["commit"]["author"]
        return f"{node['name']} <{node['email']}>"

    def get_check_suite_conclusions(self) -> Dict[int, str]:
        last_commit = self.info["commits"]["nodes"][-1]["commit"]
        rc = {}
        for node in last_commit["checkSuites"]["nodes"]:
            rc[int(node["app"]["databaseId"])] = node["conclusion"]
        return rc

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

    def get_pr_url(self) -> str:
        return f"https://github.com/{self.org}/{self.project}/pull/{self.pr_num}"

    def merge_ghstack_into(self, repo: GitRepo) -> None:
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
            if pr_num != self.pr_num:
                pr = GitHubPR(self.org, self.project, pr_num)
                if pr.is_closed():
                    print(f"Skipping {idx+1} of {len(rev_list)} PR (#{pr_num}) as its already been merged")
                    continue
                check_if_should_be_merged(pr, repo)
            repo.cherry_pick(rev)
            repo.amend_commit_message(re.sub(RE_GHSTACK_SOURCE_ID, "", msg))

    def merge_into(self, repo: GitRepo, dry_run: bool = False) -> None:
        check_if_should_be_merged(self, repo)
        if repo.current_branch() != self.default_branch():
            repo.checkout(self.default_branch())
        if not self.is_ghstack_pr():
            msg = self.get_title() + "\n" + self.get_body()
            msg += f"\nPull Request resolved: {self.get_pr_url()}\n"
            repo._run_git("merge", "--squash", f"{repo.remote}/{self.head_ref()}")
            repo._run_git("commit", f"--author=\"{self.get_author()}\"", "-m", msg)
        else:
            self.merge_ghstack_into(repo)

        if not dry_run:
            repo.push(self.default_branch(), dry_run)


@dataclass
class MergeRule:
    name: str
    patterns: List[str]
    approved_by: List[str]
    mandatory_app_id: Optional[int]


def read_merge_rules(repo: GitRepo) -> List[MergeRule]:
    from pathlib import Path
    rules_path = Path(repo.repo_dir) / ".github" / "merge_rules.json"
    if not rules_path.exists():
        print(f"{rules_path} does not exist, returning empty rules")
        return []
    with open(rules_path) as fp:
        rc = json.load(fp, object_hook=lambda x: MergeRule(**x))
    return cast(List[MergeRule], rc)


def check_if_should_be_merged(pr: GitHubPR, repo: GitRepo) -> None:
    changed_files = pr.get_changed_files()
    approved_by = set(pr.get_approved_by())
    rules = read_merge_rules(repo)
    for rule in rules:
        rule_name = rule.name
        rule_approvers_set = set(rule.approved_by)
        patterns_re = patterns_to_regex(rule.patterns)
        approvers_intersection = approved_by.intersection(rule_approvers_set)
        # If rule requires approvers but they aren't the ones that reviewed PR
        if len(approvers_intersection) == 0 and len(rule_approvers_set) > 0:
            print(f"Skipping rule {rule_name} due to no approvers overlap")
            continue
        if rule.mandatory_app_id is not None:
            cs_conslusions = pr.get_check_suite_conclusions()
            mandatory_app_id = rule.mandatory_app_id
            if mandatory_app_id not in cs_conslusions or cs_conslusions[mandatory_app_id] != "SUCCESS":
                print(f"Skipping rule {rule_name} as mandatory app {mandatory_app_id} is not in {cs_conslusions}")
                continue
        non_matching_files = []
        for fname in changed_files:
            if not patterns_re.match(fname):
                non_matching_files.append(fname)
        if len(non_matching_files) > 0:
            print(f"Skipping rule {rule_name} due to non-matching files: {non_matching_files}")
            continue
        print(f"Matched rule {rule_name} for {pr.pr_num}")
        return
    raise RuntimeError(f"PR {pr.pr_num} does not match merge rules")


def main() -> None:
    import sys
    args = parse_args()
    repo = GitRepo(get_git_repo_dir(), get_git_remote_name())
    org, project = repo.gh_owner_and_name()

    pr = GitHubPR(org, project, args.pr_num)
    if pr.is_closed():
        print(gh_post_comment(org, project, args.pr_num, f"Can't merge closed PR #{args.pr_num}", dry_run=args.dry_run))
        sys.exit(-1)

    if pr.is_cross_repo():
        print(gh_post_comment(org, project, args.pr_num, "Cross-repo merges are not supported at the moment", dry_run=args.dry_run))
        sys.exit(-1)

    try:
        pr.merge_into(repo, dry_run=args.dry_run)
    except Exception as e:
        gh_post_comment(org, project, args.pr_num, f"Merge failed due to {e}", dry_run=args.dry_run)


if __name__ == "__main__":
    main()
