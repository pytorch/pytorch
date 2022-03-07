#!/usr/bin/env python3

import json
import os
import re
from dataclasses import dataclass
from urllib.request import urlopen, Request
from urllib.error import HTTPError
from typing import cast, Any, Callable, Dict, List, Optional, Tuple, Union
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
        isPrivate
        defaultBranchRef {
          name
        }
      }
      mergeCommit {
        oid
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
      comments(last: 1) {
        nodes {
          bodyText
          author {
            login
          }
          authorAssociation
          editor {
            login
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
        raise RuntimeError(f"GraphQL query {query} failed: {rc['errors']}")
    return cast(Dict[str, Any], rc)


def gh_get_pr_info(org: str, proj: str, pr_no: int) -> Any:
    rc = gh_graphql(GH_GET_PR_INFO_QUERY, name=proj, owner=org, number=pr_no)
    return rc["data"]["repository"]["pullRequest"]


def parse_args() -> Any:
    from argparse import ArgumentParser
    parser = ArgumentParser("Merge PR into default branch")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--revert", action="store_true")
    parser.add_argument("pr_num", type=int)
    return parser.parse_args()


class GitHubPR:
    def __init__(self, org: str, project: str, pr_num: int) -> None:
        assert isinstance(pr_num, int)
        self.org = org
        self.project = project
        self.pr_num = pr_num
        self.info = gh_get_pr_info(org, project, pr_num)
        self.changed_files: Optional[List[str]] = None

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
        return int(self.info["commits"]["totalCount"])

    def get_pr_creator_login(self) -> str:
        return cast(str, self.info["author"]["login"])

    def get_committer_login(self, num: int = 0) -> str:
        user = self.info["commits"]["nodes"][num]["commit"]["author"]["user"]
        # If author is not github user, user node will be null
        if user is None:
            return ""
        return cast(str, user["login"])

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

    def get_merge_commit(self) -> Optional[str]:
        mc = self.info["mergeCommit"]
        return mc["oid"] if mc is not None else None

    def get_pr_url(self) -> str:
        return f"https://github.com/{self.org}/{self.project}/pull/{self.pr_num}"

    def get_comment_body(self, num: int = -1) -> str:
        return cast(str, self.info["comments"]["nodes"][num]["bodyText"])

    def get_comment_author_login(self, num: int = -1) -> str:
        return cast(str, self.info["comments"]["nodes"][num]["author"]["login"])

    def get_comment_editor_login(self, num: int = -1) -> Optional[str]:
        rc = self.info["comments"]["nodes"][num]["editor"]
        return rc["login"] if rc is not None else None

    def get_comment_author_association(self, num: int = -1) -> str:
        return cast(str, self.info["comments"]["nodes"][num]["authorAssociation"])

    def merge_ghstack_into(self, repo: GitRepo) -> None:
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
                find_matching_merge_rule(pr, repo)

            # Adding the url here makes it clickable within the Github UI
            approved_by_urls = ', '.join(prefix_with_github_url(login) for login in approved_by)
            repo.cherry_pick(rev)
            msg = re.sub(RE_GHSTACK_SOURCE_ID, "", msg)
            msg += f"\nApproved by: {approved_by_urls}\n"
            repo.amend_commit_message(msg)

    def merge_into(self, repo: GitRepo, dry_run: bool = False) -> None:
        # Raises exception if matching rule is not found
        find_matching_merge_rule(self, repo)
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
            self.merge_ghstack_into(repo)

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



def find_matching_merge_rule(pr: GitHubPR, repo: GitRepo) -> MergeRule:
    """Returns merge rule matching to this pr or raises an exception"""
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
        return rule
    raise RuntimeError(f"PR {pr.pr_num} does not match merge rules")


def try_revert(repo: GitRepo, pr: GitHubPR, dry_run: bool = False) -> None:
    def post_comment(msg: str) -> None:
        gh_post_comment(pr.org, pr.project, pr.pr_num, msg, dry_run=dry_run)
    if not pr.is_closed():
        return post_comment(f"Can't revert open PR #{pr.pr_num}")
    if not RE_REVERT_CMD.match(pr.get_comment_body()):
        raise RuntimeError(f"Comment {pr.get_comment_body()} does not seem to be a valid revert command")
    if pr.get_comment_editor_login() is not None:
        return post_comment("Don't want to revert based on edited command")
    author_association = pr.get_comment_author_association()
    author_login = pr.get_comment_author_login()
    # For some reason, one can not be a member of private repo, only CONTRIBUTOR
    expected_association = "CONTRIBUTOR" if pr.is_base_repo_private() else "MEMBER"
    if author_association != expected_association and author_association != "OWNER":
        return post_comment(f"Will not revert as @{author_login} is not a {expected_association}, but {author_association}")

    # Raises exception if matching rule is not found
    find_matching_merge_rule(pr, repo)
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
            try_revert(repo, pr, dry_run=args.dry_run)
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
        pr.merge_into(repo, dry_run=args.dry_run)
    except Exception as e:
        msg = f"Merge failed due to {e}"
        run_url = os.getenv("GH_RUN_URL")
        if run_url is not None:
            msg += f"\nRaised by {run_url}"
        gh_post_comment(org, project, args.pr_num, msg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
