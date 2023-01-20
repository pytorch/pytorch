#!/usr/bin/env python3

import base64
import json
import os
import re
import tempfile
import github_constants as gh_constants
import yaml
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import cast, Any, Callable, Dict, Iterator, Iterable, List, NamedTuple, Optional, Tuple, Union
from urllib.request import urlopen, Request
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from urllib.parse import quote
from warnings import warn


RE_GITHUB_URL_MATCH = re.compile("^https://.*@?github.com/(.+)/(.+)$")
RE_GHSTACK_HEAD_REF = re.compile(r"^(gh/[^/]+/[0-9]+/)head$")
RE_GHSTACK_DESC = re.compile(r'Stack.*:\r?\n(\* [^\r\n]+\r?\n)+', re.MULTILINE)
RE_PULL_REQUEST_RESOLVED = re.compile(
    r'Pull Request resolved: '
    r'https://github.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/pull/(?P<number>[0-9]+)',
    re.MULTILINE
)
RE_PR_CC_LINE = re.compile(r'^cc:? @\w+.*\r?\n?$', re.MULTILINE)
RE_DIFF_REV = re.compile(r'^Differential Revision:.+?(D[0-9]+)', re.MULTILINE)

MERGE_RULE_PATH = Path(".github") / "merge_rules.yaml"


def get_git_remote_name() -> str:
    return os.getenv("GIT_REMOTE_NAME", "origin")


def get_git_repo_dir() -> str:
    from pathlib import Path
    return os.getenv("GIT_REPO_DIR", str(Path(__file__).resolve().parent.parent.parent))


def fuzzy_list_to_dict(items: List[Tuple[str, str]]) -> Dict[str, List[str]]:
    """
    Converts list to dict preserving elements with duplicate keys
    """
    rc: Dict[str, List[str]] = defaultdict(lambda: [])
    for (key, val) in items:
        rc[key].append(val)
    return dict(rc)


def _check_output(items: List[str], encoding: str = "utf-8") -> str:
    from subprocess import check_output, CalledProcessError, STDOUT
    try:
        return check_output(items, stderr=STDOUT).decode(encoding)
    except CalledProcessError as e:
        msg = f"Command `{' '.join(e.cmd)}` returned non-zero exit code {e.returncode}"
        stdout = e.stdout.decode(encoding) if e.stdout is not None else ""
        stderr = e.stderr.decode(encoding) if e.stderr is not None else ""
        if len(stderr) == 0:
            msg += f"\n```\n{stdout}```"
        else:
            msg += f"\nstdout:\n```\n{stdout}```\nstderr:\n```\n{stderr}```"
        raise RuntimeError(msg) from e


class GitCommit:
    commit_hash: str
    title: str
    body: str
    author: str
    author_date: datetime
    commit_date: Optional[datetime]

    def __init__(self,
                 commit_hash: str,
                 author: str,
                 author_date: datetime,
                 title: str,
                 body: str,
                 commit_date: Optional[datetime] = None) -> None:
        self.commit_hash = commit_hash
        self.author = author
        self.author_date = author_date
        self.commit_date = commit_date
        self.title = title
        self.body = body

    def __repr__(self) -> str:
        return f"{self.title} ({self.commit_hash})"

    def __contains__(self, item: Any) -> bool:
        return item in self.body or item in self.title


def parse_fuller_format(lines: Union[str, List[str]]) -> GitCommit:
    """
    Expect commit message generated using `--format=fuller --date=unix` format, i.e.:
        commit <sha1>
        Author:     <author>
        AuthorDate: <author date>
        Commit:     <committer>
        CommitDate: <committer date>

        <title line>

        <full commit message>

    """
    if isinstance(lines, str):
        lines = lines.split("\n")
    # TODO: Handle merge commits correctly
    if len(lines) > 1 and lines[1].startswith("Merge:"):
        del lines[1]
    assert len(lines) > 7
    assert lines[0].startswith("commit")
    assert lines[1].startswith("Author: ")
    assert lines[2].startswith("AuthorDate: ")
    assert lines[3].startswith("Commit: ")
    assert lines[4].startswith("CommitDate: ")
    assert len(lines[5]) == 0
    return GitCommit(commit_hash=lines[0].split()[1].strip(),
                     author=lines[1].split(":", 1)[1].strip(),
                     author_date=datetime.fromtimestamp(int(lines[2].split(":", 1)[1].strip())),
                     commit_date=datetime.fromtimestamp(int(lines[4].split(":", 1)[1].strip())),
                     title=lines[6].strip(),
                     body="\n".join(lines[7:]),
                     )


class GitRepo:
    def __init__(self, path: str, remote: str = "origin", debug: bool = False) -> None:
        self.repo_dir = path
        self.remote = remote
        self.debug = debug

    def _run_git(self, *args: Any) -> str:
        if self.debug:
            print(f"+ git -C {self.repo_dir} {' '.join(args)}")
        return _check_output(["git", "-C", self.repo_dir] + list(args))

    def revlist(self, revision_range: str) -> List[str]:
        rc = self._run_git("rev-list", revision_range, "--", ".").strip()
        return rc.split("\n") if len(rc) > 0 else []

    def current_branch(self) -> str:
        return self._run_git("symbolic-ref", "--short", "HEAD").strip()

    def checkout(self, branch: str) -> None:
        self._run_git("checkout", branch)

    def fetch(self, ref: Optional[str] = None, branch: Optional[str] = None) -> None:
        if branch is None and ref is None:
            self._run_git("fetch", self.remote)
        elif branch is None:
            self._run_git("fetch", self.remote, ref)
        else:
            self._run_git("fetch", self.remote, f"{ref}:{branch}")

    def show_ref(self, name: str) -> str:
        refs = self._run_git('show-ref', '-s', name).strip().split('\n')
        if not all(refs[i] == refs[0] for i in range(1, len(refs))):
            raise RuntimeError(f"referce {name} is ambigous")
        return refs[0]

    def rev_parse(self, name: str) -> str:
        return self._run_git('rev-parse', '--verify', name).strip()

    def get_merge_base(self, from_ref: str, to_ref: str) -> str:
        return self._run_git('merge-base', from_ref, to_ref).strip()

    def patch_id(self, ref: Union[str, List[str]]) -> List[Tuple[str, str]]:
        is_list = isinstance(ref, list)
        if is_list:
            if len(ref) == 0:
                return []
            ref = " ".join(ref)
        rc = _check_output(['sh', '-c', f'git -C {self.repo_dir} show {ref}|git patch-id --stable']).strip()
        return [cast(Tuple[str, str], x.split(" ", 1)) for x in rc.split("\n")]

    def commits_resolving_gh_pr(self, pr_num: int) -> List[str]:
        owner, name = self.gh_owner_and_name()
        msg = f"Pull Request resolved: https://github.com/{owner}/{name}/pull/{pr_num}"
        rc = self._run_git('log', '--format=%H', '--grep', msg).strip()
        return rc.split("\n") if len(rc) > 0 else []

    def get_commit(self, ref: str) -> GitCommit:
        return parse_fuller_format(self._run_git('show', '--format=fuller', '--date=unix', '--shortstat', ref))

    def cherry_pick(self, ref: str) -> None:
        self._run_git('cherry-pick', '-x', ref)

    def revert(self, ref: str) -> None:
        self._run_git("revert", "--no-edit", ref)

    def compute_branch_diffs(self, from_branch: str, to_branch: str) -> Tuple[List[str], List[str]]:
        """
        Returns list of commmits that are missing in each other branch since their merge base
        Might be slow if merge base is between two branches is pretty far off
        """
        from_ref = self.rev_parse(from_branch)
        to_ref = self.rev_parse(to_branch)
        merge_base = self.get_merge_base(from_ref, to_ref)
        from_commits = self.revlist(f'{merge_base}..{from_ref}')
        to_commits = self.revlist(f'{merge_base}..{to_ref}')
        from_ids = fuzzy_list_to_dict(self.patch_id(from_commits))
        to_ids = fuzzy_list_to_dict(self.patch_id(to_commits))
        for patch_id in set(from_ids).intersection(set(to_ids)):
            from_values = from_ids[patch_id]
            to_values = to_ids[patch_id]
            if len(from_values) != len(to_values):
                # Eliminate duplicate commits+reverts from the list
                while len(from_values) > 0 and len(to_values) > 0:
                    frc = self.get_commit(from_values.pop())
                    toc = self.get_commit(to_values.pop())
                    # FRC branch might have PR number added to the title
                    if frc.title != toc.title or frc.author_date != toc.author_date:
                        # HACK: Same commit were merged, reverted and landed again
                        # which creates a tracking problem
                        if (
                            "pytorch/pytorch" not in self.remote_url() or
                            frc.commit_hash not in {"0a6a1b27a464ba5be5f587cce2ee12ab8c504dbf",
                                                    "6d0f4a1d545a8f161df459e8d4ccafd4b9017dbe",
                                                    "edf909e58f06150f7be41da2f98a3b9de3167bca",
                                                    "a58c6aea5a0c9f8759a4154e46f544c8b03b8db1",
                                                    "7106d216c29ca16a3504aa2bedad948ebcf4abc2"}
                        ):
                            raise RuntimeError(f"Unexpected differences between {frc} and {toc}")
                    from_commits.remove(frc.commit_hash)
                    to_commits.remove(toc.commit_hash)
                continue
            for commit in from_values:
                from_commits.remove(commit)
            for commit in to_values:
                to_commits.remove(commit)
        # Another HACK: Patch-id is not stable for commits with binary files or for big changes across commits
        # I.e. cherry-picking those from one branch into another will change patchid
        if "pytorch/pytorch" in self.remote_url():
            for excluded_commit in {"8e09e20c1dafcdbdb45c2d1574da68a32e54a3a5",
                                    "5f37e5c2a39c3acb776756a17730b865f0953432",
                                    "b5222584e6d6990c6585981a936defd1af14c0ba",
                                    "84d9a2e42d5ed30ec3b8b4140c38dd83abbce88d",
                                    "f211ec90a6cdc8a2a5795478b5b5c8d7d7896f7e"}:
                if excluded_commit in from_commits:
                    from_commits.remove(excluded_commit)

        return (from_commits, to_commits)

    def cherry_pick_commits(self, from_branch: str, to_branch: str) -> None:
        orig_branch = self.current_branch()
        self.checkout(to_branch)
        from_commits, to_commits = self.compute_branch_diffs(from_branch, to_branch)
        if len(from_commits) == 0:
            print("Nothing to do")
            self.checkout(orig_branch)
            return
        for commit in reversed(from_commits):
            print(f"Cherry picking commit {commit}")
            self.cherry_pick(commit)
        self.checkout(orig_branch)

    def push(self, branch: str, dry_run: bool, retry: int = 3) -> None:
        for cnt in range(retry):
            try:
                if dry_run:
                    self._run_git("push", "--dry-run", self.remote, branch)
                else:
                    self._run_git("push", self.remote, branch)
            except RuntimeError as e:
                print(f"{cnt} push attempt failed with {e}")
                self.fetch()
                self._run_git("rebase", f"{self.remote}/{branch}")

    def head_hash(self) -> str:
        return self._run_git("show-ref", "--hash", "HEAD").strip()

    def remote_url(self) -> str:
        return self._run_git("remote", "get-url", self.remote)

    def gh_owner_and_name(self) -> Tuple[str, str]:
        url = os.getenv("GIT_REMOTE_URL", None)
        if url is None:
            url = self.remote_url()
        rc = RE_GITHUB_URL_MATCH.match(url)
        if rc is None:
            raise RuntimeError(f"Unexpected url format {url}")
        return cast(Tuple[str, str], rc.groups())

    def commit_message(self, ref: str) -> str:
        return self._run_git("log", "-1", "--format=%B", ref)

    def amend_commit_message(self, msg: str) -> None:
        self._run_git("commit", "--amend", "-m", msg)


class JobCheckState(NamedTuple):
    name: str
    url: str
    status: Optional[str]

JobNameToStateDict = Dict[str, JobCheckState]


class WorkflowCheckState:
    def __init__(self, name: str, url: str, status: Optional[str]):
        self.name: str = name
        self.url: str = url
        self.status: Optional[str] = status
        self.jobs: JobNameToStateDict = {}


def fetch_url(url: str, *,
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
        url += '?' + '&'.join(f"{name}={quote(str(val))}" for name, val in params.items())
    return cast(List[Dict[str, Any]], fetch_url(url, headers=headers, data=data, reader=json.load))


def fetch_json_dict(url: str,
                    params: Optional[Dict[str, Any]] = None,
                    data: Optional[Dict[str, Any]] = None) -> Dict[str, Any] :
    headers = {'Accept': 'application/vnd.github.v3+json'}
    if params is not None and len(params) > 0:
        url += '?' + '&'.join(f"{name}={quote(str(val))}" for name, val in params.items())
    return cast(Dict[str, Any], fetch_url(url, headers=headers, data=data, reader=json.load))


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
    rc = fetch_url("https://api.github.com/graphql", data={"query": query, "variables": kwargs}, reader=json.load)
    if "errors" in rc:
        raise RuntimeError(f"GraphQL query {query}, args {kwargs} failed: {rc['errors']}")
    return cast(Dict[str, Any], rc)


def gh_get_pr_info(org: str, proj: str, pr_no: int) -> Any:
    rc = gh_graphql(gh_constants.GH_GET_PR_INFO_QUERY, name=proj, owner=org, number=pr_no)
    return rc["data"]["repository"]["pullRequest"]


def gh_get_land_check_info(org: str, proj: str, commit: str) -> Any:
    rc = gh_graphql(gh_constants.GH_GET_COMMIT_CHECKSUITES, name=proj, owner=org, commit=commit)
    return rc["data"]["repository"]["object"]


@lru_cache(maxsize=None)
def gh_get_team_members(org: str, name: str) -> List[str]:
    rc: List[str] = []
    team_members: Dict[str, Any] = {"pageInfo": {"hasNextPage": "true", "endCursor": None}}
    while bool(team_members["pageInfo"]["hasNextPage"]):
        query = gh_graphql(gh_constants.GH_GET_TEAM_MEMBERS_QUERY, org=org, name=name, cursor=team_members["pageInfo"]["endCursor"])
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


def prefix_with_github_url(suffix_str: str) -> str:
    return f"https://github.com/{suffix_str}"


def add_workflow_conclusions(
    checksuites: Any,
    get_next_checkruns_page: Callable[[List[Dict[str, Dict[str, Any]]], int, Any], Any],
    get_next_checksuites: Callable[[Any], Any]
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
                    if existing_checkrun is None or not is_passing_status(existing_checkrun.status):
                        workflow_obj.jobs[checkrun_name] = JobCheckState(
                            name=checkrun_name,
                            status=checkrun_node["conclusion"],
                            url=checkrun_node["detailsUrl"],
                        )

                if bool(checkruns["pageInfo"]["hasNextPage"]):
                    checkruns = get_next_checkruns_page(edges, edge_idx, checkruns)
                else:
                    checkruns = None

    add_conclusions(checksuites["edges"])
    while bool(checksuites["pageInfo"]["hasNextPage"]):
        checksuites = get_next_checksuites(checksuites)
        add_conclusions(checksuites["edges"])

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
                workflow.status
            )
    for job_name, job in no_workflow_obj.jobs.items():
        res[job_name] = job
    return res


def can_skip_internal_checks(pr: "GitHubPR", comment_id: Optional[int] = None) -> bool:
    if comment_id is None:
        return False
    comment = pr.get_comment_by_id(comment_id)
    if comment.editor_login is not None:
        return False
    return comment.author_login == "facebook-github-bot"


def get_ghstack_prs(repo: GitRepo, pr: "GitHubPR") -> List[Tuple["GitHubPR", str]]:
    '''
    Get the open PRs in the stack that are below this PR.  Throws error if any of the PRs are out of sync.
    '''
    assert pr.is_ghstack_pr()
    entire_stack: List[Tuple["GitHubPR", str]] = []
    # For ghstack, cherry-pick commits based from origin
    orig_ref = f"{repo.remote}/{re.sub(r'/head$', '/orig', pr.head_ref())}"
    rev_list = repo.revlist(f"{pr.default_branch()}..{orig_ref}")
    for idx, rev in enumerate(reversed(rev_list)):
        msg = repo.commit_message(rev)
        m = gh_constants.RE_PULL_REQUEST_RESOLVED.search(msg)
        if m is None:
            raise RuntimeError(f"Could not find PR-resolved string in {msg} of ghstacked PR {pr.pr_num}")
        if pr.org != m.group('owner') or pr.project != m.group('repo'):
            raise RuntimeError(f"PR {m.group('number')} resolved to wrong owner/repo pair")
        stacked_pr_num = int(m.group('number'))
        if stacked_pr_num != pr.pr_num:
            stacked_pr = GitHubPR(pr.org, pr.project, stacked_pr_num)
            if stacked_pr.is_closed():
                print(f"Skipping {idx+1} of {len(rev_list)} PR (#{stacked_pr_num}) as its already been merged")
                continue
            entire_stack.append((stacked_pr, rev))
        else:
            entire_stack.append((pr, rev))

    for stacked_pr, rev in entire_stack:
        commit_sha = stacked_pr.last_commit()['oid']
        tree_sha = repo._run_git("rev-parse", commit_sha + "^{tree}")
        if tree_sha not in repo.commit_message(rev):
            raise RuntimeError(
                f"PR {stacked_pr.pr_num} is out of sync with the corresponding revision {rev} on " +
                f"branch {orig_ref} that would be merged into master.  " +
                "This usually happens because there is a non ghstack change in the PR.  " +
                f"Please sync them and try again (ex. make the changes on {orig_ref} and run ghstack)."
            )
    return entire_stack


class MandatoryChecksMissingError(Exception):
    def __init__(self, message: str, rule: Optional['MergeRule'] = None) -> None:
        super().__init__(message)
        self.rule = rule
        
        
@dataclass
class MergeRule:
    name: str
    patterns: List[str]
    approved_by: List[str]
    mandatory_checks_name: Optional[List[str]]
        

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
        self.conclusions: Optional[JobNameToStateDict] = None
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
        return gh_constants.RE_GHSTACK_HEAD_REF.match(self.head_ref()) is not None

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
                rc = gh_graphql(gh_constants.GH_GET_PR_NEXT_FILES_QUERY,
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
                rc = gh_graphql(gh_constants.GH_GET_PR_PREV_REVIEWS_QUERY,
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
            rc = gh_graphql(gh_constants.GH_GET_PR_NEXT_AUTHORS_QUERY,
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

    def get_checkrun_conclusions(self) -> JobNameToStateDict:
        """ Returns dict of checkrun -> [conclusion, url] """
        if self.conclusions is not None:
            return self.conclusions
        orig_last_commit = self.info["commits"]["nodes"][-1]["commit"]

        def get_pr_next_check_runs(edges: List[Dict[str, Dict[str, Any]]], edge_idx: int, checkruns: Any) -> Any:
            rc = gh_graphql(gh_constants.GH_GET_PR_NEXT_CHECK_RUNS,
                            name=self.project,
                            owner=self.org,
                            number=self.pr_num,
                            cs_cursor=edges[edge_idx - 1]["cursor"] if edge_idx > 0 else None,
                            cr_cursor=checkruns["pageInfo"]["endCursor"])
            last_commit = rc["data"]["repository"]["pullRequest"]["commits"]["nodes"][-1]["commit"]
            checkruns = last_commit["checkSuites"]["nodes"][-1]["checkRuns"]
            return checkruns

        def get_pr_next_checksuites(checksuites: Any) -> Any:
            rc = gh_graphql(gh_constants.GH_GET_PR_NEXT_CHECKSUITES,
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
                self.conclusions[name] = JobCheckState(name=name, status=status["state"], url=status["targetUrl"])

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
            rc = gh_graphql(gh_constants.GH_GET_PR_PREV_COMMENTS,
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
        rc = gh_constants.RE_DIFF_REV.search(self.get_body())
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
                    land_check_commit=land_check_commit)
            repo.cherry_pick(rev)
            repo.amend_commit_message(commit_msg)
        return [x for x, _ in ghstack_prs]

    def gen_commit_message(self, filter_ghstack: bool = False) -> str:
        """ Fetches title and body from PR description
            adds reviewed by, pull request resolved and optionally
            filters out ghstack info """
        # Adding the url here makes it clickable within the Github UI
        approved_by_urls = ', '.join(prefix_with_github_url(login) for login in self.get_approved_by())
        # Remove "cc: " line from the message body
        msg_body = re.sub(gh_constants.RE_PR_CC_LINE, "", self.get_body())
        if filter_ghstack:
            msg_body = re.sub(gh_constants.RE_GHSTACK_DESC, "", msg_body)
        msg = self.get_title() + f" (#{self.pr_num})\n\n"
        msg += msg_body
        msg += f"\nPull Request resolved: {self.get_pr_url()}\n"
        msg += f"Approved by: {approved_by_urls}\n"
        return msg

    def add_numbered_label(self, label_base: str) -> None:
        labels = self.get_labels()
        label = label_base
        for i in range(len(labels) if labels is not None else 0):
            if label in labels:
                label = f"{label_base}X{i+2}"
        gh_add_labels(self.org, self.project, self.pr_num, [label])

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
        additional_merged_prs = self.merge_changes(repo, skip_mandatory_checks, comment_id, land_check_commit=land_check_commit)

        repo.push(self.default_branch(), dry_run)
        if not dry_run:
            if land_check_commit:
                self.delete_land_time_check_branch(repo)
            self.add_numbered_label("merged")
            for pr in additional_merged_prs:
                pr.add_numbered_label("merged")

    def merge_changes(self,
                      repo: GitRepo,
                      skip_mandatory_checks: bool = False,
                      comment_id: Optional[int] = None,
                      land_check_commit: Optional[str] = None,
                      branch: Optional[str] = None) -> List["GitHubPR"]:
        branch_to_merge_into = self.default_branch() if branch is None else branch
        if repo.current_branch() != branch_to_merge_into:
            repo.checkout(branch_to_merge_into)
        if not self.is_ghstack_pr():
            msg = self.gen_commit_message()
            pr_branch_name = f"__pull-request-{self.pr_num}__init__"
            repo.fetch(f"pull/{self.pr_num}/head", pr_branch_name)
            repo._run_git("merge", "--squash", pr_branch_name)
            repo._run_git("commit", f"--author=\"{self.get_author()}\"", "-m", msg)
            return []
        else:
            return self.merge_ghstack_into(
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


def read_merge_rules(repo: Optional[GitRepo], org: str, project: str) -> List[MergeRule]:
    repo_relative_rules_path = MERGE_RULE_PATH
    if repo is None:
        json_data = fetch_url(
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
    
    
def categorize_checks(check_runs: Dict[str, JobCheckState],
                      required_checks: Iterable[str]) -> Tuple[List[Tuple[str, Optional[str]]], List[Tuple[str, Optional[str]]]]:
    pending_checks: List[Tuple[str, Optional[str]]] = []
    failed_checks: List[Tuple[str, Optional[str]]] = []

    relevant_checknames = [name for name in check_runs.keys() if any([x in name for x in required_checks])]

    for checkname in required_checks:
        if all([checkname not in x for x in check_runs.keys()]):
            pending_checks.append((checkname, None))
    for checkname in relevant_checknames:
        if check_runs[checkname].status is None:
            pending_checks.append((checkname, check_runs[checkname].url))
        elif not is_passing_status(check_runs[checkname].status):
            failed_checks.append((checkname, check_runs[checkname].url))
    return (pending_checks, failed_checks)


def checks_to_markdown_bullets(checks: List[Tuple[str, Optional[str]]]) -> List[str]:
    return [f"- [{c[0]}]({c[1]})" if c[1] is not None else f"- {c[0]}" for c in checks[:5]]


def gen_new_issue_link(
    org: str,
    project: str,
    labels: List[str],
    template: str = "bug-report.yml"
) -> str:
    labels_str = ",". join(labels)
    return (f"https://github.com/{org}/{project}/issues/new?"
            f"labels={quote(labels_str)}&"
            f"template={quote(template)}")


def get_land_checkrun_conclusions(org: str, project: str, commit: str) -> JobNameToStateDict:

    def get_commit_next_check_runs(edges: List[Dict[str, Dict[str, Any]]], edge_idx: int, checkruns: Any) -> Any:
        rc = gh_graphql(gh_constants.GH_GET_COMMIT_NEXT_CHECK_RUNS,
                        name=project,
                        owner=org,
                        cs_cursor=edges[edge_idx - 1]["cursor"] if edge_idx > 0 else None,
                        cr_cursor=checkruns["pageInfo"]["endCursor"],
                        commit=commit)
        return rc["data"]["repository"]["object"]["checkSuites"]["nodes"][-1]["checkRuns"]

    def get_commit_next_checksuites(checksuites: Any) -> Any:
        rc = gh_graphql(gh_constants.GH_GET_COMMIT_NEXT_CHECKSUITES,
                        name=project,
                        owner=org,
                        commit=commit,
                        cursor=checksuites["edges"][-1]["cursor"])
        info = rc["data"]["repository"]["object"]
        return info["checkSuites"]

    land_check_info = gh_get_land_check_info(org, project, commit)
    checksuites = land_check_info["checkSuites"]

    return add_workflow_conclusions(checksuites, get_commit_next_check_runs, get_commit_next_checksuites)


def get_combined_checks_from_pr_and_land_validation(
    pr: GitHubPR,
    land_check_commit: Optional[str],
) -> JobNameToStateDict:
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


def find_matching_merge_rule(
    pr: GitHubPR,
    repo: Optional[GitRepo] = None,
    skip_mandatory_checks: bool = False,
    skip_internal_checks: bool = False,
    land_check_commit: Optional[str] = None,
) -> MergeRule:
    """Returns merge rule matching to this pr or raises an exception"""
    changed_files = pr.get_changed_files()
    approved_by = set(pr.get_approved_by())
    checks = get_combined_checks_from_pr_and_land_validation(pr, land_check_commit)

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
        required_checks = list(filter(lambda x: "EasyCLA" in x or not skip_mandatory_checks, mandatory_checks))
        [pending_checks, failed_checks] = categorize_checks(checks, required_checks)

        hud_link = f"https://hud.pytorch.org/{pr.org}/{pr.project}/commit/{pr.last_commit()['oid']}"
        if len(failed_checks) > 0:
            if reject_reason_score < 30000:
                reject_reason_score = 30000
                reject_reason = "\n".join((
                    f"{len(failed_checks)} mandatory check(s) failed (Rule `{rule_name}`).  The first few are:",
                    *checks_to_markdown_bullets(failed_checks),
                    "",
                    f"Dig deeper by [viewing the failures on hud]({hud_link})"
                ))
            continue
        elif len(pending_checks) > 0:
            if reject_reason_score < 20000:
                reject_reason_score = 20000
                reject_reason = "\n".join((
                    f"{len(pending_checks)} mandatory check(s) are pending/not yet run (Rule `{rule_name}`).  The first few are:",
                    *checks_to_markdown_bullets(pending_checks),
                    "",
                    f"Dig deeper by [viewing the pending checks on hud]({hud_link})"
                ))
            continue

        if not skip_internal_checks and pr.has_internal_changes():
            raise RuntimeError("This PR has internal changes and must be landed via Phabricator")

        return rule

    if reject_reason_score == 20000:
        raise MandatoryChecksMissingError(reject_reason, rule)
    raise RuntimeError(reject_reason)


def clone_repo(username: str, password: str, org: str, project: str) -> GitRepo:
    path = tempfile.mkdtemp()
    _check_output(['git', 'clone', f'https://{username}:{password}@github.com/{org}/{project}', path]).strip()
    return GitRepo(path=path)


class PeekableIterator(Iterator[str]):
    def __init__(self, val: str) -> None:
        self._val = val
        self._idx = -1

    def peek(self) -> Optional[str]:
        if self._idx + 1 >= len(self._val):
            return None
        return self._val[self._idx + 1]

    def __iter__(self) -> "PeekableIterator":
        return self

    def __next__(self) -> str:
        rc = self.peek()
        if rc is None:
            raise StopIteration
        self._idx += 1
        return rc


def patterns_to_regex(allowed_patterns: List[str]) -> Any:
    """
    pattern is glob-like, i.e. the only special sequences it has are:
      - ? - matches single character
      - * - matches any non-folder separator characters or no character
      - ** - matches any characters or no character
      Assuming that patterns are free of braces and backslashes
      the only character that needs to be escaped are dot and plus
    """
    rc = "("
    for idx, pattern in enumerate(allowed_patterns):
        if idx > 0:
            rc += "|"
        pattern_ = PeekableIterator(pattern)
        assert not any(c in pattern for c in "{}()[]\\")
        for c in pattern_:
            if c == ".":
                rc += "\\."
            elif c == "+":
                rc += "\\+"
            elif c == "*":
                if pattern_.peek() == "*":
                    next(pattern_)
                    rc += ".*"
                else:
                    rc += "[^/]*"
            else:
                rc += c
    rc += ")"
    return re.compile(rc)

# Modified from https://github.com/pytorch/pytorch/blob/b00206d4737d1f1e7a442c9f8a1cadccd272a386/torch/hub.py#L129
def _read_url(url: Any) -> Any:
    with urlopen(url) as r:
        return r.headers, r.read().decode(r.headers.get_content_charset('utf-8'))


def request_for_labels(url: str) -> Any:
    headers = {'Accept': 'application/vnd.github.v3+json'}
    return _read_url(Request(url, headers=headers))


def get_last_page(header: Any) -> int:
    # Link info looks like: <https://api.github.com/repositories/65600975/labels?per_page=100&page=2>;
    # rel="next", <https://api.github.com/repositories/65600975/labels?per_page=100&page=3>; rel="last"
    link_info = header['link']
    prefix = "&page="
    suffix = ">;"
    return int(link_info[link_info.rindex(prefix) + len(prefix):link_info.rindex(suffix)])


def update_labels(labels: List[str], info: str) -> None:
    labels_json = json.loads(info)
    labels.extend([x["name"] for x in labels_json])


@lru_cache()
def get_pytorch_labels() -> List[str]:
    prefix = "https://api.github.com/repos/pytorch/pytorch/labels?per_page=100"
    header, info = request_for_labels(prefix + "&page=1")
    labels: List[str] = []
    update_labels(labels, info)

    last_page = get_last_page(header)
    assert last_page > 0, "Error reading header info to determine total number of pages of labels"
    for page_number in range(2, last_page + 1):  # skip page 1
        _, info = request_for_labels(prefix + f"&page={page_number}")
        update_labels(labels, info)

    return labels


def delete_comment(comment_id: int) -> None:
    url = f"https://api.github.com/repos/pytorch/pytorch/issues/comments/{comment_id}"
    fetch_url(url, method="DELETE")
