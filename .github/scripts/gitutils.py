#!/usr/bin/env python3

import os
import re
import tempfile
from collections import defaultdict
from datetime import datetime
from functools import wraps
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

T = TypeVar("T")

RE_GITHUB_URL_MATCH = re.compile("^https://.*@?github.com/(.+)/(.+)$")


def get_git_remote_name() -> str:
    return os.getenv("GIT_REMOTE_NAME", "origin")


def get_git_repo_dir() -> str:
    from pathlib import Path

    return os.getenv("GIT_REPO_DIR", str(Path(__file__).resolve().parent.parent.parent))


def fuzzy_list_to_dict(items: List[Tuple[str, str]]) -> Dict[str, List[str]]:
    """
    Converts list to dict preserving elements with duplicate keys
    """
    rc: Dict[str, List[str]] = defaultdict(list)
    for key, val in items:
        rc[key].append(val)
    return dict(rc)


def _check_output(items: List[str], encoding: str = "utf-8") -> str:
    from subprocess import CalledProcessError, check_output, STDOUT

    try:
        return check_output(items, stderr=STDOUT).decode(encoding)
    except CalledProcessError as e:
        msg = f"Command `{' '.join(e.cmd)}` returned non-zero exit code {e.returncode}"
        stdout = e.stdout.decode(encoding) if e.stdout is not None else ""
        stderr = e.stderr.decode(encoding) if e.stderr is not None else ""
        # These get swallowed up, so print them here for debugging
        print(f"stdout: \n{stdout}")
        print(f"stderr: \n{stderr}")
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

    def __init__(
        self,
        commit_hash: str,
        author: str,
        author_date: datetime,
        title: str,
        body: str,
        commit_date: Optional[datetime] = None,
    ) -> None:
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
    return GitCommit(
        commit_hash=lines[0].split()[1].strip(),
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

    def branches_containing_ref(
        self, ref: str, *, include_remote: bool = True
    ) -> List[str]:
        rc = (
            self._run_git("branch", "--remote", "--contains", ref)
            if include_remote
            else self._run_git("branch", "--contains", ref)
        )
        return [x.strip() for x in rc.split("\n") if x.strip()] if len(rc) > 0 else []

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
        refs = self._run_git("show-ref", "-s", name).strip().split("\n")
        if not all(refs[i] == refs[0] for i in range(1, len(refs))):
            raise RuntimeError(f"reference {name} is ambiguous")
        return refs[0]

    def rev_parse(self, name: str) -> str:
        return self._run_git("rev-parse", "--verify", name).strip()

    def get_merge_base(self, from_ref: str, to_ref: str) -> str:
        return self._run_git("merge-base", from_ref, to_ref).strip()

    def patch_id(self, ref: Union[str, List[str]]) -> List[Tuple[str, str]]:
        is_list = isinstance(ref, list)
        if is_list:
            if len(ref) == 0:
                return []
            ref = " ".join(ref)
        rc = _check_output(
            ["sh", "-c", f"git -C {self.repo_dir} show {ref}|git patch-id --stable"]
        ).strip()
        return [cast(Tuple[str, str], x.split(" ", 1)) for x in rc.split("\n")]

    def commits_resolving_gh_pr(self, pr_num: int) -> List[str]:
        owner, name = self.gh_owner_and_name()
        msg = f"Pull Request resolved: https://github.com/{owner}/{name}/pull/{pr_num}"
        rc = self._run_git("log", "--format=%H", "--grep", msg).strip()
        return rc.split("\n") if len(rc) > 0 else []

    def get_commit(self, ref: str) -> GitCommit:
        return parse_fuller_format(
            self._run_git("show", "--format=fuller", "--date=unix", "--shortstat", ref)
        )

    def cherry_pick(self, ref: str) -> None:
        self._run_git("cherry-pick", "-x", ref)

    def revert(self, ref: str) -> None:
        self._run_git("revert", "--no-edit", ref)

    def compute_branch_diffs(
        self, from_branch: str, to_branch: str
    ) -> Tuple[List[str], List[str]]:
        """
        Returns list of commmits that are missing in each other branch since their merge base
        Might be slow if merge base is between two branches is pretty far off
        """
        from_ref = self.rev_parse(from_branch)
        to_ref = self.rev_parse(to_branch)
        merge_base = self.get_merge_base(from_ref, to_ref)
        from_commits = self.revlist(f"{merge_base}..{from_ref}")
        to_commits = self.revlist(f"{merge_base}..{to_ref}")
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
                            "pytorch/pytorch" not in self.remote_url()
                            or frc.commit_hash
                            not in {
                                "0a6a1b27a464ba5be5f587cce2ee12ab8c504dbf",
                                "6d0f4a1d545a8f161df459e8d4ccafd4b9017dbe",
                                "edf909e58f06150f7be41da2f98a3b9de3167bca",
                                "a58c6aea5a0c9f8759a4154e46f544c8b03b8db1",
                                "7106d216c29ca16a3504aa2bedad948ebcf4abc2",
                            }
                        ):
                            raise RuntimeError(
                                f"Unexpected differences between {frc} and {toc}"
                            )
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
            for excluded_commit in {
                "8e09e20c1dafcdbdb45c2d1574da68a32e54a3a5",
                "5f37e5c2a39c3acb776756a17730b865f0953432",
                "b5222584e6d6990c6585981a936defd1af14c0ba",
                "84d9a2e42d5ed30ec3b8b4140c38dd83abbce88d",
                "f211ec90a6cdc8a2a5795478b5b5c8d7d7896f7e",
            }:
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

    def diff(self, from_ref: str, to_ref: Optional[str] = None) -> str:
        if to_ref is None:
            return self._run_git("diff", f"{from_ref}^!")
        return self._run_git("diff", f"{from_ref}..{to_ref}")


def clone_repo(username: str, password: str, org: str, project: str) -> GitRepo:
    path = tempfile.mkdtemp()
    _check_output(
        [
            "git",
            "clone",
            f"https://{username}:{password}@github.com/{org}/{project}",
            path,
        ]
    ).strip()
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


def _shasum(value: str) -> str:
    import hashlib

    m = hashlib.sha256()
    m.update(value.encode("utf-8"))
    return m.hexdigest()


def are_ghstack_branches_in_sync(repo: GitRepo, head_ref: str) -> bool:
    """Checks that diff between base and head is the same as diff between orig and its parent"""
    orig_ref = re.sub(r"/head$", "/orig", head_ref)
    base_ref = re.sub(r"/head$", "/base", head_ref)
    orig_diff_sha = _shasum(repo.diff(f"{repo.remote}/{orig_ref}"))
    head_diff_sha = _shasum(
        repo.diff(f"{repo.remote}/{base_ref}", f"{repo.remote}/{head_ref}")
    )
    return orig_diff_sha == head_diff_sha


def retries_decorator(
    rc: Any = None, num_retries: int = 3
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    def decorator(f: Callable[..., T]) -> Callable[..., T]:
        @wraps(f)
        def wrapper(*args: List[Any], **kwargs: Dict[str, Any]) -> T:
            for idx in range(num_retries):
                try:
                    return f(*args, **kwargs)
                except Exception as e:
                    print(
                        f'Attempt {idx} of {num_retries} to call {f.__name__} failed with "{e}"'
                    )
                    pass
            return cast(T, rc)

        return wrapper

    return decorator
