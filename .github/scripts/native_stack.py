"""GitHub-native PR stacks for the merge bot: reading and validating them, finding
what landed, and building the commits that merge them."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

from github_utils import gh_graphql, GHGraphQLError


if TYPE_CHECKING:
    from gitutils import GitRepo


GH_GET_PR_STACK_QUERY = """
query ($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      stack {
        number
        baseRefName
        entries(first: 100) {
          totalCount
          pageInfo {
            hasNextPage
          }
          nodes {
            position
            pullRequest {
              number
              closed
              isCrossRepository
              headRefName
              headRefOid
              baseRefName
            }
          }
        }
      }
    }
  }
}
"""

RE_GHSTACK_HEAD_REF = re.compile(r"^gh/[^/]+/[0-9]+/head$")
PULL_REQUEST_RESOLVED = "Pull Request resolved: "
STACK_DEPENDENCIES = "Stack dependencies: "
PR_UPDATED_ERROR = "PR #{} was updated while preparing the merge, please try again"
REBASE_HINT = "Rebase the stack onto main and try again."


class NativeStackError(RuntimeError):
    pass


@dataclass(frozen=True)
class StackEntry:
    position: int
    number: int
    closed: bool
    is_cross_repository: bool
    head_ref: str
    head_oid: str
    base_ref: str


@dataclass(frozen=True)
class NativeStack:
    number: int
    base_ref: str
    entries: tuple[StackEntry, ...]


def get_native_stack(org: str, project: str, pr_num: int) -> NativeStack | None:
    try:
        rc = gh_graphql(GH_GET_PR_STACK_QUERY, owner=org, name=project, number=pr_num)
    except GHGraphQLError as e:
        # Its message has the whole query, too long for the PR comments that quote it
        errors = "; ".join(str(err.get("message")) for err in e.response["errors"])
        raise NativeStackError(f"GraphQL errors: {errors}") from e
    stack = rc["data"]["repository"]["pullRequest"]["stack"]
    if stack is None:
        return None
    page = stack["entries"]
    nodes = page["nodes"]
    if (
        page["pageInfo"]["hasNextPage"]
        or page["totalCount"] != len(nodes)
        or any(node is None or node["pullRequest"] is None for node in nodes)
    ):
        raise NativeStackError(
            f"Could not read all {page['totalCount']} PRs in the stack of PR #{pr_num}"
        )
    entries = [
        StackEntry(
            position=node["position"],
            number=node["pullRequest"]["number"],
            closed=node["pullRequest"]["closed"],
            is_cross_repository=node["pullRequest"]["isCrossRepository"],
            head_ref=node["pullRequest"]["headRefName"],
            head_oid=node["pullRequest"]["headRefOid"],
            base_ref=node["pullRequest"]["baseRefName"],
        )
        for node in nodes
    ]
    entries.sort(key=lambda entry: entry.position)
    return NativeStack(stack["number"], stack["baseRefName"], tuple(entries))


def stack_dependencies_line(numbers: list[int]) -> str:
    """The line that the merge bot adds to the landed commit of a stacked PR to name
    the PRs below it, bottom first."""
    return STACK_DEPENDENCIES + ", ".join(f"#{num}" for num in numbers)


def _pr_url(org: str, project: str, number: int) -> str:
    return f"https://github.com/{org}/{project}/pull/{number}"


def _ere_escape(text: str) -> str:
    return "".join(f"\\{c}" if c in ".[\\()*+?{|^$" else c for c in text)


def _log(repo: GitRepo, rev: str, *options: str) -> list[tuple[str, str]]:
    """The SHA and message of each commit of `rev` that `git log <options>` lists,
    newest first."""
    out = repo._run_git("log", "-z", "--format=%H%n%B", *options, rev, "--")
    commits = []
    for record in filter(None, out.split("\0")):
        sha, _, message = record.partition("\n")
        if not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise RuntimeError(f"Unexpected output from git log on {rev}:\n{out}")
        commits.append((sha, message))
    return commits


def _own_pr_url(message: str) -> str:
    """The URL of the last `Pull Request resolved` line of `message` ("" if none). A
    commit lands the PR of that URL: the merge bot appends this line to the PR's
    body, which may quote other commits' lines. Only LF ends a line, as for `git log
    --grep`: the Co-authored-by lines after the bot's lines hold author names, which
    may contain other line separators."""
    resolved = [x for x in message.split("\n") if x.startswith(PULL_REQUEST_RESOLVED)]
    return resolved[-1].removeprefix(PULL_REQUEST_RESOLVED) if resolved else ""


def _landing_candidates(
    repo: GitRepo, pr_url: str, rev: str, *options: str
) -> list[tuple[str, str]]:
    """The commits of `rev` that `git log <options>` lists and that have a `Pull
    Request resolved` line for `pr_url`, as _log returns them."""
    grep = f"--grep=^{PULL_REQUEST_RESOLVED}{_ere_escape(pr_url)}$"
    return _log(repo, rev, *options, "-E", grep)


def _newest_landing(repo: GitRepo, pr_url: str, rev: str) -> str | None:
    """Newest commit of `rev` that lands `pr_url`, reverted or not."""
    # Quoted lines are rare, so the newest candidate is nearly always the landing
    candidates = _landing_candidates(repo, pr_url, rev, "-1")
    if candidates and _own_pr_url(candidates[0][1]) != pr_url:
        candidates = _landing_candidates(repo, pr_url, rev)
    landings = (sha for sha, msg in candidates if _own_pr_url(msg) == pr_url)
    return next(landings, None)


def _revert_of(repo: GitRepo, sha: str, rev: str) -> tuple[str, str] | None:
    """Newest commit of `rev` that reverts commit `sha`, as _log returns it: the merge
    bot's reverts and plain `git revert`s name it in a `This reverts commit <sha>.`
    line, to which some manual reverts add more text."""
    grep = f"--grep=^This reverts commit {sha}([^0-9a-f]|$)"
    reverts = _log(repo, rev, "-1", "-E", grep)
    return reverts[0] if reverts else None


def find_landed_commit(repo: GitRepo, pr_url: str, ref: str) -> str | None:
    """Newest commit on `ref` that lands `pr_url` (see _own_pr_url), or None if
    there is none or a later commit reverted it."""
    landing = _newest_landing(repo, pr_url, ref)
    if landing is None or _revert_of(repo, landing, f"{landing}..{ref}") is not None:
        return None
    return landing


def landed_since(repo: GitRepo, pr_url: str, start: str, ref: str) -> bool:
    """Whether a commit in `start..ref` lands `pr_url`, reverted since or not."""
    return _newest_landing(repo, pr_url, f"{start}..{ref}") is not None


def _is_ancestor(repo: GitRepo, ancestor: str, descendant: str) -> bool:
    return repo.get_merge_base(ancestor, descendant) == ancestor


def _stack_entries(
    stack: NativeStack, target: int
) -> tuple[tuple[StackEntry, ...], int]:
    """The entries of `stack` up to PR `target` and the index of the lowest open one,
    checked to be a chain of same-repository PRs on the stack's trunk with no closed
    PR above an open one and no ghstack PR. Even a landed ghstack PR is refused: its
    revert takes the ghstack path, which would leave the PRs landed above it."""
    numbers = [entry.number for entry in stack.entries]
    if target not in numbers:
        raise NativeStackError(f"PR #{target} is not in the stack")
    entries = stack.entries[: numbers.index(target) + 1]
    if entries[-1].closed:
        raise NativeStackError(f"PR #{target} is closed")
    for idx, entry in enumerate(entries):
        if entry.is_cross_repository:
            raise NativeStackError(
                f"PR #{entry.number} is from a fork, but the bot only supports stacks "
                "of PRs from this repository"
            )
        if RE_GHSTACK_HEAD_REF.match(entry.head_ref):
            raise NativeStackError(
                f"PR #{entry.number} is a ghstack PR, and the bot does not support "
                "stacks that mix ghstack PRs with other PRs; unstack the other PRs"
            )
        if idx == 0 and entry.base_ref != stack.base_ref:
            raise NativeStackError(
                f"PR #{entry.number} is at the bottom of the stack, so its base must "
                f"be the stack's trunk ({stack.base_ref}), not {entry.base_ref}"
            )
        if idx > 0 and entry.base_ref != entries[idx - 1].head_ref:
            raise NativeStackError(
                f"PR #{entry.number} targets {entry.base_ref} instead of "
                f"{entries[idx - 1].head_ref}, the branch of PR "
                f"#{entries[idx - 1].number} below it in the stack"
            )
    first_open = next(idx for idx, entry in enumerate(entries) if not entry.closed)
    for entry in entries[first_open:]:
        if entry.closed:
            raise NativeStackError(
                f"PR #{entry.number} is closed, but PR #{entries[first_open].number} "
                "below it is still open"
            )
    return entries, first_open


def _check_closed_branch(
    repo: GitRepo, number: int, head: str, tip: str, trunk: str, default_branch: str
) -> None:
    """Refuse if the branch of closed PR `number`, at `tip`, has commits that are
    neither on `trunk` nor in `head`, the PR's head or a head based on it."""
    revs = (tip, "--not", trunk, head)
    never_landed = repo._run_git("rev-list", "--no-merges", *revs, "--").split()
    if never_landed:
        raise NativeStackError(
            f"PR #{number} is closed, but its branch has commits that "
            f"never landed on {default_branch}: {', '.join(never_landed)}"
        )


# For each repository and PR URL, the trunk commit that _trunk_landings last read up
# to and the commits it found that land the PR, newest first
_trunk_landings_read: WeakKeyDictionary[GitRepo, dict[str, tuple[str, list[str]]]] = (
    WeakKeyDictionary()
)


def _trunk_landings(repo: GitRepo, pr_url: str, trunk: str) -> list[str]:
    """Every commit of `trunk` that lands `pr_url`, reverted or not, newest first. A
    trunk only moves forward, so only the first call for a PR in `repo` reads the
    whole history, and later calls read the commits added since."""
    tip = repo.rev_parse(trunk)
    read = _trunk_landings_read.setdefault(repo, {})
    rev = tip
    landings: list[str] = []
    if pr_url in read and _is_ancestor(repo, read[pr_url][0], tip):
        start, landings = read[pr_url]
        rev = f"{start}..{tip}"
    candidates = _landing_candidates(repo, pr_url, rev)
    new = [sha for sha, msg in candidates if _own_pr_url(msg) == pr_url]
    read[pr_url] = (tip, new + landings)
    return new + landings


def _landed_in(repo: GitRepo, pr_url: str, rev: str, trunk: str) -> str | None:
    """find_landed_commit(repo, pr_url, rev) for a `rev` whose landings of the PR
    are commits of `trunk`, read through _trunk_landings."""
    landings = _trunk_landings(repo, pr_url, trunk)
    landing = next((sha for sha in landings if _is_ancestor(repo, sha, rev)), None)
    if landing is None or _revert_of(repo, landing, f"{landing}..{rev}") is not None:
        return None
    return landing


def _check_landed(
    repo: GitRepo,
    org: str,
    project: str,
    entries: tuple[StackEntry, ...],
    trunk: str,
    default_branch: str,
) -> None:
    """Refuse unless every closed entry landed on `trunk` and no open one did."""
    for entry in entries:
        landed = _landed_in(repo, _pr_url(org, project, entry.number), trunk, trunk)
        if entry.closed and landed is None:
            raise NativeStackError(
                f"PR #{entry.number} is closed but not landed on {default_branch}, "
                "or it was reverted; reopen it and try again"
            )
        if not entry.closed and landed is not None:
            raise NativeStackError(
                f"PR #{entry.number} already landed but is still open; try again "
                "once it is closed"
            )


def _check_own_landing(
    repo: GitRepo,
    org: str,
    project: str,
    number: int,
    lower: str,
    lower_name: str,
    trunk: str,
) -> None:
    """Refuse if `lower`, the commit that open PR `number`'s changes start from, has
    a landing of the PR that it does not revert: the PR would then land without
    those changes. Landings are trunk commits, so those in `lower` are the trunk's
    that it has."""
    landing = _landed_in(repo, _pr_url(org, project, number), lower, trunk)
    if landing is not None:
        raise NativeStackError(
            f"PR #{number} landed in {landing}, which {lower_name} still has, so its "
            "changes would not land again. " + REBASE_HINT
        )


def get_native_stack_landing_prs(
    repo: GitRepo,
    org: str,
    project: str,
    stack: NativeStack,
    target: int,
    default_branch: str,
) -> list[tuple[StackEntry, str]]:
    """Validate `stack` for merging PR `target` and return the open PRs to land,
    bottom first, each with the commit its changes start from: the live tip of its
    base branch, or its merge base with `default_branch` at the bottom of the stack.
    """
    if stack.base_ref != default_branch:
        raise NativeStackError(
            f"PR #{target} is in a stack based on {stack.base_ref}, but only stacks "
            f"based on {default_branch} can be merged"
        )
    entries, first_open = _stack_entries(stack, target)

    trunk = f"refs/remotes/{repo.remote}/{default_branch}"
    # The PRs to land, plus the closed PR whose branch the lowest of them is based on
    fetched = entries[max(first_open - 1, 0) :]
    refspecs = [f"+refs/heads/{default_branch}:{trunk}"]
    refspecs += [entry.head_oid for entry in fetched]
    refspecs += [
        f"+refs/heads/{entry.head_ref}:refs/remotes/{repo.remote}/{entry.head_ref}"
        for entry in fetched[:-1]
    ]
    try:
        repo._run_git("fetch", repo.remote, *refspecs)
    except RuntimeError as e:
        prs = ", ".join(f"#{entry.number}" for entry in fetched)
        raise NativeStackError(
            f"Could not fetch {default_branch} and the commits of PRs {prs}: {e}"
        ) from e
    _check_landed(repo, org, project, entries, trunk, default_branch)

    rc: list[tuple[StackEntry, str]] = []
    for idx in range(first_open, len(entries)):
        entry = entries[idx]
        if idx == 0:
            lower = repo.get_merge_base(trunk, entry.head_oid)
            lower_name = f"its merge base with {default_branch}"
            _check_own_landing(
                repo, org, project, entry.number, lower, lower_name, trunk
            )
            rc.append((entry, lower))
            continue
        lower = repo.rev_parse(f"refs/remotes/{repo.remote}/{entry.base_ref}")
        below = entries[idx - 1]
        # This PR lands only the commits above `lower`, so commits under it that the
        # PR below does not land (its pinned head if open, its landed head if closed)
        # would never land. A closed PR's branch may still have merges of the trunk,
        # which only bring in landed commits.
        if below.closed:
            _check_closed_branch(
                repo, below.number, below.head_oid, lower, trunk, default_branch
            )
        elif lower != below.head_oid:
            raise NativeStackError(PR_UPDATED_ERROR.format(below.number))
        if not _is_ancestor(repo, lower, entry.head_oid):
            raise NativeStackError(
                f"PR #{entry.number} is not based on the tip of {entry.base_ref}. "
                + REBASE_HINT
            )
        merge_bases = repo._run_git("merge-base", "--all", trunk, entry.head_oid)
        if not all(_is_ancestor(repo, mb, lower) for mb in merge_bases.split()):
            raise NativeStackError(
                f"PR #{entry.number} contains {default_branch} commits that "
                f"{entry.base_ref} does not have. " + REBASE_HINT
            )
        _check_own_landing(
            repo, org, project, entry.number, lower, entry.base_ref, trunk
        )
        rc.append((entry, lower))
    return rc


def _git(
    repo: GitRepo,
    *args: str,
    stdin: str = "",
    env: dict[str, str] | None = None,
    ok_codes: tuple[int, ...] = (0,),
) -> subprocess.CompletedProcess[str]:
    """GitRepo._run_git with stdin, extra environment and accepted exit codes. It
    runs in bytes mode, as text mode would turn each carriage return of the output
    into a newline."""
    cmd = ["git", "-C", repo.repo_dir, *args]
    if repo.debug:
        print(f"+ {' '.join(cmd)}")
    run = subprocess.run(
        cmd,
        input=stdin.encode(),
        capture_output=True,
        env=None if env is None else {**os.environ, **env},
    )
    stdout, stderr = run.stdout.decode(), run.stderr.decode()
    proc = subprocess.CompletedProcess(cmd, run.returncode, stdout, stderr)
    if proc.returncode not in ok_codes:
        print(f"stdout: \n{proc.stdout}")
        print(f"stderr: \n{proc.stderr}")
        raise RuntimeError(
            f"Command `{' '.join(cmd)}` returned non-zero exit code "
            f"{proc.returncode}\n```\n{proc.stdout}{proc.stderr}```"
        )
    return proc


def _merge_tree(
    repo: GitRepo, ours: str, theirs: str, conflict: str, merge_base: str
) -> str:
    """The tree of merging `theirs` into `ours`, or NativeStackError(`conflict`)."""
    args = ("merge-tree", "--write-tree", f"--merge-base={merge_base}", ours, theirs)
    merge = _git(repo, *args, ok_codes=(0, 1))
    if merge.returncode == 1:
        raise NativeStackError(conflict)
    return merge.stdout.split("\n", 1)[0]


def build_native_stack_commits(
    repo: GitRepo,
    base_sha: str,
    landing: list[tuple[StackEntry, str]],
    commits: list[tuple[str, str]],
) -> str:
    """Commit the changes of each PR in `landing`, as returned by
    get_native_stack_landing_prs, on top of `base_sha`, without touching the
    worktree or the index, and return the last commit. `commits` holds the author
    (`Name <email>`) and the message of each PR's commit; the message is cleaned up
    like `git commit -m` does."""
    current = base_sha
    for (entry, lower), (author, message) in zip(landing, commits, strict=True):
        conflict = f"PR #{entry.number} has conflicts with the commits below it. "
        tree = _merge_tree(repo, current, entry.head_oid, conflict + REBASE_HINT, lower)
        if tree == repo.rev_parse(f"{current}^{{tree}}"):
            raise NativeStackError(
                f"PR #{entry.number} has no changes to land: it is empty, or its "
                "changes already landed"
            )
        name, _, email = author.rpartition("<")
        env = {
            "GIT_AUTHOR_NAME": name.strip(),
            "GIT_AUTHOR_EMAIL": email.removesuffix(">"),
        }
        message = _git(repo, "stripspace", stdin=message).stdout
        commit = _git(repo, "commit-tree", tree, "-p", current, stdin=message, env=env)
        current = commit.stdout.strip()
    return current
