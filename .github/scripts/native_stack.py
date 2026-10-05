"""GitHub-native PR stacks for the merge bot: reading and validating them, and
finding what landed."""

from __future__ import annotations

import re
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
    head_ref: str
    head_oid: str
    base_ref: str


@dataclass(frozen=True)
class NativeStack:
    base_ref: str
    entries: tuple[StackEntry, ...]


def get_native_stack(org: str, project: str, pr_num: int) -> NativeStack | None:
    try:
        rc = gh_graphql(GH_GET_PR_STACK_QUERY, owner=org, name=project, number=pr_num)
    except GHGraphQLError as e:
        # Its message has the whole query, too long to quote in a PR comment
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
            head_ref=node["pullRequest"]["headRefName"],
            head_oid=node["pullRequest"]["headRefOid"],
            base_ref=node["pullRequest"]["baseRefName"],
        )
        for node in nodes
    ]
    entries.sort(key=lambda entry: entry.position)
    return NativeStack(stack["baseRefName"], tuple(entries))


def stack_dependencies_line(numbers: list[int]) -> str:
    """The `Stack dependencies` line of the landed commit of a stacked PR, which names
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


def _own_trailers(message: str) -> tuple[str, list[int]]:
    """The URL of the last `Pull Request resolved` line of `message` ("" if none) and
    the PRs, bottom first, that the `Stack dependencies` line after it names. A
    commit lands the PR of that URL: the merge bot appends that URL's line to the
    PR's body, which may quote other commits' lines. Only LF ends a line, as for `git
    log --grep`: the Co-authored-by lines after the bot's lines hold author names,
    which may contain other line separators."""
    lines = message.split("\n")
    resolved = [i for i, x in enumerate(lines) if x.startswith(PULL_REQUEST_RESOLVED)]
    if not resolved:
        return "", []
    url = lines[resolved[-1]].removeprefix(PULL_REQUEST_RESOLVED)
    for line in lines[resolved[-1] + 1 :]:
        if re.fullmatch(rf"{STACK_DEPENDENCIES}#[0-9]+(, #[0-9]+)*", line):
            return url, [int(num) for num in re.findall(r"[0-9]+", line)]
    return url, []


def _landing_candidates(
    repo: GitRepo, pr_url: str, rev: str, *options: str
) -> list[tuple[str, str]]:
    """The commits of `rev` that `git log <options>` lists and that have a `Pull
    Request resolved` line for `pr_url`, as _log returns them."""
    grep = f"--grep=^{PULL_REQUEST_RESOLVED}{_ere_escape(pr_url)}$"
    return _log(repo, rev, *options, "-E", grep)


def _revert_of(repo: GitRepo, sha: str, rev: str) -> tuple[str, str] | None:
    """Newest commit of `rev` that reverts commit `sha`, as _log returns it: the merge
    bot's reverts and plain `git revert`s name it in a `This reverts commit <sha>.`
    line, to which some manual reverts add more text."""
    grep = f"--grep=^This reverts commit {sha}([^0-9a-f]|$)"
    reverts = _log(repo, rev, "-1", "-E", grep)
    return reverts[0] if reverts else None


def _reverted(repo: GitRepo, sha: str, ref: str) -> bool:
    """Whether `ref` reverts commit `sha`, counting reverts of reverts: it does if
    the chain of reverts after `sha`, each reverting the one before, is of odd
    length."""
    reverts = 0
    while (revert := _revert_of(repo, sha, f"{sha}..{ref}")) is not None:
        sha = revert[0]
        reverts += 1
    return reverts % 2 == 1


def _is_ancestor(repo: GitRepo, ancestor: str, descendant: str) -> bool:
    return repo.get_merge_base(ancestor, descendant) == ancestor


def _stack_entries(
    stack: NativeStack, target: int
) -> tuple[tuple[StackEntry, ...], int]:
    """The entries of `stack` up to PR `target` and the index of the lowest open one,
    checked to be a chain of PRs on the stack's trunk with no closed PR above an open
    one and no ghstack PR. Even a landed ghstack PR is refused: its revert takes the
    ghstack path, which would leave the PRs landed above it."""
    numbers = [entry.number for entry in stack.entries]
    if target not in numbers:
        raise NativeStackError(f"PR #{target} is not in the stack")
    entries = stack.entries[: numbers.index(target) + 1]
    if entries[-1].closed:
        raise NativeStackError(f"PR #{target} is closed")
    for idx, entry in enumerate(entries):
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
    """Every commit of `trunk` that lands `pr_url`, reverted or not, newest first. The
    first call for a PR in `repo` reads the whole history, and later calls only the
    commits added since, unless `trunk` was rewritten and lost the commit that the
    last call read up to."""
    tip = repo.rev_parse(trunk)
    read = _trunk_landings_read.setdefault(repo, {})
    rev = tip
    landings: list[str] = []
    if pr_url in read and _is_ancestor(repo, read[pr_url][0], tip):
        start, landings = read[pr_url]
        rev = f"{start}..{tip}"
    candidates = _landing_candidates(repo, pr_url, rev)
    new = [sha for sha, msg in candidates if _own_trailers(msg)[0] == pr_url]
    read[pr_url] = (tip, new + landings)
    return new + landings


def _landed_in(repo: GitRepo, pr_url: str, rev: str, trunk: str) -> str | None:
    """The newest commit of `rev` that lands `pr_url` (see _own_trailers), or None
    if there is none or `rev` reverts it (see _reverted). The landings are read
    through _trunk_landings, so those of `rev` must be commits of `trunk`."""
    landings = _trunk_landings(repo, pr_url, trunk)
    landing = next((sha for sha in landings if _is_ancestor(repo, sha, rev)), None)
    if landing is None or _reverted(repo, landing, rev):
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
