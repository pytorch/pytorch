"""GitHub-native PR stacks for the merge bot: reading and validating them, finding
what landed, and building the commits that rebase them."""

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
REBASE_HINT = (
    "Rebase the stack onto main with `@pytorchbot rebase -b main` and try again."
)


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
    repo: GitRepo, ours: str, theirs: str, conflict: str, merge_base: str | None = None
) -> str:
    """The tree of merging `theirs` into `ours`, or NativeStackError(`conflict`)."""
    base = [] if merge_base is None else [f"--merge-base={merge_base}"]
    args = ("merge-tree", "--write-tree", *base, ours, theirs)
    merge = _git(repo, *args, ok_codes=(0, 1))
    if merge.returncode == 1:
        raise NativeStackError(conflict)
    return merge.stdout.split("\n", 1)[0]


def _commit(repo: GitRepo, tree: str, parents: list[str], message: str) -> str:
    """Commit `tree` on `parents` with the author of the first parent and the
    repository's identity as the committer: the authors of a PR's commits are the
    co-authors of its landed commit, so the bot must not author any."""
    identity = repo._run_git("log", "-1", "--format=%an%x00%ae", parents[0])
    name, email = identity.removesuffix("\n").split("\0")
    env = {"GIT_AUTHOR_NAME": name, "GIT_AUTHOR_EMAIL": email}
    args = [arg for parent in parents for arg in ("-p", parent)]
    return _git(repo, "commit-tree", tree, *args, stdin=message, env=env).stdout.strip()


def _merge_into(
    repo: GitRepo,
    number: int,
    branch: str,
    tip: str,
    base: str,
    base_name: str,
    tree: str | None = None,
) -> str:
    """A commit merging `base` into `tip`, the tip of PR `number`'s branch, with
    `tree` if given."""
    if tree is None:
        conflict = (
            f"Merging {base_name} into {branch} (PR #{number}) has conflicts; merge "
            "it locally and push the result"
        )
        tree = _merge_tree(repo, tip, base, conflict)
    return _commit(repo, tree, [tip, base], f"Merge {base_name} into {branch}\n")


def _reapply_reverted(
    repo: GitRepo,
    number: int,
    url: str,
    tip: str,
    base: str,
    merged: str,
    trunk: str,
) -> str:
    """`merged`, the merge of `base` into `tip`, plus a commit that reapplies PR
    `number`'s landing in `tip` (see _landed_in) if the merge brought in a revert of
    it: the merge then took the PR's own changes out of its branch. Only the
    landing's own changes come back, even if the revert also reverted other
    commits."""
    landing = _landed_in(repo, url, tip, trunk)
    found = None if landing is None else _revert_of(repo, landing, f"{tip}..{base}")
    if landing is None or found is None:
        return merged
    revert, message = found
    conflict = (
        f"Reapplying PR #{number} on its branch after its revert has conflicts; "
        "update the branch locally, keeping the PR's changes, and push the result"
    )
    tree = _merge_tree(repo, merged, landing, conflict, merge_base=f"{landing}^")
    if tree == repo.rev_parse(f"{merged}^{{tree}}"):
        return merged
    title = message.split("\n", 1)[0].removeprefix('Revert "').removesuffix('"')
    reapply = f'Reapply "{title}"\n\nThis reverts commit {revert}.\n'
    return _commit(repo, tree, [merged], reapply)


def _update_landed_branch(
    repo: GitRepo,
    number: int,
    url: str,
    branch: str,
    tip: str,
    onto: str,
    onto_name: str,
    trunk: str,
) -> str:
    """The tip of the branch of landed PR `number` once `onto` is merged into it.
    If `onto` has the PR's landed changes, the merge takes `onto`'s tree, which also
    skips conflicts with later changes to the same lines; otherwise (`onto` lags
    behind the landing) it is a real merge, which keeps the PR's changes."""
    if _is_ancestor(repo, onto, tip):
        return tip
    tree = None
    if _landed_in(repo, url, onto, trunk) is not None:
        tree = repo.rev_parse(f"{onto}^{{tree}}")
    return _merge_into(repo, number, branch, tip, onto, onto_name, tree)


def _update_branches(
    repo: GitRepo,
    org: str,
    project: str,
    trunk: str,
    base: str,
    base_name: str,
    branches: list[tuple[int, str, str]],
) -> list[tuple[int, str, str]]:
    """Merge `base` into the first of `branches` (PR number, branch, tip; bottom
    first), the result into the next one and so on, and return the PR number, branch
    and new tip of each branch that changed."""
    rc: list[tuple[int, str, str]] = []
    for number, branch, tip in branches:
        new = tip
        if not _is_ancestor(repo, base, tip):
            merged = _merge_into(repo, number, branch, tip, base, base_name)
            url = _pr_url(org, project, number)
            new = _reapply_reverted(repo, number, url, tip, base, merged, trunk)
            rc.append((number, branch, new))
        base, base_name = new, branch
    return rc


def build_native_stack_rebase(
    repo: GitRepo,
    org: str,
    project: str,
    stack: NativeStack,
    target: int,
    default_branch: str,
    onto: str,
) -> list[tuple[int, str, str]]:
    """Merge branch `onto` into the branches of `stack` bottom-up, from the lowest
    open PR to PR `target`, without touching the worktree or the index. The branch of
    the landed PR right below the lowest open one gets `onto` first, so each open PR
    still shows only its own changes. Returns the PR number, branch and new commit of
    each branch to fast-forward, bottom first."""
    if stack.base_ref != default_branch:
        raise NativeStackError(
            f"PR #{target} is in a stack based on {stack.base_ref}, but only stacks "
            f"based on {default_branch} can be rebased"
        )
    entries, first_open = _stack_entries(stack, target)
    below = entries[first_open - 1] if first_open else None
    # The open PRs to rebase, plus the closed PR whose branch the lowest is based on
    updated = entries[max(first_open - 1, 0) :]
    for entry in updated:
        if entry.head_ref in (default_branch, onto):
            raise NativeStackError(
                f"PR #{entry.number} is opened from {entry.head_ref}, so rebasing it "
                f"would push to {entry.head_ref}"
            )

    tracking = f"refs/remotes/{repo.remote}"
    trunk = f"{tracking}/{default_branch}"
    opened = [entry for entry in stack.entries[first_open:] if not entry.closed]
    names = [entry.head_ref for entry in updated]
    branches = list(dict.fromkeys([default_branch, onto, *names]))
    refspecs = [f"+refs/heads/{b}:{tracking}/{b}" for b in branches]
    shas = [entry.head_oid for entry in opened]
    if below is not None:
        shas.append(below.head_oid)
    try:
        repo._run_git("fetch", repo.remote, *refspecs, *shas)
    except RuntimeError as e:
        raise NativeStackError(
            f"Could not fetch {', '.join(branches)} and the open PRs of the stack: {e}"
        ) from e
    _check_landed(repo, org, project, entries, trunk, default_branch)
    onto_sha = repo.rev_parse(f"{tracking}/{onto}")
    # Such an `onto` (a lagging viable/strict) would take the PR's changes out of
    # the stack, or bring an older version of them in
    for entry in updated:
        url = _pr_url(org, project, entry.number)
        landing = _landed_in(repo, url, onto_sha, trunk)
        if landing is not None and landing != _landed_in(repo, url, trunk, trunk):
            raise NativeStackError(
                f"{onto} is behind {default_branch} for PR #{entry.number}: it still "
                f"has the PR's landing {landing}, which {default_branch} reverted or "
                "replaced since. " + REBASE_HINT
            )
    tips = {e.number: repo.rev_parse(f"{tracking}/{e.head_ref}") for e in updated}
    # Merging `onto` into the stack would also merge such a PR into the branches
    # below it
    heads = [(e.number, e.head_oid) for e in opened]
    heads += [(e.number, tips[e.number]) for e in entries[first_open:]]
    for number, head in heads:
        if _is_ancestor(repo, head, onto_sha):
            raise NativeStackError(
                f"The head of PR #{number} is already in {onto}, so it has nothing "
                "to rebase; close the PR if it landed"
            )

    rc: list[tuple[int, str, str]] = []
    base, base_name = onto_sha, onto
    if below is not None:
        tip = tips[below.number]
        _check_closed_branch(
            repo, below.number, below.head_oid, tip, trunk, default_branch
        )
        url = _pr_url(org, project, below.number)
        base = _update_landed_branch(
            repo, below.number, url, below.head_ref, tip, onto_sha, onto, trunk
        )
        base_name = below.head_ref
        if base != tip:
            rc.append((below.number, below.head_ref, base))
    rebased = [(e.number, e.head_ref, tips[e.number]) for e in entries[first_open:]]
    return rc + _update_branches(repo, org, project, trunk, base, base_name, rebased)


def push_branches(
    repo: GitRepo, updates: list[tuple[int, str, str]], dry_run: bool = False
) -> None:
    """Fast-forward the branches of `updates`, as build_native_stack_rebase returns
    them, all at once or not at all."""
    # A push without refspecs would push the current branch (push.default)
    if not updates:
        return
    refspecs = [f"{new}:refs/heads/{branch}" for _, branch, new in updates]
    dry = ["--dry-run"] if dry_run else []
    repo._run_git("push", *dry, "--atomic", repo.remote, *refspecs)


def branch_update_comments(
    stack: NativeStack,
    updates: list[tuple[int, str, str]],
    onto: str,
    pr_num: int,
    cause: str,
) -> list[tuple[int, str]]:
    """The PR number and comment for each open PR of `stack` whose branch `updates`,
    as build_native_stack_rebase returns them, fast-forwards once `onto` is merged
    into it because PR `pr_num` was `cause`."""
    closed = {entry.number for entry in stack.entries if entry.closed}
    rc: list[tuple[int, str]] = []
    for number, branch, _ in updates:
        if number in closed:
            continue
        msg = f"Merged `{onto}` into `{branch}`"
        if number != pr_num:
            msg += f" because #{pr_num} was {cause}"
        msg += (
            ", please pull locally before adding more changes (for example, via "
            f"`git checkout {branch} && git pull --rebase`)"
        )
        rc.append((number, msg))
    return rc
