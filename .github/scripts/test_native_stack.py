#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest import main, mock, TestCase

from github_utils import GHGraphQLError
from gitutils import _check_output, GitRepo
from native_stack import (
    _landed_in,
    _trunk_landings,
    get_native_stack,
    get_native_stack_landing_prs,
    GH_GET_PR_STACK_QUERY,
    NativeStack,
    NativeStackError,
    PULL_REQUEST_RESOLVED,
    StackEntry,
)


ORG = "pytorch"
PROJECT = "pytorch"
TRUNK = "main"
# Keeps the developer's git config (signing, hooks, default branch) out of the
# temporary repositories. HOME and XDG_CONFIG_HOME are pointed at the temporary
# directory too, as git reads the global ignore and attributes files from there even
# with GIT_CONFIG_GLOBAL set.
GIT_ENV = {
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    # Only local paths: a push or fetch of the code under test can never reach GitHub
    "GIT_ALLOW_PROTOCOL": "file",
    "GIT_AUTHOR_NAME": "PR Author",
    "GIT_AUTHOR_EMAIL": "author@example.com",
    "GIT_COMMITTER_NAME": "PR Author",
    "GIT_COMMITTER_EMAIL": "author@example.com",
}
TRUNK_REF = f"refs/remotes/origin/{TRUNK}"
# Path -> new content, or None to delete the file
FileChanges = dict[str, str | None]


def pr_url(number: int, host: str = "github.com") -> str:
    return f"https://{host}/{ORG}/{PROJECT}/pull/{number}"


def landing_message(url: str, body: str = "Summary") -> str:
    return (
        f"Change\n\n{body}\n\n{PULL_REQUEST_RESOLVED}{url}\n"
        "Approved by: https://github.com/reviewer\n"
    )


def revert_message(url: str, sha: str) -> str:
    return (
        f'Revert "Change"\n\nThis reverts commit {sha}.\n\n'
        f"Reverted {url} on behalf of https://github.com/reverter due to broken trunk\n"
    )


def manual_revert_message(sha: str, number: int = 700) -> str:
    """The landed message of revert PR `number`, opened by hand from `git revert`: it
    reverts commit `sha` without naming the PR it reverts"""
    return landing_message(pr_url(number), f"This reverts commit {sha}.")


def make_entry(
    position: int,
    number: int,
    head_ref: str,
    head_oid: str,
    base_ref: str,
    closed: bool = False,
) -> StackEntry:
    return StackEntry(
        position=position,
        number=number,
        closed=closed,
        head_ref=head_ref,
        head_oid=head_oid,
        base_ref=base_ref,
    )


def with_entry(stack: NativeStack, index: int, **changes: Any) -> NativeStack:
    entries = list(stack.entries)
    entries[index] = replace(entries[index], **changes)
    return replace(stack, entries=tuple(entries))


def gql_node(entry: StackEntry) -> dict[str, Any]:
    return {
        "position": entry.position,
        "pullRequest": {
            "number": entry.number,
            "title": f"PR {entry.number}",
            "state": "CLOSED" if entry.closed else "OPEN",
            "closed": entry.closed,
            "merged": False,
            "headRefName": entry.head_ref,
            "headRefOid": entry.head_oid,
            "baseRefName": entry.base_ref,
            "baseRefOid": "0" * 40,
        },
    }


def gql_stack(
    nodes: list[Any], total_count: int | None = None, has_next_page: bool = False
) -> dict[str, Any]:
    total = len(nodes) if total_count is None else total_count
    stack = {
        "size": total,
        "baseRefName": TRUNK,
        "entries": {
            "totalCount": total,
            "pageInfo": {"hasNextPage": has_next_page},
            "nodes": nodes,
        },
    }
    return {"data": {"repository": {"pullRequest": {"stack": stack}}}}


GQL_ENTRIES = (
    make_entry(1, 198532, "user/capture", "1" * 40, TRUNK, closed=True),
    make_entry(2, 198534, "user/inductor", "2" * 40, "user/capture"),
    make_entry(3, 198536, "user/tests", "3" * 40, "user/inductor"),
)
GQL_NODES = [gql_node(entry) for entry in GQL_ENTRIES]


@mock.patch("native_stack.gh_graphql")
class TestGetNativeStack(TestCase):
    def assertRejected(self, gql: Any, response: dict[str, Any]) -> None:
        gql.return_value = response
        message = r"Could not read all \d+ PRs in the stack of PR #198534"
        with self.assertRaisesRegex(NativeStackError, message):
            get_native_stack(ORG, PROJECT, 198534)

    def test_returns_none_without_stack(self, gql: Any) -> None:
        gql.return_value = {"data": {"repository": {"pullRequest": {"stack": None}}}}
        self.assertIsNone(get_native_stack(ORG, PROJECT, 199613))
        gql.assert_called_once_with(
            GH_GET_PR_STACK_QUERY, owner=ORG, name=PROJECT, number=199613
        )

    def test_parses_stack(self, gql: Any) -> None:
        gql.return_value = gql_stack(GQL_NODES)
        self.assertEqual(
            get_native_stack(ORG, PROJECT, 198534),
            NativeStack(base_ref=TRUNK, entries=GQL_ENTRIES),
        )
        gql.assert_called_once_with(
            GH_GET_PR_STACK_QUERY, owner=ORG, name=PROJECT, number=198534
        )

    def test_sorts_entries_by_position(self, gql: Any) -> None:
        gql.return_value = gql_stack([GQL_NODES[i] for i in (2, 0, 1)])
        self.assertEqual(
            get_native_stack(ORG, PROJECT, 198534),
            NativeStack(base_ref=TRUNK, entries=GQL_ENTRIES),
        )

    def test_rejects_more_than_one_page(self, gql: Any) -> None:
        self.assertRejected(gql, gql_stack(GQL_NODES, has_next_page=True))

    def test_rejects_total_count_mismatch(self, gql: Any) -> None:
        self.assertRejected(gql, gql_stack(GQL_NODES[:2], total_count=3))

    def test_rejects_null_node(self, gql: Any) -> None:
        self.assertRejected(gql, gql_stack([GQL_NODES[0], None, GQL_NODES[2]]))

    def test_rejects_null_pull_request(self, gql: Any) -> None:
        node = {"position": 2, "pullRequest": None}
        self.assertRejected(gql, gql_stack([GQL_NODES[0], node, GQL_NODES[2]]))

    def test_graphql_errors_are_reported_without_the_query(self, gql: Any) -> None:
        errors = [{"message": "Field 'stack' doesn't exist"}, {"message": "Timeout"}]
        response = {"errors": errors}
        gql.side_effect = GHGraphQLError(f"{GH_GET_PR_STACK_QUERY} failed", response)
        message = r"^GraphQL errors: Field 'stack' doesn't exist; Timeout$"
        with self.assertRaisesRegex(NativeStackError, message):
            get_native_stack(ORG, PROJECT, 198534)

    def test_query_requests_parsed_fields(self, gql: Any) -> None:
        for field in (
            "$owner",
            "$name",
            "$number",
            "stack",
            "baseRefName",
            "totalCount",
            "hasNextPage",
            "position",
            "closed",
            "headRefName",
            "headRefOid",
        ):
            self.assertIn(field, GH_GET_PR_STACK_QUERY)


class GitTestCase(TestCase):
    """A bare origin, a developer clone that pushes to it and a stale bot clone"""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        home = {"HOME": tmp.name, "XDG_CONFIG_HOME": tmp.name}
        env = mock.patch.dict(os.environ, {**GIT_ENV, **home})
        env.start()
        self.addCleanup(env.stop)
        # GIT_DIR, GIT_INDEX_FILE, ... (exported e.g. inside git hooks) would make the
        # commands below write into another repository.
        for var in _check_output(["git", "rev-parse", "--local-env-vars"]).split():
            os.environ.pop(var, None)
        origin = os.path.join(tmp.name, "origin.git")
        bot = os.path.join(tmp.name, "bot")
        self.dev = os.path.join(tmp.name, "dev")
        self.files = 0
        self.git("init", "-q", "--bare", "-b", TRUNK, origin, cwd=tmp.name)
        self.git("init", "-q", "-b", TRUNK, self.dev, cwd=tmp.name)
        self.git("remote", "add", "origin", origin)
        self.root = self.commit("Initial commit")
        self.git("push", "-q", "origin", TRUNK)
        # Cloned before any stack branch exists, so the code under test must fetch.
        self.git("clone", "-q", origin, bot, cwd=tmp.name)
        self.repo = GitRepo(bot, remote="origin")

    def git(self, *args: str, cwd: str | None = None) -> str:
        return _check_output(["git", "-C", cwd or self.dev, *args]).strip()

    def commit(self, message: str, files: FileChanges | None = None) -> str:
        if files is None:
            self.files += 1
            files = {f"file{self.files}.txt": f"{self.files}\n"}
        for name, content in files.items():
            if content is None:
                Path(self.dev, name).unlink()
            else:
                Path(self.dev, name).write_text(content)
        self.git("add", "-A")
        self.git("commit", "-q", "-m", message)
        return self.git("rev-parse", "HEAD")

    def add_commits(
        self, branch: str, count: int = 1, files: FileChanges | None = None
    ) -> str:
        self.git("checkout", "-q", branch)
        for _ in range(count):
            self.commit(f"Update {branch}", files)
        self.git("push", "-q", "origin", branch)
        return self.git("rev-parse", "HEAD")

    def push_branch(
        self, branch: str, base: str, count: int = 1, files: FileChanges | None = None
    ) -> str:
        self.git("branch", branch, base)
        return self.add_commits(branch, count, files)

    def push_stack(
        self,
        branches: tuple[str, ...] = ("user/a", "user/b", "user/c"),
        trunk: str = TRUNK,
        commits: int = 1,
        changes: tuple[FileChanges, ...] | None = None,
    ) -> NativeStack:
        entries: list[StackEntry] = []
        base = trunk
        for position, branch in enumerate(branches, start=1):
            files = None if changes is None else changes[position - 1]
            head = self.push_branch(branch, base, commits, files)
            entries.append(make_entry(position, 100 + position, branch, head, base))
            base = branch
        return NativeStack(base_ref=trunk, entries=tuple(entries))

    def landing_prs(
        self, stack: NativeStack, target: int
    ) -> list[tuple[StackEntry, str]]:
        return get_native_stack_landing_prs(
            self.repo, ORG, PROJECT, stack, target, TRUNK
        )

    def merge_into(self, branch: str, other: str) -> str:
        self.git("checkout", "-q", branch)
        self.git("merge", "-q", "--no-ff", "-m", f"Merge {other}", other)
        self.git("push", "-q", "origin", branch)
        return self.git("rev-parse", "HEAD")

    def rebase(self, branch: str, *args: str) -> str:
        self.git("rebase", "-q", *args, branch)
        self.git("push", "-q", "-f", "origin", branch)
        return self.git("rev-parse", "HEAD")

    def land(self, entry: StackEntry, body: str = "Summary") -> str:
        self.git("checkout", "-q", TRUNK)
        self.git("merge", "-q", "--squash", entry.head_ref)
        self.git("commit", "-q", "-m", landing_message(pr_url(entry.number), body))
        self.git("push", "-q", "origin", TRUNK)
        return self.git("rev-parse", "HEAD")

    def revert(self, entry: StackEntry, sha: str) -> str:
        self.git("checkout", "-q", TRUNK)
        self.git("revert", "--no-commit", sha)
        self.git("commit", "-q", "-m", revert_message(pr_url(entry.number), sha))
        self.git("push", "-q", "origin", TRUNK)
        return self.git("rev-parse", "HEAD")


class TestLandedIn(GitTestCase):
    def find(self, url: str) -> str | None:
        return _landed_in(GitRepo(self.dev), url, TRUNK, TRUNK)

    def test_finds_newest_landing_commit(self) -> None:
        landed = self.commit(landing_message(pr_url(1234)))
        self.commit(landing_message(pr_url(4321)))
        self.assertEqual(self.find(pr_url(1234)), landed)

    def test_pr_number_must_match_exactly(self) -> None:
        self.commit(landing_message(pr_url(12345)))
        self.assertIsNone(self.find(pr_url(1234)))
        self.assertIsNone(self.find(pr_url(123456)))

    def test_lines_quoted_in_a_body_are_not_landings(self) -> None:
        quoted = landing_message(pr_url(1234))
        self.commit(landing_message(pr_url(4321), quoted))
        self.assertIsNone(self.find(pr_url(1234)))
        landed = self.commit(landing_message(pr_url(1234)))
        quoting = self.commit(landing_message(pr_url(5678), quoted))
        self.assertEqual(self.find(pr_url(1234)), landed)
        self.assertEqual(self.find(pr_url(5678)), quoting)

    def test_only_lf_ends_a_line(self) -> None:
        # Co-authored-by lines, which follow the bot's lines, hold author names
        quoted = f"{PULL_REQUEST_RESOLVED}{pr_url(4321)}"
        for separator in ("\r", "\u2028"):
            with self.subTest(separator=separator):
                name = f"Name{separator}{quoted}{separator}Name"
                message = landing_message(pr_url(1234))
                coauthor = f"Co-authored-by: {name} <co@example.com>"
                landed = self.commit(f"{message}\n{coauthor}\n")
                self.assertEqual(self.find(pr_url(1234)), landed)
                self.assertIsNone(self.find(pr_url(4321)))

    def test_landed_then_reverted(self) -> None:
        landed = self.commit(landing_message(pr_url(1234)))
        self.commit(revert_message(pr_url(1234), landed))
        self.assertIsNone(self.find(pr_url(1234)))

    def test_landed_then_reverted_by_a_revert_pr_opened_by_hand(self) -> None:
        landed = self.commit(landing_message(pr_url(1234)))
        self.commit(manual_revert_message(landed))
        self.assertIsNone(self.find(pr_url(1234)))

    def test_relanded_after_revert(self) -> None:
        landed = self.commit(landing_message(pr_url(1234)))
        self.commit(revert_message(pr_url(1234), landed))
        relanded = self.commit(landing_message(pr_url(1234)))
        self.commit("Unrelated change")
        self.assertEqual(self.find(pr_url(1234)), relanded)

    def test_revert_line_needs_exactly_the_sha_at_the_start_of_a_line(self) -> None:
        landed = self.commit(landing_message(pr_url(1234)))
        lines = (
            f"This reverts commit {landed}0.\n"
            f"This reverts commit {landed[:10]}.\n"
            f"  This reverts commit {landed}.\n"
            f"> This reverts commit {landed}.\n"
        )
        self.commit(f"Revert\n\n{lines}")
        self.assertEqual(self.find(pr_url(1234)), landed)


class TestTrunkLandings(GitTestCase):
    def push_landing(self, number: int) -> str:
        landed = self.commit(landing_message(pr_url(number)))
        self.git("push", "-q", "origin", TRUNK)
        self.git("fetch", "-q", "origin", cwd=self.repo.repo_dir)
        return landed

    def read(self, number: int) -> tuple[list[str], list[str]]:
        """The landings of PR `number` on the bot's trunk, and the revisions that
        `git log` read to find them"""
        with mock.patch.object(self.repo, "_run_git", wraps=self.repo._run_git) as run:
            landings = _trunk_landings(self.repo, pr_url(number), TRUNK_REF)
        logs = [call.args for call in run.call_args_list if call.args[0] == "log"]
        return landings, [args[-2] for args in logs]

    def test_reads_the_history_once_then_only_the_commits_added_since(self) -> None:
        first = self.push_landing(101)
        self.assertEqual(self.read(101), ([first], [first]))
        other = self.push_landing(102)
        second = self.push_landing(101)
        self.assertEqual(self.read(101), ([second, first], [f"{first}..{second}"]))
        self.assertEqual(self.read(101), ([second, first], [f"{second}..{second}"]))
        self.assertEqual(self.read(102), ([other], [second]))


NOT_LANDED = r"PR #101 is closed but not landed on main, or it was reverted"
REBASE = re.escape("Rebase the stack onto main and try again.")


class TestGetNativeStackLandingPrs(GitTestCase):
    def assertRefused(self, stack: NativeStack, target: int, message: str) -> None:
        with self.assertRaisesRegex(NativeStackError, message):
            self.landing_prs(stack, target)

    def test_lands_open_entries_up_to_target(self) -> None:
        stack = self.push_stack()
        a, b, c = stack.entries
        self.assertEqual(
            self.landing_prs(stack, c.number),
            [(a, self.root), (b, a.head_oid), (c, b.head_oid)],
        )
        self.assertEqual(
            self.landing_prs(stack, b.number), [(a, self.root), (b, a.head_oid)]
        )
        self.assertEqual(self.landing_prs(stack, a.number), [(a, self.root)])

    def test_landed_entry_below_uses_live_branch_tip(self) -> None:
        stack = self.push_stack()
        self.land(stack.entries[0])
        a_head = self.merge_into("user/a", TRUNK)
        b_head = self.merge_into("user/b", "user/a")
        c_head = self.merge_into("user/c", "user/b")
        stack = with_entry(stack, 0, closed=True)
        stack = with_entry(with_entry(stack, 1, head_oid=b_head), 2, head_oid=c_head)
        _, b, c = stack.entries
        self.assertEqual(self.landing_prs(stack, c.number), [(b, a_head), (c, b_head)])

    def test_bottom_entry_merged_trunk_and_upper_entry_merged_it(self) -> None:
        stack = self.push_stack(("user/a", "user/b"))
        trunk = self.add_commits(TRUNK)
        a_head = self.merge_into("user/a", TRUNK)
        b_head = self.merge_into("user/b", "user/a")
        stack = with_entry(with_entry(stack, 0, head_oid=a_head), 1, head_oid=b_head)
        a, b = stack.entries
        self.assertEqual(self.landing_prs(stack, b.number), [(a, trunk), (b, a_head)])

    def test_refuses_unsupported_stacks_without_running_git(self) -> None:
        entries = (
            make_entry(1, 101, "user/a", "1" * 40, TRUNK),
            make_entry(2, 102, "user/b", "2" * 40, "user/a"),
            make_entry(3, 103, "user/c", "3" * 40, "user/b"),
        )
        stack = NativeStack(base_ref=TRUNK, entries=entries)
        other_trunk = replace(stack, base_ref=VIABLE_STRICT)
        ghstack = with_entry(stack, 0, head_ref="gh/someone/7/head")
        based = r"#103 is in a stack based on viable/strict, but only stacks based on"
        bottom = (
            r"PR #101 is at the bottom of the stack, so its base must be the stack's "
            r"trunk \(main\), not user/other"
        )
        middle = r"PR #{} targets {} instead of {}, the branch of PR #{} below it"
        mixed = (
            r"^PR #101 is a ghstack PR, and the bot does not support stacks that mix "
            r"ghstack PRs with other PRs; unstack the other PRs$"
        )
        below_open = r"^PR #102 is closed, but PR #101 below it is still open$"
        chains = (
            (0, "user/other", bottom),
            (1, "user/other", middle.format(102, "user/other", "user/a", 101)),
            (1, TRUNK, middle.format(102, TRUNK, "user/a", 101)),
            (2, "user/other", middle.format(103, "user/other", "user/b", 102)),
        )
        refused = [
            (with_entry(stack, index, base_ref=base), 103, message)
            for index, base, message in chains
        ]
        refused += [
            (other_trunk, 103, f"{based} main can be merged$"),
            (stack, 999, r"^PR #999 is not in the stack$"),
            (with_entry(stack, 2, closed=True), 103, r"^PR #103 is closed$"),
            (with_entry(stack, 1, closed=True), 103, below_open),
            (ghstack, 101, mixed),
            (with_entry(ghstack, 0, closed=True), 103, mixed),
        ]
        with mock.patch.object(self.repo, "_run_git", side_effect=AssertionError):
            for refused_stack, target, message in refused:
                with self.subTest(target=target, message=message):
                    self.assertRefused(refused_stack, target, message)

    def test_refuses_entry_below_target_landed_then_reverted(self) -> None:
        stack = self.push_stack()
        a, b, c = stack.entries
        closed = with_entry(stack, 0, closed=True)
        landed = self.land(a)
        self.assertEqual(
            self.landing_prs(closed, c.number), [(b, a.head_oid), (c, b.head_oid)]
        )
        self.revert(a, landed)
        self.assertRefused(closed, c.number, NOT_LANDED)

    def test_refuses_open_entry_that_already_landed(self) -> None:
        stack = self.push_stack()
        a, b, c = stack.entries
        for entry in (a, c):
            with self.subTest(landed=entry.number):
                landed = self.land(entry)
                message = rf"PR #{entry.number} already landed but is still open"
                self.assertRefused(stack, c.number, message)
                self.revert(entry, landed)
        self.assertEqual(
            self.landing_prs(stack, c.number),
            [(a, self.root), (b, a.head_oid), (c, b.head_oid)],
        )

    def test_refuses_landed_entry_branch_with_commits_that_never_landed(self) -> None:
        stack = self.push_stack(("user/a", "user/b"))
        self.land(stack.entries[0])
        extra = self.add_commits("user/a")
        b_head = self.rebase("user/b", "user/a")
        stack = with_entry(with_entry(stack, 0, closed=True), 1, head_oid=b_head)
        message = (
            "PR #101 is closed, but its branch has commits that never landed on "
            f"main: {extra}"
        )
        self.assertRefused(stack, 102, message)

    def test_refuses_deleted_branch_of_landed_entry(self) -> None:
        stack = self.push_stack(("user/a", "user/b"))
        a, b = stack.entries
        self.land(a)
        stack = with_entry(stack, 0, closed=True)
        self.assertEqual(self.landing_prs(stack, b.number), [(b, a.head_oid)])
        self.git("push", "-q", "origin", "--delete", a.head_ref)
        message = r"^Could not fetch main and the commits of PRs #101, #102: "
        self.assertRefused(stack, b.number, message)

    def test_refuses_lower_entry_updated_while_preparing_the_merge(self) -> None:
        stack = self.push_stack(("user/a",))
        self.push_branch("user/a-next", "user/a")
        b_head = self.push_branch("user/b", "user/a-next")
        b = make_entry(2, 102, "user/b", b_head, "user/a")
        stack = replace(stack, entries=(*stack.entries, b))
        self.git("push", "-q", "origin", "user/a-next:user/a")
        message = r"PR #101 was updated while preparing the merge"
        self.assertRefused(stack, b.number, message)

    def test_refuses_lower_entry_updated_after_upper_entry(self) -> None:
        stack = self.push_stack()
        stack = with_entry(stack, 0, head_oid=self.add_commits("user/a"))
        expected = rf"^PR #102 is not based on the tip of user/a\. {REBASE}$"
        for target in (102, 103):
            with self.subTest(target=target):
                self.assertRefused(stack, target, expected)

    def test_refuses_upper_entry_that_merged_trunk(self) -> None:
        stack = self.push_stack()
        a, b, _ = stack.entries
        self.add_commits(TRUNK)
        self.assertEqual(
            self.landing_prs(stack, b.number), [(a, self.root), (b, a.head_oid)]
        )
        stack = with_entry(stack, 1, head_oid=self.merge_into("user/b", TRUNK))
        expected = (
            r"^PR #102 contains main commits that user/a does not have\. "
            rf"{REBASE}$"
        )
        self.assertRefused(stack, b.number, expected)

    def merge_into_stack(self, stack: NativeStack, base: str) -> NativeStack:
        """Merges `base` into the branch of the bottom PR of `stack`, that branch
        into the next one and so on, as a rebase with merge commits does"""
        for index, entry in enumerate(stack.entries):
            head = self.merge_into(entry.head_ref, base)
            stack = with_entry(stack, index, head_oid=head)
            base = entry.head_ref
        return stack

    def test_refuses_bottom_pr_whose_merge_base_has_its_landing(self) -> None:
        stack = self.push_stack()
        a, _, c = stack.entries
        landed = self.land(a)
        self.revert(a, landed)
        # As a rebase onto a viable/strict that has the landing but not the revert
        stack = self.merge_into_stack(stack, landed)
        message = (
            rf"^PR #101 landed in {landed}, which its merge base with main still "
            rf"has, so its changes would not land again\. {REBASE}$"
        )
        self.assertRefused(stack, c.number, message)
        stack = self.merge_into_stack(stack, TRUNK)
        landing = [entry.number for entry, _ in self.landing_prs(stack, c.number)]
        self.assertEqual(landing, [101, 102, 103])

    def test_refuses_upper_pr_whose_base_has_its_landing(self) -> None:
        stack = self.push_stack()
        _, b, c = stack.entries
        landed = self.land(b)
        self.revert(b, landed)
        stack = self.merge_into_stack(stack, landed)
        message = (
            rf"^PR #102 landed in {landed}, which user/a still has, so its changes "
            rf"would not land again\. {REBASE}$"
        )
        for target in (102, 103):
            with self.subTest(target=target):
                self.assertRefused(stack, target, message)
        stack = self.merge_into_stack(stack, TRUNK)
        landing = [entry.number for entry, _ in self.landing_prs(stack, c.number)]
        self.assertEqual(landing, [101, 102, 103])


VIABLE_STRICT = "viable/strict"


if __name__ == "__main__":
    main()
