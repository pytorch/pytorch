#!/usr/bin/env python3
from __future__ import annotations

import io
import os
import re
import shutil
import tempfile
from contextlib import contextmanager, redirect_stdout
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from typing import Any, TYPE_CHECKING
from unittest import main, mock, TestCase

from github_utils import GHGraphQLError
from gitutils import _check_output, GitRepo
from native_stack import (
    _trunk_landings,
    build_native_stack_commits,
    build_native_stack_rebase,
    find_landed_commit,
    find_stack_dependents,
    get_native_stack,
    get_native_stack_landing_prs,
    GH_GET_PR_STACK_QUERY,
    NativeStack,
    NativeStackError,
    PULL_REQUEST_RESOLVED,
    push_branches,
    stack_dependencies_line,
    StackEntry,
)


if TYPE_CHECKING:
    from collections.abc import Iterator


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
BOT_NAME = "PyTorch MergeBot"
BOT_EMAIL = "pytorchmergebot@users.noreply.github.com"
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


def author(number: int) -> str:
    return f"Author {number} <author{number}@example.com>"


def make_entry(
    position: int,
    number: int,
    head_ref: str,
    head_oid: str,
    base_ref: str,
    closed: bool = False,
    cross_repo: bool = False,
) -> StackEntry:
    return StackEntry(
        position=position,
        number=number,
        closed=closed,
        is_cross_repository=cross_repo,
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
            "isCrossRepository": entry.is_cross_repository,
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
        "number": 197570,
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
    make_entry(2, 198534, "user/inductor", "2" * 40, "user/capture", cross_repo=True),
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
            NativeStack(number=197570, base_ref=TRUNK, entries=GQL_ENTRIES),
        )
        gql.assert_called_once_with(
            GH_GET_PR_STACK_QUERY, owner=ORG, name=PROJECT, number=198534
        )

    def test_sorts_entries_by_position(self, gql: Any) -> None:
        gql.return_value = gql_stack([GQL_NODES[i] for i in (2, 0, 1)])
        self.assertEqual(
            get_native_stack(ORG, PROJECT, 198534),
            NativeStack(number=197570, base_ref=TRUNK, entries=GQL_ENTRIES),
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
            "isCrossRepository",
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
        return NativeStack(number=100, base_ref=trunk, entries=tuple(entries))

    def landing_prs(
        self, stack: NativeStack, target: int, default_branch: str = TRUNK
    ) -> list[tuple[StackEntry, str]]:
        return get_native_stack_landing_prs(
            self.repo, ORG, PROJECT, stack, target, default_branch
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

    def revert_manually(self, sha: str) -> str:
        self.git("checkout", "-q", TRUNK)
        self.git("revert", "--no-commit", sha)
        self.git("commit", "-q", "-m", manual_revert_message(sha))
        self.git("push", "-q", "origin", TRUNK)
        return self.git("rev-parse", "HEAD")


class TestFindLandedCommit(GitTestCase):
    def find(self, url: str, ref: str = TRUNK) -> str | None:
        return find_landed_commit(GitRepo(self.dev), url, ref)

    def test_finds_newest_landing_commit(self) -> None:
        landed = self.commit(landing_message(pr_url(1234)))
        self.commit(landing_message(pr_url(4321)))
        self.assertEqual(self.find(pr_url(1234)), landed)

    def test_no_landing_commit(self) -> None:
        self.commit(landing_message(pr_url(4321)))
        self.assertIsNone(self.find(pr_url(1234)))

    def test_pr_number_must_match_exactly(self) -> None:
        self.commit(landing_message(pr_url(12345)))
        self.assertIsNone(self.find(pr_url(1234)))
        self.assertIsNone(self.find(pr_url(123456)))

    def test_url_dots_are_literal(self) -> None:
        self.commit(landing_message(pr_url(1234, host="githubXcom")))
        self.assertIsNone(self.find(pr_url(1234)))

    def test_whole_line_must_match(self) -> None:
        self.commit(f"Change\n\nSee {PULL_REQUEST_RESOLVED}{pr_url(1234)}\n")
        self.assertIsNone(self.find(pr_url(1234)))

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

    def test_line_quoted_after_a_revert_is_not_a_reland(self) -> None:
        landed = self.commit(landing_message(pr_url(1234)))
        self.commit(revert_message(pr_url(1234), landed))
        self.commit(landing_message(pr_url(4321), landing_message(pr_url(1234))))
        self.assertIsNone(self.find(pr_url(1234)))

    def test_relanded_after_revert(self) -> None:
        landed = self.commit(landing_message(pr_url(1234)))
        self.commit(revert_message(pr_url(1234), landed))
        relanded = self.commit(landing_message(pr_url(1234)))
        self.commit("Unrelated change")
        self.assertEqual(self.find(pr_url(1234)), relanded)

    def test_reverts_of_other_prs_are_ignored(self) -> None:
        landed = self.commit(landing_message(pr_url(1234)))
        for url in (pr_url(12345), pr_url(1234, host="githubXcom")):
            other = self.commit(landing_message(url))
            self.commit(revert_message(url, other))
        self.assertEqual(self.find(pr_url(1234)), landed)

    def test_revert_lines_without_the_landed_commit_do_not_revert(self) -> None:
        landed = self.commit(landing_message(pr_url(1234)))
        other = self.commit(landing_message(pr_url(4321)))
        self.commit(revert_message(pr_url(1234), other))
        line = f"Reverted {pr_url(1234)} on behalf of https://github.com/reverter"
        self.commit(landing_message(pr_url(5678), line))
        self.assertEqual(self.find(pr_url(1234)), landed)
        self.assertIsNone(self.find(pr_url(4321)))

    def test_revert_line_may_go_on_after_the_sha(self) -> None:
        rests = {1234: ", which was wrong", 1235: " because it broke trunk.", 1236: ""}
        for number, rest in rests.items():
            landed = self.commit(landing_message(pr_url(number)))
            self.commit(f"Revert\n\nThis reverts commit {landed}{rest}\n")
            self.assertIsNone(self.find(pr_url(number)))

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

    def test_only_commits_reachable_from_ref_count(self) -> None:
        landed = self.commit(landing_message(pr_url(1234)))
        self.git("checkout", "-q", "-b", "reverted")
        self.commit(revert_message(pr_url(1234), landed))
        self.git("checkout", "-q", "-b", "other", TRUNK)
        other = self.commit(landing_message(pr_url(4321)))
        self.assertEqual(self.find(pr_url(1234)), landed)
        self.assertIsNone(self.find(pr_url(1234), ref="reverted"))
        self.assertIsNone(self.find(pr_url(4321)))
        self.assertEqual(self.find(pr_url(4321), ref="other"), other)

    def test_ref_named_like_a_file(self) -> None:
        landed = self.commit(landing_message(pr_url(1234)))
        Path(self.dev, TRUNK).write_text("untracked\n")
        self.assertEqual(self.find(pr_url(1234)), landed)

    def test_rejects_output_not_starting_with_a_sha(self) -> None:
        repo = GitRepo(self.dev)
        output = f"warning: refname '{TRUNK}' is ambiguous.\n{'a' * 40}\nChange\n"
        message = rf"^Unexpected output from git log on {TRUNK}"
        with mock.patch.object(repo, "_run_git", return_value=output):
            with self.assertRaisesRegex(RuntimeError, message):
                find_landed_commit(repo, pr_url(1234), TRUNK)


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

    def test_reads_the_whole_history_again_after_a_forced_update(self) -> None:
        landed = self.push_landing(101)
        self.assertEqual(self.read(101), ([landed], [landed]))
        self.git("push", "-q", "-f", "origin", f"{self.root}:{TRUNK}")
        self.git("fetch", "-q", "origin", cwd=self.repo.repo_dir)
        self.assertEqual(self.read(101), ([], [self.root]))

    def test_each_repository_reads_its_own_history(self) -> None:
        landed = self.push_landing(101)
        self.assertEqual(self.read(101), ([landed], [landed]))
        self.repo = GitRepo(self.repo.repo_dir, remote="origin")
        self.assertEqual(self.read(101), ([landed], [landed]))


NOT_LANDED = r"PR #101 is closed but not landed on main, or it was reverted"
REBASE = re.escape(
    "Rebase the stack onto main with `@pytorchbot rebase -b main` and try again."
)


class TestGetNativeStackLandingPrs(GitTestCase):
    def assertRefused(
        self,
        stack: NativeStack,
        target: int,
        message: str,
        default_branch: str = TRUNK,
    ) -> str:
        with self.assertRaisesRegex(NativeStackError, message) as refusal:
            self.landing_prs(stack, target, default_branch)
        return str(refusal.exception)

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

    def test_trunk_moved_forward(self) -> None:
        stack = self.push_stack()
        self.add_commits(TRUNK, 2)
        a, b, c = stack.entries
        self.assertEqual(
            self.landing_prs(stack, c.number),
            [(a, self.root), (b, a.head_oid), (c, b.head_oid)],
        )

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

    def test_landed_entry_branch_reset_to_trunk(self) -> None:
        stack = self.push_stack(("user/a", "user/b"))
        a = stack.entries[0]
        landed = self.land(a)
        # GitHub keeps a closed PR's head reachable from refs/pull/<number>/head.
        self.git("push", "-q", "origin", f"{a.head_oid}:refs/pull/{a.number}/head")
        self.git("push", "-q", "-f", "origin", f"{TRUNK}:{a.head_ref}")
        b_head = self.rebase("user/b", "--onto", TRUNK, a.head_oid)
        stack = with_entry(with_entry(stack, 0, closed=True), 1, head_oid=b_head)
        b = stack.entries[1]
        self.assertEqual(self.landing_prs(stack, b.number), [(b, landed)])

    def test_skips_several_landed_entries(self) -> None:
        stack = self.push_stack()
        a, b, c = stack.entries
        self.land(a)
        self.land(b)
        stack = with_entry(with_entry(stack, 0, closed=True), 1, closed=True)
        self.assertEqual(self.landing_prs(stack, c.number), [(c, b.head_oid)])

    def test_bottom_entry_merged_trunk_and_upper_entry_merged_it(self) -> None:
        stack = self.push_stack(("user/a", "user/b"))
        trunk = self.add_commits(TRUNK)
        a_head = self.merge_into("user/a", TRUNK)
        b_head = self.merge_into("user/b", "user/a")
        stack = with_entry(with_entry(stack, 0, head_oid=a_head), 1, head_oid=b_head)
        a, b = stack.entries
        self.assertEqual(self.landing_prs(stack, b.number), [(a, trunk), (b, a_head)])

    def test_cross_repo_and_ghstack_entries_above_target_are_ignored(self) -> None:
        stack = self.push_stack(("user/a", "user/b", "gh/someone/7/head"))
        stack = with_entry(stack, 2, is_cross_repository=True)
        a, b, _ = stack.entries
        self.assertEqual(
            self.landing_prs(stack, b.number), [(a, self.root), (b, a.head_oid)]
        )

    def test_closed_entry_above_target_is_ignored(self) -> None:
        stack = with_entry(self.push_stack(), 2, closed=True)
        a, b, _ = stack.entries
        self.assertEqual(
            self.landing_prs(stack, b.number), [(a, self.root), (b, a.head_oid)]
        )

    def test_refuses_trunk_other_than_default_branch(self) -> None:
        self.git("branch", "viable/strict", TRUNK)
        self.git("push", "-q", "origin", "viable/strict")
        stack = self.push_stack(trunk="viable/strict")
        a, b, c = stack.entries
        self.assertEqual(
            self.landing_prs(stack, c.number, default_branch="viable/strict"),
            [(a, self.root), (b, a.head_oid), (c, b.head_oid)],
        )
        message = (
            r"PR #103 is in a stack based on viable/strict, but only stacks based on "
            r"main can be merged"
        )
        self.assertRefused(stack, c.number, message)

    def test_refuses_target_not_in_stack(self) -> None:
        self.assertRefused(self.push_stack(), 999, r"PR #999 is not in the stack")

    def test_refuses_closed_target(self) -> None:
        stack = self.push_stack()
        message = r"^PR #103 is closed$"
        self.assertRefused(with_entry(stack, 2, closed=True), 103, message)
        for entry in stack.entries:
            self.land(entry)
        landed = tuple(replace(entry, closed=True) for entry in stack.entries)
        self.assertRefused(replace(stack, entries=landed), 103, message)

    def test_refuses_cross_repo_entry_at_or_below_target(self) -> None:
        stack = self.push_stack()
        for index, entry in enumerate(stack.entries):
            with self.subTest(index=index):
                cross_repo = with_entry(stack, index, is_cross_repository=True)
                message = rf"PR #{entry.number} is from a fork"
                self.assertRefused(cross_repo, 103, message)

    def test_refuses_broken_chain(self) -> None:
        self.git("branch", "user/other", TRUNK)
        self.git("push", "-q", "origin", "user/other")
        stack = self.push_stack()
        bottom = (
            r"PR #101 is at the bottom of the stack, so its base must be the stack's "
            r"trunk \(main\), not user/other"
        )
        middle = r"PR #{} targets {} instead of {}, the branch of PR #{} below it"
        broken = (
            (0, "user/other", bottom),
            (1, "user/other", middle.format(102, "user/other", "user/a", 101)),
            (1, TRUNK, middle.format(102, TRUNK, "user/a", 101)),
            (2, "user/other", middle.format(103, "user/other", "user/b", 102)),
        )
        for index, base, message in broken:
            with self.subTest(index=index, base=base):
                chain = with_entry(stack, index, base_ref=base)
                self.assertRefused(chain, 103, message)

    def test_refuses_closed_entry_below_target_that_did_not_land(self) -> None:
        stack = with_entry(self.push_stack(), 0, closed=True)
        self.assertRefused(stack, 103, NOT_LANDED)

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

    def test_refuses_open_entry_below_closed_entry(self) -> None:
        stack = self.push_stack()
        self.land(stack.entries[1])
        message = r"PR #102 is closed, but PR #101 below it is still open"
        self.assertRefused(with_entry(stack, 1, closed=True), 103, message)

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

    def test_refuses_ghstack_entry_at_or_below_target(self) -> None:
        stack = self.push_stack(("gh/someone/7/head", "user/b", "user/c"))
        message = (
            r"^PR #101 is a ghstack PR, and the bot does not support stacks that mix "
            r"ghstack PRs with other PRs; unstack the other PRs$"
        )
        self.assertRefused(stack, 101, message)
        self.land(stack.entries[0])
        self.assertRefused(with_entry(stack, 0, closed=True), 103, message)

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

    def test_relanded_pr_counts_its_newest_landing_in_its_base(self) -> None:
        stack = self.push_stack()
        a, _, c = stack.entries
        self.revert(a, self.land(a))
        relanded = self.land(a)
        stack = self.merge_into_stack(stack, relanded)
        self.revert(a, relanded)
        message = rf"^PR #101 landed in {relanded}, which its merge base"
        self.assertRefused(stack, c.number, message)
        stack = self.merge_into_stack(stack, TRUNK)
        self.assertEqual(len(self.landing_prs(stack, c.number)), 3)


class TestBuildNativeStackCommits(GitTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.origin = self.git("remote", "get-url", "origin")
        self.bot_git("config", "user.name", BOT_NAME)
        self.bot_git("config", "user.email", BOT_EMAIL)

    def bot_git(self, *args: str) -> str:
        return self.git(*args, cwd=self.repo.repo_dir)

    def repo_state(self) -> list[str]:
        state = [
            self.bot_git(*args)
            for args in (
                ("symbolic-ref", "HEAD"),
                ("rev-parse", "HEAD"),
                ("for-each-ref",),
                ("ls-files", "--stage"),
                ("status", "--porcelain", "--untracked-files=all"),
            )
        ]
        return state + [self.git("for-each-ref", cwd=self.origin)]

    def build(
        self,
        stack: NativeStack,
        target: int,
        commits: list[tuple[str, str]] | None = None,
    ) -> list[str]:
        """Builds the commits landing `target` and returns them oldest first, after
        checking that they form a linear history on trunk and that neither the bot
        clone (HEAD, refs, index, worktree) nor the origin changed."""
        landing = self.landing_prs(stack, target)
        base = self.repo.rev_parse(TRUNK_REF)
        if commits is None:
            commits = [
                (author(entry.number), landing_message(pr_url(entry.number)))
                for entry, _ in landing
            ]
        before = self.repo_state()
        try:
            # trymerge.yml only sets user.name and user.email; these variables would
            # override them and an author given through git config.
            with mock.patch.dict(os.environ):
                for var in GIT_ENV:
                    if var.startswith(("GIT_AUTHOR_", "GIT_COMMITTER_")):
                        del os.environ[var]
                last = build_native_stack_commits(self.repo, base, landing, commits)
        finally:
            self.assertEqual(self.repo_state(), before)
        built = [last]
        for _ in landing:
            parents = self.bot_git("log", "-1", "--format=%P", built[0]).split()
            self.assertEqual(len(parents), 1, f"{built[0]} has parents {parents}")
            built.insert(0, parents[0])
        self.assertEqual(built.pop(0), base)
        return built

    def tree(self, rev: str) -> str:
        return self.bot_git("rev-parse", f"{rev}^{{tree}}")

    def files_at(self, rev: str) -> dict[str, str]:
        lines = self.bot_git("ls-tree", "-r", rev).splitlines()
        return {path: meta.split()[2] for meta, path in (x.split("\t") for x in lines)}

    def show(self, rev: str, path: str) -> str:
        return self.repo._run_git("show", f"{rev}:{path}")

    def message(self, commit: str) -> str:
        return self.repo._run_git("cat-file", "commit", commit).partition("\n\n")[2]

    def diffs(self, *revs: str) -> list[str]:
        return [self.bot_git("diff", old, new) for old, new in pairwise(revs)]

    def assertTrunkPlusHeads(self, built: list[str], heads: list[str]) -> None:
        trunk = self.files_at(TRUNK_REF)
        self.assertEqual(
            [self.files_at(commit) for commit in built],
            [{**trunk, **self.files_at(head)} for head in heads],
        )

    def test_reproduces_heads_on_unmoved_trunk(self) -> None:
        stack = self.push_stack()
        for count in (1, 2, 3):
            with self.subTest(target=100 + count):
                built = self.build(stack, 100 + count)
                heads = [entry.head_oid for entry in stack.entries[:count]]
                self.assertEqual(
                    [self.tree(commit) for commit in built],
                    [self.tree(head) for head in heads],
                )

    def test_applies_prs_on_top_of_moved_trunk(self) -> None:
        stack = self.push_stack(commits=2)
        self.add_commits(TRUNK, 2)
        built = self.build(stack, 103)
        self.assertTrunkPlusHeads(built, [entry.head_oid for entry in stack.entries])

    def test_lands_only_prs_above_landed_pr(self) -> None:
        stack = self.push_stack(commits=2)
        a, b, c = stack.entries
        self.land(a)
        built = self.build(with_entry(stack, 0, closed=True), 103)
        expected = self.diffs(a.head_oid, b.head_oid, c.head_oid)
        self.assertEqual(self.diffs(TRUNK_REF, *built), expected)

    def test_keeps_trunk_edits_to_files_of_landed_pr(self) -> None:
        stack = self.push_stack(
            ("user/a", "user/b"), changes=({"shared.txt": "a\n"}, {"b.txt": "b\n"})
        )
        a, b = stack.entries
        self.land(a)
        self.add_commits(TRUNK, files={"shared.txt": "a\nfixed on trunk\n"})
        built = self.build(with_entry(stack, 0, closed=True), b.number)
        expected = self.diffs(a.head_oid, b.head_oid)
        self.assertEqual(self.diffs(TRUNK_REF, *built), expected)

    def test_upper_pr_undoes_part_of_lower_pr(self) -> None:
        stack = self.push_stack(
            ("user/a", "user/b"),
            changes=(
                {"shared.txt": "keep\ndrop\n", "gone.txt": "gone\n"},
                {"shared.txt": "keep\n", "gone.txt": None},
            ),
        )
        self.add_commits(TRUNK)
        built = self.build(stack, 102)
        self.assertTrunkPlusHeads(built, [entry.head_oid for entry in stack.entries])
        self.assertEqual(self.show(built[1], "shared.txt"), "keep\n")

    def test_merges_trunk_and_pr_edits_to_same_file(self) -> None:
        self.add_commits(TRUNK, files={"shared.txt": "1\n2\n3\n4\n5\n6\n7\n"})
        stack = self.push_stack(
            ("user/a", "user/b"),
            changes=(
                {"shared.txt": "A\n2\n3\n4\n5\n6\n7\n"},
                {"shared.txt": "A\n2\n3\nB\n5\n6\n7\n"},
            ),
        )
        self.add_commits(TRUNK, files={"shared.txt": "1\n2\n3\n4\n5\n6\nT\n"})
        built = self.build(stack, 102)
        self.assertEqual(
            [self.show(commit, "shared.txt") for commit in built],
            ["A\n2\n3\n4\n5\n6\nT\n", "A\n2\n3\nB\n5\n6\nT\n"],
        )

    def test_lower_branches_advanced_by_merge_commits(self) -> None:
        stack = self.push_stack(("user/a", "user/b"))
        self.add_commits(TRUNK)
        a_head = self.merge_into("user/a", TRUNK)
        b_head = self.merge_into("user/b", "user/a")
        self.add_commits(TRUNK)
        stack = with_entry(with_entry(stack, 0, head_oid=a_head), 1, head_oid=b_head)
        self.assertTrunkPlusHeads(self.build(stack, 102), [a_head, b_head])

    def test_landed_lower_branch_advanced_by_merge_commits(self) -> None:
        stack = self.push_stack(("user/a", "user/b"))
        self.land(stack.entries[0])
        self.merge_into("user/a", TRUNK)
        b_head = self.merge_into("user/b", "user/a")
        self.add_commits(TRUNK)
        stack = with_entry(with_entry(stack, 0, closed=True), 1, head_oid=b_head)
        self.assertTrunkPlusHeads(self.build(stack, 102), [b_head])

    def test_refuses_pr_conflicting_with_trunk(self) -> None:
        stack = self.push_stack(
            ("user/a", "user/b"), changes=({"file1.txt": "a\n"}, {"b.txt": "b\n"})
        )
        self.add_commits(TRUNK, files={"file1.txt": "trunk\n"})
        with self.assertRaisesRegex(NativeStackError, r"#101\b"):
            self.build(stack, 102)

    def test_refuses_empty_bottom_pr(self) -> None:
        stack = self.push_stack(("user/a", "user/b"), commits=0)
        with self.assertRaisesRegex(NativeStackError, r"#101\b"):
            self.build(stack, 102)

    def test_refuses_pr_already_on_trunk(self) -> None:
        stack = self.push_stack()
        a, b, _ = stack.entries
        self.git("checkout", "-q", TRUNK)
        for entry in (b, a):
            self.git("cherry-pick", entry.head_oid)
            self.git("push", "-q", "origin", TRUNK)
            with self.subTest(on_trunk=entry.number):
                with self.assertRaisesRegex(NativeStackError, rf"#{entry.number}\b"):
                    self.build(stack, 103)

    def test_sets_author_committer_and_message(self) -> None:
        stack = self.push_stack()
        commits = [
            ("Ren\u00e9e O'Brien <renee@example.com>", landing_message(pr_url(101))),
            (
                author(102),
                f"{landing_message(pr_url(102))}{stack_dependencies_line([101])}\n",
            ),
            (
                author(103),
                "Change (#103)\n\n## Test plan\n\n```\npython test.py\n```\n\n"
                f"{PULL_REQUEST_RESOLVED}{pr_url(103)}\n"
                f"{stack_dependencies_line([101, 102])}\n",
            ),
        ]
        built = self.build(stack, 103, commits)
        for commit, (name, message) in zip(built, commits, strict=True):
            people = self.bot_git("log", "-1", "--format=%an <%ae>%n%cn <%ce>", commit)
            self.assertEqual(people.splitlines(), [name, f"{BOT_NAME} <{BOT_EMAIL}>"])
            self.assertEqual(self.message(commit), message)

    def test_cleans_up_message_like_git_commit(self) -> None:
        stack = self.push_stack(("user/a",))
        message = (
            "Change (#101)\r\n\r\n## Summary\r\nDetails   \r\n\r\n\r\n\r\n"
            f"{PULL_REQUEST_RESOLVED}{pr_url(101)}\r\n\r\n"
            "Co-authored-by: Author 2 <author2@example.com>"
        )
        built = self.build(stack, 101, [(author(101), message)])
        self.assertEqual(
            [self.message(commit) for commit in built],
            [
                "Change (#101)\n\n## Summary\nDetails\n\n"
                f"{PULL_REQUEST_RESOLVED}{pr_url(101)}\n\n"
                "Co-authored-by: Author 2 <author2@example.com>\n"
            ],
        )

    def test_keeps_carriage_returns_inside_lines(self) -> None:
        stack = self.push_stack(("user/a",))
        name = f"Name\r{PULL_REQUEST_RESOLVED}{pr_url(4321)}\rName"
        message = landing_message(pr_url(101))
        message += f"\nCo-authored-by: {name} <co@example.com>\n"
        built = self.build(stack, 101, [(author(101), message)])
        self.assertEqual([self.message(commit) for commit in built], [message])
        landed = find_landed_commit(self.repo, pr_url(101), built[0])
        self.assertEqual(landed, built[0])

    def test_logs_git_commands_in_debug_mode(self) -> None:
        stack = self.push_stack(("user/a",))
        self.repo.debug = True
        with redirect_stdout(io.StringIO()) as out:
            self.build(stack, 101)
        prefix = f"+ git -C {self.repo.repo_dir} "
        logged = [x for x in out.getvalue().splitlines() if x.startswith(prefix)]
        commands = {line.removeprefix(prefix).split()[0] for line in logged}
        self.assertLessEqual({"merge-tree", "stripspace", "commit-tree"}, commands)

    def test_failed_git_command_is_printed_with_its_repository(self) -> None:
        stack = self.push_stack(("user/a",))
        landing = self.landing_prs(stack, 101)
        commits = [(author(101), landing_message(pr_url(101)))]
        with redirect_stdout(io.StringIO()) as out:
            with self.assertRaises(RuntimeError) as failure:
                build_native_stack_commits(self.repo, "0" * 40, landing, commits)
        command = f"Command `git -C {self.repo.repo_dir} merge-tree --write-tree "
        self.assertTrue(str(failure.exception).startswith(command), failure.exception)
        self.assertIn("stderr: \n", out.getvalue())

    def test_leaves_dirty_worktree_and_index_alone(self) -> None:
        stack = self.push_stack()
        bot = Path(self.repo.repo_dir)
        (bot / "file1.txt").write_text("local edit\n")
        (bot / "staged.txt").write_text("staged\n")
        self.bot_git("add", "staged.txt")
        (bot / "untracked.txt").write_text("untracked\n")
        self.build(stack, 103)
        self.assertEqual((bot / "file1.txt").read_text(), "local edit\n")


class TestFindStackDependents(GitTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.landings: dict[int, str] = {}

    def land_pr(self, number: int, *deps: int) -> str:
        message = landing_message(pr_url(number))
        if deps:
            message += f"{stack_dependencies_line(list(deps))}\n"
        self.landings[number] = self.commit(message)
        return self.landings[number]

    def revert_pr(self, number: int) -> str:
        return self.commit(revert_message(pr_url(number), self.landings[number]))

    def land_stack(self) -> tuple[str, str, str]:
        return self.land_pr(101), self.land_pr(102, 101), self.land_pr(103, 101, 102)

    def found(self, number: int, ref: str = TRUNK) -> tuple[str | None, list[Any]]:
        return find_stack_dependents(GitRepo(self.dev), ORG, PROJECT, number, ref)

    def dependents(self, number: int, ref: str = TRUNK) -> list[tuple[str, int]]:
        return self.found(number, ref)[1]

    def test_bottom_pr_has_every_pr_above_it_top_first(self) -> None:
        a, b, c = self.land_stack()
        self.assertEqual(self.found(101), (a, [(c, 103), (b, 102)]))

    def test_reverts_of_reverts_reapply_their_commit(self) -> None:
        a, b, c = self.land_stack()

        def revert(sha: str) -> str:
            return self.commit(f"Revert\n\nThis reverts commit {sha}.\n")

        revert(revert(b))
        self.assertEqual(self.found(101), (a, [(c, 103), (b, 102)]))
        reverted = revert(a)
        self.assertEqual(self.found(101), (None, []))
        reapplied = revert(reverted)
        self.assertEqual(self.found(101), (a, [(c, 103), (b, 102)]))
        # Merges and rebases still read a reverted revert as a revert
        self.assertIsNone(find_landed_commit(GitRepo(self.dev), pr_url(101), TRUNK))
        revert(reapplied)
        self.assertEqual(self.found(101), (None, []))

    def test_dependents_are_ordered_by_their_position_in_the_stack(self) -> None:
        _, _, c = self.land_stack()
        self.revert_pr(102)
        b = self.land_pr(102, 101)
        self.assertEqual(self.dependents(101), [(c, 103), (b, 102)])

    def test_dependents_at_the_same_position_are_newest_first(self) -> None:
        _, b, c = self.land_stack()
        d = self.land_pr(104, 101)
        self.assertEqual(self.dependents(101), [(c, 103), (d, 104), (b, 102)])

    def test_middle_pr_has_only_the_prs_above_it(self) -> None:
        _, _, c = self.land_stack()
        self.assertEqual(self.dependents(102), [(c, 103)])
        self.assertEqual(self.dependents(103), [])

    def test_skips_commits_of_other_prs_and_stacks(self) -> None:
        self.land_pr(101)
        self.land_pr(200)
        b = self.land_pr(102, 101)
        self.land_pr(300)
        other = self.land_pr(301, 300)
        c = self.land_pr(103, 101, 102)
        self.commit("Unrelated change")
        self.assertEqual(self.dependents(101), [(c, 103), (b, 102)])
        self.assertEqual(self.dependents(300), [(other, 301)])

    def test_skips_dependents_reverted_since(self) -> None:
        _, _, c = self.land_stack()
        self.land_pr(104, 101, 102, 103)
        self.revert_pr(104)
        self.revert_pr(102)
        self.assertEqual(self.dependents(101), [(c, 103)])
        self.assertEqual(self.dependents(103), [])

    def test_later_commit_quoting_a_dependent_does_not_replace_it(self) -> None:
        self.land_pr(101)
        b = self.land_pr(102, 101)
        quoted = f"{landing_message(pr_url(102))}{stack_dependencies_line([101])}"
        self.commit(landing_message(pr_url(500), quoted))
        self.assertEqual(self.dependents(101), [(b, 102)])
        self.assertEqual(find_landed_commit(GitRepo(self.dev), pr_url(102), TRUNK), b)

    def test_stack_dependencies_line_quoted_in_a_body_does_not_count(self) -> None:
        self.land_pr(101)
        self.land_pr(300)
        quoted = stack_dependencies_line([101])
        self.commit(landing_message(pr_url(600), f"Landed commits end with\n{quoted}"))
        message = landing_message(pr_url(301), quoted)
        on_300 = self.commit(f"{message}{stack_dependencies_line([300])}\n")
        self.assertEqual(self.dependents(101), [])
        self.assertEqual(self.dependents(300), [(on_300, 301)])

    def test_dependent_relanded_after_a_revert_counts_at_its_reland(self) -> None:
        _, b, _ = self.land_stack()
        self.revert_pr(103)
        self.land_pr(200)
        c = self.land_pr(103, 101, 102)
        self.assertEqual(self.dependents(101), [(c, 103), (b, 102)])

    def test_dependent_relanded_outside_the_stack_is_not_a_dependent(self) -> None:
        _, b, _ = self.land_stack()
        self.revert_pr(103)
        self.land_pr(103)
        self.assertEqual(self.dependents(101), [(b, 102)])
        self.assertEqual(self.dependents(102), [])

    def test_pr_numbers_in_stack_dependencies_must_match_exactly(self) -> None:
        for number in (10, 101, 1011):
            self.land_pr(number)
        on_10 = self.land_pr(201, 10)
        on_101 = self.land_pr(202, 101)
        on_1011 = self.land_pr(203, 1011)
        on_all = self.land_pr(204, 10, 101, 1011)
        self.assertEqual(self.dependents(10), [(on_all, 204), (on_10, 201)])
        self.assertEqual(self.dependents(101), [(on_all, 204), (on_101, 202)])
        self.assertEqual(self.dependents(1011), [(on_all, 204), (on_1011, 203)])

    def test_pr_numbers_in_urls_must_match_exactly(self) -> None:
        self.land_pr(101)
        self.land_pr(103, 10)
        self.land_pr(10)
        dependent = self.land_pr(102, 10)
        self.revert_pr(101)
        self.land_pr(1021)
        self.revert_pr(1021)
        self.assertEqual(self.dependents(10), [(dependent, 102)])

    def test_dependent_needs_an_exact_pull_request_resolved_line(self) -> None:
        self.land_pr(101)
        for message in (
            landing_message(f"https://github.com/{ORG}/vision/pull/102"),
            landing_message(pr_url(103, host="githubXcom")),
            f"Change\n\nSee {PULL_REQUEST_RESOLVED}{pr_url(104)}\n",
        ):
            self.commit(f"{message}{stack_dependencies_line([101])}\n")
        self.assertEqual(self.dependents(101), [])

    def test_only_lf_ends_a_line(self) -> None:
        self.land_pr(101)
        self.land_pr(300)
        dependents: list[tuple[str, int]] = []
        for number, separator in ((102, "\r"), (103, "\u2028")):
            # A co-author named like the bot's lines, which it follows
            resolved = f"{PULL_REQUEST_RESOLVED}{pr_url(number)}"
            on_300 = stack_dependencies_line([300])
            name = separator.join(("Name", resolved, on_300, "Name"))
            message = landing_message(pr_url(number), on_300)
            message += f"{stack_dependencies_line([101])}\n\n"
            message += f"Co-authored-by: {name} <co@example.com>\n"
            dependents.insert(0, (self.commit(message), number))
        self.assertEqual(self.dependents(101), dependents)
        self.assertEqual(self.dependents(300), [])

    def test_dependent_body_quoting_another_pr_resolves_to_its_own_pr(self) -> None:
        self.land_pr(101)
        quoted = landing_message(pr_url(90))
        dependent = self.commit(
            f"Reland (#102)\n\n{quoted}\n{PULL_REQUEST_RESOLVED}{pr_url(102)}\n"
            f"{stack_dependencies_line([101])}\n"
        )
        self.assertEqual(self.dependents(101), [(dependent, 102)])

    def test_only_a_stack_dependencies_line_counts(self) -> None:
        self.land_pr(101)
        for number, line in (
            (102, "ghstack dependencies: #101"),
            (103, f"See {stack_dependencies_line([101])}"),
            (104, "Follow-up to #101"),
        ):
            self.commit(f"{landing_message(pr_url(number))}{line}\n")
        self.assertEqual(self.dependents(101), [])

    def test_dependents_landed_before_a_reland_still_count(self) -> None:
        _, b, c = self.land_stack()
        self.revert_pr(101)
        self.land_pr(101)
        self.assertEqual(self.dependents(101), [(c, 103), (b, 102)])

    def test_dependents_reverted_before_a_reland_stay_excluded(self) -> None:
        self.land_stack()
        for number in (103, 102, 101):
            self.revert_pr(number)
        self.land_pr(101)
        self.assertEqual(self.dependents(101), [])
        d = self.land_pr(104, 101)
        self.assertEqual(self.dependents(101), [(d, 104)])

    def test_commits_before_the_earliest_landing_never_count(self) -> None:
        self.land_pr(102, 101)
        self.land_pr(101)
        c = self.land_pr(103, 101)
        self.assertEqual(self.dependents(101), [(c, 103)])

    def test_pr_without_a_landing_has_no_dependents(self) -> None:
        self.land_pr(102, 101)
        self.assertEqual(self.dependents(101), [])
        self.land_pr(101)
        c = self.land_pr(103, 101)
        self.assertEqual(self.dependents(101), [(c, 103)])
        self.revert_pr(101)
        self.assertEqual(self.dependents(101), [])

    def test_regular_pr_has_no_dependents(self) -> None:
        for number in (101, 102, 103):
            self.land_pr(number)
        self.assertEqual(self.dependents(101), [])


VIABLE_STRICT = "viable/strict"
# Branch, line of f.txt and label of each PR of a line stack
LINE_STACK = (("user/a", 2, "A"), ("user/b", 6, "B"), ("user/c", 9, "C"))
# Branch updates as build_native_stack_rebase returns them
Updates = list[tuple[int, str, str, str]]


def lines(edits: dict[int, str]) -> str:
    """f.txt, with each line in `edits` followed by its label"""
    return "".join(
        f"line {n} {edits[n]}\n" if n in edits else f"line {n}\n" for n in range(1, 11)
    )


def identity(person: str) -> dict[str, str]:
    """The environment that makes git author and commit as `person`"""
    name, _, email = person.rpartition(" <")
    email = email.removesuffix(">")
    return {
        "GIT_AUTHOR_NAME": name,
        "GIT_AUTHOR_EMAIL": email,
        "GIT_COMMITTER_NAME": name,
        "GIT_COMMITTER_EMAIL": email,
    }


@contextmanager
def without_identity_variables() -> Iterator[None]:
    """trymerge.yml and tryrebase.yml only set user.name and user.email; these
    variables would override them and any identity the code under test sets
    through git config"""
    with mock.patch.dict(os.environ):
        for var in GIT_ENV:
            if var.startswith(("GIT_AUTHOR_", "GIT_COMMITTER_")):
                del os.environ[var]
        yield


class LineStackTestCase(GitTestCase):
    """A stack whose PRs each change their own line of f.txt and add their own file,
    in commits by the PR's author, on an origin that refuses non-fast-forward pushes,
    even forced ones"""

    def setUp(self) -> None:
        super().setUp()
        self.origin = self.git("remote", "get-url", "origin")
        self.origin_git("config", "receive.denyNonFastForwards", "true")
        for key, value in (("user.name", BOT_NAME), ("user.email", BOT_EMAIL)):
            self.git("config", key, value, cwd=self.repo.repo_dir)
        self.add_commits(TRUNK, files={"f.txt": lines({})})
        self.landings: dict[int, str] = {}

    def origin_git(self, *args: str) -> str:
        return self.git(*args, cwd=self.origin)

    def push_line_stack(self) -> NativeStack:
        entries: list[StackEntry] = []
        edits: dict[int, str] = {}
        base = TRUNK
        for position, (branch, line, label) in enumerate(LINE_STACK, start=1):
            edits[line] = label
            files = {"f.txt": lines(edits), f"{label}.txt": f"{label}\n"}
            with mock.patch.dict(os.environ, identity(author(100 + position))):
                head = self.push_branch(branch, base, files=files)
            entries.append(make_entry(position, 100 + position, branch, head, base))
            base = branch
        return NativeStack(number=100, base_ref=TRUNK, entries=tuple(entries))

    def heads(self, stack: NativeStack) -> dict[str, str]:
        """The tip of each PR's branch on the origin, bottom first"""
        refs = [f"refs/heads/{entry.head_ref}" for entry in stack.entries]
        fmt = "--format=%(refname:lstrip=2) %(objectname)"
        listed = self.origin_git("for-each-ref", fmt, *refs)
        tips = dict(line.split() for line in listed.splitlines())
        return {entry.head_ref: tips[entry.head_ref] for entry in stack.entries}

    def current(self, stack: NativeStack) -> NativeStack:
        """`stack` as GitHub reports it now: open PRs at the tips of their branches,
        closed ones at the head they were closed with"""
        heads = self.heads(stack)
        entries = [
            entry if entry.closed else replace(entry, head_oid=heads[entry.head_ref])
            for entry in stack.entries
        ]
        return replace(stack, entries=tuple(entries))

    def changes(self, base: str, head: str, cwd: str | None = None) -> list[str]:
        """The changes GitHub shows for a PR of `head` into `base`: the diff of
        `base...head`, without line numbers and blob ids"""
        diff = self.git("diff", "-U0", f"{base}...{head}", cwd=cwd or self.origin)
        return [x for x in diff.splitlines() if not x.startswith(("index ", "@@"))]

    def pr_changes(self, stack: NativeStack) -> dict[int, list[str]]:
        return {e.number: self.changes(e.base_ref, e.head_ref) for e in stack.entries}

    def move_trunk(self, files: FileChanges | None = None) -> str:
        self.git("checkout", "-q", TRUNK)
        self.git("pull", "-q", "--ff-only", "origin", TRUNK)
        return self.add_commits(TRUNK, files=files)

    def land_pr(self, stack: NativeStack, index: int) -> NativeStack:
        """Squashes the PR at `index` onto trunk with the message of a native merge,
        and closes it"""
        entry = stack.entries[index]
        message = landing_message(pr_url(entry.number))
        if index:
            deps = [below.number for below in stack.entries[:index]]
            message += f"{stack_dependencies_line(deps)}\n"
        self.git("fetch", "-q", "origin")
        self.git("checkout", "-q", TRUNK)
        self.git("merge", "-q", "--ff-only", f"origin/{TRUNK}")
        self.git("merge", "-q", "--squash", f"origin/{entry.head_ref}")
        self.git("commit", "-q", "-m", message)
        self.git("push", "-q", "origin", TRUNK)
        self.landings[entry.number] = self.git("rev-parse", "HEAD")
        return with_entry(self.current(stack), index, closed=True)

    def revert_pr(self, number: int) -> str:
        """Reverts the landed commit of PR `number` on trunk like do_revert_prs"""
        self.git("checkout", "-q", TRUNK)
        self.git("pull", "-q", "--ff-only", "origin", TRUNK)
        self.git("revert", "--no-edit", self.landings[number])
        message = self.git("log", "-1", "--format=%B")
        reverter = "https://github.com/reverter"
        message += f"\n\nReverted {pr_url(number)} on behalf of {reverter}\n"
        self.git("commit", "-q", "--amend", "-m", message)
        self.git("push", "-q", "origin", TRUNK)
        return self.git("rev-parse", "HEAD")

    def landed_stack(self, landed: int = 1) -> tuple[NativeStack, dict[int, list[str]]]:
        """A line stack whose `landed` lowest PRs landed on a trunk that moved before
        and after, with the changes each PR showed before it landed"""
        stack = self.push_line_stack()
        own = self.pr_changes(stack)
        self.move_trunk()
        for index in range(landed):
            stack = self.land_pr(stack, index)
        self.move_trunk()
        return stack, own

    def reject_pushes_to(self, branch: str) -> None:
        """Makes the origin refuse updates of `branch`; the other branches of a push
        are still updated unless it is atomic"""
        hook = Path(self.origin, "hooks", "update")
        hook.write_text(f'#!/bin/sh\ntest "$1" != refs/heads/{branch}\n')
        hook.chmod(0o755)

    def rebase_stack(
        self, stack: NativeStack, target: int, onto: str = TRUNK
    ) -> Updates:
        """Rebases `stack` from PR `target` onto `onto` like tryrebase.py does"""
        with without_identity_variables():
            updates = build_native_stack_rebase(
                self.repo, ORG, PROJECT, self.current(stack), target, TRUNK, onto
            )
            if updates:
                push_branches(self.repo, updates)
        return updates

    def assertFastForwarded(
        self, before: dict[str, str], after: dict[str, str], *pushed: str
    ) -> None:
        """The branches `pushed`, bottom first, moved forward from `before` to
        `after`, and no other branch moved"""
        moved = [branch for branch in before if before[branch] != after[branch]]
        self.assertEqual(moved, list(pushed))
        for branch in pushed:
            old, new = before[branch], after[branch]
            self.assertEqual(self.origin_git("merge-base", old, new), old, branch)

    def assertNoBranchHasTheHeadsAboveIt(
        self, stack: NativeStack, *heads: dict[str, str]
    ) -> None:
        refs = [f"refs/heads/{entry.head_ref}" for entry in stack.entries]
        for index, entry in enumerate(stack.entries[1:], start=1):
            for head in {h[entry.head_ref] for h in heads}:
                args = ("--contains", head, "--format=%(refname)", *refs[:index])
                having = self.origin_git("for-each-ref", *args)
                self.assertEqual(having, "", f"{entry.head_ref} at {head}")

    def assertNewCommitsBy(
        self, stack: NativeStack, before: dict[str, str], after: dict[str, str]
    ) -> None:
        """The commits each PR's branch got, other than those of the branches below
        it and of the trunks, are authored by the PR's author and committed by the
        bot"""
        trunks = [f"refs/heads/{TRUNK}", f"refs/heads/{VIABLE_STRICT}"]
        trunks = self.origin_git("for-each-ref", "--format=^%(refname)", *trunks)
        for index, entry in enumerate(stack.entries):
            lower = stack.entries[:index]
            below = [h[e.head_ref] for e in lower for h in (before, after)]
            revs = [after[entry.head_ref], f"^{before[entry.head_ref]}"]
            revs += [*trunks.split(), *(f"^{head}" for head in below)]
            people = self.origin_git("log", "--format=%an <%ae>|%cn <%ce>", *revs, "--")
            expected = f"{author(entry.number)}|{BOT_NAME} <{BOT_EMAIL}>"
            for line in people.splitlines():
                self.assertEqual(line, expected, entry.head_ref)

    def assertUpdated(
        self,
        stack: NativeStack,
        before: dict[str, str],
        own: dict[int, list[str]],
        *pushed: str,
    ) -> None:
        """Only the branches `pushed` were fast-forwarded, with commits of their PR's
        author, and every open PR of `stack` still shows only its own changes"""
        after = self.heads(stack)
        self.assertFastForwarded(before, after, *pushed)
        self.assertNoBranchHasTheHeadsAboveIt(stack, before, after)
        self.assertNewCommitsBy(stack, before, after)
        changes = self.pr_changes(stack)
        opened = [entry.number for entry in stack.entries if not entry.closed]
        self.assertEqual({n: changes[n] for n in opened}, {n: own[n] for n in opened})

    def assertMergesCleanly(
        self, stack: NativeStack, target: int, own: dict[int, list[str]]
    ) -> None:
        """The native merge of PR `target` accepts the stack and lands exactly the
        changes of each PR"""
        landing = self.landing_prs(self.current(stack), target)
        base = self.repo.rev_parse(TRUNK_REF)
        commits = [
            (author(entry.number), landing_message(pr_url(entry.number)))
            for entry, _ in landing
        ]
        top = build_native_stack_commits(self.repo, base, landing, commits)
        built = [base, *reversed(self.repo.revlist(f"{base}..{top}"))]
        bot = self.repo.repo_dir
        landed = [self.changes(old, new, bot) for old, new in pairwise(built)]
        self.assertEqual(landed, [own[entry.number] for entry, _ in landing])


class TestBuildNativeStackRebase(LineStackTestCase):
    def tree(self, ref: str) -> str:
        return self.origin_git("rev-parse", f"{ref}^{{tree}}")

    def show(self, ref: str, path: str) -> str:
        return self.origin_git("show", f"{ref}:{path}") + "\n"

    def test_merges_target_into_landed_branch_and_the_prs_above_it(self) -> None:
        stack, own = self.landed_stack()
        before = self.heads(stack)
        updates = self.rebase_stack(stack, 103)
        after = self.heads(stack)
        refs = [entry.head_ref for entry in stack.entries]
        expected = [
            (number, ref, before[ref], after[ref])
            for number, ref in enumerate(refs, start=101)
        ]
        self.assertEqual(updates, expected)
        self.assertUpdated(stack, before, own, *refs)
        # The trunk has the landed PR, so its branch takes the trunk's tree
        self.assertEqual(self.tree("user/a"), self.tree(TRUNK))
        self.assertMergesCleanly(stack, 103, own)
        self.assertEqual(self.rebase_stack(stack, 103), [])
        self.assertEqual(self.heads(stack), after)

    def test_landed_pr_with_several_commits_on_the_same_line(self) -> None:
        with mock.patch.dict(os.environ, identity(author(101))):
            files = {"f.txt": lines({2: "A1"}), "A.txt": "1\n"}
            self.push_branch("user/a", TRUNK, files=files)
            files = {"f.txt": lines({2: "A2", 3: "A2"}), "A.txt": "2\n"}
            a_head = self.add_commits("user/a", files=files)
        with mock.patch.dict(os.environ, identity(author(102))):
            files = {"f.txt": lines({2: "A2", 3: "A2", 6: "B"}), "B.txt": "B\n"}
            b_head = self.push_branch("user/b", "user/a", files=files)
        a = make_entry(1, 101, "user/a", a_head, TRUNK)
        b = make_entry(2, 102, "user/b", b_head, "user/a")
        stack = NativeStack(100, TRUNK, (a, b))
        own = self.pr_changes(stack)
        self.move_trunk()
        stack = self.land_pr(stack, 0)
        self.move_trunk()
        before = self.heads(stack)
        self.rebase_stack(stack, 102)
        self.assertUpdated(stack, before, own, "user/a", "user/b")
        self.assertMergesCleanly(stack, 102, own)

    def test_trunk_changing_the_lines_of_the_landed_pr_does_not_conflict(self) -> None:
        stack, own = self.landed_stack()
        self.move_trunk({"f.txt": lines({2: "trunk"})})
        before = self.heads(stack)
        self.rebase_stack(stack, 103)
        self.assertUpdated(stack, before, own, "user/a", "user/b", "user/c")
        expected = lines({2: "trunk", 6: "B", 9: "C"})
        self.assertEqual(self.show("user/c", "f.txt"), expected)
        self.assertMergesCleanly(stack, 103, own)

    def test_target_without_the_landed_pr_keeps_its_changes(self) -> None:
        stack = self.push_line_stack()
        own = self.pr_changes(stack)
        self.move_trunk()
        self.git("push", "-q", "origin", f"{TRUNK}:refs/heads/{VIABLE_STRICT}")
        stack = self.land_pr(stack, 0)
        self.move_trunk()
        before = self.heads(stack)
        self.rebase_stack(stack, 103, onto=VIABLE_STRICT)
        self.assertUpdated(stack, before, own, "user/a", "user/b", "user/c")
        for ref, edits in (("user/a", {2: "A"}), ("user/b", {2: "A", 6: "B"})):
            self.assertEqual(self.show(ref, "f.txt"), lines(edits))
            self.assertEqual(self.show(ref, "A.txt"), "A\n")
        self.assertMergesCleanly(stack, 103, own)
        # Once the target has the landed PR, the landed branch takes the target's tree
        self.git("push", "-q", "origin", f"{TRUNK}:refs/heads/{VIABLE_STRICT}")
        before = self.heads(stack)
        self.rebase_stack(stack, 103, onto=VIABLE_STRICT)
        self.assertUpdated(stack, before, own, "user/a", "user/b", "user/c")
        self.assertEqual(self.tree("user/a"), self.tree(VIABLE_STRICT))

    def test_bottom_pr_merges_the_target(self) -> None:
        stack = self.push_line_stack()
        own = self.pr_changes(stack)
        self.move_trunk()
        before = self.heads(stack)
        self.rebase_stack(stack, 103)
        self.assertUpdated(stack, before, own, "user/a", "user/b", "user/c")
        self.assertMergesCleanly(stack, 103, own)

    def test_prs_above_the_target_are_untouched(self) -> None:
        stack = self.push_line_stack()
        own = self.pr_changes(stack)
        self.move_trunk()
        before = self.heads(stack)
        self.rebase_stack(stack, 101)
        self.assertUpdated(stack, before, own, "user/a")
        stack = self.land_pr(stack, 0)
        self.move_trunk()
        before = self.heads(stack)
        self.rebase_stack(stack, 102)
        self.assertUpdated(stack, before, own, "user/a", "user/b")
        self.assertMergesCleanly(stack, 102, own)
        before = self.heads(stack)
        self.rebase_stack(stack, 103)
        self.assertUpdated(stack, before, own, "user/c")
        self.assertMergesCleanly(stack, 103, own)

    def test_only_the_branch_below_the_lowest_open_pr_merges_the_target(self) -> None:
        stack, own = self.landed_stack(landed=2)
        before = self.heads(stack)
        self.rebase_stack(stack, 103)
        self.assertUpdated(stack, before, own, "user/b", "user/c")
        self.assertMergesCleanly(stack, 103, own)

    def test_reapplies_reverted_pr_whose_branch_has_its_landed_commit(self) -> None:
        stack, own = self.landed_stack()
        self.rebase_stack(stack, 103)
        self.revert_pr(101)
        stack = with_entry(stack, 0, closed=False)
        self.assertEqual(self.pr_changes(stack)[101], [])
        before = self.heads(stack)
        self.rebase_stack(stack, 103)
        self.assertUpdated(stack, before, own, "user/a", "user/b", "user/c")

    def test_reapplies_pr_reverted_by_a_revert_pr_opened_by_hand(self) -> None:
        stack, own = self.landed_stack()
        other = self.move_trunk({"other.txt": "other\n"})
        self.rebase_stack(stack, 103)
        # The revert PR also reverts another commit, which stays reverted
        landed = self.landings[101]
        self.git("revert", "--no-commit", landed, other)
        body = f"This reverts commit {landed}.\nThis reverts commit {other}."
        self.git("commit", "-q", "-m", landing_message(pr_url(700), body))
        self.git("push", "-q", "origin", TRUNK)
        stack = with_entry(stack, 0, closed=False)
        self.assertEqual(self.pr_changes(stack)[101], [])
        before = self.heads(stack)
        self.rebase_stack(stack, 103)
        self.assertUpdated(stack, before, own, "user/a", "user/b", "user/c")

    def test_conflict_names_the_pr_and_nothing_is_pushed(self) -> None:
        stack, _ = self.landed_stack()
        self.move_trunk({"f.txt": lines({2: "A", 6: "trunk"})})
        before = self.heads(stack)
        message = r"^Merging user/a into user/b \(PR #102\) has conflicts"
        with self.assertRaisesRegex(NativeStackError, message):
            self.rebase_stack(stack, 103)
        self.assertEqual(self.heads(stack), before)

    def test_refuses_stack_based_on_another_branch(self) -> None:
        stack = replace(self.push_line_stack(), base_ref=VIABLE_STRICT)
        message = (
            r"^PR #103 is in a stack based on viable/strict, but only stacks based on "
            r"main can be rebased$"
        )
        with self.assertRaisesRegex(NativeStackError, message):
            self.rebase_stack(stack, 103)

    def test_refuses_closed_pr_below_that_did_not_land(self) -> None:
        stack = with_entry(self.push_line_stack(), 0, closed=True)
        before = self.heads(stack)
        with self.assertRaisesRegex(NativeStackError, NOT_LANDED):
            self.rebase_stack(stack, 103)
        self.assertEqual(self.heads(stack), before)

    def test_refuses_landed_pr_below_that_was_reverted(self) -> None:
        stack, _ = self.landed_stack()
        self.revert_pr(101)
        before = self.heads(stack)
        with self.assertRaisesRegex(NativeStackError, NOT_LANDED):
            self.rebase_stack(stack, 103)
        self.assertEqual(self.heads(stack), before)

    def test_refuses_landed_branch_with_commits_that_never_landed(self) -> None:
        stack, _ = self.landed_stack()
        extra = self.add_commits("user/a")
        message = (
            "PR #101 is closed, but its branch has commits that never landed on "
            f"main: {extra}"
        )
        with self.assertRaisesRegex(NativeStackError, message):
            self.rebase_stack(stack, 103)

    def test_refuses_target_behind_main_for_a_pr_it_rebases(self) -> None:
        stack, _ = self.landed_stack()
        first = self.landings[101]
        self.git("push", "-q", "origin", f"{TRUNK}:refs/heads/{VIABLE_STRICT}")
        self.revert_pr(101)
        reopened = with_entry(stack, 0, closed=False)
        before = self.heads(stack)
        message = (
            rf"^viable/strict is behind main for PR #101: it still has the PR's "
            rf"landing {first}, which main reverted or replaced since\. {REBASE}$"
        )
        with self.assertRaisesRegex(NativeStackError, message):
            self.rebase_stack(reopened, 103, onto=VIABLE_STRICT)
        relanded = self.land_pr(reopened, 0)
        with self.assertRaisesRegex(NativeStackError, message):
            self.rebase_stack(relanded, 103, onto=VIABLE_STRICT)
        self.assertEqual(self.heads(stack), before)

    def test_refuses_target_that_has_an_open_pr(self) -> None:
        stack = self.push_line_stack()
        self.git("checkout", "-q", TRUNK)
        self.git("merge", "-q", "--ff-only", "user/a")
        self.git("push", "-q", "origin", TRUNK)
        message = (
            r"^The head of PR #101 is already in main, so it has nothing to rebase; "
            r"close the PR if it landed$"
        )
        with self.assertRaisesRegex(NativeStackError, message):
            self.rebase_stack(stack, 103)

    def test_refuses_pr_opened_from_the_target(self) -> None:
        head = self.push_branch(VIABLE_STRICT, TRUNK)
        entry = make_entry(1, 101, VIABLE_STRICT, head, TRUNK)
        stack = NativeStack(100, TRUNK, (entry,))
        message = (
            r"^PR #101 is opened from viable/strict, so rebasing it would push to "
            r"viable/strict$"
        )
        with self.assertRaisesRegex(NativeStackError, message):
            self.rebase_stack(stack, 101, onto=VIABLE_STRICT)


class TestPushBranches(LineStackTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.stack, _ = self.landed_stack()
        with without_identity_variables():
            self.updates = build_native_stack_rebase(
                self.repo, ORG, PROJECT, self.stack, 103, TRUNK, TRUNK
            )
        self.before = self.heads(self.stack)

    def test_fast_forwards_every_branch_in_one_atomic_push(self) -> None:
        log = Path(self.repo.repo_dir).parent / "git.log"
        wrapper = log.parent / "bin" / "git"
        wrapper.parent.mkdir()
        git = shutil.which("git")
        wrapper.write_text(f'#!/bin/sh\necho "$*" >> {log}\nexec {git} "$@"\n')
        wrapper.chmod(0o755)
        path = f"{wrapper.parent}{os.pathsep}{os.environ['PATH']}"
        with mock.patch.dict(os.environ, {"PATH": path}):
            push_branches(self.repo, self.updates)
        after = {branch: new for _, branch, _, new in self.updates}
        self.assertEqual(self.heads(self.stack), after)
        commands = [line.split() for line in log.read_text().splitlines()]
        pushes = [args[args.index("push") + 1 :] for args in commands if "push" in args]
        self.assertEqual(len(pushes), 1, pushes)
        self.assertIn("--atomic", pushes[0])
        forced = [arg for arg in pushes[0] if arg.startswith(("-f", "--force", "+"))]
        leases = [
            f"--force-with-lease=refs/heads/{branch}:{old}"
            for _, branch, old, _ in self.updates
        ]
        self.assertEqual(forced, leases)

    def test_pushes_nothing_if_one_branch_is_refused(self) -> None:
        self.reject_pushes_to("user/c")
        with self.assertRaisesRegex(RuntimeError, "user/c"):
            push_branches(self.repo, self.updates)
        self.assertEqual(self.heads(self.stack), self.before)

    def test_dry_run_pushes_nothing(self) -> None:
        push_branches(self.repo, self.updates, dry_run=True)
        self.assertEqual(self.heads(self.stack), self.before)

    def test_runs_no_git_command_without_updates(self) -> None:
        with mock.patch.object(self.repo, "_run_git") as run:
            push_branches(self.repo, [])
        run.assert_not_called()

    def test_refuses_branch_reset_since_it_was_read(self) -> None:
        dropped = self.before["user/b"]
        self.origin_git("update-ref", "refs/heads/user/b", f"{dropped}^")
        reset = self.heads(self.stack)
        with self.assertRaisesRegex(RuntimeError, r"-> user/b \(stale info\)"):
            push_branches(self.repo, self.updates)
        self.assertEqual(self.heads(self.stack), reset)

    def test_refuses_update_that_is_not_a_fast_forward(self) -> None:
        # The lease matches and the origin takes forced updates, so only the check
        # that the new commit has the old tip stops this push
        self.origin_git("config", "receive.denyNonFastForwards", "false")
        number, branch, old, _ = self.updates[-1]
        updates = [*self.updates[:-1], (number, branch, old, self.root)]
        message = (
            rf"^The update of {branch} from {old} to {self.root} is not a "
            "fast-forward$"
        )
        with self.assertRaisesRegex(NativeStackError, message):
            push_branches(self.repo, updates)
        self.assertEqual(self.heads(self.stack), self.before)


if __name__ == "__main__":
    main()
