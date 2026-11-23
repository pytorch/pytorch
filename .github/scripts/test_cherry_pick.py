from __future__ import annotations

from typing import Any
from unittest import main, TestCase
from unittest.mock import MagicMock, patch

from cherry_pick import (
    create_cherry_pick_branch,
    find_revert_commit_sha,
    get_merge_commit_sha,
    is_ancestor,
    post_tracker_issue_comment,
    resolve_trunk_revert,
)


LANDED = "a" * 40
REVERT = "b" * 40
OTHER_LANDED = "c" * 40


def make_repo(revert_log: str = "", on_branch: tuple[str, ...] = ()) -> Any:
    """
    A GitRepo stub. `revert_log` is what `git log --grep` returns and `on_branch`
    lists the SHAs that the target branch already contains.
    """
    repo = MagicMock()
    repo.remote = "origin"
    repo._run_git.return_value = revert_log
    repo.rev_parse.side_effect = lambda ref: ref
    # get_merge_base returns the ref itself only when the branch contains it
    repo.get_merge_base.side_effect = lambda ref, _branch: (
        ref if ref in on_branch else "base"
    )
    return repo


def make_pr(pr_num: int = 42, default_branch: str = "main") -> Any:
    pr = MagicMock()
    pr.pr_num = pr_num
    pr.default_branch.return_value = default_branch
    return pr


class TestGetMergeCommitSha(TestCase):
    def _pr(self, closed: bool) -> Any:
        pr = make_pr()
        pr.is_closed.return_value = closed
        return pr

    def test_closed_pr_returns_its_landed_commit(self) -> None:
        with patch("cherry_pick.get_pr_commit_sha", return_value=LANDED):
            self.assertEqual(
                get_merge_commit_sha(make_repo(""), self._pr(closed=True)), LANDED
            )

    def test_open_never_merged_pr_is_refused(self) -> None:
        # No revert names this commit, so the PR is genuinely not merged.
        with patch("cherry_pick.get_pr_commit_sha", return_value=LANDED):
            self.assertIsNone(
                get_merge_commit_sha(make_repo(""), self._pr(closed=False))
            )

    def test_reopened_after_revert_is_accepted(self) -> None:
        # The revert bot reopens the PR it reverts, so the one case the revert
        # path exists for arrives here open. Gating on is_closed() alone made
        # the feature unreachable.
        with patch("cherry_pick.get_pr_commit_sha", return_value=LANDED):
            self.assertEqual(
                get_merge_commit_sha(make_repo(f"{REVERT}\n"), self._pr(closed=False)),
                LANDED,
            )


class TestFindRevertCommitSha(TestCase):
    def test_returns_none_when_not_reverted(self) -> None:
        self.assertIsNone(find_revert_commit_sha(make_repo(""), make_pr(), LANDED))

    def test_finds_the_revert(self) -> None:
        self.assertEqual(
            find_revert_commit_sha(make_repo(f"{REVERT}\n"), make_pr(), LANDED), REVERT
        )

    def test_greps_the_default_branch_for_the_exact_landed_sha(self) -> None:
        repo = make_repo(f"{REVERT}\n")
        find_revert_commit_sha(repo, make_pr(default_branch="main"), LANDED)
        args = repo._run_git.call_args[0]
        self.assertIn(f"--grep=This reverts commit {LANDED}.", args)
        self.assertIn("--fixed-strings", args)
        self.assertIn("origin/main", args)

    def test_multiple_reverts_resolve_to_the_newest(self) -> None:
        # git log is newest first
        newest = "d" * 40
        self.assertEqual(
            find_revert_commit_sha(
                make_repo(f"{newest}\n{REVERT}\n"), make_pr(), LANDED
            ),
            newest,
        )


class TestIsAncestor(TestCase):
    def test_true_when_branch_contains_ref(self) -> None:
        self.assertTrue(is_ancestor(make_repo(on_branch=(LANDED,)), LANDED, "release"))

    def test_false_when_branch_does_not_contain_ref(self) -> None:
        self.assertFalse(is_ancestor(make_repo(), LANDED, "release"))


class TestResolveTrunkRevert(TestCase):
    def test_never_reverted_is_a_plain_cherry_pick(self) -> None:
        self.assertIsNone(
            resolve_trunk_revert(make_repo(""), make_pr(), "release/2.14", LANDED)
        )

    def test_reverted_returns_the_trunk_revert(self) -> None:
        repo = make_repo(f"{REVERT}\n", on_branch=(LANDED,))
        self.assertEqual(
            resolve_trunk_revert(repo, make_pr(), "release/2.14", LANDED), REVERT
        )

    def test_refuses_when_landed_commit_is_not_on_the_release_branch(self) -> None:
        # Reverted on trunk but the change never reached the release branch, so
        # there is nothing there to undo
        repo = make_repo(f"{REVERT}\n", on_branch=())
        with self.assertRaises(RuntimeError) as err:
            resolve_trunk_revert(repo, make_pr(), "release/2.14", LANDED)
        self.assertIn("nothing to revert", str(err.exception))

    def test_refuses_when_the_revert_is_already_on_the_release_branch(self) -> None:
        repo = make_repo(f"{REVERT}\n", on_branch=(LANDED, REVERT))
        with self.assertRaises(RuntimeError) as err:
            resolve_trunk_revert(repo, make_pr(), "release/2.14", LANDED)
        self.assertIn("already on", str(err.exception))

    def test_relanded_pr_keyed_on_the_land_that_reached_the_branch(self) -> None:
        # A PR that landed twice has one revert per land. Only the revert naming
        # the land that is on the release branch should be picked up, so a lookup
        # for the other land finds nothing and it stays a plain cherry pick.
        def run_git(*args: Any) -> str:
            grep = next(a for a in args if a.startswith("--grep="))
            return f"{REVERT}\n" if LANDED in grep else ""

        repo = make_repo(on_branch=(OTHER_LANDED,))
        repo._run_git.side_effect = run_git

        self.assertIsNone(
            resolve_trunk_revert(repo, make_pr(), "release/2.14", OTHER_LANDED)
        )

    def test_ancestry_is_checked_against_the_remote_tracking_ref(self) -> None:
        # This runs before the branch is checked out, so it only exists as
        # refs/remotes/origin/<branch>. git rev-parse's DWIM ladder does not
        # reach that from a bare "release/2.14", so both guards would die with
        # exit 128 on exactly the PRs this feature exists for.
        seen: list[str] = []
        repo = make_repo(f"{REVERT}\n", on_branch=(LANDED,))
        repo.get_merge_base.side_effect = lambda ref, branch: (
            seen.append(branch) or (ref if ref == LANDED else "base")
        )

        resolve_trunk_revert(repo, make_pr(), "release/2.14", LANDED)

        self.assertTrue(seen)
        for branch in seen:
            self.assertEqual(branch, "origin/release/2.14")


class TestCreateCherryPickBranch(TestCase):
    def _repo(self) -> Any:
        repo = MagicMock()
        repo.remote = "origin"
        repo.commit_message.return_value = 'Revert "thing"\n\nThis reverts commit x.'
        return repo

    def test_plain_cherry_pick_uses_cherry_pick(self) -> None:
        repo = self._repo()
        create_cherry_pick_branch("actor", repo, make_pr(), LANDED, "release/2.14")
        repo.revert.assert_not_called()
        repo._run_git.assert_any_call("cherry-pick", "-x", LANDED)
        repo.push.assert_called_once()

    def test_revert_case_reverts_the_landed_commit_on_the_release_branch(self) -> None:
        repo = self._repo()
        create_cherry_pick_branch(
            "actor", repo, make_pr(), LANDED, "release/2.14", REVERT
        )
        # The landed commit is reverted here, not trunk's revert cherry picked
        repo.revert.assert_called_once_with(LANDED)
        for call in repo._run_git.call_args_list:
            self.assertNotIn("cherry-pick", call[0])
        repo.push.assert_called_once()

    def test_revert_records_the_trunk_revert_in_the_message(self) -> None:
        repo = self._repo()
        create_cherry_pick_branch(
            "actor", repo, make_pr(), LANDED, "release/2.14", REVERT
        )
        amended = repo.amend_commit_message.call_args[0][0]
        self.assertIn(f"Reverted on main by {REVERT}.", amended)

    def test_conflict_aborts_and_does_not_push(self) -> None:
        repo = self._repo()
        repo.revert.side_effect = RuntimeError("CONFLICT (content): merge conflict")

        with self.assertRaises(RuntimeError) as err:
            create_cherry_pick_branch(
                "actor", repo, make_pr(), LANDED, "release/2.14", REVERT
            )

        self.assertIn("hit a conflict", str(err.exception))
        self.assertIn("by hand", str(err.exception))
        repo._run_git.assert_any_call("revert", "--abort")
        repo.push.assert_not_called()

    def test_conflict_still_reports_when_abort_also_fails(self) -> None:
        repo = self._repo()
        repo.revert.side_effect = RuntimeError("CONFLICT (content): merge conflict")

        def run_git(*args: Any) -> str:
            if args[:2] == ("revert", "--abort"):
                raise RuntimeError("nothing to abort")
            return ""

        repo._run_git.side_effect = run_git

        with self.assertRaises(RuntimeError) as err:
            create_cherry_pick_branch(
                "actor", repo, make_pr(), LANDED, "release/2.14", REVERT
            )
        self.assertIn("hit a conflict", str(err.exception))
        repo.push.assert_not_called()


class TestTrackerIssueComment(TestCase):
    def _comment_body(self, is_revert: bool) -> str:
        posted: dict[str, str] = {}

        import cherry_pick

        original = cherry_pick.gh_post_pr_comment

        def capture(
            org: str, project: str, num: int, comment: str, dry_run: bool = False
        ) -> list[dict[str, Any]]:
            posted["comment"] = comment
            return []

        cherry_pick.gh_post_pr_comment = capture  # type: ignore[assignment]
        try:
            post_tracker_issue_comment(
                "pytorch",
                "pytorch",
                1,
                42,
                "https://github.com/pytorch/pytorch/pull/43",
                "critical",
                "",
                True,
                is_revert,
            )
        finally:
            cherry_pick.gh_post_pr_comment = original  # type: ignore[assignment]
        return posted["comment"]

    def test_revert_is_flagged_in_the_criteria(self) -> None:
        self.assertIn("Cherry-pick revert - Critical", self._comment_body(True))

    def test_plain_cherry_pick_is_not_flagged(self) -> None:
        body = self._comment_body(False)
        self.assertNotIn("Cherry-pick revert", body)
        self.assertIn("Critical", body)

    def test_unlinked_cherry_pick_has_no_dangling_separator(self) -> None:
        # fixes is empty here, so the criteria line must not end in " - "
        for is_revert in (True, False):
            criteria = self._comment_body(is_revert).splitlines()[-1]
            self.assertFalse(criteria.rstrip().endswith("-"), criteria)

    def test_links_the_reverted_pr_and_the_release_branch_pr(self) -> None:
        body = self._comment_body(True)
        self.assertIn("https://github.com/pytorch/pytorch/pull/42", body)
        self.assertIn("https://github.com/pytorch/pytorch/pull/43", body)


if __name__ == "__main__":
    main()
