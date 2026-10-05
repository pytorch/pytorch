import os
from argparse import Namespace
from typing import Any
from unittest import main, mock, TestCase

from gitutils import get_git_remote_name, get_git_repo_dir, GitRepo
from native_stack import NativeStackError
from test_native_stack import LineStackTestCase, without_identity_variables
from test_trymerge import mocked_gh_graphql, NATIVE_STACK, NoNetworkTestCase, stacked_pr
from trymerge import GitHubPR
from tryrebase import (
    additional_rebase_failure_info,
    main as tryrebase_main,
    rebase_ghstack_onto,
    rebase_onto,
)


def mocked_rev_parse(branch: str) -> str:
    return branch


MAIN_BRANCH = "refs/remotes/origin/main"
VIABLE_STRICT_BRANCH = "refs/remotes/origin/viable/strict"
# A full 40-char hex SHA, as `git merge-base` actually outputs; value is opaque
# to the code (passed through verbatim to the rebase call).
FORK_POINT = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"


def make_mocked_run_git(push_result: str = "") -> Any:
    """_run_git stub: merge-base yields a fork point, everything else push_result."""

    def run_git(*args: str) -> str:
        if args and args[0] == "merge-base":
            return FORK_POINT
        return push_result

    return run_git


class TestRebase(TestCase):
    @mock.patch("trymerge.gh_graphql", side_effect=mocked_gh_graphql)
    @mock.patch("gitutils.GitRepo._run_git")
    @mock.patch("gitutils.GitRepo.rev_parse", side_effect=mocked_rev_parse)
    @mock.patch("tryrebase.gh_post_comment")
    def test_rebase(
        self,
        mocked_post_comment: Any,
        mocked_rp: Any,
        mocked_run_git: Any,
        mocked_gql: Any,
    ) -> None:
        "Tests rebase successfully"
        mocked_run_git.side_effect = make_mocked_run_git()
        pr = GitHubPR("pytorch", "pytorch", 31093)
        repo = GitRepo(get_git_repo_dir(), get_git_remote_name())
        rebase_onto(pr, repo, MAIN_BRANCH)
        base_ref = f"refs/remotes/origin/{pr.base_ref()}"
        calls = [
            mock.call("fetch", "origin", "pull/31093/head:pull/31093/head"),
            mock.call("merge-base", base_ref, "pull/31093/head"),
            mock.call("rebase", "--onto", MAIN_BRANCH, FORK_POINT, "pull/31093/head"),
            mock.call(
                "push",
                "-f",
                "https://github.com/mingxiaoh/pytorch.git",
                "pull/31093/head:master",
            ),
        ]
        mocked_run_git.assert_has_calls(calls)
        self.assertIn(
            f"Successfully rebased `master` onto `{MAIN_BRANCH}`",
            mocked_post_comment.call_args[0][3],
        )

    @mock.patch("trymerge.gh_graphql", side_effect=mocked_gh_graphql)
    @mock.patch("gitutils.GitRepo._run_git")
    @mock.patch("gitutils.GitRepo.rev_parse", side_effect=mocked_rev_parse)
    @mock.patch("tryrebase.gh_post_comment")
    def test_rebase_to_stable(
        self,
        mocked_post_comment: Any,
        mocked_rp: Any,
        mocked_run_git: Any,
        mocked_gql: Any,
    ) -> None:
        "Tests rebase to viable/strict successfully"
        mocked_run_git.side_effect = make_mocked_run_git()
        pr = GitHubPR("pytorch", "pytorch", 31093)
        repo = GitRepo(get_git_repo_dir(), get_git_remote_name())
        rebase_onto(pr, repo, VIABLE_STRICT_BRANCH, False)
        base_ref = f"refs/remotes/origin/{pr.base_ref()}"
        calls = [
            mock.call("fetch", "origin", "pull/31093/head:pull/31093/head"),
            mock.call("merge-base", base_ref, "pull/31093/head"),
            mock.call(
                "rebase", "--onto", VIABLE_STRICT_BRANCH, FORK_POINT, "pull/31093/head"
            ),
            mock.call(
                "push",
                "-f",
                "https://github.com/mingxiaoh/pytorch.git",
                "pull/31093/head:master",
            ),
        ]
        mocked_run_git.assert_has_calls(calls)
        self.assertIn(
            f"Successfully rebased `master` onto `{VIABLE_STRICT_BRANCH}`",
            mocked_post_comment.call_args[0][3],
        )

    @mock.patch("trymerge.gh_graphql", side_effect=mocked_gh_graphql)
    @mock.patch("gitutils.GitRepo._run_git")
    @mock.patch("gitutils.GitRepo.rev_parse", side_effect=mocked_rev_parse)
    @mock.patch("tryrebase.gh_post_comment")
    def test_no_need_to_rebase(
        self,
        mocked_post_comment: Any,
        mocked_rp: Any,
        mocked_run_git: Any,
        mocked_gql: Any,
    ) -> None:
        "Tests branch already up to date"
        mocked_run_git.side_effect = make_mocked_run_git("Everything up-to-date")
        pr = GitHubPR("pytorch", "pytorch", 31093)
        repo = GitRepo(get_git_repo_dir(), get_git_remote_name())
        rebase_onto(pr, repo, MAIN_BRANCH)
        base_ref = f"refs/remotes/origin/{pr.base_ref()}"
        calls = [
            mock.call("fetch", "origin", "pull/31093/head:pull/31093/head"),
            mock.call("merge-base", base_ref, "pull/31093/head"),
            mock.call("rebase", "--onto", MAIN_BRANCH, FORK_POINT, "pull/31093/head"),
            mock.call(
                "push",
                "-f",
                "https://github.com/mingxiaoh/pytorch.git",
                "pull/31093/head:master",
            ),
        ]
        mocked_run_git.assert_has_calls(calls)
        self.assertIn(
            "Tried to rebase and push PR #31093, but it was already up to date",
            mocked_post_comment.call_args[0][3],
        )
        self.assertNotIn(
            "Try rebasing against [main]",
            mocked_post_comment.call_args[0][3],
        )

    @mock.patch("trymerge.gh_graphql", side_effect=mocked_gh_graphql)
    @mock.patch("gitutils.GitRepo._run_git", return_value="Everything up-to-date")
    @mock.patch("gitutils.GitRepo.rev_parse", side_effect=mocked_rev_parse)
    @mock.patch("tryrebase.gh_post_comment")
    def test_no_need_to_rebase_try_main(
        self,
        mocked_post_comment: Any,
        mocked_rp: Any,
        mocked_run_git: Any,
        mocked_gql: Any,
    ) -> None:
        "Tests branch already up to date again viable/strict"
        pr = GitHubPR("pytorch", "pytorch", 31093)
        repo = GitRepo(get_git_repo_dir(), get_git_remote_name())
        rebase_onto(pr, repo, VIABLE_STRICT_BRANCH)
        self.assertIn(
            "Tried to rebase and push PR #31093, but it was already up to date. Try rebasing against [main]",
            mocked_post_comment.call_args[0][3],
        )

    @mock.patch("trymerge.gh_graphql", side_effect=mocked_gh_graphql)
    @mock.patch("gitutils.GitRepo._run_git")
    @mock.patch("gitutils.GitRepo.rev_parse", side_effect=lambda branch: "same sha")
    @mock.patch("tryrebase.gh_post_comment")
    def test_same_sha(
        self,
        mocked_post_comment: Any,
        mocked_rp: Any,
        mocked_run_git: Any,
        mocked_gql: Any,
    ) -> None:
        "Tests rebase results in same sha"
        pr = GitHubPR("pytorch", "pytorch", 31093)
        repo = GitRepo(get_git_repo_dir(), get_git_remote_name())
        with self.assertRaisesRegex(Exception, "same sha as the target branch"):
            rebase_onto(pr, repo, MAIN_BRANCH)
        with self.assertRaisesRegex(Exception, "same sha as the target branch"):
            rebase_ghstack_onto(pr, repo, MAIN_BRANCH)

    def test_additional_rebase_failure_info(self) -> None:
        error = (
            "Command `git -C /Users/csl/zzzzzzzz/pytorch push --dry-run -f "
            "https://github.com/Lightning-Sandbox/pytorch.git pull/106089/head:fix/spaces` returned non-zero exit code 128\n"
            "```\n"
            "remote: Permission to Lightning-Sandbox/pytorch.git denied to clee2000.\n"
            "fatal: unable to access 'https://github.com/Lightning-Sandbox/pytorch.git/': The requested URL returned error: 403\n"
            "```"
        )
        additional_msg = additional_rebase_failure_info(Exception(error))
        self.assertTrue("This is likely because" in additional_msg)

    @mock.patch("trymerge.gh_graphql", side_effect=mocked_gh_graphql)
    @mock.patch("gitutils.GitRepo._run_git")
    @mock.patch("gitutils.GitRepo.rev_parse", side_effect=mocked_rev_parse)
    @mock.patch("tryrebase.gh_post_comment")
    def test_rebase_does_not_replay_trunk_commits(
        self,
        mocked_post_comment: Any,
        mocked_rp: Any,
        mocked_run_git: Any,
        mocked_gql: Any,
    ) -> None:
        """#187374: rebase must anchor on the fork point (--onto) and never use
        the 2-arg form, which grafts trunk commits onto the PR."""
        mocked_run_git.side_effect = make_mocked_run_git()
        pr = GitHubPR("pytorch", "pytorch", 31093)
        repo = GitRepo(get_git_repo_dir(), get_git_remote_name())
        rebase_onto(pr, repo, VIABLE_STRICT_BRANCH)
        base_ref = f"refs/remotes/origin/{pr.base_ref()}"

        self.assertIn(
            mock.call("merge-base", base_ref, "pull/31093/head"),
            mocked_run_git.call_args_list,
        )
        rebase_calls = [
            c for c in mocked_run_git.call_args_list if c.args and c.args[0] == "rebase"
        ]
        self.assertEqual(
            rebase_calls,
            [
                mock.call(
                    "rebase",
                    "--onto",
                    VIABLE_STRICT_BRANCH,
                    FORK_POINT,
                    "pull/31093/head",
                )
            ],
        )


def merged(branch: str, onto: str = MAIN_BRANCH, rebased: int | None = None) -> str:
    """The comment on the PR of `branch` once `onto` was merged into it because PR
    `rebased`, if given, was rebased"""
    because = "" if rebased is None else f" because #{rebased} was rebased"
    return (
        f"Merged `{onto}` into `{branch}`{because}, please pull locally before adding "
        f"more changes (for example, via `git checkout {branch} && git pull --rebase`)"
    )


class TryRebaseMainTestCase(NoNetworkTestCase):
    """tryrebase.main() on PR #1002 of NATIVE_STACK, with GitHub patched"""

    def setUp(self) -> None:
        super().setUp()
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("GH_RUN_URL", None)
        self.args = Namespace(pr_num=1002, branch=None, dry_run=False)
        self.patch("tryrebase.parse_args", return_value=self.args)
        self.pr = stacked_pr(1002)
        self.pr.is_cross_repo.return_value = False
        self.pr.head_ref.return_value = "user/1002"
        self.patch("tryrebase.GitHubPR", return_value=self.pr)
        self.post = self.patch("tryrebase.gh_post_comment")
        # The PRs based on PR #1002's branch
        self.pulls = self.patch("tryrebase.gh_fetch_json_list", return_value=[])
        self.get_stack = self.patch(
            "tryrebase.get_native_stack", return_value=NATIVE_STACK
        )

    def run_main(self) -> object:
        """Runs tryrebase.main() and returns the exit code of the process"""
        try:
            tryrebase_main()
        except SystemExit as e:
            return e.code
        return 0

    def comments(self, dry_run: bool = False) -> list[Any]:
        """The comments after the one that starts the job"""
        started = self.post.call_args_list[0]
        self.assertEqual(started.args[2], self.args.pr_num)
        self.assertRegex(started.args[3], "^@pytorchbot started a rebase job onto")
        self.assertEqual(started.kwargs, {"dry_run": dry_run})
        return self.post.call_args_list[1:]

    def comment(self, number: int, message: str, dry_run: bool = False) -> Any:
        return mock.call("pytorch", "pytorch", number, message, dry_run=dry_run)


class TestNativeStackRebaseMain(TryRebaseMainTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.repo = mock.MagicMock(spec=GitRepo)
        self.repo.remote = "origin"
        self.repo.gh_owner_and_name.return_value = ("pytorch", "pytorch")
        self.repo.rev_parse.return_value = "onto-sha"
        self.patch("tryrebase.GitRepo", return_value=self.repo)
        self.updates = [(n, f"user/{n}", f"new-{n}") for n in (1000, 1001, 1002)]
        self.build = self.patch(
            "tryrebase.build_native_stack_rebase", return_value=self.updates
        )
        self.push = self.patch("tryrebase.push_branches")
        self.rebase_onto = self.patch("tryrebase.rebase_onto", return_value=True)
        self.rebase_ghstack_onto = self.patch(
            "tryrebase.rebase_ghstack_onto", return_value=True
        )

    def assertNotRebased(self) -> None:
        self.build.assert_not_called()
        self.push.assert_not_called()
        self.rebase_onto.assert_not_called()
        self.rebase_ghstack_onto.assert_not_called()

    def test_merges_the_target_into_the_stack_and_fast_forwards_it(self) -> None:
        self.assertEqual(self.run_main(), 0)
        self.get_stack.assert_called_once_with("pytorch", "pytorch", 1002)
        self.build.assert_called_once_with(
            self.repo, "pytorch", "pytorch", NATIVE_STACK, 1002, "main", "main"
        )
        self.push.assert_called_once_with(self.repo, self.updates, False)
        self.rebase_onto.assert_not_called()
        # The landed PR whose branch got the target is not told about it
        self.assertEqual(
            self.comments(),
            [
                self.comment(1001, merged("user/1001", rebased=1002)),
                self.comment(1002, merged("user/1002")),
            ],
        )

    def test_rebases_onto_viable_strict(self) -> None:
        self.args.branch = "viable/strict"
        self.assertEqual(self.run_main(), 0)
        self.build.assert_called_once_with(
            self.repo, "pytorch", "pytorch", NATIVE_STACK, 1002, "main", "viable/strict"
        )
        self.assertEqual(
            self.comments()[-1],
            self.comment(1002, merged("user/1002", VIABLE_STRICT_BRANCH)),
        )

    def test_refuses_other_targets(self) -> None:
        self.args.branch = "release/2.9"
        self.assertEqual(self.run_main(), 0)
        self.assertNotRebased()
        message = (
            "Rebase failed due to PR #1002 is in a stack, so it can only be rebased "
            "onto main or viable/strict, not release/2.9"
        )
        self.assertEqual(self.comments(), [self.comment(1002, message)])

    def test_up_to_date_stack_exits_with_success(self) -> None:
        self.build.return_value = []
        self.assertEqual(self.run_main(), 0)
        self.push.assert_not_called()
        message = "Tried to rebase and push PR #1002, but it was already up to date."
        self.assertEqual(self.comments(), [self.comment(1002, message)])

    def test_stack_errors_are_commented(self) -> None:
        error = "PR #1000 is closed but not landed on main, or it was reverted"
        self.build.side_effect = NativeStackError(error)
        self.assertEqual(self.run_main(), 0)
        self.push.assert_not_called()
        message = f"Rebase failed due to {error}"
        self.assertEqual(self.comments(), [self.comment(1002, message)])

    def test_dry_run(self) -> None:
        self.args.dry_run = True
        self.assertEqual(self.run_main(), 0)
        self.push.assert_called_once_with(self.repo, self.updates, True)
        comments = self.comments(dry_run=True)
        self.assertEqual([call.kwargs for call in comments], [{"dry_run": True}] * 2)

    def test_reads_the_stack_only_of_prs_that_may_be_stacked(self) -> None:
        for cross_repo, base, upper, read in (
            (True, "user/1001", [], False),
            (False, "main", [], False),
            (False, "main", [{"number": 1003}], True),
            (False, "user/1001", [], True),
        ):
            with self.subTest(cross_repo=cross_repo, base=base, upper=upper):
                self.pr.is_cross_repo.return_value = cross_repo
                self.pr.base_ref.return_value = base
                self.pulls.return_value = upper
                for patched in (self.pulls, self.get_stack, self.build):
                    patched.reset_mock()
                self.rebase_onto.reset_mock()
                self.assertEqual(self.run_main(), 0)
                self.assertEqual(self.get_stack.called, read)
                self.assertEqual(self.build.called, read)
                self.assertEqual(self.rebase_onto.called, not read)
                if not cross_repo and base == "main":
                    self.pulls.assert_called_once_with(
                        "https://api.github.com/repos/pytorch/pytorch/pulls",
                        {"base": "user/1002", "state": "all", "per_page": 1},
                    )
                else:
                    self.pulls.assert_not_called()

    def test_pr_without_a_stack_keeps_its_rebase(self) -> None:
        self.get_stack.return_value = None
        self.assertEqual(self.run_main(), 0)
        self.rebase_onto.assert_called_once_with(
            self.pr, self.repo, MAIN_BRANCH, dry_run=False
        )
        self.build.assert_not_called()

    def test_pr_whose_stack_cannot_be_read_is_not_rebased(self) -> None:
        self.get_stack.side_effect = NativeStackError("GraphQL errors: Timeout")
        self.pulls.return_value = [{"number": 1003}]
        message = (
            "Rebase failed due to Could not read the stack of PR #1002, so it was "
            "not rebased: GraphQL errors: Timeout"
        )
        for base in ("main", "user/1001"):
            with self.subTest(base=base):
                self.pr.base_ref.return_value = base
                self.post.reset_mock()
                self.assertEqual(self.run_main(), 0)
                self.assertNotRebased()
                self.assertEqual(self.comments(), [self.comment(1002, message)])

    def test_ghstack_pr_keeps_its_rebase(self) -> None:
        self.pr.is_ghstack_pr.return_value = True
        self.assertEqual(self.run_main(), 0)
        self.rebase_ghstack_onto.assert_called_once_with(
            self.pr, self.repo, MAIN_BRANCH, dry_run=False
        )
        self.get_stack.assert_not_called()
        self.build.assert_not_called()

    def test_closed_pr_is_not_rebased(self) -> None:
        self.pr.is_closed.return_value = True
        self.assertEqual(self.run_main(), 0)
        self.get_stack.assert_not_called()
        self.assertNotRebased()
        message = "PR #1002 is closed, won't rebase"
        self.assertEqual(self.comments(), [self.comment(1002, message)])


class TestNativeStackRebaseEndToEnd(TryRebaseMainTestCase, LineStackTestCase):
    """tryrebase.main() rebases a real stack, from PR #103, in the bot clone and pushes
    it to a local origin; only GitHub is patched"""

    def setUp(self) -> None:
        LineStackTestCase.setUp(self)
        TryRebaseMainTestCase.setUp(self)
        self.stack, self.own = self.landed_stack()
        os.environ.update(GIT_REPO_DIR=self.repo.repo_dir, GIT_REMOTE_NAME="origin")
        self.args.pr_num = 103
        self.pr.pr_num = 103
        self.pr.base_ref.return_value = "user/b"
        self.get_stack.side_effect = lambda *_: self.current(self.stack)

    def test_rebases_the_stack_with_fast_forwards(self) -> None:
        before = self.heads(self.stack)
        with without_identity_variables():
            self.assertEqual(self.run_main(), 0)
        self.assertUpdated(self.stack, before, self.own, "user/a", "user/b", "user/c")
        self.assertEqual(
            self.comments(),
            [
                self.comment(102, merged("user/b", rebased=103)),
                self.comment(103, merged("user/c")),
            ],
        )
        self.post.reset_mock()
        with without_identity_variables():
            self.assertEqual(self.run_main(), 0)
        message = "Tried to rebase and push PR #103, but it was already up to date."
        self.assertEqual(self.comments(), [self.comment(103, message)])


if __name__ == "__main__":
    main()
