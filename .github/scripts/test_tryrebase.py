from typing import Any
from unittest import main, mock, TestCase

from gitutils import get_git_remote_name, get_git_repo_dir, GitRepo
from test_trymerge import mocked_gh_graphql
from trymerge import GitHubPR
from tryrebase import additional_rebase_failure_info, rebase_ghstack_onto, rebase_onto


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


if __name__ == "__main__":
    main()
