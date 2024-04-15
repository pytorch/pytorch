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
        pr = GitHubPR("pytorch", "pytorch", 31093)
        repo = GitRepo(get_git_repo_dir(), get_git_remote_name())
        rebase_onto(pr, repo, MAIN_BRANCH)
        calls = [
            mock.call("fetch", "origin", "pull/31093/head:pull/31093/head"),
            mock.call("rebase", MAIN_BRANCH, "pull/31093/head"),
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
        pr = GitHubPR("pytorch", "pytorch", 31093)
        repo = GitRepo(get_git_repo_dir(), get_git_remote_name())
        rebase_onto(pr, repo, VIABLE_STRICT_BRANCH, False)
        calls = [
            mock.call("fetch", "origin", "pull/31093/head:pull/31093/head"),
            mock.call("rebase", VIABLE_STRICT_BRANCH, "pull/31093/head"),
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
    @mock.patch("gitutils.GitRepo._run_git", return_value="Everything up-to-date")
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
        pr = GitHubPR("pytorch", "pytorch", 31093)
        repo = GitRepo(get_git_repo_dir(), get_git_remote_name())
        rebase_onto(pr, repo, MAIN_BRANCH)
        calls = [
            mock.call("fetch", "origin", "pull/31093/head:pull/31093/head"),
            mock.call("rebase", MAIN_BRANCH, "pull/31093/head"),
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


if __name__ == "__main__":
    main()
