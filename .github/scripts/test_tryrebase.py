from typing import Any
from unittest import main, mock, TestCase

from github_utils import GitHubComment
from gitutils import get_git_remote_name, get_git_repo_dir, GitRepo
from test_trymerge import mocked_gh_graphql
from trymerge import GitHubPR
from tryrebase import (
    additional_rebase_failure_info,
    approve_pending_ci,
    maintainer_approved_sha,
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

    @mock.patch("trymerge.gh_graphql", side_effect=mocked_gh_graphql)
    @mock.patch("gitutils.GitRepo._run_git")
    @mock.patch("gitutils.GitRepo.rev_parse")
    @mock.patch("tryrebase.gh_post_comment")
    @mock.patch("tryrebase.approve_pending_ci")
    def test_rebase_approves_ci(
        self,
        mocked_approve: Any,
        mocked_post_comment: Any,
        mocked_rp: Any,
        mocked_run_git: Any,
        mocked_gql: Any,
    ) -> None:
        "Tests CI of the rebased commit is approved only if the original was trusted"
        rebased = False

        def run_git(*args: str) -> str:
            nonlocal rebased
            rebased = rebased or args[0] == "rebase"
            return make_mocked_run_git()(*args)

        def rev_parse(branch: str) -> str:
            if branch != "pull/31093/head":
                return branch
            return "rebased sha" if rebased else "orig sha"

        mocked_run_git.side_effect = run_git
        mocked_rp.side_effect = rev_parse
        pr = GitHubPR("pytorch", "pytorch", 31093)
        repo = GitRepo(get_git_repo_dir(), get_git_remote_name())
        for kwargs in [
            {},
            {"approve_ci_sha": "other sha"},
            {"approve_ci_sha": "orig sha", "dry_run": True},
        ]:
            rebased = False
            rebase_onto(pr, repo, MAIN_BRANCH, **kwargs)
        mocked_approve.assert_not_called()

        rebased = False
        mocked_approve.side_effect = RuntimeError("API error")
        self.assertTrue(rebase_onto(pr, repo, MAIN_BRANCH, approve_ci_sha="orig sha"))
        mocked_approve.assert_called_once_with(pr, "rebased sha")

    @mock.patch("trymerge.gh_graphql", side_effect=mocked_gh_graphql)
    @mock.patch("tryrebase.time")
    @mock.patch("tryrebase.gh_fetch_url")
    @mock.patch("tryrebase.gh_fetch_json_dict")
    def test_approve_pending_ci(
        self,
        mocked_fetch_runs: Any,
        mocked_fetch_url: Any,
        mocked_time: Any,
        mocked_gql: Any,
    ) -> None:
        "Tests held runs of the PR are approved once, across polls"
        runs_url = "https://api.github.com/repos/pytorch/pytorch/actions/runs"

        def run(
            id: int, branch: str = "master", repo: str = "mingxiaoh/pytorch"
        ) -> Any:
            return {
                "id": id,
                "html_url": f"{runs_url}/{id}",
                "head_branch": branch,
                "head_repository": {"full_name": repo},
            }

        mocked_fetch_runs.side_effect = [
            {"workflow_runs": [run(1)]},
            {"workflow_runs": [run(1), run(2), run(3, branch="x"), run(4, repo="x")]},
        ]
        mocked_fetch_url.side_effect = [RuntimeError("API error"), None]
        mocked_time.monotonic.side_effect = [0, 10, 60]
        pr = GitHubPR("pytorch", "pytorch", 31093)
        approve_pending_ci(pr, "sha")

        params = {
            "head_sha": "sha",
            "event": "pull_request",
            "status": "action_required",
            "per_page": 100,
        }
        self.assertEqual(
            mocked_fetch_runs.call_args_list, [mock.call(runs_url, params)] * 2
        )
        self.assertEqual(
            [c.args[0] for c in mocked_fetch_url.call_args_list],
            [f"{runs_url}/1/approve", f"{runs_url}/2/approve"],
        )

    @mock.patch("trymerge.gh_graphql", side_effect=mocked_gh_graphql)
    @mock.patch("tryrebase.pr_runs")
    @mock.patch("tryrebase.gh_fetch_json_dict")
    def test_maintainer_approved_sha(
        self, mocked_fetch_perm: Any, mocked_runs: Any, mocked_gql: Any
    ) -> None:
        "Tests CI is only approved for unedited merge comments from maintainers"
        pr = GitHubPR("pytorch", "pytorch", 31093)
        for editor, permission, sha, run_created, expected in [
            (None, "write", "sha", "t0", "sha"),
            (None, "admin", "sha", "t0", "sha"),
            (None, "read", "sha", "t0", None),
            (None, "none", "sha", "t0", None),
            ("someone", "write", "sha", "t0", None),
            (None, "write", None, "t0", None),
            # CI of sha was triggered after the comment, e.g. by a backdated commit
            (None, "write", "sha", "t2", None),
            (None, RuntimeError("404"), "sha", "t0", None),
        ]:
            comment = GitHubComment(
                body_text="@pytorchbot merge -r",
                created_at="t1",
                author_login="maintainer",
                author_url=None,
                author_association="CONTRIBUTOR",
                editor_login=editor,
                database_id=1,
                url="",
            )
            if isinstance(permission, Exception):
                mocked_fetch_perm.side_effect = permission
            else:
                mocked_fetch_perm.side_effect = None
                mocked_fetch_perm.return_value = {"permission": permission}
            mocked_runs.return_value = [{"created_at": run_created}]
            with (
                mock.patch.object(pr, "get_comment_by_id", return_value=comment),
                mock.patch.object(pr, "get_commit_sha_at_comment", return_value=sha),
            ):
                self.assertEqual(maintainer_approved_sha(pr, 1), expected)
        self.assertIsNone(maintainer_approved_sha(pr, None))
        mocked_fetch_perm.assert_called_with(
            "https://api.github.com/repos/pytorch/pytorch/collaborators/maintainer/permission"  # @lint-ignore
        )
        mocked_runs.assert_called_with(pr, "sha")


if __name__ == "__main__":
    main()
