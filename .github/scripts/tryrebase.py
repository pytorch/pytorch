#!/usr/bin/env python3

import contextlib
import os
import re
import subprocess
import sys
import time
from collections.abc import Generator
from typing import Any

from github_utils import (
    gh_fetch_json_dict,
    gh_fetch_url,
    gh_post_pr_comment as gh_post_comment,
)
from gitutils import get_git_remote_name, get_git_repo_dir, GitRepo
from trymerge import GitHubPR


SAME_SHA_ERROR = (
    "\n```\nAborting rebase because rebasing the branch resulted in the same sha as the target branch.\n"
    + "This usually happens because the PR has already been merged.  Please rebase locally and push.\n```"
)


def parse_args() -> Any:
    from argparse import ArgumentParser

    parser = ArgumentParser("Rebase PR into branch")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--branch", type=str)
    parser.add_argument("--comment-id", type=int)
    parser.add_argument("pr_num", type=int)
    return parser.parse_args()


def post_already_uptodate(
    pr: GitHubPR, repo: GitRepo, onto_branch: str, dry_run: bool
) -> None:
    msg = f"Tried to rebase and push PR #{pr.pr_num}, but it was already up to date."
    def_branch = pr.default_branch()
    def_branch_fcn = f"refs/remotes/{repo.remote}/{def_branch}"
    if onto_branch != def_branch_fcn and repo.rev_parse(
        def_branch_fcn
    ) != repo.rev_parse(onto_branch):
        def_branch_url = f"https://github.com/{pr.org}/{pr.project}/tree/{def_branch}"
        msg += f" Try rebasing against [{def_branch}]({def_branch_url}) by issuing:"
        msg += f"\n`@pytorchbot rebase -b {def_branch}`"

    gh_post_comment(
        pr.org,
        pr.project,
        pr.pr_num,
        msg,
        dry_run=dry_run,
    )


def pr_runs(pr: GitHubPR, sha: str, **params: Any) -> list[dict[str, Any]]:
    """pull_request workflow runs of the PR's commit sha."""
    head_repo = pr.info["headRepository"]["nameWithOwner"]
    runs = gh_fetch_json_dict(
        f"https://api.github.com/repos/{pr.org}/{pr.project}/actions/runs",
        {"head_sha": sha, "event": "pull_request", "per_page": 100, **params},
    )["workflow_runs"]
    return [
        r
        for r in runs
        if r["head_branch"] == pr.head_ref()
        and (r["head_repository"] or {}).get("full_name") == head_repo
    ]


def maintainer_approved_sha(pr: GitHubPR, comment_id: int | None) -> str | None:
    """PR head when comment_id was posted, if it was posted by a maintainer."""
    if comment_id is None:
        return None
    try:
        comment = pr.get_comment_by_id(comment_id)
        if comment.editor_login is not None:
            return None
        perm = gh_fetch_json_dict(
            f"https://api.github.com/repos/{pr.org}/{pr.project}/collaborators/{comment.author_login}/permission"
        )
        if perm["permission"] not in ("admin", "write"):
            print(f"@{comment.author_login} is not a maintainer, not approving CI")
            return None
        sha = pr.get_commit_sha_at_comment(comment_id)
        # The timeline orders commits by their author-controlled dates, so also
        # require CI of sha to have been triggered before the comment.
        if sha is None or not any(
            r["created_at"] < comment.created_at for r in pr_runs(pr, sha)
        ):
            print(f"Can't tell which commit comment {comment_id} was posted on")
            return None
        return sha
    except Exception as e:
        print(f"Failed to check comment {comment_id}: {e}")
        return None


def approve_pending_ci(
    pr: GitHubPR, sha: str, timeout: float = 60, interval: float = 10
) -> None:
    """Approve held CI of the PR's commit sha.

    GitHub holds outside contributors' CI for maintainer approval again after the
    rebase is pushed. Runs are created gradually after a push, so keep polling until
    the timeout.
    """
    runs_url = f"https://api.github.com/repos/{pr.org}/{pr.project}/actions/runs"
    approved: set[int] = set()
    deadline = time.monotonic() + timeout
    while True:
        for run in pr_runs(pr, sha, status="action_required"):
            if run["id"] in approved:
                continue
            approved.add(run["id"])
            print(f"Approving {run['html_url']}")
            try:
                gh_fetch_url(
                    f"{runs_url}/{run['id']}/approve", method="POST", reader=lambda x: x
                )
            except Exception as e:
                print(f"Failed to approve {run['html_url']}: {e}")
        if time.monotonic() >= deadline:
            return
        time.sleep(interval)


def rebase_onto(
    pr: GitHubPR,
    repo: GitRepo,
    onto_branch: str,
    dry_run: bool = False,
    approve_ci_sha: str | None = None,
) -> bool:
    branch = f"pull/{pr.pr_num}/head"
    head_repo = pr.info["headRepository"]
    if head_repo is None:
        raise RuntimeError(
            f"Can't determine the head repository of PR #{pr.pr_num}; its org likely "
            "forbids access via mergebot's token (classic PAT), so the rebased branch "
            "can't be pushed back. Please rebase locally and push."
        )
    remote_url = f"https://github.com/{head_repo['nameWithOwner']}.git"
    refspec = f"{branch}:{pr.head_ref()}"

    repo.fetch(branch, branch)
    orig_sha = repo.rev_parse(branch)
    # Rebase only the PR's own commits. The 2-arg `git rebase <onto_branch>
    # <branch>` form replays all of onto_branch..branch, which grafts in trunk
    # commits when the PR has merged its base in and onto_branch (e.g.
    # viable/strict) lags that base. Anchoring on the fork point from the PR's
    # base branch excludes anything already on the base. See #187374.
    base_branch = f"refs/remotes/{repo.remote}/{pr.base_ref()}"
    fork_point = repo.get_merge_base(base_branch, branch)
    repo._run_git("rebase", "--onto", onto_branch, fork_point, branch)

    if repo.rev_parse(branch) == repo.rev_parse(onto_branch):
        raise Exception(SAME_SHA_ERROR)  # noqa: TRY002

    if dry_run:
        push_result = repo._run_git("push", "--dry-run", "-f", remote_url, refspec)
    else:
        push_result = repo._run_git("push", "-f", remote_url, refspec)
    if "Everything up-to-date" in push_result:
        post_already_uptodate(pr, repo, onto_branch, dry_run)
        return False
    else:
        gh_post_comment(
            pr.org,
            pr.project,
            pr.pr_num,
            f"Successfully rebased `{pr.head_ref()}` onto `{onto_branch}`, please pull locally "
            + f"before adding more changes (for example, via `git checkout {pr.head_ref()} && "
            + "git pull --rebase`)",
            dry_run=dry_run,
        )
        if not dry_run and approve_ci_sha is not None and orig_sha == approve_ci_sha:
            try:
                approve_pending_ci(pr, repo.rev_parse(branch))
            except Exception as e:
                print(f"Failed to approve CI: {e}")
        return True


def rebase_ghstack_onto(
    pr: GitHubPR, repo: GitRepo, onto_branch: str, dry_run: bool = False
) -> bool:
    if (
        subprocess.run(
            [sys.executable, "-m", "ghstack", "--help"],
            capture_output=True,
            check=False,
        ).returncode
        != 0
    ):
        subprocess.run([sys.executable, "-m", "pip", "install", "ghstack"], check=True)
    orig_ref = f"{re.sub(r'/head$', '/orig', pr.head_ref())}"

    repo.fetch(orig_ref, orig_ref)
    repo._run_git("rebase", onto_branch, orig_ref)

    if repo.rev_parse(orig_ref) == repo.rev_parse(onto_branch):
        raise Exception(SAME_SHA_ERROR)  # noqa: TRY002

    # steal the identity of the committer of the commit on the orig branch
    email = repo._run_git("log", orig_ref, "--pretty=format:%ae", "-1")
    name = repo._run_git("log", orig_ref, "--pretty=format:%an", "-1")
    repo._run_git("config", "--global", "user.email", email)
    repo._run_git("config", "--global", "user.name", name)

    os.environ["OAUTH_TOKEN"] = os.environ["GITHUB_TOKEN"]
    with open(".ghstackrc", "w+") as f:
        f.write(
            "[ghstack]\n"
            + "github_url=github.com\n"
            + "github_username=pytorchmergebot\n"
            + "remote_name=origin"
        )

    if dry_run:
        print("Don't know how to dry-run ghstack")
        return False
    else:
        ghstack_result = subprocess.run(["ghstack"], capture_output=True, check=True)
        push_result = ghstack_result.stdout.decode("utf-8")
        print(push_result)
        if ghstack_result.returncode != 0:
            print(ghstack_result.stderr.decode("utf-8"))
            raise Exception(f"\n```{push_result}```")  # noqa: TRY002
        # The contents of a successful push result should look like:
        # Summary of changes (ghstack 0.6.0)

        #  - Updated https://github.com/clee2000/random-testing-public/pull/2
        #  - Updated https://github.com/clee2000/random-testing-public/pull/1

        # Facebook employees can import your changes by running
        # (on a Facebook machine):

        #     ghimport -s https://github.com/clee2000/random-testing-public/pull/2

        # If you want to work on this diff stack on another machine:

        #     ghstack checkout https://github.com/clee2000/random-testing-public/pull/2
        org, project = repo.gh_owner_and_name()
        for line in push_result.splitlines():
            if "Updated" in line:
                pr_num = int(line.split("/")[-1])
                if pr_num != pr.pr_num:
                    gh_post_comment(
                        pr.org,
                        pr.project,
                        pr_num,
                        f"Rebased `{orig_ref}` onto `{onto_branch}` because #{pr.pr_num} was rebased, "
                        "please pull locally before adding more changes (for example, via `ghstack "
                        + f"checkout https://github.com/{org}/{project}/pull/{pr_num}`)",
                        dry_run=dry_run,
                    )
                else:
                    gh_post_comment(
                        pr.org,
                        pr.project,
                        pr_num,
                        f"Successfully rebased `{orig_ref}` onto `{onto_branch}`, please pull locally "
                        + "before adding more changes (for example, via `ghstack "
                        + f"checkout https://github.com/{org}/{project}/pull/{pr.pr_num}`)",
                        dry_run=dry_run,
                    )

        if (
            f"Skipped https://github.com/{org}/{project}/pull/{pr.pr_num}"
            in push_result
        ):
            post_already_uptodate(pr, repo, onto_branch, dry_run)
            return False
        return True


def additional_rebase_failure_info(e: Exception) -> str:
    if re.search(
        r"remote: Permission to .* denied to .*\.\nfatal: unable to access", str(e)
    ):
        return (
            "\nThis is likely because the author did not allow edits from maintainers on the PR or because the "
            "repo has additional permissions settings that mergebot does not qualify."
        )
    return ""


@contextlib.contextmanager
def git_config_guard(repo: GitRepo) -> Generator[None, None, None]:
    """Restores user.name and user.email global properties after context is finished"""
    user_email = repo._run_git("config", "user.email")
    user_name = repo._run_git("config", "user.name")
    try:
        yield
    finally:
        if user_email:
            repo._run_git("config", "--global", "user.email", user_email)
        if user_name:
            repo._run_git("config", "--global", "user.name", user_name)


def main() -> None:
    args = parse_args()
    repo = GitRepo(get_git_repo_dir(), get_git_remote_name(), debug=True)
    org, project = repo.gh_owner_and_name()

    pr = GitHubPR(org, project, args.pr_num)
    onto_branch = args.branch if args.branch else pr.default_branch()
    onto_branch = f"refs/remotes/{repo.remote}/{onto_branch}"
    onto_branch_url = (
        f"https://github.com/{org}/{project}/commit/{repo.rev_parse(onto_branch)}"
    )

    msg = f"@pytorchbot started a rebase job onto [{onto_branch}]({onto_branch_url})."
    msg += f" Check the current status [here]({os.getenv('GH_RUN_URL')})"
    gh_post_comment(org, project, args.pr_num, msg, dry_run=args.dry_run)

    if pr.is_closed():
        gh_post_comment(
            org,
            project,
            args.pr_num,
            f"PR #{args.pr_num} is closed, won't rebase",
            dry_run=args.dry_run,
        )
        return

    try:
        if pr.is_ghstack_pr():
            with git_config_guard(repo):
                rc = rebase_ghstack_onto(pr, repo, onto_branch, dry_run=args.dry_run)
        else:
            rc = rebase_onto(
                pr,
                repo,
                onto_branch,
                dry_run=args.dry_run,
                approve_ci_sha=maintainer_approved_sha(pr, args.comment_id),
            )
        sys.exit(0 if rc else 1)

    except Exception as e:
        msg = f"Rebase failed due to {e}"
        msg += additional_rebase_failure_info(e)
        run_url = os.getenv("GH_RUN_URL")
        if run_url is not None:
            msg += f"\nRaised by {run_url}"
        gh_post_comment(org, project, args.pr_num, msg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
