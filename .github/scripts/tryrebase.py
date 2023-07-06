#!/usr/bin/env python3

import contextlib
import os
import re
import subprocess
import sys
from typing import Any, Generator

from github_utils import gh_post_pr_comment as gh_post_comment
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


def rebase_onto(
    pr: GitHubPR, repo: GitRepo, onto_branch: str, dry_run: bool = False
) -> None:
    branch = f"pull/{pr.pr_num}/head"
    remote_url = f"https://github.com/{pr.info['headRepository']['nameWithOwner']}.git"
    refspec = f"{branch}:{pr.head_ref()}"

    repo.fetch(branch, branch)
    repo._run_git("rebase", onto_branch, branch)

    if repo.rev_parse(branch) == repo.rev_parse(onto_branch):
        raise Exception(SAME_SHA_ERROR)

    if dry_run:
        push_result = repo._run_git("push", "--dry-run", "-f", remote_url, refspec)
    else:
        push_result = repo._run_git("push", "-f", remote_url, refspec)
    if "Everything up-to-date" in push_result:
        post_already_uptodate(pr, repo, onto_branch, dry_run)
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


def rebase_ghstack_onto(
    pr: GitHubPR, repo: GitRepo, onto_branch: str, dry_run: bool = False
) -> None:
    if (
        subprocess.run(
            [sys.executable, "-m", "ghstack", "--help"], capture_output=True
        ).returncode
        != 0
    ):
        subprocess.run([sys.executable, "-m", "pip", "install", "ghstack"])
    orig_ref = f"{re.sub(r'/head$', '/orig', pr.head_ref())}"

    repo.fetch(orig_ref, orig_ref)
    repo._run_git("rebase", onto_branch, orig_ref)

    if repo.rev_parse(orig_ref) == repo.rev_parse(onto_branch):
        raise Exception(SAME_SHA_ERROR)

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
    else:
        ghstack_result = subprocess.run(["ghstack"], capture_output=True)
        push_result = ghstack_result.stdout.decode("utf-8")
        print(push_result)
        if ghstack_result.returncode != 0:
            print(ghstack_result.stderr.decode("utf-8"))
            raise Exception(f"\n```{push_result}```")
        # The contents of a successful push result should look like:
        # Summary of changes (ghstack 0.6.0)

        #  - Updated https://github.com/clee2000/random-testing/pull/2
        #  - Updated https://github.com/clee2000/random-testing/pull/1

        # Facebook employees can import your changes by running
        # (on a Facebook machine):

        #     ghimport -s https://github.com/clee2000/random-testing/pull/2

        # If you want to work on this diff stack on another machine:

        #     ghstack checkout https://github.com/clee2000/random-testing/pull/2
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
                rebase_ghstack_onto(pr, repo, onto_branch, dry_run=args.dry_run)
        else:
            rebase_onto(pr, repo, onto_branch, dry_run=args.dry_run)

    except Exception as e:
        msg = f"Rebase failed due to {e}"
        run_url = os.getenv("GH_RUN_URL")
        if run_url is not None:
            msg += f"\nRaised by {run_url}"
        gh_post_comment(org, project, args.pr_num, msg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
