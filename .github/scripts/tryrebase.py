#!/usr/bin/env python3

import os
from typing import Any
from gitutils import get_git_remote_name, get_git_repo_dir, GitRepo
from trymerge import gh_post_comment, GitHubPR


def parse_args() -> Any:
    from argparse import ArgumentParser
    parser = ArgumentParser("Rebase PR into branch")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("pr_num", type=int)
    return parser.parse_args()


def rebase_onto(pr: GitHubPR, repo: GitRepo, dry_run: bool = False) -> None:
    branch = f"pull/{pr.pr_num}/head"
    onto_branch = pr.default_branch()
    remote_url = f"https://github.com/{pr.info['headRepository']['nameWithOwner']}.git"
    refspec = f"{branch}:{pr.head_ref()}"

    repo.fetch(branch, branch)
    repo._run_git("rebase", onto_branch, branch)
    if dry_run:
        repo._run_git("push", "--dry-run", "-f", remote_url, refspec)
    else:
        push_result = repo._run_git("push", "-f", remote_url, refspec)
        if "Everything up-to-date" in push_result:
            gh_post_comment(pr.org, pr.project, pr.pr_num,
                            f"Tried to rebase and push PR #{pr.pr_num}, but it was already up to date", dry_run=dry_run)
        else:
            gh_post_comment(pr.org, pr.project, pr.pr_num,
                            f"Successfully rebased {pr.head_ref()} onto {onto_branch}, please pull locally before adding more changes", dry_run=dry_run)


def main() -> None:
    args = parse_args()
    repo = GitRepo(get_git_repo_dir(), get_git_remote_name(), debug=True)
    org, project = repo.gh_owner_and_name()

    pr = GitHubPR(org, project, args.pr_num)

    if pr.is_closed():
        gh_post_comment(org, project, args.pr_num, f"PR #{args.pr_num} is closed, won't rebase", dry_run=args.dry_run)
        return

    if pr.is_ghstack_pr():
        gh_post_comment(org, project, args.pr_num,
                        f"PR #{args.pr_num} is a ghstack, which is currently not supported", dry_run=args.dry_run)
        return

    try:
        rebase_onto(pr, repo, dry_run=args.dry_run)
    except Exception as e:
        msg = f"Rebase failed due to {e}"
        run_url = os.getenv("GH_RUN_URL")
        if run_url is not None:
            msg += f"\nRaised by {run_url}"
        gh_post_comment(org, project, args.pr_num, msg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
