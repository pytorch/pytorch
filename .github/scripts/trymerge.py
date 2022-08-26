#!/usr/bin/env python3

import os

from gitutils import (
    GitRepo,
    get_git_remote_name,
    get_git_repo_dir
)
from trymerge_explainer import (
    get_land_check_troubleshooting_message,
    get_revert_message
)
from trymerge_utils import GitHubPR, gh_post_pr_comment, parse_args, try_revert, merge


def main() -> None:
    args = parse_args()
    repo = GitRepo(get_git_repo_dir(), get_git_remote_name())
    org, project = repo.gh_owner_and_name()
    pr = GitHubPR(org, project, args.pr_num)

    def handle_exception(e: Exception, msg: str = "Merge failed") -> None:
        msg += f"\nReason: {e}"
        run_url = os.getenv("GH_RUN_URL")
        if run_url is not None:
            msg += f"\nRaised by [workflow job]({run_url})"
        if args.land_checks:
            msg += get_land_check_troubleshooting_message()
        gh_post_pr_comment(org, project, args.pr_num, msg, dry_run=args.dry_run)
        import traceback

        traceback.print_exc()

    if args.revert:
        try:
            gh_post_pr_comment(
                org,
                project,
                args.pr_num,
                get_revert_message(org, project, pr.pr_num),
                args.dry_run,
            )
            try_revert(
                repo,
                pr,
                dry_run=args.dry_run,
                comment_id=args.comment_id,
                reason=args.reason,
            )
        except Exception as e:
            handle_exception(e, f"Reverting PR {args.pr_num} failed")
        return

    if pr.is_closed():
        gh_post_pr_comment(
            org,
            project,
            args.pr_num,
            f"Can't merge closed PR #{args.pr_num}",
            dry_run=args.dry_run,
        )
        return

    if pr.is_cross_repo() and pr.is_ghstack_pr():
        gh_post_pr_comment(
            org,
            project,
            args.pr_num,
            "Cross-repo ghstack merges are not supported",
            dry_run=args.dry_run,
        )
        return

    try:
        merge(
            args.pr_num,
            repo,
            dry_run=args.dry_run,
            force=args.force,
            comment_id=args.comment_id,
            on_green=args.on_green,
            mandatory_only=args.on_mandatory,
            land_checks=args.land_checks,
        )
    except Exception as e:
        handle_exception(e)


if __name__ == "__main__":
    main()
