from typing import Any
from trymerge import gh_post_pr_comment
from gitutils import get_git_remote_name, get_git_repo_dir, GitRepo
from trymerge_explainer import BOT_COMMANDS_WIKI
import os


def parse_args() -> Any:
    from argparse import ArgumentParser

    parser = ArgumentParser("Comment on a PR")
    parser.add_argument("pr_num", type=int)
    parser.add_argument("action", type=str)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = GitRepo(get_git_repo_dir(), get_git_remote_name(), debug=True)
    org, project = repo.gh_owner_and_name()
    run_url = os.environ.get("GH_RUN_URL")

    job_link = f"[job]({run_url})" if run_url is not None else "job"
    msg = (
        f"The {args.action} {job_link} was canceled. If you believe this is a mistake,"
        + f"then you can re trigger it through [pytorch-bot]({BOT_COMMANDS_WIKI})."
    )

    gh_post_pr_comment(org, project, args.pr_num, msg)
    print(org, project, args.pr_num, msg)


if __name__ == "__main__":
    main()
