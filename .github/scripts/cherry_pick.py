#!/usr/bin/env python3

import json
import os
import re
from typing import Any, cast, Dict, List, Optional
from urllib.error import HTTPError

from github_utils import gh_fetch_url, gh_post_pr_comment, gh_query_issues_by_labels
from gitutils import get_git_remote_name, get_git_repo_dir, GitRepo
from trymerge import get_pr_commit_sha, GitHubPR


# This is only a suggestion for now, not a strict requirement
REQUIRES_ISSUE = {
    "regression",
    "critical",
    "fixnewfeature",
}
RELEASE_BRANCH_REGEX = re.compile(r"release/(?P<version>.+)")


def parse_args() -> Any:
    from argparse import ArgumentParser

    parser = ArgumentParser("cherry pick a landed PR onto a release branch")
    parser.add_argument(
        "--onto-branch", type=str, required=True, help="the target release branch"
    )
    parser.add_argument(
        "--github-actor", type=str, required=True, help="all the world's a stage"
    )
    parser.add_argument(
        "--classification",
        choices=["regression", "critical", "fixnewfeature", "docs", "release"],
        required=True,
        help="the cherry pick category",
    )
    parser.add_argument("pr_num", type=int)
    parser.add_argument(
        "--fixes",
        type=str,
        default="",
        help="the GitHub issue that the cherry pick fixes",
    )
    parser.add_argument("--dry-run", action="store_true")

    return parser.parse_args()


def get_merge_commit_sha(repo: GitRepo, pr: GitHubPR) -> Optional[str]:
    """
    Return the merge commit SHA iff the PR has been merged. For simplicity, we
    will only cherry pick PRs that have been merged into main
    """
    commit_sha = get_pr_commit_sha(repo, pr)
    return commit_sha if pr.is_closed() else None


def get_release_version(onto_branch: str) -> Optional[str]:
    """
    Return the release version if the target branch is a release branch
    """
    m = re.match(RELEASE_BRANCH_REGEX, onto_branch)
    return m.group("version") if m else ""


def get_tracker_issues(
    org: str, project: str, onto_branch: str
) -> List[Dict[str, Any]]:
    """
    Find the tracker issue from the repo. The tracker issue needs to have the title
    like [VERSION] Release Tracker following the convention on PyTorch
    """
    version = get_release_version(onto_branch)
    if not version:
        return []

    tracker_issues = gh_query_issues_by_labels(org, project, labels=["release tracker"])
    if not tracker_issues:
        return []

    # Figure out the tracker issue from the list by looking at the title
    return [issue for issue in tracker_issues if version in issue.get("title", "")]


def cherry_pick(
    github_actor: str,
    repo: GitRepo,
    pr: GitHubPR,
    commit_sha: str,
    onto_branch: str,
    classification: str,
    fixes: str,
    dry_run: bool = False,
) -> None:
    """
    Create a local branch to cherry pick the commit and submit it as a pull request
    """
    current_branch = repo.current_branch()
    cherry_pick_branch = create_cherry_pick_branch(
        github_actor, repo, pr, commit_sha, onto_branch
    )

    try:
        org, project = repo.gh_owner_and_name()

        cherry_pick_pr = ""
        if not dry_run:
            cherry_pick_pr = submit_pr(repo, pr, cherry_pick_branch, onto_branch)

        tracker_issues_comments = []
        tracker_issues = get_tracker_issues(org, project, onto_branch)
        for issue in tracker_issues:
            issue_number = int(str(issue.get("number", "0")))
            if not issue_number:
                continue

            res = cast(
                Dict[str, Any],
                post_tracker_issue_comment(
                    org,
                    project,
                    issue_number,
                    pr.pr_num,
                    cherry_pick_pr,
                    classification,
                    fixes,
                    dry_run,
                ),
            )

            comment_url = res.get("html_url", "")
            if comment_url:
                tracker_issues_comments.append(comment_url)

        msg = f"The cherry pick PR is at {cherry_pick_pr}"
        if fixes:
            msg += f" and it is linked with issue {fixes}."
        elif classification in REQUIRES_ISSUE:
            msg += f" and it is recommended to link a {classification} cherry pick PR with an issue."

        if tracker_issues_comments:
            msg += " The following tracker issues are updated:\n"
            for tracker_issues_comment in tracker_issues_comments:
                msg += f"* {tracker_issues_comment}\n"

        post_pr_comment(org, project, pr.pr_num, msg, dry_run)

    finally:
        if current_branch:
            repo.checkout(branch=current_branch)


def create_cherry_pick_branch(
    github_actor: str, repo: GitRepo, pr: GitHubPR, commit_sha: str, onto_branch: str
) -> str:
    """
    Create a local branch and cherry pick the commit. Return the name of the local
    cherry picking branch.
    """
    repo.checkout(branch=onto_branch)
    repo._run_git("submodule", "update", "--init", "--recursive")

    # Remove all special characters if we want to include the actor in the branch name
    github_actor = re.sub("[^0-9a-zA-Z]+", "_", github_actor)

    cherry_pick_branch = f"cherry-pick-{pr.pr_num}-by-{github_actor}"
    repo.create_branch_and_checkout(branch=cherry_pick_branch)

    # We might want to support ghstack later
    # We don't want to resolve conflicts here.
    repo._run_git("cherry-pick", "-x", commit_sha)
    repo.push(branch=cherry_pick_branch, dry_run=False)

    return cherry_pick_branch


def submit_pr(
    repo: GitRepo,
    pr: GitHubPR,
    cherry_pick_branch: str,
    onto_branch: str,
) -> str:
    """
    Submit the cherry pick PR and return the link to the PR
    """
    org, project = repo.gh_owner_and_name()

    default_msg = f"Cherry pick #{pr.pr_num} onto {onto_branch} branch"
    title = pr.info.get("title", default_msg)
    body = pr.info.get("body", default_msg)

    try:
        response = gh_fetch_url(
            f"https://api.github.com/repos/{org}/{project}/pulls",
            method="POST",
            data={
                "title": title,
                "body": body,
                "head": cherry_pick_branch,
                "base": onto_branch,
            },
            headers={"Accept": "application/vnd.github.v3+json"},
            reader=json.load,
        )

        cherry_pick_pr = response.get("html_url", "")
        if not cherry_pick_pr:
            raise RuntimeError(
                f"Fail to find the cherry pick PR: {json.dumps(response)}"
            )

        return str(cherry_pick_pr)

    except HTTPError as error:
        msg = f"Fail to submit the cherry pick PR: {error}"
        raise RuntimeError(msg) from error


def post_pr_comment(
    org: str, project: str, pr_num: int, msg: str, dry_run: bool = False
) -> List[Dict[str, Any]]:
    """
    Post a comment on the PR itself to point to the cherry picking PR when success
    or print the error when failure
    """
    internal_debugging = ""

    run_url = os.getenv("GH_RUN_URL")
    # Post a comment to tell folks that the PR is being cherry picked
    if run_url is not None:
        internal_debugging = "\n".join(
            line
            for line in (
                "<details><summary>Details for Dev Infra team</summary>",
                f'Raised by <a href="{run_url}">workflow job</a>\n',
                "</details>",
            )
            if line
        )

    comment = "\n".join(
        (f"### Cherry picking #{pr_num}", f"{msg}", "", f"{internal_debugging}")
    )
    return gh_post_pr_comment(org, project, pr_num, comment, dry_run)


def post_tracker_issue_comment(
    org: str,
    project: str,
    issue_num: int,
    pr_num: int,
    cherry_pick_pr: str,
    classification: str,
    fixes: str,
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    """
    Post a comment on the tracker issue (if any) to record the cherry pick
    """
    comment = "\n".join(
        (
            "Link to landed trunk PR (if applicable):",
            f"* https://github.com/{org}/{project}/pull/{pr_num}",
            "",
            "Link to release branch PR:",
            f"* {cherry_pick_pr}",
            "",
            "Criteria Category:",
            " - ".join((classification.capitalize(), fixes.capitalize())),
        )
    )
    return gh_post_pr_comment(org, project, issue_num, comment, dry_run)


def main() -> None:
    args = parse_args()
    pr_num = args.pr_num

    repo = GitRepo(get_git_repo_dir(), get_git_remote_name())
    org, project = repo.gh_owner_and_name()

    pr = GitHubPR(org, project, pr_num)

    try:
        commit_sha = get_merge_commit_sha(repo, pr)
        if not commit_sha:
            raise RuntimeError(
                f"Refuse to cherry pick #{pr_num} because it hasn't been merged yet"
            )

        cherry_pick(
            args.github_actor,
            repo,
            pr,
            commit_sha,
            args.onto_branch,
            args.classification,
            args.fixes,
            args.dry_run,
        )

    except RuntimeError as error:
        if not args.dry_run:
            post_pr_comment(org, project, pr_num, str(error))
        else:
            raise error


if __name__ == "__main__":
    main()
