#!/usr/bin/env python3

from __future__ import annotations

import contextlib
import json
import os
import re
from typing import Any, cast
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


def get_merge_commit_sha(repo: GitRepo, pr: GitHubPR) -> str | None:
    """
    Return the merge commit SHA iff the PR has been merged. For simplicity, we
    will only cherry pick PRs that have been merged into main.

    A merged PR that was later reverted is reopened by the revert bot, so
    is_closed() alone would reject exactly the PRs the revert path exists for.
    Accept an open PR only when its landed commit was in fact reverted.
    """
    commit_sha = get_pr_commit_sha(repo, pr)
    if pr.is_closed():
        return commit_sha
    return commit_sha if find_revert_commit_sha(repo, pr, commit_sha) else None


def is_ancestor(repo: GitRepo, ref: str, branch: str) -> bool:
    """
    Return True if ref is already part of branch's history
    """
    return repo.get_merge_base(ref, branch) == repo.rev_parse(ref)


def find_revert_commit_sha(repo: GitRepo, pr: GitHubPR, commit_sha: str) -> str | None:
    """
    Return the commit on the default branch that reverts commit_sha, or None if
    the landed commit was never reverted.

    The lookup is keyed on the exact landed SHA rather than the PR number: a PR
    that landed and was reverted several times has one revert per land, and only
    the one undoing the land that reached the release branch is the right pick.
    """
    default_branch = f"{repo.remote}/{pr.default_branch()}"
    # `git revert`, which is what the revert bot runs, always records this line
    revert_shas = repo._run_git(
        "log",
        "--format=%H",
        "--fixed-strings",
        f"--grep=This reverts commit {commit_sha}.",
        default_branch,
    ).strip()

    if not revert_shas:
        return None

    # Newest first, so a re-reverted commit resolves to the most recent revert
    return revert_shas.split("\n")[0]


def resolve_trunk_revert(
    repo: GitRepo, pr: GitHubPR, onto_branch: str, commit_sha: str
) -> str | None:
    """
    Return the default-branch revert that the release branch is missing, or None
    when the PR is a plain cherry pick.

    When the PR landed and was later reverted on the default branch, applying the
    landed commit would re-add a change trunk has already thrown out, to a branch
    that already has it. What the release branch needs is for that change to be
    undone, so the release branch ends up matching trunk.
    """
    revert_sha = find_revert_commit_sha(repo, pr, commit_sha)
    if not revert_sha:
        return None

    # This runs before create_cherry_pick_branch checks the branch out, so it
    # only exists as a remote-tracking ref. git rev-parse's DWIM does not reach
    # refs/remotes/origin/<branch> from a bare name, so qualify it.
    onto_ref = f"{repo.remote}/{onto_branch}"

    if not is_ancestor(repo, commit_sha, onto_ref):
        raise RuntimeError(
            f"Refuse to cherry pick #{pr.pr_num} onto {onto_branch}: it was reverted "
            f"on {pr.default_branch()} by {revert_sha}, and its landed commit "
            f"{commit_sha} is not on {onto_branch}, so there is nothing to revert there."
        )

    if is_ancestor(repo, revert_sha, onto_ref):
        raise RuntimeError(
            f"Refuse to cherry pick #{pr.pr_num} onto {onto_branch}: the revert "
            f"{revert_sha} is already on {onto_branch}."
        )

    return revert_sha


def get_release_version(onto_branch: str) -> str | None:
    """
    Return the release version if the target branch is a release branch
    """
    m = re.match(RELEASE_BRANCH_REGEX, onto_branch)
    return m.group("version") if m else ""


def get_tracker_issues(
    org: str, project: str, onto_branch: str
) -> list[dict[str, Any]]:
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
    trunk_revert_sha: str = "",
) -> None:
    """
    Create a local branch to cherry pick the commit and submit it as a pull request
    """
    is_revert = bool(trunk_revert_sha)
    current_branch = repo.current_branch()
    cherry_pick_branch = create_cherry_pick_branch(
        github_actor, repo, pr, commit_sha, onto_branch, trunk_revert_sha
    )

    try:
        org, project = repo.gh_owner_and_name()

        cherry_pick_pr = ""
        if not dry_run:
            cherry_pick_pr = submit_pr(
                repo, pr, cherry_pick_branch, onto_branch, commit_sha, trunk_revert_sha
            )

        tracker_issues_comments = []
        tracker_issues = get_tracker_issues(org, project, onto_branch)
        for issue in tracker_issues:
            issue_number = int(str(issue.get("number", "0")))
            if not issue_number:
                continue

            res = cast(
                dict[str, Any],
                post_tracker_issue_comment(
                    org,
                    project,
                    issue_number,
                    pr.pr_num,
                    cherry_pick_pr,
                    classification,
                    fixes,
                    dry_run,
                    is_revert,
                ),
            )

            comment_url = res.get("html_url", "")
            if comment_url:
                tracker_issues_comments.append(comment_url)

        msg = f"The cherry pick PR is at {cherry_pick_pr}"
        if is_revert:
            msg = (
                f"This PR was reverted on {pr.default_branch()}, so the change was "
                f"reverted on {onto_branch} instead of being applied there. {msg}"
            )
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
    github_actor: str,
    repo: GitRepo,
    pr: GitHubPR,
    commit_sha: str,
    onto_branch: str,
    trunk_revert_sha: str = "",
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
    if trunk_revert_sha:
        # Revert on the release branch rather than cherry picking the default
        # branch's revert commit. The revert is then computed against the tree
        # the release branch actually has, which is what makes it apply on a
        # branch that has drifted from trunk, and it records the landed SHA that
        # is really present here instead of trunk's.
        try:
            repo.revert(commit_sha)
        except RuntimeError as error:
            # Deciding what a conflicted revert should look like on a release
            # branch is a judgement call, so hand it back rather than guess.
            with contextlib.suppress(RuntimeError):
                repo._run_git("revert", "--abort")
            raise RuntimeError(
                f"Reverting {commit_sha} on {onto_branch} hit a conflict, so nothing "
                f"was pushed. Please revert #{pr.pr_num} on {onto_branch} by hand and "
                f"open the pull request manually.\n\n{error}"
            ) from error
        repo.amend_commit_message(
            "\n".join(
                (
                    repo.commit_message("HEAD").rstrip(),
                    "",
                    f"Reverted on {pr.default_branch()} by {trunk_revert_sha}.",
                )
            )
        )
    else:
        repo._run_git("cherry-pick", "-x", commit_sha)
    repo.push(branch=cherry_pick_branch, dry_run=False)

    return cherry_pick_branch


def submit_pr(
    repo: GitRepo,
    pr: GitHubPR,
    cherry_pick_branch: str,
    onto_branch: str,
    commit_sha: str = "",
    trunk_revert_sha: str = "",
) -> str:
    """
    Submit the cherry pick PR and return the link to the PR
    """
    org, project = repo.gh_owner_and_name()

    default_msg = f"Cherry pick #{pr.pr_num} onto {onto_branch} branch"
    title = pr.info.get("title", default_msg)
    body = pr.info.get("body", default_msg)

    if trunk_revert_sha:
        # Reusing the PR's own title and body would describe the change being
        # undone, which reads as though the feature is being added to the release
        title = f'[{onto_branch}] Revert "{title}"'
        body = "\n".join(
            (
                f"Reverts #{pr.pr_num} on {onto_branch}.",
                "",
                f"#{pr.pr_num} landed before {onto_branch} was cut, so the change is "
                f"on the release branch, but it was then reverted on "
                f"{pr.default_branch()} by {trunk_revert_sha}. This brings "
                f"{onto_branch} back in line with trunk.",
                "",
                f"The revert was generated on {onto_branch} rather than cherry picked "
                f"from {pr.default_branch()}, so it undoes {commit_sha} as that commit "
                "exists on the release branch.",
                "",
                repo.commit_message(trunk_revert_sha),
            )
        )

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
) -> list[dict[str, Any]]:
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
    is_revert: bool = False,
) -> list[dict[str, Any]]:
    """
    Post a comment on the tracker issue (if any) to record the cherry pick
    """
    # Skip the empty parts, otherwise an unlinked cherry pick renders a dangling
    # separator like "Critical - "
    criteria = " - ".join(
        part
        for part in (
            "Cherry-pick revert" if is_revert else "",
            classification.capitalize(),
            fixes.capitalize(),
        )
        if part
    )

    comment = "\n".join(
        (
            "Link to landed trunk PR (if applicable):",
            f"* https://github.com/{org}/{project}/pull/{pr_num}",
            "",
            "Link to release branch PR:",
            f"* {cherry_pick_pr}",
            "",
            "Criteria Category:",
            criteria,
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

        trunk_revert_sha = resolve_trunk_revert(repo, pr, args.onto_branch, commit_sha)

        cherry_pick(
            args.github_actor,
            repo,
            pr,
            commit_sha,
            args.onto_branch,
            args.classification,
            args.fixes,
            args.dry_run,
            trunk_revert_sha or "",
        )

    except RuntimeError as error:
        if not args.dry_run:
            post_pr_comment(org, project, pr_num, str(error))
        else:
            raise error


if __name__ == "__main__":
    main()
