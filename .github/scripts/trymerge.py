#!/usr/bin/env python3

import os
import re
import time
from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    cast,
)

from gitutils import (
    can_skip_internal_checks,
    get_combined_checks_from_pr_and_land_validation,
    get_git_remote_name,
    get_git_repo_dir,
    get_ghstack_prs,
    get_land_checkrun_conclusions,
    gh_post_pr_comment,
    gh_post_commit_comment,
    gh_add_labels,
    fetch_json,
    find_matching_merge_rule,
    categorize_checks,
    JobNameToStateDict,
    JobCheckState,
    GitRepo,
    GitHubPR,
    MandatoryChecksMissingError,
    prefix_with_github_url,
)
from trymerge_explainer import (
    TryMergeExplainer,
    get_revert_message,
)
import github_constants as gh_constants
from label_utils import (
    has_required_labels,
    delete_all_label_err_comments,
    LABEL_ERR_MSG,
)


def parse_args() -> Any:
    from argparse import ArgumentParser
    parser = ArgumentParser("Merge PR into default branch")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--on-green", action="store_true")
    parser.add_argument("--on-mandatory", action="store_true")
    parser.add_argument("--land-checks", action="store_true")
    parser.add_argument("--revert", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--comment-id", type=int)
    parser.add_argument("--reason", type=str)
    parser.add_argument("pr_num", type=int)
    return parser.parse_args()


class PostCommentError(Exception):
    pass


def checks_to_str(checks: List[Tuple[str, Optional[str]]]) -> str:
    return ", ".join(f"[{c[0]}]({c[1]})" if c[1] is not None else c[0] for c in checks)


def filter_checks_with_lambda(
    checks: JobNameToStateDict,
    status_filter: Callable[[Optional[str]], bool]
) -> List[JobCheckState]:
    return [check for check in checks.values() if status_filter(check.status)]

def validate_revert(repo: GitRepo, pr: GitHubPR, *,
                    comment_id: Optional[int] = None) -> Tuple[str, str]:
    comment = pr.get_last_comment() if comment_id is None else pr.get_comment_by_id(comment_id)
    if comment.editor_login is not None:
        raise PostCommentError("Don't want to revert based on edited command")
    author_association = comment.author_association
    author_login = comment.author_login
    allowed_reverters = ["COLLABORATOR", "MEMBER", "OWNER"]
    # For some reason, one can not be a member of private repo, only CONTRIBUTOR
    if pr.is_base_repo_private():
        allowed_reverters.append("CONTRIBUTOR")
    if author_association not in allowed_reverters:
        raise PostCommentError((
            f"Will not revert as @{author_login} is not one of "
            f"[{', '.join(allowed_reverters)}], but instead is {author_association}."
        ))
    skip_internal_checks = can_skip_internal_checks(pr, comment_id)

    # Raises exception if matching rule is not found, but ignores all status checks
    find_matching_merge_rule(pr, repo, skip_mandatory_checks=True, skip_internal_checks=skip_internal_checks)
    commit_sha = pr.get_merge_commit()
    if commit_sha is None:
        commits = repo.commits_resolving_gh_pr(pr.pr_num)
        if len(commits) == 0:
            raise PostCommentError("Can't find any commits resolving PR")
        commit_sha = commits[0]
    msg = repo.commit_message(commit_sha)
    rc = gh_constants.RE_DIFF_REV.search(msg)
    if rc is not None and not skip_internal_checks:
        raise PostCommentError(
            f"Can't revert PR that was landed via phabricator as {rc.group(1)}.  " +
            "Please revert by going to the internal diff and clicking Unland."
        )
    return (author_login, commit_sha)


def try_revert(repo: GitRepo, pr: GitHubPR, *,
               dry_run: bool = False,
               comment_id: Optional[int] = None,
               reason: Optional[str] = None) -> None:
    def post_comment(msg: str) -> None:
        gh_post_pr_comment(pr.org, pr.project, pr.pr_num, msg, dry_run=dry_run)
    try:
        author_login, commit_sha = validate_revert(repo, pr, comment_id=comment_id)
    except PostCommentError as e:
        return post_comment(str(e))
    revert_msg = f"\nReverted {pr.get_pr_url()} on behalf of {prefix_with_github_url(author_login)}"
    revert_msg += f" due to {reason}\n" if reason is not None else "\n"
    repo.checkout(pr.default_branch())
    repo.revert(commit_sha)
    msg = repo.commit_message("HEAD")
    msg = re.sub(gh_constants.RE_PULL_REQUEST_RESOLVED, "", msg)
    msg += revert_msg
    repo.amend_commit_message(msg)
    repo.push(pr.default_branch(), dry_run)
    post_comment(f"@{pr.get_pr_creator_login()} your PR has been successfully reverted.")
    if not dry_run:
        pr.add_numbered_label("reverted")
        gh_post_commit_comment(pr.org, pr.project, commit_sha, revert_msg)


def check_for_sev(org: str, project: str, skip_mandatory_checks: bool) -> None:
    if skip_mandatory_checks:
        return
    response = cast(
        Dict[str, Any],
        fetch_json(
            "https://api.github.com/search/issues",
            params={"q": f'repo:{org}/{project} is:open is:issue label:"ci: sev"'},
        ),
    )
    if response["total_count"] != 0:
        for item in response["items"]:
            if "merge blocking" in item["body"].lower():
                raise RuntimeError(
                    "Not merging any PRs at the moment because there is a "
                    + "merge blocking https://github.com/pytorch/pytorch/labels/ci:%20sev issue open at: \n"
                    + f"{item['html_url']}"
                )
    return

def validate_land_time_checks(org: str, project: str, commit: str) -> None:
    checks = get_land_checkrun_conclusions(org, project, commit)
    if len(checks) == 0:
        raise MandatoryChecksMissingError("Refusing to merge as land check(s) are not yet run")

    [pending_checks, failed_checks] = categorize_checks(checks, list(checks.keys()))

    if len(failed_checks) > 0:
        raise RuntimeError(f"Failed to merge; some land checks failed: {checks_to_str(failed_checks)}")
    if len(pending_checks) > 0:
        raise MandatoryChecksMissingError(f"Refusing to merge as land check(s) {checks_to_str(pending_checks)} are not yet run")

def merge(pr_num: int, repo: GitRepo,
          dry_run: bool = False,
          skip_mandatory_checks: bool = False,
          comment_id: Optional[int] = None,
          mandatory_only: bool = False,
          on_green: bool = False,
          land_checks: bool = False,
          timeout_minutes: int = 400,
          stale_pr_days: int = 3) -> None:
    repo = GitRepo(get_git_repo_dir(), get_git_remote_name())
    org, project = repo.gh_owner_and_name()
    pr = GitHubPR(org, project, pr_num)
    initial_commit_sha = pr.last_commit()['oid']
    print(f"Attempting merge of {initial_commit_sha}")

    explainer = TryMergeExplainer(skip_mandatory_checks, on_green, land_checks, pr.get_labels(), pr.pr_num, org, project)
    on_green, land_checks = explainer.get_flags()
    land_check_commit = None

    if pr.is_ghstack_pr():
        get_ghstack_prs(repo, pr)  # raises error if out of sync

    check_for_sev(org, project, skip_mandatory_checks)

    if skip_mandatory_checks or can_skip_internal_checks(pr, comment_id):
        # do not wait for any pending signals if PR is closed as part of co-development process
        gh_post_pr_comment(org, project, pr.pr_num, explainer.get_merge_message(), dry_run=dry_run)
        return pr.merge_into(
            repo,
            dry_run=dry_run,
            skip_mandatory_checks=skip_mandatory_checks,
            comment_id=comment_id
        )

    # Important: check for merge rule once before starting land checks
    # because we want to make sure that only approved PRs can start CI
    # jobs. If there's missing approval, a RuntimeError will be raised
    # here to stop the merge process right away
    find_matching_merge_rule(pr, repo, skip_mandatory_checks=True)

    if not has_required_labels(pr):
        raise RuntimeError(LABEL_ERR_MSG.lstrip(" #"))
    else:
        delete_all_label_err_comments(pr)

    if land_checks and not dry_run:
        land_check_commit = pr.create_land_time_check_branch(
            repo,
            'viable/strict',
            skip_mandatory_checks=skip_mandatory_checks,
            comment_id=comment_id
        )

    gh_post_pr_comment(org, project, pr.pr_num, explainer.get_merge_message(land_check_commit), dry_run=dry_run)
    if (datetime.utcnow() - pr.last_pushed_at()).days > stale_pr_days:
        if land_checks and not dry_run:
            pr.delete_land_time_check_branch(repo)
        raise RuntimeError(f"This PR is too stale; the last push date was more than {stale_pr_days} days ago. "
                           "Please rebase and try again. You can rebase by leaving the following comment on this PR:\n"
                           "`@pytorchbot rebase`")

    start_time = time.time()
    last_exception = ''
    elapsed_time = 0.0
    while elapsed_time < timeout_minutes * 60:
        check_for_sev(org, project, skip_mandatory_checks)
        current_time = time.time()
        elapsed_time = current_time - start_time
        print(f"Attempting merge of https://github.com/{org}/{project}/pull/{pr_num} ({elapsed_time / 60} minutes elapsed)")
        pr = GitHubPR(org, project, pr_num)
        if initial_commit_sha != pr.last_commit()['oid']:
            if land_checks and not dry_run:
                pr.delete_land_time_check_branch(repo)
            raise RuntimeError("New commits were pushed while merging. Please rerun the merge command.")
        try:
            required_checks = []
            failed_rule_message = None
            try:
                find_matching_merge_rule(pr, repo)
            except MandatoryChecksMissingError as ex:
                if ex.rule is not None and ex.rule.mandatory_checks_name is not None:
                    required_checks = ex.rule.mandatory_checks_name
                failed_rule_message = ex

            checks = get_combined_checks_from_pr_and_land_validation(pr, land_check_commit)
            pending, failing = categorize_checks(checks, required_checks + [x for x in checks.keys() if x not in required_checks])
            # HACK until GitHub will be better about surfacing those
            startup_failures = filter_checks_with_lambda(checks, lambda status: status == "STARTUP_FAILURE")
            if len(startup_failures) > 0:
                raise RuntimeError(f"{len(startup_failures)} STARTUP failures reported, please check workflows syntax! " +
                                   ', '.join(f"[{x.name}]({x.url})" for x in startup_failures[:5]))
            # END of HACK

            if len(failing) > 0:
                raise RuntimeError(f"{len(failing)} jobs have failed, first few of them are: " +
                                   ', '.join(f"[{x[0]}]({x[1]})" for x in failing[:5]))
            if len(pending) > 0:
                if failed_rule_message is not None:
                    raise failed_rule_message
                else:
                    raise MandatoryChecksMissingError(f"Still waiting for {len(pending)} jobs to finish, " +
                                                      f"first few of them are: {', '.join(x[0] for x in pending[:5])}")
            if land_checks and land_check_commit is not None:
                validate_land_time_checks(org, project, land_check_commit)

            return pr.merge_into(
                repo,
                dry_run=dry_run,
                skip_mandatory_checks=skip_mandatory_checks,
                comment_id=comment_id,
                land_check_commit=land_check_commit
            )
        except MandatoryChecksMissingError as ex:
            last_exception = str(ex)
            print(f"Merge of https://github.com/{org}/{project}/pull/{pr_num} failed due to: {ex}. Retrying in 5 min")
            time.sleep(5 * 60)
        except RuntimeError:
            if land_checks and not dry_run:
                pr.delete_land_time_check_branch(repo)
            raise
    # Finally report timeout back
    msg = f"Merged timed out after {timeout_minutes} minutes. Please contact the pytorch_dev_infra team."
    msg += f"The last exception was: {last_exception}"
    if not dry_run:
        if land_checks:
            pr.delete_land_time_check_branch(repo)
        gh_add_labels(org, project, pr_num, ["land-failed"])
    raise RuntimeError(msg)


def main() -> None:
    args = parse_args()
    repo = GitRepo(get_git_repo_dir(), get_git_remote_name())
    org, project = repo.gh_owner_and_name()
    pr = GitHubPR(org, project, args.pr_num)

    def handle_exception(e: Exception, title: str = "Merge failed") -> None:
        exception = f"**Reason**: {e}"

        internal_debugging = ""
        run_url = os.getenv("GH_RUN_URL")
        if run_url is not None:
            # Hide this behind a collapsed bullet since it's not helpful to most devs
            internal_debugging = "\n".join((
                "<details><summary>Details for Dev Infra team</summary>",
                f"Raised by <a href=\"{run_url}\">workflow job</a>",
                "</details>"
            ))

        msg = "\n".join((
            f"## {title}",
            f"{exception}",
            "",
            f"{internal_debugging}"
        ))

        gh_post_pr_comment(org, project, args.pr_num, msg, dry_run=args.dry_run)
        import traceback
        traceback.print_exc()

    if args.revert:
        try:
            gh_post_pr_comment(org, project, args.pr_num, get_revert_message(org, project, pr.pr_num), args.dry_run)
            try_revert(repo, pr, dry_run=args.dry_run, comment_id=args.comment_id, reason=args.reason)
        except Exception as e:
            handle_exception(e, f"Reverting PR {args.pr_num} failed")
        return

    if pr.is_closed():
        gh_post_pr_comment(org, project, args.pr_num, f"Can't merge closed PR #{args.pr_num}", dry_run=args.dry_run)
        return

    if pr.is_cross_repo() and pr.is_ghstack_pr():
        gh_post_pr_comment(org, project, args.pr_num, "Cross-repo ghstack merges are not supported", dry_run=args.dry_run)
        return

    try:
        merge(args.pr_num, repo,
              dry_run=args.dry_run,
              skip_mandatory_checks=args.force,
              comment_id=args.comment_id,
              on_green=args.on_green,
              mandatory_only=args.on_mandatory,
              land_checks=args.land_checks)
    except Exception as e:
        handle_exception(e)


if __name__ == "__main__":
    main()
