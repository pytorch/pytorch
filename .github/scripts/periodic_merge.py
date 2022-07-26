from typing import Any, List

from gitutils import GitRepo, get_git_remote_name, get_git_repo_dir
from trymerge import (
    GH_CHECKSUITES_FRAGMENT,
    GH_COMMIT_AUTHORS_FRAGMENT,
    GH_PR_REVIEWS_FRAGMENT,
    GH_PULL_REQUEST_FRAGMENT,
    GitHubPR,
    MandatoryChecksMissingError,
    check_for_sev,
    find_matching_merge_rule,
    gh_add_labels,
    gh_graphql,
    gh_remove_label,
    handle_exception,
    pr_get_failed_checks,
    pr_get_pending_checks,
    validate_land_time_checks,
)

# land state variables to determine if land needs to be merged or the end state
LAND_QUEUED_LABEL = "land-queued"
LAND_PENDING_LABEL = "land-pending"
LAND_FAILED_LABEL = "land-failed"
LAND_SUCCEEDED_LABEL = "land-succeeded"

# land options for checking land checks or all green
LAND_CHECK_LABEL = 'land-checks'
ALL_GREEN_LABEL = 'all-green'

PRS_WITH_LABEL_QUERY = (GH_PULL_REQUEST_FRAGMENT
                        + GH_PR_REVIEWS_FRAGMENT
                        + GH_CHECKSUITES_FRAGMENT
                        + GH_COMMIT_AUTHORS_FRAGMENT) + """
query ($owner: String!, $name: String!, $labels: [String!], $with_labels: Boolean = false) {
  repository(owner: $owner, name: $name) {
    pullRequests(first: 10, labels: $labels, states: OPEN){
        nodes{
        ...PullRequestFragment
      }
    }
  }
}
"""


def fetch_land_pending_pr_numbers(org: str, project: str) -> Any:
    pr_query = gh_graphql(PRS_WITH_LABEL_QUERY,
                          owner=org,
                          name=project,
                          labels=[LAND_PENDING_LABEL, LAND_QUEUED_LABEL],
                          with_labels=True)
    return pr_query['data']['repository']['pullRequests']['nodes']


def get_github_prs(org: str, project: str, pr_query: Any) -> List[GitHubPR]:
    prs = []
    for pr_info in pr_query:
        prs.append(GitHubPR(org, project, pr_info['number'], pr_info))
    return prs


def is_land_check(labels: List[str]) -> bool:
    return LAND_CHECK_LABEL in labels


def is_all_green(labels: List[str]) -> bool:
    return ALL_GREEN_LABEL in labels

def has_label(labels: List[str], label: str) -> bool:
    return ALL_GREEN_LABEL in labels

def validate_all_green(pr: GitHubPR) -> None:
    pending = pr_get_pending_checks(pr)
    failing = pr_get_failed_checks(pr)
    if len(failing) > 0:
        raise RuntimeError(f"{len(failing)} additional jobs have failed, first few of them are: " +
                           ' ,'.join(f"[{x[0]}]({x[1]})" for x in failing[:5]))
    if len(pending) > 0:
        raise MandatoryChecksMissingError(f"Still waiting for {len(pending)} additional jobs to finish, " +
                                          f"first few of them are: {' ,'.join(x[0] for x in pending[:5])}")

def update_label_state(org: str, project: str, pr_num: int, old_state: str, new_state: str) -> None:
    try:
        gh_remove_label(org, project, pr_num, old_state)
        gh_add_labels(org, project, pr_num, [new_state])
    except Exception:
        return


def get_pr_link(org: str, project: str, pr_num: int) -> str:
    return f"[PR {pr_num}](https://github.com/{org}/{project}/pull/{str(pr_num)})"

def main() -> None:
    # Find all the PRs that have land-pending label
    repo = GitRepo(get_git_repo_dir(), get_git_remote_name())
    org, project = repo.gh_owner_and_name()
    pr_query = fetch_land_pending_pr_numbers(org, project)
    prs = get_github_prs(org, project, pr_query)

    pending_prs = ", ".join([str(pr.pr_num) for pr in prs])
    print(f"Attempting to merge PRs: {pending_prs}" if len(prs) > 0
          else "No land-pending pull requests found. Attempting to merge later")
    check_for_sev(org, project, False)

    # Validate each PR and merge them in
    for pr in prs:
        pr_num = pr.pr_num
        labels = pr.get_labels()
        is_queued = has_label(labels, LAND_QUEUED_LABEL)
        pr_link = get_pr_link(org, project, pr_num)
        try:
            if is_queued:
                update_label_state(org, project, pr_num, LAND_QUEUED_LABEL, LAND_PENDING_LABEL)

            if is_all_green(labels):
                validate_all_green(pr)
                print(f"All on green checks passed for {pr_link}.")

            if is_land_check(labels):
                if is_queued:
                    print(f"Creating land check branch for {pr_link}")
                    pr.create_land_time_check_branch(repo, 'viable/strict')

                else:
                    validate_land_time_checks(org, project, 'landchecks/' + str(pr_num))
                    print(f"All land checks passed for {pr_link}.")

            find_matching_merge_rule(pr, repo)
            print(f"All mandatory checks passed for {pr_link}.")

            pr.merge_into(repo)
            update_label_state(org, project, pr_num, LAND_PENDING_LABEL, LAND_SUCCEEDED_LABEL)
            print(f"Successfully landed {pr_link}.")
        except RuntimeError as ex:
            handle_exception(ex, org, project, pr_num)
            update_label_state(org, project, pr_num, LAND_PENDING_LABEL, LAND_FAILED_LABEL)
            continue
        except MandatoryChecksMissingError as ex:
            print(f"Merge of {pr_link} failed due to: {ex}. Retrying in 5 min")
            continue


if __name__ == "__main__":
    main()
