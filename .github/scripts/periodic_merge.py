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

LAND_PENDING_LABEL = "ciflow/trunk"
LAND_FAILED_LABEL = "land-failed"

PRS_WITH_LABEL_QUERY = GH_PULL_REQUEST_FRAGMENT + GH_PR_REVIEWS_FRAGMENT + GH_CHECKSUITES_FRAGMENT + GH_COMMIT_AUTHORS_FRAGMENT + """
query ($owner: String!, $name: String!, $labels: [String!], $with_labels: Boolean = false) {
  repository(owner: $owner, name: $name) {
    pullRequests(first: 5, labels: $labels, states: OPEN){
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
                          labels=[LAND_PENDING_LABEL],
                          with_labels=True)
    return pr_query['data']['repository']['pullRequests']['nodes']


def get_github_prs(org: str, project: str, pr_query: Any) -> List[GitHubPR]:
    prs = []
    for pr_info in pr_query:
        prs.append(GitHubPR(org, project, pr_info['number'], pr_info))
    return prs


def is_land_check(labels: List[str]) -> bool:
    return "land-checks" in labels


def is_all_green(labels: List[str]) -> bool:
    return "all-green" in labels


def validate_all_green(pr: GitHubPR):
    pending = pr_get_pending_checks(pr)
    failing = pr_get_failed_checks(pr)
    if len(failing) > 0:
        raise RuntimeError(f"{len(failing)} additional jobs have failed, first few of them are: " +
                           ' ,'.join(f"[{x[0]}]({x[1]})" for x in failing[:5]))
    if len(pending) > 0:
        raise MandatoryChecksMissingError(f"Still waiting for {len(pending)} additional jobs to finish, " +
                                          f"first few of them are: {' ,'.join(x[0] for x in pending[:5])}")


def modify_labels(org, project, pr_num):
    try:
        gh_add_labels(org, project, pr_num, [LAND_FAILED_LABEL])
        gh_remove_label(org, project, pr_num, LAND_PENDING_LABEL)
    except:
        return


def main() -> None:
    gh_remove_label('pytorch', 'pytorch', 81706, 'land-failed')
    repo = GitRepo(get_git_repo_dir(), get_git_remote_name())
    org, project = repo.gh_owner_and_name()

    pr_query = fetch_land_pending_pr_numbers(org, project)
    prs = get_github_prs(org, project, pr_query)

    check_for_sev(org, project, False)

    for pr in prs:
        pr_num = pr.pr_num
        try:
            labels = pr.get_labels()

            if is_all_green(labels):
                validate_all_green(pr)

            if is_land_check(labels):
                validate_land_time_checks(org, project, 'landchecks/' + pr_num)

            find_matching_merge_rule(pr, repo)
            pr.merge_into(repo)
        except RuntimeError as e:
            handle_exception(e)
            modify_labels(org, project, pr_num)
            continue
        except MandatoryChecksMissingError:
            continue


if __name__ == "__main__":
    main()
