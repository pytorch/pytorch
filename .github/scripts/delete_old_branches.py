# Delete old branches
import os
import re
from collections.abc import Callable
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from github_utils import gh_fetch_json_dict, gh_graphql
from gitutils import GitRepo


SEC_IN_DAY = 24 * 60 * 60
CLOSED_PR_RETENTION = 30 * SEC_IN_DAY
NO_PR_RETENTION = 1.5 * 365 * SEC_IN_DAY
PR_WINDOW = 90 * SEC_IN_DAY  # Set to None to look at all PRs (may take a lot of tokens)
REPO_OWNER = "pytorch"
REPO_NAME = "pytorch"
ESTIMATED_TOKENS = [0]

TOKEN = os.environ["GITHUB_TOKEN"]
if not TOKEN:
    raise Exception("GITHUB_TOKEN is not set")  # noqa: TRY002

REPO_ROOT = Path(__file__).parents[2]

# Query for all PRs instead of just closed/merged because it's faster
GRAPHQL_ALL_PRS_BY_UPDATED_AT = """
query ($owner: String!, $repo: String!, $cursor: String) {
  repository(owner: $owner, name: $repo) {
    pullRequests(
      first: 100
      after: $cursor
      orderBy: {field: UPDATED_AT, direction: DESC}
    ) {
      totalCount
      pageInfo {
        hasNextPage
        endCursor
      }
      nodes {
        headRefName
        number
        updatedAt
        state
      }
    }
  }
}
"""

GRAPHQL_OPEN_PRS = """
query ($owner: String!, $repo: String!, $cursor: String) {
  repository(owner: $owner, name: $repo) {
    pullRequests(
      first: 100
      after: $cursor
      states: [OPEN]
    ) {
      totalCount
      pageInfo {
        hasNextPage
        endCursor
      }
      nodes {
        headRefName
        number
        updatedAt
        state
      }
    }
  }
}
"""

GRAPHQL_NO_DELETE_BRANCH_LABEL = """
query ($owner: String!, $repo: String!, $cursor: String) {
  repository(owner: $owner, name: $repo) {
    label(name: "no-delete-branch") {
      pullRequests(first: 100, after: $cursor) {
        totalCount
        pageInfo {
          hasNextPage
          endCursor
        }
        nodes {
          headRefName
          number
          updatedAt
          state
        }
      }
    }
  }
}
"""


def is_protected(branch: str) -> bool:
    try:
        ESTIMATED_TOKENS[0] += 1
        res = gh_fetch_json_dict(
            f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/branches/{branch}"
        )
        return bool(res["protected"])
    except Exception as e:
        print(f"[{branch}] Failed to fetch branch protections: {e}")
        return True


def convert_gh_timestamp(date: str) -> float:
    return datetime.strptime(date, "%Y-%m-%dT%H:%M:%SZ").timestamp()


def get_branches(repo: GitRepo) -> dict[str, Any]:
    # Query locally for branches, group by branch base name (e.g. gh/blah/base -> gh/blah), and get the most recent branch
    git_response = repo._run_git(
        "for-each-ref",
        "--sort=creatordate",
        "--format=%(refname) %(committerdate:iso-strict)",
        "refs/remotes/origin",
    )
    branches_by_base_name: dict[str, Any] = {}
    for line in git_response.splitlines():
        branch, date = line.split(" ")
        re_branch = re.match(r"refs/remotes/origin/(.*)", branch)
        if not re_branch:
            raise AssertionError(
                f"Branch name '{branch}' does not match expected pattern"
            )
        branch = branch_base_name = re_branch.group(1)
        if x := re.match(r"(gh\/.+)\/(head|base|orig)", branch):
            branch_base_name = x.group(1)
        date = datetime.fromisoformat(date).timestamp()
        if branch_base_name not in branches_by_base_name:
            branches_by_base_name[branch_base_name] = [date, [branch]]
        else:
            branches_by_base_name[branch_base_name][1].append(branch)
            if date > branches_by_base_name[branch_base_name][0]:
                branches_by_base_name[branch_base_name][0] = date
    return branches_by_base_name


def paginate_graphql(
    query: str,
    kwargs: dict[str, Any],
    termination_func: Callable[[list[dict[str, Any]]], bool],
    get_data: Callable[[dict[str, Any]], list[dict[str, Any]]],
    get_page_info: Callable[[dict[str, Any]], dict[str, Any]],
) -> list[Any]:
    hasNextPage = True
    endCursor = None
    data: list[dict[str, Any]] = []
    while hasNextPage:
        ESTIMATED_TOKENS[0] += 1
        res = gh_graphql(query, cursor=endCursor, **kwargs)
        data.extend(get_data(res))
        hasNextPage = get_page_info(res)["hasNextPage"]
        endCursor = get_page_info(res)["endCursor"]
        if termination_func(data):
            break
    return data


def get_recent_prs() -> dict[str, Any]:
    now = datetime.now().timestamp()

    # Grab all PRs updated in last CLOSED_PR_RETENTION days
    pr_infos: list[dict[str, Any]] = paginate_graphql(
        GRAPHQL_ALL_PRS_BY_UPDATED_AT,
        {"owner": "pytorch", "repo": "pytorch"},
        lambda data: (
            PR_WINDOW is not None
            and (now - convert_gh_timestamp(data[-1]["updatedAt"]) > PR_WINDOW)
        ),
        lambda res: res["data"]["repository"]["pullRequests"]["nodes"],
        lambda res: res["data"]["repository"]["pullRequests"]["pageInfo"],
    )

    # Get the most recent PR for each branch base (group gh together)
    prs_by_branch_base = {}
    for pr in pr_infos:
        pr["updatedAt"] = convert_gh_timestamp(pr["updatedAt"])
        branch_base_name = pr["headRefName"]
        if x := re.match(r"(gh\/.+)\/(head|base|orig)", branch_base_name):
            branch_base_name = x.group(1)
        if branch_base_name not in prs_by_branch_base:
            prs_by_branch_base[branch_base_name] = pr
        else:
            if pr["updatedAt"] > prs_by_branch_base[branch_base_name]["updatedAt"]:
                prs_by_branch_base[branch_base_name] = pr
    return prs_by_branch_base


@lru_cache(maxsize=1)
def get_open_prs() -> list[dict[str, Any]]:
    return paginate_graphql(
        GRAPHQL_OPEN_PRS,
        {"owner": "pytorch", "repo": "pytorch"},
        lambda data: False,
        lambda res: res["data"]["repository"]["pullRequests"]["nodes"],
        lambda res: res["data"]["repository"]["pullRequests"]["pageInfo"],
    )


def get_branches_with_magic_label_or_open_pr() -> set[str]:
    pr_infos: list[dict[str, Any]] = paginate_graphql(
        GRAPHQL_NO_DELETE_BRANCH_LABEL,
        {"owner": "pytorch", "repo": "pytorch"},
        lambda data: False,
        lambda res: res["data"]["repository"]["label"]["pullRequests"]["nodes"],
        lambda res: res["data"]["repository"]["label"]["pullRequests"]["pageInfo"],
    )

    pr_infos.extend(get_open_prs())

    # Get the most recent PR for each branch base (group gh together)
    branch_bases = set()
    for pr in pr_infos:
        branch_base_name = pr["headRefName"]
        if x := re.match(r"(gh\/.+)\/(head|base|orig)", branch_base_name):
            branch_base_name = x.group(1)
        branch_bases.add(branch_base_name)
    return branch_bases


def delete_branch(repo: GitRepo, branch: str) -> None:
    repo._run_git("push", "origin", "-d", branch)


def delete_branches() -> None:
    now = datetime.now().timestamp()
    git_repo = GitRepo(str(REPO_ROOT), "origin", debug=True)
    branches = get_branches(git_repo)
    prs_by_branch = get_recent_prs()
    keep_branches = get_branches_with_magic_label_or_open_pr()

    delete = []
    # Do not delete if:
    # * associated PR is open, closed but updated recently, or contains the magic string
    # * no associated PR and branch was updated in last 1.5 years
    # * is protected
    # Setting different values of PR_WINDOW will change how branches with closed
    # PRs are treated depending on how old the branch is.  The default value of
    # 90 will allow branches with closed PRs to be deleted if the PR hasn't been
    # updated in 90 days and the branch hasn't been updated in 1.5 years
    for base_branch, (date, sub_branches) in branches.items():
        print(f"[{base_branch}] Updated {(now - date) / SEC_IN_DAY} days ago")
        if base_branch in keep_branches:
            print(f"[{base_branch}] Has magic label or open PR, skipping")
            continue
        pr = prs_by_branch.get(base_branch)
        if pr:
            print(
                f"[{base_branch}] Has PR {pr['number']}: {pr['state']}, updated {(now - pr['updatedAt']) / SEC_IN_DAY} days ago"
            )
            if (
                now - pr["updatedAt"] < CLOSED_PR_RETENTION
                or (now - date) < CLOSED_PR_RETENTION
            ):
                continue
        elif now - date < NO_PR_RETENTION:
            continue
        print(f"[{base_branch}] Checking for branch protections")
        if any(is_protected(sub_branch) for sub_branch in sub_branches):
            print(f"[{base_branch}] Is protected")
            continue
        for sub_branch in sub_branches:
            print(f"[{base_branch}] Deleting {sub_branch}")
            delete.append(sub_branch)
        if ESTIMATED_TOKENS[0] > 400:
            print("Estimated tokens exceeded, exiting")
            break

    print(f"To delete ({len(delete)}):")
    for branch in delete:
        print(f"About to delete branch {branch}")
        delete_branch(git_repo, branch)


def delete_old_tags() -> None:
    # Deletes ciflow tags if they are associated with a closed PR or a specific
    # commit.  Lightweight tags don't have information about the date they were
    # created, so we can't check how old they are.  The script just assumes that
    # ciflow tags should be deleted regardless of creation date.
    git_repo = GitRepo(str(REPO_ROOT), "origin", debug=True)

    def delete_tag(tag: str) -> None:
        print(f"Deleting tag {tag}")
        ESTIMATED_TOKENS[0] += 1
        delete_branch(git_repo, f"refs/tags/{tag}")

    tags = git_repo._run_git("tag").splitlines()

    CIFLOW_TAG_REGEX = re.compile(r"^ciflow\/.*\/(\d{5,6}|[0-9a-f]{40})$")
    AUTO_REVERT_TAG_REGEX = re.compile(r"^trunk\/[0-9a-f]{40}$")
    for tag in tags:
        try:
            if ESTIMATED_TOKENS[0] > 400:
                print("Estimated tokens exceeded, exiting")
                break

            if not CIFLOW_TAG_REGEX.match(tag) and not AUTO_REVERT_TAG_REGEX.match(tag):
                continue

            # This checks the date of the commit associated with the tag instead
            # of the tag itself since lightweight tags don't have this
            # information.  I think it should be ok since this only runs once a
            # day
            tag_info = git_repo._run_git("show", "-s", "--format=%ct", tag)
            tag_timestamp = int(tag_info.strip())
            # Maybe some timezone issues, but a few hours shouldn't matter
            tag_age_days = (datetime.now().timestamp() - tag_timestamp) / SEC_IN_DAY

            if tag_age_days > 7:
                print(f"[{tag}] Tag is older than 7 days, deleting")
                delete_tag(tag)
        except Exception as e:
            print(f"Failed to check tag {tag}: {e}")


if __name__ == "__main__":
    delete_branches()
    delete_old_tags()
