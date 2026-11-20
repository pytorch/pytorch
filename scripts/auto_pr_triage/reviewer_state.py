"""Read reviewer state for planning."""

from __future__ import annotations

from assess_intake import fetch_user_permissions
from github_api import PullRequestRef, TIMELINE_PAGE_SIZE


# Team attribution is omitted because its GraphQL fields require read:org,
# which the repository-scoped workflow token intentionally does not receive.
SUBMITTED_REVIEWS_QUERY = """
query($owner: String!, $name: String!, $number: Int!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      reviews(first: 100, after: $cursor) {
        nodes {
          author {
            login
          }
          state
        }
        pageInfo {
          endCursor
          hasNextPage
        }
      }
    }
  }
}
""".strip()


def fetch_requested_reviewer_handles(pr: PullRequestRef, /) -> set[str]:
    """Return all currently requested user and team handles."""

    response = pr.github.json(f"repos/{pr.repo}/pulls/{pr.number}/requested_reviewers")
    if not isinstance(response, dict):
        raise RuntimeError("requested reviewers response is invalid")
    owner, _ = pr.repo.split("/", 1)
    try:
        users = {f"@{user['login']}" for user in response["users"]}
        teams = {f"@{owner}/{team['slug']}" for team in response["teams"]}
    except (KeyError, TypeError, AttributeError) as exc:
        raise RuntimeError("requested reviewers response is incomplete") from exc
    return users | teams


def fetch_submitted_review_state(pr: PullRequestRef, /) -> frozenset[str]:
    """Return users with a qualifying submitted review."""

    owner, name = pr.repo.split("/", 1)
    cursor: str | None = None
    reviewers: set[str] = set()
    while True:
        data = pr.github.graphql(
            query=SUBMITTED_REVIEWS_QUERY,
            variables={
                "owner": owner,
                "name": name,
                "number": pr.number,
                "cursor": cursor,
            },
        )
        try:
            repository = data.get("repository")
            pull_request = repository.get("pullRequest") if repository else None
            if pull_request is None:
                raise RuntimeError(f"pull request not found: {pr.repo}#{pr.number}")
            reviews = pull_request["reviews"]
            nodes = reviews["nodes"]
            page_info = reviews["pageInfo"]
            if not isinstance(nodes, list) or not isinstance(page_info, dict):
                raise RuntimeError("submitted reviews response is invalid")
            for review in nodes:
                # COMMENTED is a routing handoff, not an approval or ownership claim.
                if review.get("state") not in {
                    "APPROVED",
                    "CHANGES_REQUESTED",
                    "COMMENTED",
                }:
                    continue
                author = review.get("author")
                if author is None:
                    continue
                login = author["login"]
                if not isinstance(login, str) or not login:
                    raise RuntimeError("submitted reviews response is invalid")
                reviewers.add(f"@{login}")
            has_next_page = page_info["hasNextPage"]
            end_cursor = page_info.get("endCursor")
        except (KeyError, TypeError, AttributeError) as exc:
            raise RuntimeError("submitted reviews response is incomplete") from exc
        if not isinstance(has_next_page, bool) or (
            has_next_page and (not isinstance(end_cursor, str) or not end_cursor)
        ):
            raise RuntimeError("submitted reviews response is invalid")
        if not has_next_page:
            return frozenset(reviewers)
        cursor = end_cursor


def previously_requested_users(timeline: list[dict]) -> frozenset[str]:
    """Return every user a review was ever requested from, as casefolded handles."""

    return frozenset(
        f"@{event['requested_reviewer']['login']}".casefold()
        for event in timeline
        if event.get("event") == "review_requested" and event.get("requested_reviewer")
    )


def fetch_previously_requested_users(pr: PullRequestRef, /) -> frozenset[str]:
    """Return every user ever requested on the PR, from its whole timeline.

    A removed request leaves its review_requested event behind, so this is how a
    later run knows not to ask that person again. Unlike intake's fetch_timeline
    there is no page limit: a missed early request would re-request them.
    """

    endpoint = f"repos/{pr.repo}/issues/{pr.number}/timeline"
    users: set[str] = set()
    page = 1
    while True:
        events = pr.github.json(f"{endpoint}?per_page={TIMELINE_PAGE_SIZE}&page={page}")
        if not isinstance(events, list) or not all(isinstance(e, dict) for e in events):
            raise RuntimeError("pull request timeline response is invalid")
        users |= previously_requested_users(events)
        if len(events) < TIMELINE_PAGE_SIZE:
            return frozenset(users)
        page += 1


def has_push_access(pr: PullRequestRef, /, *, login: str) -> bool:
    """Return whether one user has push access, which GitHub required of code owners.

    gh reports a deleted or renamed account as HTTP 404. It can't be requested
    either, so it counts as no access instead of failing the run.
    """

    try:
        access = fetch_user_permissions(github=pr.github, repo=pr.repo, login=login)
    except RuntimeError as exc:
        if "(HTTP 404)" not in str(exc):
            raise
        return False
    return access["push"]
