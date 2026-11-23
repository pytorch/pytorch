"""Read reviewer state for planning."""

from __future__ import annotations

from github_api import PullRequestRef


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
