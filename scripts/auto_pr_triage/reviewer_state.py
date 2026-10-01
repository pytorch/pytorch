"""Read reviewer and round-robin state for planning."""

from __future__ import annotations

import hashlib
from typing import Any
from urllib.parse import quote

from github_api import fetch_timeline, PullRequestRef, TIMELINE_PAGE_SIZE
from schemas import RoundRobinCursor


MAX_REPOSITORY_EVENT_PAGES = 10
MAX_TEAM_STATE_PAGES = 10
MAX_REVIEW_REQUEST_PAGES = 10


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


CODEOWNER_REVIEW_REQUESTS_QUERY = """
query($owner: String!, $name: String!, $number: Int!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      reviewRequests(first: 100, after: $cursor) {
        nodes {
          asCodeOwner
          requestedReviewer {
            __typename
            ... on User {
              login
            }
            ... on Team {
              slug
            }
          }
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


def fetch_requested_codeowner_handles(pr: PullRequestRef, /) -> frozenset[str]:
    """Return active review requests GitHub identifies as CODEOWNERS-derived."""

    owner, name = pr.repo.split("/", 1)
    cursor: str | None = None
    reviewers: set[str] = set()
    for _ in range(MAX_REVIEW_REQUEST_PAGES):
        data = pr.github.graphql(
            query=CODEOWNER_REVIEW_REQUESTS_QUERY,
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
            requests = pull_request["reviewRequests"]
            nodes = requests["nodes"]
            page_info = requests["pageInfo"]
            if not isinstance(nodes, list) or not isinstance(page_info, dict):
                raise RuntimeError("CODEOWNERS review requests response is invalid")
            for request in nodes:
                as_code_owner = request.get("asCodeOwner")
                if not isinstance(as_code_owner, bool):
                    raise RuntimeError(
                        "CODEOWNERS review request provenance is invalid"
                    )
                if not as_code_owner:
                    continue
                reviewer = request.get("requestedReviewer")
                if not isinstance(reviewer, dict):
                    raise RuntimeError("CODEOWNERS review request has no reviewer")
                if reviewer.get("__typename") == "User":
                    login = reviewer.get("login")
                    if not isinstance(login, str) or not login:
                        raise RuntimeError("CODEOWNERS user request is invalid")
                    reviewers.add(f"@{login}")
                elif reviewer.get("__typename") == "Team":
                    slug = reviewer.get("slug")
                    if not isinstance(slug, str) or not slug:
                        raise RuntimeError("CODEOWNERS team request is invalid")
                    reviewers.add(f"@{owner}/{slug}")
                else:
                    raise RuntimeError("CODEOWNERS reviewer type is unsupported")
            has_next_page = page_info["hasNextPage"]
            end_cursor = page_info.get("endCursor")
        except (KeyError, TypeError, AttributeError) as exc:
            raise RuntimeError(
                "CODEOWNERS review requests response is incomplete"
            ) from exc
        if not isinstance(has_next_page, bool) or (
            has_next_page and (not isinstance(end_cursor, str) or not end_cursor)
        ):
            raise RuntimeError("CODEOWNERS review requests pagination is invalid")
        if not has_next_page:
            return frozenset(reviewers)
        cursor = end_cursor
    raise RuntimeError("CODEOWNERS review requests exceed the collection limit")


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


def fetch_latest_labeled_pull_requests(
    pr: PullRequestRef,
    /,
    *,
    team_labels: dict[str, str],
) -> tuple[dict[str, tuple[int, int]], dict[int, list[dict[str, Any]]]]:
    """Return the latest prior PR and label-event ID for each team."""

    teams_by_label = {label.casefold(): team for team, label in team_labels.items()}
    found: dict[str, tuple[int, int]] = {}
    timelines: dict[int, list[dict[str, Any]]] = {}
    removed: set[tuple[int, str]] = set()
    for page in range(1, MAX_REPOSITORY_EVENT_PAGES + 1):
        events = pr.github.json(
            f"repos/{pr.repo}/issues/events?per_page={TIMELINE_PAGE_SIZE}&page={page}"
        )
        if not isinstance(events, list):
            raise RuntimeError("repository issue events response is invalid")
        try:
            for event in events:
                event_name = event.get("event")
                if event_name not in {"labeled", "unlabeled"}:
                    continue
                label = event.get("label")
                issue = event.get("issue")
                if not isinstance(label, dict) or not isinstance(issue, dict):
                    raise RuntimeError("repository issue event is incomplete")
                name = label.get("name")
                number = issue.get("number")
                event_id = event.get("id")
                if (
                    not isinstance(name, str)
                    or not isinstance(number, int)
                    or not isinstance(event_id, int)
                ):
                    raise RuntimeError("repository issue event is incomplete")
                team = teams_by_label.get(name.casefold())
                if (
                    team is None
                    or number == pr.number
                    or not isinstance(issue.get("pull_request"), dict)
                ):
                    continue
                key = (number, team)
                if event_name == "unlabeled":
                    removed.add(key)
                elif team not in found and key not in removed:
                    found[team] = (number, event_id)
        except (TypeError, AttributeError) as exc:
            raise RuntimeError("repository issue event is incomplete") from exc
        if len(found) == len(team_labels) or len(events) < TIMELINE_PAGE_SIZE:
            return found, timelines

    for team, label in team_labels.items():
        if team in found:
            continue
        prior = fetch_latest_pull_request_for_label(
            pr,
            label=label,
            timelines=timelines,
        )
        if prior is not None:
            found[team] = prior
    return found, timelines


def fetch_latest_pull_request_for_label(
    pr: PullRequestRef,
    /,
    *,
    label: str,
    timelines: dict[int, list[dict[str, Any]]],
) -> tuple[int, int] | None:
    """Recover the newest durable owner-label event from labeled pull requests."""

    encoded_label = quote(label, safe="")
    latest: tuple[int, int] | None = None
    for page in range(1, MAX_TEAM_STATE_PAGES + 1):
        issues = pr.github.json(
            f"repos/{pr.repo}/issues?state=all&labels={encoded_label}"
            f"&per_page={TIMELINE_PAGE_SIZE}&page={page}"
        )
        if not isinstance(issues, list) or any(
            not isinstance(issue, dict) for issue in issues
        ):
            raise RuntimeError("owner label usage response is invalid")
        for issue in issues:
            number = issue.get("number")
            if not isinstance(number, int) or not isinstance(
                issue.get("pull_request"), dict
            ):
                raise RuntimeError("owner label is not dedicated to pull requests")
            if number == pr.number:
                continue
            timeline = timelines.get(number)
            if timeline is None:
                timeline = fetch_timeline(github=pr.github, repo=pr.repo, number=number)
                timelines[number] = timeline
            matching_ids: list[int] = []
            for event in timeline:
                event_label = event.get("label")
                name = (
                    event_label.get("name") if isinstance(event_label, dict) else None
                )
                event_id = event.get("id")
                if (
                    event.get("event") == "labeled"
                    and isinstance(name, str)
                    and name.casefold() == label.casefold()
                    and isinstance(event_id, int)
                ):
                    matching_ids.append(event_id)
            if not matching_ids:
                raise RuntimeError(
                    "owner label event is absent from labeled pull request"
                )
            candidate = (number, max(matching_ids))
            if latest is None or candidate[1] > latest[1]:
                latest = candidate
        if len(issues) < TIMELINE_PAGE_SIZE:
            return latest
    raise RuntimeError("owner label state exceeds the collection limit")


def fetch_assigned_member(
    *,
    timeline: list[dict[str, Any]],
    label_event_id: int,
    members: list[str] | tuple[str, ...],
) -> str | None:
    """Return the roster member requested before an owner label, if present."""

    members_by_key = {member.casefold(): member for member in members}
    label_index = next(
        (
            index
            for index, event in enumerate(timeline)
            if event.get("event") == "labeled" and event.get("id") == label_event_id
        ),
        None,
    )
    if label_index is None:
        raise RuntimeError("owner label event is absent from pull request timeline")
    for event in reversed(timeline[:label_index]):
        if event.get("event") != "review_requested":
            continue
        reviewer = event.get("requested_reviewer")
        if not isinstance(reviewer, dict):
            continue
        login = reviewer.get("login")
        if not isinstance(login, str):
            raise RuntimeError("review request event is incomplete")
        member = members_by_key.get(f"@{login}".casefold())
        if member is not None:
            return member
    return None


def stable_fallback_member(
    *,
    repo: str,
    current_number: int,
    owner: str,
    members: list[str] | tuple[str, ...],
    ineligible_reviewers: set[str],
) -> str | None:
    """Choose a reproducible fallback when an owner's saved cursor is invalid."""

    ineligible_keys = {reviewer.casefold() for reviewer in ineligible_reviewers}
    eligible = [
        member for member in members if member.casefold() not in ineligible_keys
    ]
    if not eligible:
        return None
    key = f"{repo}:{current_number}:{owner}".encode()
    index = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % len(eligible)
    return eligible[index]


def next_round_robin_member(
    *,
    members: list[str] | tuple[str, ...],
    latest_reviewer: str | None,
    ineligible_reviewers: set[str],
) -> str | None:
    """Choose the next eligible roster member after the latest reviewer."""

    if not members:
        return None
    ineligible_keys = {reviewer.casefold() for reviewer in ineligible_reviewers}
    latest_key = latest_reviewer.casefold() if latest_reviewer else None
    start = 0
    if latest_key is not None:
        for index, member in enumerate(members):
            if member.casefold() == latest_key:
                start = (index + 1) % len(members)
                break
    for offset in range(len(members)):
        candidate = members[(start + offset) % len(members)]
        if candidate.casefold() not in ineligible_keys:
            return candidate
    return None


def fetch_round_robin_cursors(
    pr: PullRequestRef,
    /,
    *,
    owners: dict[str, tuple[str, tuple[str, ...]]],
) -> dict[str, RoundRobinCursor]:
    """Read each owner's rotation state from label and review-request history.

    owners maps a team owner ID to its routing label and roster.
    """

    if not owners:
        return {}
    latest_prs, timelines = fetch_latest_labeled_pull_requests(
        pr,
        team_labels={owner: label for owner, (label, _) in owners.items()},
    )
    cursors: dict[str, RoundRobinCursor] = {}
    for owner, (_, members) in owners.items():
        prior = latest_prs.get(owner)
        if prior is None:
            cursors[owner] = RoundRobinCursor(None, None)
            continue
        prior_pr, label_event_id = prior
        timeline = timelines.get(prior_pr)
        if timeline is None:
            timeline = fetch_timeline(github=pr.github, repo=pr.repo, number=prior_pr)
            timelines[prior_pr] = timeline
        assigned = fetch_assigned_member(
            timeline=timeline, label_event_id=label_event_id, members=members
        )
        cursors[owner] = RoundRobinCursor(prior_pr, assigned)
    return cursors


def choose_round_robin_member(
    *,
    repo: str,
    current_number: int,
    owner: str,
    members: tuple[str, ...],
    cursor: RoundRobinCursor,
    ineligible_reviewers: set[str],
) -> tuple[str, str] | None:
    """Return the next member and why, or None when nobody is eligible."""

    if cursor.marker_found and cursor.last_assigned is None:
        # The latest marker has no attributable assignment; the fallback is
        # stable per PR, and requesting that member repairs the rotation.
        selected = stable_fallback_member(
            repo=repo,
            current_number=current_number,
            owner=owner,
            members=members,
            ineligible_reviewers=ineligible_reviewers,
        )
        reason = "stable_fallback"
    else:
        selected = next_round_robin_member(
            members=members,
            latest_reviewer=cursor.last_assigned,
            ineligible_reviewers=ineligible_reviewers,
        )
        reason = "round_robin_next" if cursor.marker_found else "round_robin_initial"
    return None if selected is None else (selected, reason)
