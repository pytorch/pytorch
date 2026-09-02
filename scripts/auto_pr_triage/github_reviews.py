"""Shared read-only GitHub review collection for Auto PR Triage."""

from __future__ import annotations

import hashlib
from typing import Any, Protocol
from urllib.parse import quote


MAX_TIMELINE_PAGES = 10
TIMELINE_PAGE_SIZE = 100
MAX_REPOSITORY_EVENT_PAGES = 10
MAX_TEAM_STATE_PAGES = 10
MAX_REVIEW_REQUEST_PAGES = 10
MAX_ENGAGEMENT_CANDIDATES = 100
SUBMITTED_REVIEW_STATES = frozenset(
    {"approved", "changes_requested", "commented", "dismissed"}
)
NON_HUMAN_LOGINS = frozenset({"pytorchbot", "pytorchmergebot"})


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


class GitHubClient(Protocol):
    """Minimal read interface required to collect reviewer state."""

    def json(self, endpoint: str) -> Any: ...
    def graphql(self, query: str, variables: dict[str, Any]) -> dict[str, Any]: ...


def fetch_user_has_triage_permission(
    github: GitHubClient,
    repo: str,
    login: str,
) -> bool:
    """Return whether GitHub grants one user triage-or-higher repository access."""

    encoded_login = quote(login, safe="")
    response = github.json(
        f"repos/{repo}/collaborators/{encoded_login}/permission"
    )
    if not isinstance(response, dict):
        raise RuntimeError("collaborator permission response is invalid")
    try:
        user = response["user"]
        returned_login = user["login"]
        permissions = user["permissions"]
        access = {
            name: permissions[name]
            for name in ("triage", "push", "maintain", "admin")
        }
    except (KeyError, TypeError, AttributeError) as exc:
        raise RuntimeError("collaborator permission response is incomplete") from exc
    if (
        not isinstance(returned_login, str)
        or returned_login.casefold() != login.casefold()
        or not all(isinstance(value, bool) for value in access.values())
    ):
        raise RuntimeError("collaborator permission response is inconsistent")
    return any(access.values())


def fetch_requested_reviewer_handles(
    github: GitHubClient,
    repo: str,
    number: int,
) -> set[str]:
    """Return all currently requested user and team handles."""

    response = github.json(f"repos/{repo}/pulls/{number}/requested_reviewers")
    if not isinstance(response, dict):
        raise RuntimeError("requested reviewers response is invalid")
    owner, _ = repo.split("/", 1)
    try:
        users = {f"@{user['login']}" for user in response["users"]}
        teams = {f"@{owner}/{team['slug']}" for team in response["teams"]}
    except (KeyError, TypeError, AttributeError) as exc:
        raise RuntimeError("requested reviewers response is incomplete") from exc
    return users | teams


def fetch_requested_codeowner_handles(
    github: GitHubClient,
    repo: str,
    number: int,
) -> frozenset[str]:
    """Return active review requests GitHub identifies as CODEOWNERS-derived."""

    owner, name = repo.split("/", 1)
    cursor: str | None = None
    reviewers: set[str] = set()
    for _ in range(MAX_REVIEW_REQUEST_PAGES):
        data = github.graphql(
            CODEOWNER_REVIEW_REQUESTS_QUERY,
            {
                "owner": owner,
                "name": name,
                "number": number,
                "cursor": cursor,
            },
        )
        try:
            repository = data.get("repository")
            pull_request = repository.get("pullRequest") if repository else None
            if pull_request is None:
                raise RuntimeError(f"pull request not found: {repo}#{number}")
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


def fetch_submitted_review_state(
    github: GitHubClient,
    repo: str,
    number: int,
) -> frozenset[str]:
    """Return users with a qualifying submitted review."""

    owner, name = repo.split("/", 1)
    cursor: str | None = None
    reviewers: set[str] = set()
    while True:
        data = github.graphql(
            SUBMITTED_REVIEWS_QUERY,
            {
                "owner": owner,
                "name": name,
                "number": number,
                "cursor": cursor,
            },
        )
        try:
            repository = data.get("repository")
            pull_request = repository.get("pullRequest") if repository else None
            if pull_request is None:
                raise RuntimeError(f"pull request not found: {repo}#{number}")
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
    github: GitHubClient,
    repo: str,
    team_labels: dict[str, str],
    current_number: int,
) -> tuple[dict[str, tuple[int, int]], dict[int, list[dict[str, Any]]]]:
    """Return the latest prior PR and label-event ID for each team."""

    teams_by_label = {label.casefold(): team for team, label in team_labels.items()}
    found: dict[str, tuple[int, int]] = {}
    timelines: dict[int, list[dict[str, Any]]] = {}
    removed: set[tuple[int, str]] = set()
    for page in range(1, MAX_REPOSITORY_EVENT_PAGES + 1):
        events = github.json(
            f"repos/{repo}/issues/events?per_page={TIMELINE_PAGE_SIZE}&page={page}"
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
                    or number == current_number
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
            github,
            repo,
            label,
            current_number,
            timelines,
        )
        if prior is not None:
            found[team] = prior
    return found, timelines


def fetch_pull_request_timeline(
    github: GitHubClient,
    repo: str,
    number: int,
) -> list[dict[str, Any]]:
    """Return one complete pull-request timeline within the configured bound."""

    timeline: list[dict[str, Any]] = []
    for page in range(1, MAX_TIMELINE_PAGES + 1):
        events = github.json(
            f"repos/{repo}/issues/{number}/timeline"
            f"?per_page={TIMELINE_PAGE_SIZE}&page={page}"
        )
        if not isinstance(events, list) or any(
            not isinstance(event, dict) for event in events
        ):
            raise RuntimeError("pull request timeline response is invalid")
        timeline.extend(events)
        if len(events) < TIMELINE_PAGE_SIZE:
            return timeline
    raise RuntimeError("pull request timeline exceeds the collection limit")


def fetch_maintainer_activity(
    github: GitHubClient,
    repo: str,
    number: int,
    author_login: str,
) -> tuple[str, tuple[str, ...]] | None:
    """Return one triage-or-higher maintainer with visible PR activity."""

    candidates: dict[str, set[str]] = {}
    author_key = author_login.casefold()

    def human_login(value: Any, context: str) -> str | None:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise RuntimeError(f"{context} user is invalid")
        login = value.get("login")
        account_type = value.get("type")
        if not isinstance(login, str) or not login:
            raise RuntimeError(f"{context} user is invalid")
        if not isinstance(account_type, str):
            raise RuntimeError(f"{context} user is invalid")
        key = login.casefold()
        if (
            account_type != "User"
            or key == author_key
            or key in NON_HUMAN_LOGINS
            or key.endswith("[bot]")
        ):
            return None
        return login

    for event in fetch_pull_request_timeline(github, repo, number):
        event_name = event.get("event")
        login: str | None = None
        signal: str | None = None
        if event_name in {"commented", "reviewed"}:
            if event_name == "reviewed":
                state = event.get("state")
                if not isinstance(state, str):
                    raise RuntimeError("review event state is invalid")
                if state.casefold() not in SUBMITTED_REVIEW_STATES:
                    continue
            login = human_login(event.get("user"), "engagement")
            signal = "comment" if event_name == "commented" else "review"
        elif event_name == "review_requested":
            requested = event.get("requested_reviewer")
            if requested is None:
                continue
            actor = human_login(event.get("actor"), "review-request actor")
            reviewer = human_login(requested, "requested reviewer")
            if (
                actor is None
                or reviewer is None
                or actor.casefold() != reviewer.casefold()
            ):
                continue
            login = actor
            signal = "self_review_request"
        if login is None or signal is None:
            continue
        candidates.setdefault(login.casefold(), set()).add(signal)
        if len(candidates) > MAX_ENGAGEMENT_CANDIDATES:
            raise RuntimeError("maintainer engagement exceeds the candidate limit")

    for login in sorted(candidates):
        if fetch_user_has_triage_permission(github, repo, login):
            return f"@{login}", tuple(sorted(candidates[login]))
    return None


def fetch_latest_pull_request_for_label(
    github: GitHubClient,
    repo: str,
    label: str,
    current_number: int,
    timelines: dict[int, list[dict[str, Any]]],
) -> tuple[int, int] | None:
    """Recover the newest durable owner-label event from labeled pull requests."""

    encoded_label = quote(label, safe="")
    latest: tuple[int, int] | None = None
    for page in range(1, MAX_TEAM_STATE_PAGES + 1):
        issues = github.json(
            f"repos/{repo}/issues?state=all&labels={encoded_label}"
            f"&per_page={TIMELINE_PAGE_SIZE}&page={page}"
        )
        if not isinstance(issues, list) or any(
            not isinstance(issue, dict) for issue in issues
        ):
            raise RuntimeError("owner label usage response is invalid")
        for issue in issues:
            number = issue.get("number")
            if (
                not isinstance(number, int)
                or not isinstance(issue.get("pull_request"), dict)
            ):
                raise RuntimeError("owner label is not dedicated to pull requests")
            if number == current_number:
                continue
            timeline = timelines.get(number)
            if timeline is None:
                timeline = fetch_pull_request_timeline(github, repo, number)
                timelines[number] = timeline
            matching_ids: list[int] = []
            for event in timeline:
                event_label = event.get("label")
                name = (
                    event_label.get("name")
                    if isinstance(event_label, dict)
                    else None
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
                raise RuntimeError("owner label event is absent from labeled pull request")
            candidate = (number, max(matching_ids))
            if latest is None or candidate[1] > latest[1]:
                latest = candidate
        if len(issues) < TIMELINE_PAGE_SIZE:
            return latest
    raise RuntimeError("owner label state exceeds the collection limit")


def fetch_assigned_member(
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


def fetch_round_robin_reviewers(
    github: GitHubClient,
    repo: str,
    current_number: int,
    owners: dict[str, dict[str, Any]],
    ineligible_reviewers: set[str],
) -> dict[str, dict[str, str]]:
    """Derive per-owner assignments from labels and reviewer-request history."""

    if not owners:
        return {}

    for entry in owners.values():
        label = entry["label"]
        response = github.json(
            f"repos/{repo}/labels/{quote(label, safe='')}"
        )
        actual = response.get("name") if isinstance(response, dict) else None
        if not isinstance(actual, str) or actual.casefold() != label.casefold():
            raise RuntimeError("configured owner label is unavailable")

    latest_prs, timelines = fetch_latest_labeled_pull_requests(
        github,
        repo,
        {owner: entry["label"] for owner, entry in owners.items()},
        current_number,
    )
    latest_assignments: dict[tuple[int, int, tuple[str, ...]], str | None] = {}
    selections: dict[str, dict[str, str]] = {}
    for owner, entry in owners.items():
        members = tuple(entry["members"])
        prior = latest_prs.get(owner)
        latest_reviewer = None
        if prior is not None:
            prior_pr, label_event_id = prior
            timeline = timelines.get(prior_pr)
            if timeline is None:
                timeline = fetch_pull_request_timeline(github, repo, prior_pr)
                timelines[prior_pr] = timeline
            cache_key = (
                prior_pr,
                label_event_id,
                tuple(member.casefold() for member in members),
            )
            if cache_key not in latest_assignments:
                latest_assignments[cache_key] = fetch_assigned_member(
                    timeline,
                    label_event_id,
                    members,
                )
            latest_reviewer = latest_assignments[cache_key]
        if prior is not None and latest_reviewer is None:
            selected = stable_fallback_member(
                repo,
                current_number,
                owner,
                members,
                ineligible_reviewers,
            )
            selection_reason = "stable_fallback"
            if selected is not None:
                print(
                    "Auto PR Triage owner marker has no current-roster assignment; "
                    f"using stable fallback for {owner}: {selected}."
                )
        else:
            selected = next_round_robin_member(
                members,
                latest_reviewer,
                ineligible_reviewers,
            )
            selection_reason = (
                "round_robin_next" if prior is not None else "round_robin_initial"
            )
        if selected is not None:
            selections[owner] = {
                "reviewer": selected,
                "selection_reason": selection_reason,
            }
    return selections
