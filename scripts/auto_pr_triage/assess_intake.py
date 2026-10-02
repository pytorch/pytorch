#!/usr/bin/env python3
"""Stage 1: decide whether a pull request is admitted to routing.

Intake reads the PR once and records the gate facts, verified description
claims, and handoff reviewers. Later stages reuse its PR snapshot.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote

from github_api import (
    fetch_timeline,
    GitHubClient,
    GitHubReader,
    PullRequestRef,
    SUBMITTED_REVIEW_STATES,
)
from identifiers import TARGET_BASE_REF
from schemas import (
    IntakeFacts,
    IntakeResult,
    MAX_HANDOFF_REVIEWERS,
    MAX_PART_OF_ISSUES,
    passes_intake,
    PullRequestIdentity,
)


MAX_ENGAGEMENT_CANDIDATES = 100
NON_HUMAN_LOGINS = frozenset({"pytorchbot", "pytorchmergebot"})
PASSIVE_TIMELINE_EVENTS = frozenset(
    {
        "cross-referenced",
        "deployed",
        "deployment_environment_changed",
        "mentioned",
        "referenced",
        "subscribed",
        "unsubscribed",
    }
)
TRIGGER_LABEL = "open source"
HANDLED_LABELS = frozenset(
    {
        "triaged",
        "bot-triaged",
        "bot-triage-error",
        "no automated triage",
    }
)
# GitHub does not link references inside comments or code.
UNRENDERED_MARKDOWN_RE = re.compile(
    r"<!--.*?(?:-->|\Z)|(`{3,}|~{3,}).*?(?:\1|\Z)|`[^`\n]*`", re.DOTALL
)
# Like GitHub closing keywords: keyword, optional colon, reference, anywhere
# after a word boundary, case-insensitive.
PART_OF_RE = re.compile(
    r"(?<![\w-])part\s+of:?\s+(?P<repo>[\w.-]+/[\w.-]+)?#(?P<number>[1-9][0-9]*)(?![\w/-])",
    re.IGNORECASE,
)
SUPPORTED_BY_RE = re.compile(
    r"(?<![\w-])supported\s+by:?\s+"
    r"@(?P<login>[A-Za-z0-9](?:[A-Za-z0-9]|-(?=[A-Za-z0-9])){0,38})(?![\w/-])",
    re.IGNORECASE,
)


LINKED_ISSUES_QUERY = """
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      closingIssuesReferences(first: 100) {
        nodes {
          number
          repository {
            nameWithOwner
          }
          labels(first: 100) {
            nodes {
              name
            }
            pageInfo {
              hasNextPage
            }
          }
        }
        pageInfo {
          hasNextPage
        }
      }
    }
  }
}
""".strip()


ISSUE_QUERY = """
query($url: URI!) {
  resource(url: $url) {
    ... on Issue {
      number
      url
      labels(first: 100) {
        nodes {
          name
        }
        pageInfo {
          hasNextPage
        }
      }
    }
  }
}
""".strip()


def fetch_user_has_triage_permission(
    *,
    github: GitHubClient,
    repo: str,
    login: str,
) -> bool:
    """Return whether GitHub grants one user triage-or-higher repository access."""

    encoded_login = quote(login, safe="")
    response = github.json(f"repos/{repo}/collaborators/{encoded_login}/permission")
    if not isinstance(response, dict):
        raise RuntimeError("collaborator permission response is invalid")
    try:
        user = response["user"]
        returned_login = user["login"]
        permissions = user["permissions"]
        access = {
            name: permissions[name] for name in ("triage", "push", "maintain", "admin")
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


def _human_login(
    *,
    value: Any,
    context: str,
    excluded_login: str | None = None,
) -> str | None:
    """Return one human login, excluding a caller-selected identity and bots."""

    if value is None:
        return None
    if not isinstance(value, dict):
        raise RuntimeError(f"{context} user is invalid")
    login = value.get("login")
    account_type = value.get("type")
    if not isinstance(login, str) or not login or not isinstance(account_type, str):
        raise RuntimeError(f"{context} user is invalid")
    key = login.casefold()
    if (
        account_type != "User"
        or (excluded_login is not None and key == excluded_login.casefold())
        or key in NON_HUMAN_LOGINS
        or key.endswith("[bot]")
    ):
        return None
    return login


def fetch_maintainer_activity(
    pr: PullRequestRef,
    /,
    *,
    author_login: str,
) -> tuple[str, tuple[str, ...]] | None:
    """Return one triage-or-higher maintainer with deliberate activity on a PR."""

    candidates: dict[str, set[str]] = {}

    for event in fetch_timeline(github=pr.github, repo=pr.repo, number=pr.number):
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
            login = _human_login(
                value=event.get("user"),
                context="engagement",
                excluded_login=author_login,
            )
            signal = "comment" if event_name == "commented" else "review"
        elif isinstance(event_name, str) and event_name not in PASSIVE_TIMELINE_EVENTS:
            label = event.get("label")
            if (
                event_name in {"labeled", "unlabeled"}
                and isinstance(label, dict)
                and label.get("name") == TRIGGER_LABEL
            ):
                continue
            login = _human_login(
                value=event.get("actor"),
                context="activity actor",
                excluded_login=author_login,
            )
            signal = {
                "labeled": "label_change",
                "review_requested": "review_request",
                "unlabeled": "label_change",
            }.get(event_name, event_name)
        if login is None or signal is None:
            continue
        candidates.setdefault(login.casefold(), set()).add(signal)
        if len(candidates) > MAX_ENGAGEMENT_CANDIDATES:
            raise RuntimeError("maintainer engagement exceeds the candidate limit")

    for login in sorted(candidates):
        if fetch_user_has_triage_permission(
            github=pr.github, repo=pr.repo, login=login
        ):
            return f"@{login}", tuple(sorted(candidates[login]))
    return None


def fetch_maintainer_requested_reviewers(
    pr: PullRequestRef,
    /,
    *,
    author_login: str,
    limit: int = MAX_HANDOFF_REVIEWERS,
) -> list[str]:
    """Return triage-or-higher users another such maintainer asked to review."""

    requests: dict[str, tuple[str, str]] = {}
    for event in fetch_timeline(github=pr.github, repo=pr.repo, number=pr.number):
        event_name = event.get("event")
        if event_name not in {"review_requested", "review_request_removed"}:
            continue
        # Team requests carry requested_team instead of requested_reviewer.
        reviewer = _human_login(
            value=event.get("requested_reviewer"),
            context="requested reviewer",
            excluded_login=author_login,
        )
        if reviewer is None:
            continue
        key = reviewer.casefold()
        requests.pop(key, None)
        requester = _human_login(
            value=event.get("actor"),
            context="review requester",
            excluded_login=author_login,
        )
        if (
            event_name == "review_requested"
            and requester is not None
            and requester.casefold() != key
        ):
            requests[key] = (reviewer, requester)
        if len(requests) > MAX_ENGAGEMENT_CANDIDATES:
            raise RuntimeError("review requests exceed the candidate limit")

    permissions: dict[str, bool] = {}

    def has_triage(login: str) -> bool:
        key = login.casefold()
        if key not in permissions:
            permissions[key] = fetch_user_has_triage_permission(
                github=pr.github, repo=pr.repo, login=login
            )
        return permissions[key]

    reviewers: list[str] = []
    for key in sorted(requests):
        reviewer, requester = requests[key]
        if len(reviewers) < limit and has_triage(requester) and has_triage(reviewer):
            reviewers.append(reviewer)
    return reviewers


def fetch_actionable_labelers(
    *,
    github: GitHubClient,
    repo: str,
    issues: list[int],
    author_login: str,
    limit: int = MAX_HANDOFF_REVIEWERS,
) -> list[str]:
    """Return triage-or-higher users who labeled any of the issues actionable."""

    labelers: dict[str, str] = {}
    for issue in issues:
        labeler: str | None = None
        for event in fetch_timeline(github=github, repo=repo, number=issue):
            label = event.get("label")
            if (
                event.get("event") in {"labeled", "unlabeled"}
                and isinstance(label, dict)
                and isinstance(label.get("name"), str)
                and label["name"].casefold() == "actionable"
            ):
                labeler = (
                    _human_login(
                        value=event.get("actor"),
                        context="issue labeler",
                        excluded_login=author_login,
                    )
                    if event["event"] == "labeled"
                    else None
                )
        if labeler is not None:
            labelers.setdefault(labeler.casefold(), labeler)

    return [
        login
        for _, login in sorted(labelers.items())
        if fetch_user_has_triage_permission(github=github, repo=repo, login=login)
    ][:limit]


def is_already_handled(labels: Any) -> bool:
    """Return whether a prior triage outcome or an opt-out label makes this run a no-op."""

    if not isinstance(labels, list) or any(
        not isinstance(label, dict)
        or not isinstance(label.get("name"), str)
        or not label["name"]
        for label in labels
    ):
        raise RuntimeError("pull request labels are invalid")
    names = {label["name"].casefold() for label in labels}
    return bool(names & HANDLED_LABELS)


def fetch_issue_labels(
    *,
    github: GitHubClient,
    repo: str,
    number: int,
) -> frozenset[str] | None:
    """Return casefolded labels for one same-repository issue, or None if absent."""

    issue_url = f"https://github.com/{repo}/issues/{number}"
    issue = github.graphql(query=ISSUE_QUERY, variables={"url": issue_url}).get(
        "resource"
    )
    # Pull requests and transferred issues do not resolve to this Issue URL.
    if issue is None or issue == {}:
        return None
    try:
        if issue["url"].casefold() != issue_url.casefold():
            return None
        labels = issue["labels"]
        if issue["number"] != number:
            raise RuntimeError("issue response is incomplete")
        if labels["pageInfo"]["hasNextPage"] is not False:
            raise RuntimeError("issue has more than 100 labels")
        return frozenset(label["name"].casefold() for label in labels["nodes"])
    except (KeyError, TypeError, AttributeError) as exc:
        raise RuntimeError("issue response is incomplete") from exc


def fetch_actionable_linked_issues(pr: PullRequestRef, /) -> list[int]:
    """Return same-repository closing references labeled actionable."""

    owner, name = pr.repo.split("/", 1)
    data = pr.github.graphql(
        query=LINKED_ISSUES_QUERY,
        variables={"owner": owner, "name": name, "number": pr.number},
    )
    repository = data.get("repository")
    pull_request = repository.get("pullRequest") if repository else None
    if pull_request is None:
        raise RuntimeError(f"pull request not found: {pr.repo}#{pr.number}")
    references = pull_request["closingIssuesReferences"]
    if references["pageInfo"]["hasNextPage"]:
        raise RuntimeError("pull request has more than 100 linked issues")

    actionable: list[int] = []
    try:
        for issue in references["nodes"]:
            labels = issue["labels"]
            if labels["pageInfo"]["hasNextPage"]:
                raise RuntimeError("linked issue has more than 100 labels")
            if issue["repository"]["nameWithOwner"].casefold() != pr.repo.casefold():
                continue
            names = {label["name"].casefold() for label in labels["nodes"]}
            if "actionable" in names:
                actionable.append(issue["number"])
    except (KeyError, TypeError, AttributeError) as exc:
        raise RuntimeError("linked issue response is incomplete") from exc
    if any(type(issue) is not int for issue in actionable):
        raise RuntimeError("linked issue response is incomplete")
    return sorted(set(actionable))


def parse_description_claims(*, body: str, repo: str) -> tuple[list[str], list[int]]:
    """Return claimed supporter logins and same-repository part-of issues."""

    text = UNRENDERED_MARKDOWN_RE.sub(" ", body)
    logins: dict[str, str] = {}
    for match in SUPPORTED_BY_RE.finditer(text):
        logins.setdefault(match["login"].casefold(), match["login"])
    issues: dict[int, None] = {}
    for match in PART_OF_RE.finditer(text):
        if match["repo"] is None or match["repo"].casefold() == repo.casefold():
            issues.setdefault(int(match["number"]))
    return list(logins.values())[:MAX_HANDOFF_REVIEWERS], list(issues)[
        :MAX_PART_OF_ISSUES
    ]


def verify_description_claims(
    *, github: GitHubClient, repo: str, logins: list[str], issues: list[int]
) -> tuple[list[str], list[int]]:
    """Return claimed supporters with triage access and claimed actionable issues."""

    supporters: list[str] = []
    for login in logins:
        if login.casefold() in NON_HUMAN_LOGINS:
            continue
        try:
            if fetch_user_has_triage_permission(github=github, repo=repo, login=login):
                supporters.append(login)
        except RuntimeError as exc:
            # gh reports an unknown user as HTTP 404; that claim is unverified.
            if "(HTTP 404)" not in str(exc):
                raise
    actionable_issues = [
        number
        for number in issues
        if "actionable"
        in (fetch_issue_labels(github=github, repo=repo, number=number) or ())
    ]
    return sorted(supporters, key=str.casefold), actionable_issues


def assess_intake(
    pr: PullRequestRef,
    /,
    *,
    workflow_sha: str,
) -> IntakeResult:
    """Read the PR once and derive every gate fact and handoff reviewer."""

    pr_data = pr.github.json(f"repos/{pr.repo}/pulls/{pr.number}")
    try:
        pr_number = pr_data["number"]
        base_repo = pr_data["base"]["repo"]["full_name"]
        base_ref = pr_data["base"]["ref"]
        head_sha = pr_data["head"]["sha"]
        state = pr_data["state"]
        draft = pr_data["draft"]
        author_login = pr_data["user"]["login"]
    except (KeyError, TypeError, AttributeError) as exc:
        raise RuntimeError("pull request response is incomplete") from exc
    if (
        type(pr_number) is not int
        or not isinstance(base_repo, str)
        or not isinstance(base_ref, str)
        or not base_ref
        or not isinstance(head_sha, str)
        or not head_sha
        or not isinstance(state, str)
        or state not in {"open", "closed"}
        or type(draft) is not bool
        or not isinstance(author_login, str)
        or not author_login
    ):
        raise RuntimeError("pull request response is incomplete")
    if pr_number != pr.number or base_repo.casefold() != pr.repo.casefold():
        raise RuntimeError("pull request identity does not match the target")
    is_open_non_draft_pr_against_main = (
        base_ref == TARGET_BASE_REF and state == "open" and not draft
    )

    already_handled = is_already_handled(pr_data.get("labels"))
    body = pr_data.get("body") or ""
    author_has_triage_permission = False
    actionable_linked_issues: list[int] = []
    maintainer_activity = None
    supporters: list[str] = []
    actionable_part_of_issues: list[int] = []
    # Every fact is gathered for an active, unhandled PR; none can matter otherwise.
    if is_open_non_draft_pr_against_main and not already_handled:
        author_has_triage_permission = fetch_user_has_triage_permission(
            github=pr.github,
            repo=pr.repo,
            login=author_login,
        )
        actionable_linked_issues = fetch_actionable_linked_issues(pr)
        maintainer_activity = fetch_maintainer_activity(pr, author_login=author_login)
        print(
            "Auto PR Triage maintainer activity: "
            + json.dumps(
                {
                    "maintainer": maintainer_activity[0]
                    if maintainer_activity
                    else None,
                    "found": maintainer_activity is not None,
                    "signals": list(maintainer_activity[1])
                    if maintainer_activity
                    else [],
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
        claimed_logins, claimed_issues = parse_description_claims(
            body=body, repo=pr.repo
        )
        supporters, actionable_part_of_issues = verify_description_claims(
            github=pr.github, repo=pr.repo, logins=claimed_logins, issues=claimed_issues
        )
        print(
            "Auto PR Triage description claims: "
            + json.dumps(
                {
                    "claimed_supporters": claimed_logins,
                    "supporters": supporters,
                    "claimed_part_of_issues": claimed_issues,
                    "actionable_part_of_issues": actionable_part_of_issues,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
    facts = {
        "is_open_non_draft_pr_against_main": is_open_non_draft_pr_against_main,
        "is_already_handled": already_handled,
        "author_has_triage_permission": author_has_triage_permission,
        "has_actionable_linked_issue": bool(actionable_linked_issues),
        "has_maintainer_activity": maintainer_activity is not None,
        "has_supporter": bool(supporters),
        "has_related_actionable_issue": bool(actionable_part_of_issues),
    }
    admitted = passes_intake(**facts)
    actionable_labelers: list[str] = []
    maintainer_requested_reviewers: list[str] = []
    if admitted:
        # Handoff reviewers let planning triage a PR with uncovered concerns.
        actionable_labelers = fetch_actionable_labelers(
            github=pr.github,
            repo=pr.repo,
            issues=(actionable_linked_issues + actionable_part_of_issues)[
                :MAX_PART_OF_ISSUES
            ],
            author_login=author_login,
        )
        maintainer_requested_reviewers = fetch_maintainer_requested_reviewers(
            pr,
            author_login=author_login,
        )
        print(
            "Auto PR Triage handoff reviewers: "
            + json.dumps(
                {
                    "actionable_labelers": actionable_labelers,
                    "maintainer_requested_reviewers": (maintainer_requested_reviewers),
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
    identity = PullRequestIdentity(pr.repo, pr.number, head_sha, workflow_sha)
    intake_facts = IntakeFacts(
        **facts,
        supporters=tuple(supporters),
        actionable_labelers=tuple(actionable_labelers),
        maintainer_requested_reviewers=tuple(maintainer_requested_reviewers),
    )
    return IntakeResult(identity, intake_facts, author_login, pr_data["title"], body)


def write_json(*, path: Path, value: Any) -> None:
    """Overwrite a path with deterministic, newline-terminated JSON."""

    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    """Parse the event identity and output destinations."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pr", type=int, help="pull request number")
    parser.add_argument("--repository", required=True)
    parser.add_argument("--workflow-sha", required=True)
    parser.add_argument("--expected-base-ref", required=True)
    parser.add_argument("--proxy", default=os.environ.get("HTTPS_PROXY"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--github-output", type=Path)
    return parser.parse_args()


def main() -> int:
    """Write intake.json and expose whether the PR is active for later stages."""

    args = parse_args()
    if args.pr < 1:
        raise SystemExit("PR number must be positive")
    if args.expected_base_ref != TARGET_BASE_REF:
        raise SystemExit(f"--expected-base-ref must be {TARGET_BASE_REF}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        intake = assess_intake(
            PullRequestRef(
                github=GitHubReader(args.proxy), repo=args.repository, number=args.pr
            ),
            workflow_sha=args.workflow_sha,
        )
        write_json(path=args.output_dir / "intake.json", value=intake.to_dict())
        facts = intake.facts
        if args.github_output:
            with args.github_output.open("a") as output:
                output.write(f"active={str(facts.is_active).lower()}\n")
    except Exception as exc:
        write_json(
            path=args.output_dir / "error.json",
            value={"error": str(exc), "stage": "intake", "type": type(exc).__name__},
        )
        print(
            f"{args.repository}#{args.pr}: intake failed; keeping PR open; details withheld",
            file=sys.stderr,
            flush=True,
        )
        return 1
    if facts.passes_intake:
        status = "passes intake"
    elif not facts.is_open_non_draft_pr_against_main:
        status = "PR is outside the active target state"
    elif facts.is_already_handled:
        status = "existing triage outcome or opt-out label recorded"
    else:
        status = "fails intake unless a team's bypass matches"
    print(f"{args.repository}#{args.pr}: {status}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
