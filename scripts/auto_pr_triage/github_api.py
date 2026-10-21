"""Read-only GitHub access shared by the Auto PR Triage analysis stages."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Any, Protocol


MAX_TIMELINE_PAGES = 10
TIMELINE_PAGE_SIZE = 100
SUBMITTED_REVIEW_STATES = frozenset(
    {"approved", "changes_requested", "commented", "dismissed"}
)


def run_command(
    command: list[str],
    *,
    input_text: str | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run an argv directly with a timeout, raising on any nonzero exit."""

    result = subprocess.run(
        command,
        input=input_text,
        text=True,
        capture_output=True,
        env=env,
        timeout=300,
        check=False,
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            f"{command[0]} failed with exit {result.returncode}: {detail}"
        )
    return result


@dataclass(frozen=True)
class PullRequestRef:
    """Identify one pull request and the client that reads or writes it."""

    github: GitHubClient
    repo: str
    number: int


class GitHubReader:
    """Read GitHub REST and GraphQL data through a noninteractive gh process."""

    def __init__(self, proxy: str | None) -> None:
        """Snapshot the environment and optionally route HTTP through a proxy."""

        self.env = os.environ.copy()
        if proxy:
            self.env["HTTPS_PROXY"] = proxy
            self.env["HTTP_PROXY"] = proxy

    def json(self, endpoint: str) -> Any:
        """Return decoded JSON from one GitHub REST endpoint."""

        result = run_command(["gh", "api", endpoint], env=self.env)
        return json.loads(result.stdout)

    def graphql(self, *, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        """Execute a parameterized GraphQL query and reject reported errors."""

        request = json.dumps({"query": query, "variables": variables})
        result = run_command(
            ["gh", "api", "graphql", "--input", "-"],
            input_text=request,
            env=self.env,
        )
        response = json.loads(result.stdout)
        if response.get("errors"):
            raise RuntimeError(
                f"GitHub GraphQL failed: {json.dumps(response['errors'])}"
            )
        return response["data"]


class GitHubClient(Protocol):
    """Minimal read interface required to collect reviewer state."""

    def json(self, endpoint: str) -> Any: ...
    def graphql(self, *, query: str, variables: dict[str, Any]) -> dict[str, Any]: ...


def fetch_timeline(
    *, github: GitHubClient, repo: str, number: int
) -> list[dict[str, Any]]:
    """Return one complete issue or pull-request timeline within the bound."""

    timeline: list[dict[str, Any]] = []
    for page in range(1, MAX_TIMELINE_PAGES + 1):
        events = github.json(
            f"repos/{repo}/issues/{number}/timeline?per_page={TIMELINE_PAGE_SIZE}&page={page}"
        )
        if not isinstance(events, list) or any(
            not isinstance(event, dict) for event in events
        ):
            raise RuntimeError("pull request timeline response is invalid")
        timeline.extend(events)
        if len(events) < TIMELINE_PAGE_SIZE:
            return timeline
    raise RuntimeError("pull request timeline exceeds the collection limit")
