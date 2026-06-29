#!/usr/bin/env python3
"""Report stats on continuously failing PyTorch CI jobs from the HUD.

A standalone tool for the pt2 oncall to run by hand. It pulls the recent job
grid that backs https://hud.pytorch.org for a branch or commit, finds jobs that
are currently on a failure streak, and prints the streak length, overall
failure rate, when each started failing, and the distinct failure signatures.

This intentionally does not share code with tools/alerts/create_alerts.py: that
script is automation that files GitHub issues using alert-firing thresholds,
whereas this is a read-only human-facing stats view. The only contract here is
the public HUD API response shape.

Authentication:
    hud.pytorch.org sits behind a Vercel WAF that rejects anonymous bot traffic
    with a 429 on the first request. Set the HUD_INTERNAL_BOT_TOKEN env var to
    the internal bypass token and it is sent as the x-hud-internal-bot header.
    Without it every request 429s. It must be exported (not just a shell var)
    so the process inherits it, e.g. in ~/.bashrc:

        export HUD_INTERNAL_BOT_TOKEN=<token>

Examples:
    scripts/hud/analyze_failing_jobs.py
    scripts/hud/analyze_failing_jobs.py main --commits 100
    scripts/hud/analyze_failing_jobs.py 44b82837a1 --filter-job-name "trunk.*"
    scripts/hud/analyze_failing_jobs.py --filter-min-streak 5
    scripts/hud/analyze_failing_jobs.py --commits 1000 --csv > scripts/hud/analyze_failing_jobs.csv
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any


HUD_API = "https://hud.pytorch.org/api/hud/pytorch/pytorch"

# The HUD paginates commits in fixed-size pages and caps per_page at this value;
# we fetch whole pages and trim to the exact commit count requested.
PER_PAGE = 100

# Conclusions that don't represent a real signal for a commit/job pair.
SKIPPED_CONCLUSIONS = {"neutral", "skipped"}
# Two failures whose captured error text is at least this similar are treated
# as the same underlying failure.
SIMILARITY_THRESHOLD = 0.75


def is_skipped(job: dict[str, Any]) -> bool:
    return job.get("conclusion") in SKIPPED_CONCLUSIONS or job.get("conclusion") is None


def is_failure(job: dict[str, Any]) -> bool:
    conclusion = job.get("conclusion")
    return conclusion is not None and conclusion not in ("success", "pending")


def transpose_grid(
    pages: list[tuple[list[str], list[dict[str, Any]]]],
) -> dict[str, list[dict[str, Any]]]:
    """Turn the HUD sha-grids (rows = commits) into per-job columns.

    Each page is a ``(job_names, sha_grid)`` pair. A row's ``jobs`` list is
    positionally parallel to that page's ``job_names`` only -- the job set
    varies across pages -- so we transpose per page and key columns by job name.
    Pages and rows arrive newest-first, which we preserve. We stamp the sha onto
    each job so a job column is self-contained.

    Positional alignment is only sound when a row has exactly one entry per job
    name; a length mismatch means a result could be attributed to the wrong job,
    so we fail loudly rather than guess.
    """
    columns: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for job_names, sha_grid in pages:
        for row in sha_grid:
            jobs = row.get("jobs", [])
            if len(jobs) != len(job_names):
                raise SystemExit(
                    f"HUD row {row.get('sha')} has {len(jobs)} job results but the "
                    f"page lists {len(job_names)} job names; cannot align them."
                )
            for name, job in zip(job_names, jobs):
                columns[name].append({**job, "sha": row.get("sha")})
    return columns


def current_failure_streak(statuses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Most recent run of consecutive failures, ignoring skipped runs.

    Walks from newest to oldest; the streak ends at the first non-failing
    (non-skipped) result. Empty if the latest real result isn't a failure.
    """
    streak: list[dict[str, Any]] = []
    for job in statuses:
        if is_skipped(job):
            continue
        if is_failure(job):
            streak.append(job)
        else:
            break
    return streak


def group_failures(jobs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group failing jobs by similarity of their captured error text."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for job in jobs:
        captures = job.get("failureCaptures") or []
        signature = " ".join(captures) if captures else "unclassified"
        for existing in groups:
            ratio = SequenceMatcher(None, signature, existing).ratio()
            if ratio > SIMILARITY_THRESHOLD:
                groups[existing].append(job)
                break
        else:
            groups[signature].append(job)
    return groups


def summarize_job(
    name: str, statuses: list[dict[str, Any]], oldest_sha: str | None
) -> dict[str, Any]:
    real = [j for j in statuses if not is_skipped(j)]
    streak = current_failure_streak(statuses)
    groups = group_failures(streak)
    distinct = sorted(
        ({"capture": sig, "count": len(js)} for sig, js in groups.items()),
        key=lambda d: d["count"],
        reverse=True,
    )
    latest_real = real[0] if real else None
    return {
        "job_name": name,
        "current_conclusion": latest_real.get("conclusion") if latest_real else None,
        "total_runs": len(real),
        "current_streak": len(streak),
        "failing_since_sha": streak[-1].get("sha") if streak else None,
        # Clipped means the true start may predate the window, so more --commits
        # could help. That needs two things: (A) no success/pending anywhere in
        # the observed runs, so nothing pins the start (len(streak) == len(real)),
        # and (B) the job still has data at the oldest fetched commit, so older
        # data plausibly exists. (B) uses the oldest cell, not the oldest failure,
        # to tolerate no-data gaps at the window edge. A job that simply stopped
        # running earlier fails (B) and is not clipped.
        "failing_since_clipped": (
            bool(streak)
            and len(streak) == len(real)
            and bool(statuses)
            and statuses[-1].get("sha") == oldest_sha
        ),
        "distinct_failures": distinct,
    }


def collect_stats(
    pages: list[tuple[list[str], list[dict[str, Any]]]],
    filter_job_name: str,
    min_streak: int,
) -> list[dict[str, Any]]:
    import re

    pattern = re.compile(filter_job_name) if filter_job_name else None
    # Pages arrive newest-first. The newest commit anchors "current" streaks;
    # the oldest commit fetched detects streaks running off the window.
    newest_sha = next((grid[0].get("sha") for _, grid in pages if grid), None)
    oldest_sha = next(
        (grid[-1].get("sha") for _, grid in reversed(pages) if grid), None
    )
    stats = []
    for name, statuses in transpose_grid(pages).items():
        if pattern and not pattern.match(name):
            continue
        # Skip stale jobs: a job not present at the newest commit stopped running,
        # so its "streak" is ancient history, not a current failure.
        if not statuses or statuses[0].get("sha") != newest_sha:
            continue
        summary = summarize_job(name, statuses, oldest_sha)
        if summary["current_streak"] < min_streak:
            continue
        stats.append(summary)
    stats.sort(key=lambda s: (s["current_streak"], s["total_runs"]), reverse=True)
    return stats


def fetch_hud(ref: str, commits: int) -> Any:
    import math
    import os
    import sys

    import requests

    # hud.pytorch.org sits behind a Vercel WAF that challenges anonymous bot
    # traffic with a 429 on the first request. The internal bypass token, when
    # provided via env, is sent as the x-hud-internal-bot header to skip it.
    headers = {"User-Agent": "pytorch-oncall-hud-stats"}
    bot_token = os.environ.get("HUD_INTERNAL_BOT_TOKEN", "")
    if bot_token:
        headers["x-hud-internal-bot"] = bot_token

    pages: list[tuple[list[str], list[dict[str, Any]]]] = []
    remaining = commits
    num_pages = math.ceil(commits / PER_PAGE)
    for page in range(num_pages):
        print(f"Fetching page {page + 1}/{num_pages}...", file=sys.stderr)
        resp = requests.get(
            f"{HUD_API}/{ref}/{page}",
            params={"per_page": PER_PAGE},
            headers=headers,
        )
        if resp.status_code == 429:
            hint = (
                "set it (see HUD_INTERNAL_BOT_TOKEN in this script's docstring)"
                if not bot_token
                else "the token may be invalid or expired"
            )
            raise SystemExit(
                f"HUD returned 429 (Vercel WAF). The x-hud-internal-bot bypass "
                f"token is required; {hint}."
            )
        resp.raise_for_status()
        data = resp.json()
        # The job set varies per page, so each page keeps its own jobNames.
        sha_grid = data["shaGrid"][:remaining]
        pages.append((data["jobNames"], sha_grid))
        remaining -= len(sha_grid)
        if remaining <= 0:
            break
    return pages


def print_report(stats: list[dict[str, Any]], ref: str) -> None:
    print(f"\nFound {len(stats)} continuously failing jobs at ref {ref}\n")
    for s in stats:
        print(f"- {s['job_name']}")
        print(
            f"    current: {s['current_conclusion']} | streak: {s['current_streak']} | "
            f"runs: {s['total_runs']}"
        )
        if s["failing_since_sha"]:
            print(f"    failing since: {s['failing_since_sha'][:10]}")
            if s["failing_since_clipped"]:
                print(
                    "        (or earlier; failing throughout window, rerun with more --commits)"
                )
        for failure in s["distinct_failures"]:
            capture = failure["capture"]
            if len(capture) > 100:
                capture = capture[:97] + "..."
            print(f"    [{failure['count']}x] {capture}")
        print()


def write_csv(stats: list[dict[str, Any]]) -> None:
    import csv
    import sys

    writer = csv.writer(sys.stdout)
    writer.writerow(
        [
            "job_name",
            "current_conclusion",
            "total_runs",
            "current_streak",
            "failing_since_sha",
            "failing_since_clipped",
            "distinct_failures",
        ]
    )
    for s in stats:
        failures = " | ".join(
            f"{f['count']}x {f['capture']}" for f in s["distinct_failures"]
        )
        writer.writerow(
            [
                s["job_name"],
                s["current_conclusion"],
                s["total_runs"],
                s["current_streak"],
                s["failing_since_sha"],
                s["failing_since_clipped"],
                failures,
            ]
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "ref", nargs="?", default="main", help="Branch or commit sha (default: main)"
    )
    parser.add_argument(
        "--commits",
        type=int,
        default=PER_PAGE,
        help=f"Commits of history to analyze (default: {PER_PAGE})",
    )
    parser.add_argument(
        "--filter-job-name",
        default="",
        help="Only report jobs whose name matches this regex",
    )
    parser.add_argument(
        "--filter-min-streak",
        type=int,
        default=1,
        help="Only report jobs that failed at least this many consecutive most-recent "
        "commits (default: 1, i.e. currently failing)",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Emit CSV to stdout instead of the text report",
    )
    return parser.parse_args()


def main() -> None:
    import sys

    args = parse_args()
    pages = fetch_hud(args.ref, args.commits)
    stats = collect_stats(
        pages,
        args.filter_job_name,
        args.filter_min_streak,
    )
    if args.csv:
        write_csv(stats)
    else:
        print_report(stats, args.ref)

    clipped = sum(1 for s in stats if s["failing_since_clipped"])
    if clipped:
        print(
            f"\n{clipped} streak(s) fill the {args.commits}-commit window; their true "
            f"start is unknown. Rerun with --commits {args.commits * 2} to look further back.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
