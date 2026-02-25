#!/usr/bin/env python3
"""Parse cherry-pick comments from a PyTorch GitHub issue and extract trunk PR info.

Usage:
    python scripts/release_notes/parse_cherry_picks.py \\
        https://github.com/pytorch/pytorch/issues/170119 \\
        --commitlist scripts/release_notes/results/commitlist.csv

Outputs a CSV file to scripts/release_notes/results/cherry_picks_<issue_number>.csv.
Columns: comment_id, pr_number, pr_title, commit_sha
Validates that each commit_sha matches an entry in the commitlist and logs
warnings for any mismatches.
"""

import argparse
import csv
import json
import logging
import re
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_issue_url(url: str) -> tuple[str, str]:
    """Extract owner/repo and issue number from a GitHub issue URL."""
    m = re.match(r"https://github\.com/([^/]+/[^/]+)/issues/(\d+)", url)
    if not m:
        raise ValueError(f"Invalid GitHub issue URL: {url}")
    return m.group(1), m.group(2)


def gh_api(endpoint: str) -> list | dict:
    """Call gh api with pagination and return parsed JSON."""
    result = subprocess.run(
        ["gh", "api", endpoint, "--paginate"],
        capture_output=True,
        text=True,
        check=True,
    )
    # --paginate may return multiple JSON arrays concatenated; we need to handle that
    # gh api --paginate with --jq is cleaner, but let's parse raw output
    # When paginating, gh outputs one JSON array per page on separate "lines"
    output = result.stdout.strip()
    if not output:
        return []

    # Try parsing as a single JSON value first
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        pass

    # If that fails, it's multiple JSON arrays concatenated
    # Split on ][ boundaries and merge
    all_items = []
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(output):
        # Skip whitespace
        while pos < len(output) and output[pos] in " \t\n\r":
            pos += 1
        if pos >= len(output):
            break
        obj, end_pos = decoder.raw_decode(output, pos)
        if isinstance(obj, list):
            all_items.extend(obj)
        else:
            all_items.append(obj)
        pos = end_pos
    return all_items


def fetch_comments(repo: str, issue_number: str) -> list[dict]:
    """Fetch all comments from a GitHub issue."""
    endpoint = f"repos/{repo}/issues/{issue_number}/comments?per_page=100"
    comments = gh_api(endpoint)
    logger.info(f"Fetched {len(comments)} comments from {repo}#{issue_number}")
    return comments


def fetch_pr_title(repo: str, pr_number: str) -> str:
    """Fetch the title of a PR."""
    try:
        result = subprocess.run(
            ["gh", "api", f"repos/{repo}/pulls/{pr_number}", "--jq", ".title"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to fetch title for PR #{pr_number}: {e}")
        return ""


def fetch_landed_commit(repo: str, pr_number: str) -> str:
    """Fetch the actual landed commit SHA for a PR.

    PyTorch uses a merge bot that squash-merges outside of GitHub's standard
    merge mechanism, so the PR API's merge_commit_sha is unreliable.

    Strategy:
    1. Issue events API: look for the 'closed' event with a commit_id
    2. Commit search API: search for commits containing '(#NNNNN)' in message
    """
    # Strategy 1: Issue events API
    try:
        result = subprocess.run(
            [
                "gh", "api",
                f"repos/{repo}/issues/{pr_number}/events?per_page=100",
                "--paginate",
                "--jq",
                '.[] | select(.event == "closed" and .commit_id) | .commit_id',
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        commit_ids = result.stdout.strip().splitlines()
        if commit_ids and commit_ids[0]:
            return commit_ids[0]
    except subprocess.CalledProcessError:
        pass

    # Strategy 2: Commit search API
    logger.info(f"  Events API had no commit for PR #{pr_number}, trying search API...")
    try:
        result = subprocess.run(
            [
                "gh", "api",
                f'search/commits?q=repo:{repo}+"(#{pr_number})"',
                "--jq",
                ".items[0].sha",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        sha = result.stdout.strip()
        if sha and sha != "null":
            return sha
    except subprocess.CalledProcessError:
        pass

    logger.warning(f"Could not find landed commit for PR #{pr_number}")
    return ""


def extract_trunk_prs(comment_body: str) -> list[dict]:
    """Extract trunk PR numbers/commit hashes from a comment body.

    Returns a list of dicts with keys: pr_number (str or ""), raw_commit (str or "").
    Returns an empty list if the comment doesn't contain the trunk PR section.
    """
    # Find the "Link to landed trunk PR" section
    # Handle variations: "Link to landed trunk PR", "Link to the landed trunk PR"
    trunk_pattern = re.compile(
        r"Link to (?:the )?landed trunk PR[^:]*:\s*\n(.*?)(?=Li[nn][kt] to (?:the )?release branch PR|Criteria Category:|$)",
        re.DOTALL | re.IGNORECASE,
    )
    match = trunk_pattern.search(comment_body)
    if not match:
        return []

    trunk_section = match.group(1).strip()

    # Check for NA/N/A
    if re.match(r"^\*?\s*-?\s*N/?A\s*$", trunk_section, re.IGNORECASE):
        return [{"pr_number": "", "raw_commit": ""}]

    results = []

    # Extract PR URLs
    pr_urls = re.findall(
        r"https://github\.com/pytorch/pytorch/pull/(\d+)", trunk_section
    )
    for pr_num in pr_urls:
        results.append({"pr_number": pr_num, "raw_commit": ""})

    # Extract raw commit hashes (40-char hex) that aren't part of a URL
    # Remove URLs first to avoid matching hashes embedded in URLs
    section_no_urls = re.sub(r"https://\S+", "", trunk_section)
    raw_commits = re.findall(r"\b([0-9a-f]{40})\b", section_no_urls)
    for commit_hash in raw_commits:
        results.append({"pr_number": "", "raw_commit": commit_hash})

    # If we found the section but nothing matched, treat as empty/NA
    if not results:
        return [{"pr_number": "", "raw_commit": ""}]

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Parse cherry-pick comments from a PyTorch GitHub issue"
    )
    parser.add_argument("issue_url", help="GitHub issue URL to parse")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output CSV file path (default: results/cherry_picks_<issue_number>.csv relative to script)",
    )
    parser.add_argument(
        "--commitlist",
        required=True,
        help="Path to commitlist.csv to validate commit hashes against",
    )
    args = parser.parse_args()

    repo, issue_number = parse_issue_url(args.issue_url)
    if args.output:
        output_path = args.output
    else:
        script_dir = Path(__file__).resolve().parent
        results_dir = script_dir / "results"
        results_dir.mkdir(exist_ok=True)
        output_path = str(results_dir / f"cherry_picks_{issue_number}.csv")

    comments = fetch_comments(repo, issue_number)

    rows = []
    for comment in comments:
        comment_id = comment["id"]
        body = comment["body"] or ""

        trunk_prs = extract_trunk_prs(body)
        if not trunk_prs:
            # Comment doesn't contain the trunk PR section — skip
            continue

        if len(trunk_prs) > 1:
            logger.warning(
                f"Comment {comment_id} has {len(trunk_prs)} trunk PRs: "
                + ", ".join(
                    p["pr_number"] or p["raw_commit"] or "N/A" for p in trunk_prs
                )
            )

        for pr_info in trunk_prs:
            pr_number = pr_info["pr_number"]
            raw_commit = pr_info["raw_commit"]
            pr_title = ""
            commit_sha = raw_commit  # Use raw commit if provided

            if pr_number:
                logger.info(f"Fetching info for PR #{pr_number}...")
                pr_title = fetch_pr_title(repo, pr_number)
                commit_sha = fetch_landed_commit(repo, pr_number)

            rows.append(
                {
                    "comment_id": comment_id,
                    "pr_number": pr_number,
                    "pr_title": pr_title,
                    "commit_sha": commit_sha,
                }
            )

    # Validate against commitlist
    commitlist_hashes = set()
    with open(args.commitlist, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if row:
                commitlist_hashes.add(row[0])

    logger.info(f"Loaded {len(commitlist_hashes)} hashes from {args.commitlist}")

    matched = 0
    mismatched = 0
    skipped = 0
    for row in rows:
        sha = row["commit_sha"]
        if not sha:
            skipped += 1
            continue
        # commitlist uses abbreviated hashes; check if any is a prefix of
        # our full hash, or vice versa
        if any(
            sha.startswith(cl_hash) or cl_hash.startswith(sha)
            for cl_hash in commitlist_hashes
        ):
            matched += 1
        else:
            mismatched += 1
            logger.warning(
                f"Commit {sha[:11]} (PR #{row['pr_number'] or 'N/A'}) "
                f"not found in commitlist"
            )

    logger.info(
        f"Commitlist validation: {matched} matched, {mismatched} not found, "
        f"{skipped} skipped (no hash)"
    )

    # Write CSV
    fieldnames = ["comment_id", "pr_number", "pr_title", "commit_sha"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
