#!/usr/bin/env python3
"""
Generate email to GitHub username mappings from Git commits and PR author data.

Usage:
    python scripts/generate_contributors.py --since "1 day ago" --pr-authors ~/Downloads/pr-author.csv
    python scripts/generate_contributors.py --since "6 months ago" --pr-authors ~/Downloads/pr-author.csv -o .github/contributors.txt

To get PR author data, you can use the GitHub CLI:
    gh pr list --repo pytorch/pytorch --state merged --limit 1000 --json number,author \\
        | jq -r '["author","number"], (.[] | [.author.login, .number]) | @csv' > pr-author.csv

Or use --fetch-missing to automatically fetch missing PR authors.
"""

import argparse
import csv
import json
import re
import subprocess
import sys
from collections import defaultdict


def load_pr_authors(csv_path: str) -> dict[int, str]:
    """Load PR number -> GitHub username mapping from CSV."""
    pr_authors = {}
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pr_num = int(row["number"])
            author = row["author"]
            pr_authors[pr_num] = author
    return pr_authors


def fetch_pr_author(pr_number: int, repo: str = "pytorch/pytorch") -> str | None:
    """Fetch the author of a PR using the GitHub CLI."""
    try:
        result = subprocess.run(
            ["gh", "pr", "view", str(pr_number), "--repo", repo, "--json", "author"],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(result.stdout)
        return data.get("author", {}).get("login")
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return None


def fetch_missing_authors(
    commits: list[dict],
    email_to_github: dict[str, set[str]],
    pr_authors: dict[int, str],
    repo: str = "pytorch/pytorch",
    verbose: bool = False,
) -> dict[int, str]:
    """
    Fetch authors for PRs with unmapped emails using GitHub CLI.
    Only fetches what's needed - stops once all emails have mappings.
    """
    # Find emails that still need mapping
    unmapped_emails = set()
    for commit in commits:
        email = commit["author_email"]
        if email not in email_to_github or not email_to_github[email]:
            unmapped_emails.add(email)

    if not unmapped_emails:
        return {}

    # Build a mapping of email -> list of PR numbers for that email
    email_to_prs: dict[str, list[int]] = defaultdict(list)
    for commit in commits:
        email = commit["author_email"]
        pr_num = commit["pr_number"]
        if email in unmapped_emails and pr_num and pr_num not in pr_authors:
            email_to_prs[email].append(pr_num)

    # Deduplicate PR numbers per email and flatten
    prs_to_fetch = []
    for email in unmapped_emails:
        prs = email_to_prs.get(email, [])
        if prs:
            # Only need to fetch one PR per email
            prs_to_fetch.append((email, prs[0]))

    if verbose:
        print(
            f"  {len(unmapped_emails)} unmapped emails, fetching {len(prs_to_fetch)} PRs...",
            file=sys.stderr,
        )

    fetched = {}
    for i, (email, pr_num) in enumerate(prs_to_fetch, 1):
        if verbose:
            print(
                f"  Fetching PR #{pr_num} for {email} ({i}/{len(prs_to_fetch)})...",
                file=sys.stderr,
                end="",
            )
        author = fetch_pr_author(pr_num, repo)
        if author:
            fetched[pr_num] = author
            if verbose:
                print(f" {author}", file=sys.stderr)
        elif verbose:
            print(" (not found)", file=sys.stderr)

    return fetched


def get_commits(since: str, until: str | None = None) -> list[dict]:
    """Get commits since the given time, extracting author info and PR number."""
    # Format: hash|author name|author email|subject
    cmd = ["git", "log", f"--since={since}", "--format=%H|%an|%ae|%s"]
    if until:
        cmd.append(f"--until={until}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )

    commits = []
    pr_pattern = re.compile(r"\(#(\d+)\)\s*$")

    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        parts = line.split("|", 3)
        if len(parts) != 4:
            continue

        commit_hash, author_name, author_email, subject = parts

        # Extract PR number from subject
        match = pr_pattern.search(subject)
        pr_number = int(match.group(1)) if match else None

        commits.append(
            {
                "hash": commit_hash,
                "author_name": author_name,
                "author_email": author_email,
                "subject": subject,
                "pr_number": pr_number,
            }
        )

    return commits


def generate_mailmap(commits: list[dict], pr_authors: dict[int, str]) -> dict[str, str]:
    """
    Generate mailmap entries: email -> GitHub username.

    Returns a dict mapping email addresses to GitHub usernames.
    """
    # Track email -> GitHub username mappings
    # Use a dict of sets to handle potential conflicts
    email_to_github: dict[str, set[str]] = defaultdict(set)

    for commit in commits:
        pr_num = commit["pr_number"]
        email = commit["author_email"]

        if pr_num is None:
            continue

        github_user = pr_authors.get(pr_num)
        if github_user:
            email_to_github[email].add(github_user)

    return email_to_github


def format_output(email_to_github: dict[str, set[str]]) -> str:
    """Format the output as simple text: email github_username"""
    lines = []

    for email, github_users in sorted(email_to_github.items()):
        if len(github_users) == 1:
            github_user = next(iter(github_users))
            lines.append(f"{email} {github_user}")
        else:
            # Conflict - multiple GitHub users for same email
            lines.append(f"# CONFLICT for {email}: {', '.join(sorted(github_users))}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate email to GitHub username mappings from Git commits and PR author data"
    )
    parser.add_argument(
        "--since",
        default="1 day ago",
        help="Git log --since parameter (default: '1 day ago')",
    )
    parser.add_argument(
        "--until",
        help="Git log --until parameter (optional)",
    )
    parser.add_argument(
        "--pr-authors",
        required=True,
        help="Path to CSV file with columns: author,number",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file (default: stdout)",
    )
    parser.add_argument(
        "--missing-prs",
        help="Output file for missing PR numbers (one per line)",
    )
    parser.add_argument(
        "--fetch-missing",
        action="store_true",
        help="Fetch missing PR authors using GitHub CLI (requires 'gh' command)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print verbose information",
    )
    args = parser.parse_args()

    # Load PR authors
    pr_authors = load_pr_authors(args.pr_authors)
    if args.verbose:
        print(f"Loaded {len(pr_authors)} PR author mappings", file=sys.stderr)

    # Get commits
    commits = get_commits(args.since, args.until)
    if args.verbose:
        print(f"Found {len(commits)} commits since '{args.since}'", file=sys.stderr)
        commits_with_pr = sum(1 for c in commits if c["pr_number"] is not None)
        print(f"  {commits_with_pr} commits have PR numbers", file=sys.stderr)

    # Generate mailmap
    email_to_github = generate_mailmap(commits, pr_authors)
    if args.verbose:
        print(f"Generated {len(email_to_github)} email mappings", file=sys.stderr)

    # Check for missing PR authors
    missing_prs = set()
    for commit in commits:
        pr_num = commit["pr_number"]
        if pr_num and pr_num not in pr_authors:
            missing_prs.add(pr_num)

    if missing_prs and args.verbose:
        print(
            f"WARNING: {len(missing_prs)} PRs not found in CSV: {sorted(missing_prs)[:10]}...",
            file=sys.stderr,
        )

    # Fetch missing PR authors if requested
    if args.fetch_missing:
        if args.verbose:
            print("Fetching missing PR authors...", file=sys.stderr)
        fetched_authors = fetch_missing_authors(
            commits, email_to_github, pr_authors, verbose=args.verbose
        )
        pr_authors.update(fetched_authors)
        if args.verbose:
            print(f"Fetched {len(fetched_authors)} authors", file=sys.stderr)

        # Regenerate mailmap with the new data
        email_to_github = generate_mailmap(commits, pr_authors)
        if args.verbose:
            print(
                f"Generated {len(email_to_github)} email mappings (after fetch)",
                file=sys.stderr,
            )

        # Update missing PRs set
        missing_prs -= set(fetched_authors.keys())

    if args.missing_prs and missing_prs:
        with open(args.missing_prs, "w") as f:
            for pr_num in sorted(missing_prs):
                f.write(f"{pr_num}\n")
        if args.verbose:
            print(
                f"Wrote {len(missing_prs)} missing PR numbers to {args.missing_prs}",
                file=sys.stderr,
            )

    # Format and output
    output_content = format_output(email_to_github)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output_content + "\n")
        if args.verbose:
            print(f"Wrote mappings to {args.output}", file=sys.stderr)
    else:
        print(output_content)


if __name__ == "__main__":
    main()
