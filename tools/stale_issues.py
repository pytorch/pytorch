import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import date, timedelta


def parse_older_than(s):
    """Parse a relative time string like '2 months' into a cutoff date."""
    m = re.fullmatch(r"(\d+)\s*(days?|weeks?|months?|years?)", s.strip())
    if not m:
        raise argparse.ArgumentTypeError(
            f"invalid time format: {s!r} (expected e.g. '30 days', '2 months', '1 year')"
        )
    n, unit = int(m.group(1)), m.group(2).rstrip("s")
    today = date.today()
    if unit == "day":
        return today - timedelta(days=n)
    elif unit == "week":
        return today - timedelta(weeks=n)
    elif unit == "month":
        month = today.month - n
        year = today.year + (month - 1) // 12
        month = (month - 1) % 12 + 1
        day = min(
            today.day, [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1]
        )
        return date(year, month, day)
    elif unit == "year":
        return date(today.year - n, today.month, min(today.day, 28))


def gh_issue_list(search, label, limit):
    """Fetch issues from gh issue list for a single label (or no label)."""
    cmd = [
        "gh",
        "issue",
        "list",
        "-R",
        "pytorch/pytorch",
        "-S",
        search,
        "-L",
        str(limit),
        "--json",
        "number,title,updatedAt,labels,url",
    ]
    if label:
        cmd += ["-l", label]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        sys.exit(1)
    return json.loads(result.stdout)


def gh_issue_count(search_query):
    """Get total issue count via GitHub search API."""
    result = subprocess.run(
        [
            "gh",
            "api",
            "search/issues",
            "-q",
            ".total_count",
            "--method",
            "GET",
            "-f",
            f"q={search_query}",
            "-f",
            "per_page=1",
        ],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "?"


def cmd_list(args):
    search = "sort:updated-asc"
    if args.older_than:
        cutoff = parse_older_than(args.older_than)
        search += f" updated:<{cutoff.isoformat()}"

    labels = args.label or [None]
    if len(labels) == 1:
        issues = gh_issue_list(search, labels[0], args.limit)
    else:
        # Query per label and merge (gh -l does AND, we want OR)
        seen = set()
        issues = []
        for label in labels:
            for issue in gh_issue_list(search, label, args.limit):
                if issue["number"] not in seen:
                    seen.add(issue["number"])
                    issues.append(issue)
        issues.sort(key=lambda i: i["updatedAt"])
        issues = issues[: args.limit]

    for issue in issues:
        issue_labels = ", ".join(l["name"] for l in issue["labels"])
        print(
            f"#{issue['number']:>6}  {issue['updatedAt'][:10]}  {issue['title'][:80]}"
        )
        print(f"         {issue['url']}")
        if issue_labels:
            print(f"         labels: {issue_labels}")

    # Get total count via GitHub search API
    base_query = "repo:pytorch/pytorch is:issue is:open"
    if args.older_than:
        cutoff = parse_older_than(args.older_than)
        base_query += f" updated:<{cutoff.isoformat()}"
    if not args.label:
        total = gh_issue_count(base_query)
    elif len(args.label) == 1:
        total = gh_issue_count(base_query + f' label:"{args.label[0]}"')
    else:
        # Sum per-label counts (may slightly overcount shared issues)
        total = 0
        for label in args.label:
            count = gh_issue_count(base_query + f' label:"{label}"')
            try:
                total += int(count)
            except ValueError:
                total = "?"
                break
        if isinstance(total, int):
            total = f"~{total}"
    print(f"\nShowing {len(issues)} of {total} issues.")


def cmd_labels(args):
    page = 1
    labels = []
    while True:
        result = subprocess.run(
            [
                "gh",
                "api",
                "repos/pytorch/pytorch/labels",
                "--method",
                "GET",
                "-f",
                "per_page=100",
                "-f",
                f"page={page}",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            sys.exit(1)
        batch = json.loads(result.stdout)
        if not batch:
            break
        labels.extend(batch)
        page += 1

    labels.sort(key=lambda l: l["name"].lower())
    for label in labels:
        desc = f"  - {label['description']}" if label.get("description") else ""
        print(f"{label['name']}{desc}")
    print(f"\n{len(labels)} labels total.")


SUBSCRIPTION_DIR = os.path.join(
    os.environ.get("XDG_RUNTIME_DIR", os.path.join("/tmp", f"user-{os.getuid()}")),
    "gh_subscriptions",
)


def gh_graphql(query):
    result = subprocess.run(
        ["gh", "api", "graphql", "-f", f"query={query}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        sys.exit(1)
    return json.loads(result.stdout)


def subscription_state_path(issue_number):
    return os.path.join(SUBSCRIPTION_DIR, f"{issue_number}.json")


def cmd_subscription_save(args):
    result = subprocess.run(
        ["gh", "api", f"repos/pytorch/pytorch/issues/{args.issue}", "--jq", ".node_id"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        sys.exit(1)
    node_id = result.stdout.strip()

    data = gh_graphql(
        f'{{ node(id: "{node_id}") {{ ... on Issue {{ viewerSubscription }} }} }}'
    )
    state = data["data"]["node"]["viewerSubscription"]

    os.makedirs(SUBSCRIPTION_DIR, exist_ok=True)
    with open(subscription_state_path(args.issue), "w") as f:
        json.dump({"node_id": node_id, "state": state}, f)
    print(f"#{args.issue}: {state} (node_id={node_id})")


def cmd_subscription_restore(args):
    path = subscription_state_path(args.issue)
    if not os.path.exists(path):
        print(f"No saved state for #{args.issue}", file=sys.stderr)
        sys.exit(1)

    with open(path) as f:
        saved = json.load(f)
    os.remove(path)

    if saved["state"] == "SUBSCRIBED":
        print(f"#{args.issue}: was already SUBSCRIBED, nothing to restore")
        return

    node_id = saved["node_id"]

    # Poll until auto-subscribe has propagated, then unsubscribe.
    for _ in range(30):
        data = gh_graphql(
            f'{{ node(id: "{node_id}") {{ ... on Issue {{ viewerSubscription }} }} }}'
        )
        if data["data"]["node"]["viewerSubscription"] == "SUBSCRIBED":
            break
        time.sleep(1)

    gh_graphql(
        f'mutation {{ updateSubscription(input: {{subscribableId: "{node_id}", '
        f"state: UNSUBSCRIBED}}) {{ subscribable {{ ... on Issue {{ viewerSubscription }} }} }} }}"
    )
    print(f"#{args.issue}: restored to UNSUBSCRIBED")


def main():
    parser = argparse.ArgumentParser(
        description="Tools for managing stale issues in pytorch/pytorch"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser(
        "list", help="list least recently updated issues"
    )
    list_parser.add_argument(
        "-L", "--limit", type=int, default=5, help="max issues to fetch (default: 5)"
    )
    list_parser.add_argument(
        "-l", "--label", action="append", help="filter by label (multiple for OR)"
    )
    list_parser.add_argument(
        "--older-than",
        type=str,
        default="3 months",
        metavar="TIME",
        help="only show issues not updated in this long, or '' for no cutoff (default: '3 months')",
    )
    list_parser.set_defaults(func=cmd_list)

    labels_parser = subparsers.add_parser("labels", help="list all known labels")
    labels_parser.set_defaults(func=cmd_labels)

    sub_parser = subparsers.add_parser(
        "subscription", help="save/restore issue notification subscription state"
    )
    sub_sub = sub_parser.add_subparsers(dest="sub_command", required=True)
    save_parser = sub_sub.add_parser("save", help="save current subscription state")
    save_parser.add_argument("issue", type=int, help="issue number")
    save_parser.set_defaults(func=cmd_subscription_save)
    restore_parser = sub_sub.add_parser(
        "restore", help="restore saved subscription state"
    )
    restore_parser.add_argument("issue", type=int, help="issue number")
    restore_parser.set_defaults(func=cmd_subscription_restore)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
