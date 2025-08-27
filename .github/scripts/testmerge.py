import argparse
import re
import sys
from typing import Any, Optional
from urllib.parse import urlparse

from github_utils import (
    gh_close_pr,
    gh_fetch_json_list,
    gh_fetch_merge_base,
    gh_fetch_url,
    gh_graphql,
    gh_post_commit_comment,
    gh_post_pr_comment,
    gh_update_pr_state,
    GitHubComment,
)


# NOTE: These functions are duplicated from trymerge.py to avoid importing the entire module
# which has heavy dependencies (like yaml). Keep these in sync with the version in trymerge.py.

def sha_from_committed_event(ev: dict[str, Any]) -> Optional[str]:
    """Extract SHA from committed event in timeline"""
    return ev.get("sha")


def sha_from_force_push_after(ev: dict[str, Any]) -> Optional[str]:
    """Extract SHA from force push event in timeline"""
    after = ev.get("after") or ev.get("after_commit") or {}
    if isinstance(after, dict):
        return after.get("sha") or after.get("oid")
    return ev.get("after_sha") or ev.get("head_sha")


def iter_issue_timeline_until_comment(
    org: str, repo: str, issue_number: int, target_comment_id: int, max_pages: int = 1000
) -> Any:
    """
    Yield timeline entries in order until (and excluding) the entry whose id == target_comment_id
    *for a 'commented' event*. Stops as soon as the target comment is encountered.
    """
    page = 1

    while page <= max_pages:
        url = (
            f"https://api.github.com/repos/{org}/{repo}/issues/{issue_number}/timeline"
        )
        headers = {
            "Accept": "application/vnd.github+json, application/vnd.github.mockingbird-preview+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        params = {"per_page": 100, "page": page}

        # Build URL with parameters
        if params:
            param_str = "&".join(f"{k}={v}" for k, v in params.items())
            url = f"{url}?{param_str}"

        batch = gh_fetch_url(url, headers=headers)

        if not batch:
            return
        for ev in batch:
            # The target is the *issue comment* row with event == "commented" and id == issue_comment_id
            if ev.get("event") == "commented" and ev.get("id") == target_comment_id:
                return  # stop BEFORE yielding this comment
            yield ev
        if len(batch) < 100:
            return
        page += 1
        print("fetching next page...")

    # If we got here without finding the comment, then we either hit a bug or some github PR has a _really_
    # long timeline.
    # The max # of pages on any PR on pytorch/pytorch that found at the time of this change was 41 pages.
    raise RuntimeError(
        f"Could not find a merge commit in the first {max_pages} pages of the timeline at url {url}."
        f"This is most likely a bug, please report it to the @pytorch/pytorch-dev-infra team."
    )


def reconstruct_head_before_comment(
    org: str, repo: str, pr_number: int, issue_comment_id: int
) -> Optional[str]:
    """
    Reconstruct the PR head commit SHA that was present when a specific comment was posted.
    Returns None if no head-changing events found before the comment.
    """
    head = None
    found_any_event = False

    try:
        for event in iter_issue_timeline_until_comment(
            org, repo, pr_number, issue_comment_id
        ):
            etype = event.get("event")
            if etype == "committed":
                sha = sha_from_committed_event(event)
                if sha:
                    head = sha
                    found_any_event = True
                    print(f"Timeline: Found committed event with SHA {sha}")
            elif etype == "head_ref_force_pushed":
                sha = sha_from_force_push_after(event)
                if sha:
                    head = sha
                    found_any_event = True
                    print(f"Timeline: Found force push event with SHA {sha}")
            # Handle other head-changing events if needed
    except Exception as e:
        print(
            f"Warning: Failed to reconstruct timeline for comment {issue_comment_id}: {e}"
        )
        return None

    return head if found_any_event else None


def parse_github_pr_url(url: str) -> tuple[str, str, int]:
    """
    Parse a GitHub PR URL and extract org, repo, and PR number.

    Accepts URLs like:
    - https://github.com/pytorch/pytorch/pull/123
    - https://github.com/pytorch/pytorch/pull/123#issuecomment-456789

    Returns:
        tuple[str, str, int]: (org, repo, pr_number)
    """
    parsed = urlparse(url)
    if parsed.netloc != "github.com":
        raise ValueError(f"Invalid GitHub URL: {url}")

    # Match pattern: /org/repo/pull/number
    match = re.match(r'^/([^/]+)/([^/]+)/pull/(\d+)', parsed.path)
    if not match:
        raise ValueError(f"Invalid GitHub PR URL format: {url}")

    org, repo, pr_number = match.groups()
    return org, repo, int(pr_number)


def parse_comment_id_from_url(url: str) -> Optional[int]:
    """
    Extract comment ID from GitHub PR URL fragment if present.

    Example: https://github.com/pytorch/pytorch/pull/123#issuecomment-456789
    Returns: 456789
    """
    parsed = urlparse(url)
    if parsed.fragment and parsed.fragment.startswith('issuecomment-'):
        try:
            return int(parsed.fragment.split('-')[1])
        except (IndexError, ValueError):
            pass
    return None


def main():
    """
    Main command to invoke reconstruct_head_before_comment using a GitHub PR URL.
    """
    parser = argparse.ArgumentParser(
        description="Reconstruct the PR head commit SHA that was present when a specific comment was posted",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python testmerge.py https://github.com/pytorch/pytorch/pull/123 --comment-id 456789
  python testmerge.py https://github.com/pytorch/pytorch/pull/123#issuecomment-456789
        """
    )

    parser.add_argument(
        "pr_url",
        help="GitHub PR URL (e.g., https://github.com/pytorch/pytorch/pull/123)"
    )
    parser.add_argument(
        "--comment-id",
        type=int,
        help="Target comment ID to stop at (can also be specified in URL fragment)"
    )
    parser.add_argument(
        "--include-timeline",
        action="store_true",
        help="Also show timeline entries leading up to the comment"
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "summary"],
        default="summary",
        help="Output format (default: summary)"
    )

    args = parser.parse_args()

    try:
        # Parse the GitHub PR URL
        org, repo, pr_number = parse_github_pr_url(args.pr_url)

        # Get comment ID from URL fragment or command line argument
        comment_id = args.comment_id or parse_comment_id_from_url(args.pr_url)

        if comment_id is None:
            print("Error: Comment ID must be provided either via --comment-id or in the URL fragment", file=sys.stderr)
            print("Example: https://github.com/pytorch/pytorch/pull/123#issuecomment-456789", file=sys.stderr)
            sys.exit(1)

        print(f"Reconstructing head commit for {org}/{repo} PR #{pr_number} before comment {comment_id}")

        # Call the function to get the head SHA
        head_sha = reconstruct_head_before_comment(
            org=org,
            repo=repo,
            pr_number=pr_number,
            issue_comment_id=comment_id
        )

        if head_sha:
            print(f"\nHead commit SHA before comment {comment_id}: {head_sha}")

            if args.include_timeline:
                print(f"\nTimeline entries leading up to comment {comment_id}:")
                timeline_entries = list(iter_issue_timeline_until_comment(
                    org=org,
                    repo=repo,
                    issue_number=pr_number,
                    target_comment_id=comment_id
                ))

                if args.output_format == "json":
                    import json
                    result = {
                        "head_sha": head_sha,
                        "timeline": timeline_entries
                    }
                    print(json.dumps(result, indent=2))
                else:
                    for i, entry in enumerate(timeline_entries, 1):
                        event_type = entry.get('event', 'unknown')
                        created_at = entry.get('created_at', 'unknown')
                        actor = entry.get('actor', {}).get('login', 'unknown') if entry.get('actor') else 'unknown'
                        print(f"  {i}. [{event_type}] by {actor} at {created_at}")

                        # Show additional details for certain event types
                        if event_type == 'commented' and 'body' in entry:
                            body_preview = entry['body'][:100].replace('\n', ' ')
                            if len(entry['body']) > 100:
                                body_preview += '...'
                            print(f"     Comment: {body_preview}")
                        elif event_type == 'committed' and 'sha' in entry:
                            print(f"     SHA: {entry['sha']}")
                        elif event_type == 'head_ref_force_pushed':
                            sha = sha_from_force_push_after(entry)
                            if sha:
                                print(f"     Force push to SHA: {sha}")
                        elif event_type in ['labeled', 'unlabeled'] and 'label' in entry:
                            label_name = entry['label'].get('name', 'unknown')
                            print(f"     Label: {label_name}")
        else:
            print(f"\nNo head-changing events found before comment {comment_id}")

        if args.output_format == "json" and not args.include_timeline:
            import json
            result = {"head_sha": head_sha}
            print(json.dumps(result, indent=2))

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()