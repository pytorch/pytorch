"""
Main script to policy-check PRs and report results in a comment. This is the
core of the bot's logic: it loads policy.yml, validates the pull request
(description, forbidden files, unit tests), waits for the
required CI checks, posts a single results-table comment, and manages the
"Not ready to Review" label.
"""

#!/usr/bin/env python3

import argparse
import fnmatch
import json
import os
import re
import sys
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
import yaml

NOT_READY_LABEL = "Not ready to Review"

# Authors can opt a PR OUT of the bot entirely by putting this tag anywhere in
# the PR description. When present, the bot does NOT run any checks — it simply
# removes the "Not ready to Review" label and posts a short skip notice.
SKIP_TAG = "@skip-pr-bot"

# Anchor file paths to THIS script's location rather than the current working
# directory or a ".git"/".github" walk-up (which breaks with nested repos /
# submodules). See the Python style guide:
# https://github.com/ROCm/TheRock/blob/main/docs/development/style_guides/python_style_guide.md#dont-make-assumptions-about-the-current-working-directory
THIS_SCRIPT_DIR = Path(__file__).resolve().parent


def _env_flag(name: str, default: bool = True) -> bool:
    """Read a boolean capability flag set by the workflow.

    On PRs from FORKS the GitHub App secrets are unavailable, so the fallback
    GITHUB_TOKEN is READ-ONLY and every write (comment / label) returns HTTP
    403. The workflow sets POST_COMMENTS / MUTATE_PR / READ_SECURITY_ALERTS to
    'false' in that case so the bot can DEGRADE GRACEFULLY: it still evaluates
    policy and sets the check's pass/fail exit code, but skips the writes it is
    not permitted to perform instead of crashing.
    """
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


# Whether this run may write to the PR (comments / labels).
CAN_POST_COMMENTS = _env_flag("POST_COMMENTS")
CAN_MUTATE_PR = _env_flag("MUTATE_PR")

# The "Not ready to Review" label is added ONLY when the JIRA/ISSUE ID
# reference is missing from the PR description (detected in main() via
# `jira_issue_failed`). The Unit Test and Forbidden Files checks are
# WARNING-ONLY: they never block the workflow and never add the label. All
# other failures (description length/checklist, pre-commit, …) do NOT add the
# label either.

# Fixed display order for rows in the results table (by check name). Any row
# whose name is not listed here is appended after these, in its original order.
TABLE_ORDER = [
    "PR Description",
    "Forbidden Files",
    "Unit Test",
    "pre-commit",
    "Draft PR",
    "Feature Flag",
    "Code Coverage",
    "therock-pr-bot",
]


@dataclass(frozen=True)
class FailureComment:
    title: str
    body: str


@dataclass
class CheckResult:
    name: str
    icon: str
    passed: bool
    details: List[str]
    pending: bool = False
    wip: bool = False
    tbe: bool = False
    warn: bool = False
    note: Optional[str] = None


@dataclass(frozen=True)
class Policy:
    description_min_length: int
    description_issue_patterns: List[re.Pattern[str]]
    description_checklist_patterns: List[re.Pattern[str]]
    block_draft: bool
    forbidden_paths: List[str]
    unit_test_code_extensions: List[str]
    unit_test_patterns: List[str]
    unit_test_exempt_paths: List[str]
    bump_bot_authors: List[str]
    required_checks: List[str]
    precommit_failure_comment: Optional[FailureComment]


def load_policy(policy_path: Path) -> Policy:
    """Parse `policy.yml` into a typed, immutable `Policy` object.

    Reads the `pr`, `diff`, and `checks` sections, compiles all regex patterns,
    and applies sensible defaults for any missing fields. Raises ValueError if
    the file is not a mapping.
    """
    raw = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("policy.yml must be a mapping/object")

    pr = raw.get("pr", {}) or {}
    diff = raw.get("diff", {}) or {}
    checks = raw.get("checks", {}) or {}

    # PR description rules.
    description_cfg = pr.get("description", {}) or {}
    description_min_length = int(description_cfg.get("min_length", 0) or 0)
    description_issue_raw = description_cfg.get("issue_reference_patterns", []) or []
    description_issue_patterns = [re.compile(str(p)) for p in description_issue_raw]
    description_checklist_raw = (
        description_cfg.get("required_checklist_patterns", []) or []
    )
    description_checklist_patterns = [
        re.compile(str(p)) for p in description_checklist_raw
    ]

    # Block drafts / WIP titles.
    block_draft = bool(pr.get("block_draft", False))

    forbidden_paths = [str(p) for p in (diff.get("forbidden_paths", []) or [])]

    # Unit test rules live under pr.unit_tests.
    unit_cfg = pr.get("unit_tests", {}) or {}
    unit_test_code_extensions = [
        str(e).lower() for e in (unit_cfg.get("code_extensions", []) or [])
    ]
    unit_test_patterns = [
        str(p) for p in (unit_cfg.get("test_file_patterns", []) or [])
    ]
    unit_test_exempt_paths = [str(p) for p in (unit_cfg.get("exempt_paths", []) or [])]

    # Bump-PR bot authors that bypass all policy checks.
    bump_bot_authors = [str(a) for a in (pr.get("bump_bot_authors", []) or [])]

    required_checks = [str(x) for x in (checks.get("required_check_runs", []) or [])]

    fc = ((checks.get("failure_comments", {}) or {}).get("pre-commit")) or None
    precommit_failure_comment = None
    if isinstance(fc, dict) and "title" in fc and "body" in fc:
        precommit_failure_comment = FailureComment(
            title=str(fc["title"]),
            body=str(fc["body"]),
        )

    return Policy(
        description_min_length=description_min_length,
        description_issue_patterns=description_issue_patterns,
        description_checklist_patterns=description_checklist_patterns,
        block_draft=block_draft,
        forbidden_paths=forbidden_paths,
        unit_test_code_extensions=unit_test_code_extensions,
        unit_test_patterns=unit_test_patterns,
        unit_test_exempt_paths=unit_test_exempt_paths,
        bump_bot_authors=bump_bot_authors,
        required_checks=required_checks,
        precommit_failure_comment=precommit_failure_comment,
    )


def gh_headers(token: str) -> Dict[str, str]:
    """Build the standard GitHub REST API request headers for the given token."""
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "therock-pr-bot",
    }


def gh_get(url: str, token: str) -> Any:
    """Perform an authenticated GET and return the parsed JSON.

    Raises RuntimeError on any non-2xx response.
    """
    r = requests.get(url, headers=gh_headers(token), timeout=30)
    if r.status_code >= 300:
        raise RuntimeError(f"GET {url} -> {r.status_code}: {r.text}")
    return r.json()


def gh_post(url: str, token: str, payload: Dict[str, Any]) -> Any:
    """Perform an authenticated POST with a JSON body and return parsed JSON.

    Raises RuntimeError on any non-2xx response.
    """
    r = requests.post(url, headers=gh_headers(token), json=payload, timeout=30)
    if r.status_code >= 300:
        raise RuntimeError(f"POST {url} -> {r.status_code}: {r.text}")
    return r.json()


def gh_patch(url: str, token: str, payload: Dict[str, Any]) -> Any:
    """Perform an authenticated PATCH with a JSON body and return parsed JSON.

    Raises RuntimeError on any non-2xx response.
    """
    r = requests.patch(url, headers=gh_headers(token), json=payload, timeout=30)
    if r.status_code >= 300:
        raise RuntimeError(f"PATCH {url} -> {r.status_code}: {r.text}")
    return r.json()


def get_pr(owner: str, repo: str, pr_number: int, token: str) -> Dict[str, Any]:
    """Fetch a single pull request's metadata (title, body, head/base, etc.)."""
    data = gh_get(
        f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}", token
    )
    if not isinstance(data, dict):
        raise RuntimeError("Unexpected PR payload")
    return data


def iter_pr_files(
    owner: str, repo: str, pr_number: int, token: str
) -> Iterable[Dict[str, Any]]:
    """Yield every file object changed in the PR, transparently paginating.

    Each yielded dict includes keys such as `filename`, `status`, `additions`,
    `deletions`, and `changes`.
    """
    page = 1
    while True:
        data = gh_get(
            f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/files?per_page=100&page={page}",
            token,
        )
        if not isinstance(data, list):
            raise RuntimeError("Unexpected PR files payload")
        if not data:
            return
        for item in data:
            if isinstance(item, dict):
                yield item
        page += 1


def get_check_runs(owner: str, repo: str, sha: str, token: str) -> List[Dict[str, Any]]:
    """Return the list of check-runs associated with a commit SHA."""
    data = gh_get(
        f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}/check-runs?per_page=100",
        token,
    )
    if not isinstance(data, dict):
        raise RuntimeError("Unexpected check-runs payload")
    runs = data.get("check_runs", [])
    return runs if isinstance(runs, list) else []


def ensure_pr_not_draft(policy: Policy, is_draft: bool, errors: List[str]) -> None:
    """Block draft PRs when `policy.block_draft` is enabled.

    Appends a message to `errors` if the PR is still a draft.
    """
    if policy.block_draft and is_draft:
        errors.append(
            "This PR is a draft. Please mark it as 'Ready for review' before "
            "it can pass policy checks."
        )


def _strip_markdown_comments(text: str) -> str:
    """Remove HTML comment blocks from Markdown text."""
    # Note: using re.DOTALL to match _any_ character, including newlines so this
    # can handle multiline comments.
    return re.sub(r"<!--.*?(?:-->|$)", "", text, flags=re.DOTALL)


def ensure_pr_description(policy: Policy, body: str, errors: List[str]) -> None:
    """Validate the PR description (minimum length, JIRA/ISSUE reference, checklist).

    Appends a structured message to `errors` if the body is too short, does
    not contain a recognised tracking reference, or has an unticked checklist item.
    """
    # Strip comments so we only check the visible text against the policies.
    # This lets pull request templates use examples that _would_ pass the check.
    body = _strip_markdown_comments(body or "").strip()

    if policy.description_min_length and len(body) < policy.description_min_length:
        errors.append(
            f"**Error:** PR description is too short ({len(body)} characters).\n"
            f"**Expected:** at least {policy.description_min_length} characters.\n"
            "**Current:** please provide a meaningful description of your changes"
        )

    if policy.description_issue_patterns and not any(
        p.search(body) for p in policy.description_issue_patterns
    ):
        errors.append(
            "**Error:** PR description must reference a JIRA ID, ISSUE ID, or a "
            "GitHub closing keyword.\n"
            "**Expected:** include a `JIRA ID` / `ISSUE ID` line (separator `:` "
            "or `-`, or omitted; value may be a JIRA key, a number with/without "
            "`#`, or a link), OR a closing keyword + issue reference. "
            "Accepted examples:\n"
            "• `JIRA ID : TESTAUTO-6039`\n"
            "• `JIRA ID - #330`\n"
            "• `JIRA ID #330`\n"
            "• `JIRA ID` (on separate line)\n"
            "  `ROCM-25757`\n"
            "• `ISSUE ID : TESTUTO-3334`\n"
            "• `ISSUE ID #3334`\n"
            "• `ISSUE ID - TESTAUTO-3433`\n"
            "• `ISSUE ID` (on separate line)\n"
            "  `AIRUNTIME-2352`\n"
            "• `ISSUE ID : https://github.com/<org_name>/<repo_name>/issues/1234`\n"
            "• `Closes #10`\n"
            "• `Fixes octo-org/octo-repo#100`\n"
            "• `Resolves: #123`\n"
            "• `#123`\n"
            "• `https://github.com/<org_name>/<repo_name>/issues/123`\n"
            "**Current:** no valid JIRA/ISSUE/closing-keyword reference found"
        )

    # Submission Checklist: every configured checklist item must be ticked.
    for pat in policy.description_checklist_patterns:
        if not pat.search(body):
            errors.append(
                "**Error:** Submission Checklist item is not completed.\n"
                "**Expected:** read the contributing guidelines and tick the box "
                "by changing `- [ ]` to `- [x]`:\n"
                "- [x] Look over the contributing guidelines …\n"
                "**Current:** the required checklist item is unchecked"
            )


def _matches_forbidden(filename: str, pattern: str) -> bool:
    """Return True if `filename` matches a forbidden glob `pattern`.

    Handles `**/<x>` patterns so they also match root-level files (e.g. `.env`).
    """
    # GitHub returns POSIX-style paths.
    if fnmatch.fnmatch(filename, pattern):
        return True
    # Allow '**/<x>' patterns to also match root-level files (e.g. '.env').
    if pattern.startswith("**/") and fnmatch.fnmatch(filename, pattern[3:]):
        return True
    return False


def _is_test_file(filename: str, patterns: Iterable[str]) -> bool:
    """Return True if `filename` is recognised as a test file.

    Patterns that contain a '/' are treated as FULL-PATH globs (e.g.
    '**/test/gtest/**' matches any file under a test/gtest/ directory).
    Patterns without a '/' are matched against the BASENAME only
    (e.g. 'test_*', '*_test.*').
    """
    base = Path(filename).name
    for pat in patterns:
        if "/" in pat:
            if _matches_forbidden(filename, pat):
                return True
        elif fnmatch.fnmatch(base, pat):
            return True
    return False


def ensure_no_forbidden_files(
    policy: Policy, pr_files: Iterable[Dict[str, Any]], errors: List[str]
) -> None:
    """Flag any added/modified file matching a forbidden path pattern.

    Removed files are ignored. Appends one message per offending file to
    `errors`.
    """
    if not policy.forbidden_paths:
        return
    for f in pr_files:
        filename = str(f.get("filename") or "")
        status = str(f.get("status") or "")
        if not filename or status == "removed":
            continue
        norm = Path(filename).as_posix()
        for pattern in policy.forbidden_paths:
            if _matches_forbidden(norm, pattern):
                errors.append(
                    f"Forbidden file present in PR: `{norm}` (matched `{pattern}`)."
                )
                break


def ensure_unit_tests(
    policy: Policy, pr_files: Iterable[Dict[str, Any]], errors: List[str]
) -> None:
    """
    Require a unit test when real source code changes.

    - Doc/config files (anything NOT in code_extensions, e.g. .md/.txt/.yml/.ini)
      never trigger the requirement — a doc/config-only PR passes automatically.
    - If any code file is changed, the PR must also add/modify at least one
      test file (basename matching test_file_patterns, e.g. test_xxx.py).
    """
    if not policy.unit_test_code_extensions:
        return

    code_files: List[str] = []
    has_test = False

    for f in pr_files:
        status = str(f.get("status") or "")
        if status == "removed":
            continue
        filename = Path(str(f.get("filename") or "")).as_posix()
        if not filename:
            continue

        # Files under exempt paths never require an accompanying unit test.
        if any(
            _matches_forbidden(filename, pat) for pat in policy.unit_test_exempt_paths
        ):
            continue

        base = Path(filename).name
        ext = Path(filename).suffix.lower()

        # A test file satisfies the requirement.
        if _is_test_file(filename, policy.unit_test_patterns):
            has_test = True
            continue

        # A real source/code file triggers the requirement.
        if ext in policy.unit_test_code_extensions:
            code_files.append(filename)

    if code_files and not has_test:
        listed = ", ".join(f"`{c}`" for c in code_files[:5])
        more = "" if len(code_files) <= 5 else f" (+{len(code_files) - 5} more)"
        errors.append(
            "**Error:** Source/code files changed without an accompanying unit test.\n"
            "**Expected:** add at least one test file named like "
            "`test_<name>.py` / `test_<name>.cpp` (or `<name>_test.*`).\n"
            f"**Current:** code file(s) changed: {listed}{more}; no test file found"
        )


def pr_has_code_files(policy: Policy, pr_files: Iterable[Dict[str, Any]]) -> bool:
    """True if the PR changes at least one real source/code file.

    Doc/config files (.md, .txt, .yml, .ini, ...), exempt paths, and test files
    do NOT count as code.
    """
    for f in pr_files:
        status = str(f.get("status") or "")
        if status == "removed":
            continue
        filename = Path(str(f.get("filename") or "")).as_posix()
        if not filename:
            continue
        if any(
            _matches_forbidden(filename, pat) for pat in policy.unit_test_exempt_paths
        ):
            continue
        base = Path(filename).name
        ext = Path(filename).suffix.lower()
        if _is_test_file(filename, policy.unit_test_patterns):
            continue
        if ext in policy.unit_test_code_extensions:
            return True
    return False


def summarize_required_checks(
    policy: Policy,
    check_runs: List[Dict[str, Any]],
) -> Tuple[List[str], List[str], Dict[str, str]]:
    """Summarise the state of the required check-runs.

    Returns:
      - missing: required checks not present
      - failing: required checks that concluded not-success
      - conc_by_name: name -> conclusion (string; 'null' if none)
    """
    by_name: Dict[str, Dict[str, Any]] = {}
    for r in check_runs:
        name = r.get("name")
        if isinstance(name, str):
            by_name[name] = r

    conc_by_name: Dict[str, str] = {}
    for name, r in by_name.items():
        conc = r.get("conclusion")
        conc_by_name[name] = str(conc) if conc is not None else "null"

    missing = [n for n in policy.required_checks if n not in by_name]

    ok = {"success", "neutral", "skipped"}
    failing: List[str] = []
    for n in policy.required_checks:
        r = by_name.get(n)
        if not r:
            continue
        conc = r.get("conclusion")
        if conc is None:
            continue  # still running
        if str(conc) not in ok:
            failing.append(f"{n}={conc}")

    return missing, failing, conc_by_name


def upsert_comment(
    owner: str, repo: str, pr_number: int, token: str, marker: str, body: str
) -> None:
    """Create or update a single marker-tagged bot comment on the PR.

    Looks for an existing comment whose body contains `marker`; if found it is
    PATCHed in place, otherwise a new comment is POSTed. This keeps the bot to
    one self-updating comment per marker instead of spamming new ones.
    """
    if not CAN_POST_COMMENTS:
        print("ℹ️  Skipping PR comment — no write access (e.g. fork PR).")
        return
    try:
        comments = gh_get(
            f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/comments?per_page=100",
            token,
        )
        if isinstance(comments, list):
            for c in comments:
                if isinstance(c, dict) and marker in str(c.get("body", "")):
                    gh_patch(c["url"], token, {"body": body})
                    return
        gh_post(
            f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/comments",
            token,
            {"body": body},
        )
    except RuntimeError as exc:
        print(f"⚠️  Could not post/update comment (continuing): {exc}", file=sys.stderr)


def update_comment_if_exists(
    owner: str, repo: str, pr_number: int, token: str, marker: str, body: str
) -> None:
    """Edit an EXISTING bot comment in place — never create or delete.

    Finds the comment whose body contains the given bot `marker` and PATCHes it
    to `body`. If no such comment exists, this does nothing.

    We deliberately do NOT delete comments: bot code must never risk removing a
    developer's comment. Only comments authored by THIS bot (identified by an
    HTML marker) are ever touched, and only via an in-place edit.
    """
    if not CAN_POST_COMMENTS:
        return
    try:
        comments = gh_get(
            f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/comments?per_page=100",
            token,
        )
    except RuntimeError as exc:
        print(f"⚠️  Could not list comments (continuing): {exc}", file=sys.stderr)
        return
    if isinstance(comments, list):
        for c in comments:
            if isinstance(c, dict) and marker in str(c.get("body", "")):
                try:
                    gh_patch(c["url"], token, {"body": body})
                except RuntimeError as exc:
                    print(
                        f"⚠️  Could not update comment (continuing): {exc}",
                        file=sys.stderr,
                    )
                return


def build_policy_table_comment(
    results: List[CheckResult],
    marker: str,
    ready: bool = False,
    note: Optional[str] = None,
) -> str:
    """Render the results as a single Markdown comment with a status table.

    Rows are sorted into `TABLE_ORDER`, each showing ✅/❌/⏳/🚧/🔜 status and
    full details. The footer summarises failures (or success) and a FAQ link is
    appended. `ready` switches the heading to "Ready for Review"; `note` adds an
    optional banner (used for the bump-PR special case).
    """
    def _format_details(details: List[str]) -> str:
        """Join a check's detail parts into one table-cell-safe string.

        Each part's non-empty lines are joined with <br>, parts are separated by
        a divider, and any literal '|' is escaped as '&#124;' so it does not end
        the Markdown table cell early (which would truncate the text).
        """
        blocks: List[str] = []
        for part in details:
            lines = [ln.strip() for ln in part.splitlines() if ln.strip()]
            if lines:
                blocks.append("<br>".join(lines))
        return "<br>───<br>".join(blocks).replace("|", "&#124;")

    # Render rows in a fixed, human-friendly order regardless of the order in
    # which they were appended (policy rows + required-check rows).
    order_index = {name: i for i, name in enumerate(TABLE_ORDER)}
    results = sorted(results, key=lambda r: order_index.get(r.name, len(TABLE_ORDER)))

    all_passed = all(r.passed for r in results)
    if all_passed and ready:
        heading = "### ✅ All Checks Passed — Ready for Review"
    elif all_passed:
        heading = "### ✅ All Policy Checks Passed"
    else:
        heading = "### ❌ PR Check — Action Required"
    rows = []
    for r in results:
        if r.warn:
            status = "⚠️ Warning"
        elif r.wip:
            status = "🚧 WIP"
        elif r.tbe:
            status = "🔜 To Be Enabled"
        elif r.pending:
            status = "⏳ Pending"
        elif r.passed:
            status = "✅ Pass"
        else:
            status = "❌ Fail"

        if r.warn and r.details:
            # Warning rows still show their details (what is wrong) even though
            # they do NOT fail the workflow.
            detail = _format_details(r.details)
        elif r.passed and r.note:
            detail = r.note
        elif r.passed or r.wip or r.tbe or not r.details:
            detail = "—"
        else:
            detail = _format_details(r.details)
        rows.append(f"| {r.icon} **{r.name}** | {status} | {detail} |")

    table = "| Check | Status | Details |\n" "|---|:---:|---|\n" + "\n".join(rows)
    # WIP, TBE and Warning rows are neither pass nor fail — exclude from counts.
    failing_count = sum(
        1
        for r in results
        if not r.passed and not r.pending and not r.wip and not r.tbe and not r.warn
    )
    if not all_passed:
        failing_names = [
            r.name
            for r in results
            if not r.passed and not r.pending and not r.wip and not r.tbe and not r.warn
        ]
        failing_list = "\n".join(f"> - ❌ {n}" for n in failing_names)
        footer = (
            f"\n\n> ⚠️ **{failing_count} policy check(s) failed.** "
            "Please address the issues above before this PR can be Reviewed.\n>\n"
            "> 🚫 **Please fix the failed policies**\n"
            f"{failing_list}\n>\n"
            f"> The **`{NOT_READY_LABEL}`** label was added to this PR. Once all "
            "policies pass, the label is removed automatically."
        )
    elif ready:
        footer = "\n\n> 🎉 All checks passed! This PR is ready for review."
    else:
        footer = "\n\n> 🎉 All policy checks passed!"

    faq_url = "https://github.com/ROCm/pytorch/blob/develop/docs/PYTORCH_PR_BOT_FAQ.md"

    faq_link = (
        "\n\n📖 **Need help?** See the "
        f"[Policy FAQ]({faq_url}) "
        "for details on every check and how to fix failures."
    )

    override_url = (
        "https://github.com/ROCm/pytorch/blob/develop/docs/"
        "PYTORCH_PR_BOT_FAQ.md#-wish-to-override-the-policy-process-and-get-unblocked"
    )
    override_link = f"\n\n🙋 **[Wish to Override Policy?]({override_url})**"

    note_block = f"\n\n{note}" if note else ""
    return (
        f"{marker}\n{heading}{note_block}\n\n{table}{footer}{faq_link}{override_link}"
    )


def build_check_results(
    policy: Policy,
    check_runs: List[Dict[str, Any]],
    include_self: bool = False,
) -> List[CheckResult]:
    """Turn required check-runs into table rows (so they appear in one table).

    Required CI workflows (e.g. pre-commit, and any security workflow such as
    CodeQL) are reported purely by their OWN check-run conclusion. This bot does
    NOT query the Code Scanning Alerts API — that is owned by codeql.yml /
    pre_commit_security.yml. To gate on such a workflow, add its check-run name
    to `checks.required_check_runs` in policy.yml.
    """
    ok = {"success", "neutral", "skipped"}
    by_name = {
        r.get("name"): r
        for r in check_runs
        if isinstance(r, dict) and isinstance(r.get("name"), str)
    }

    def status_of(name: str) -> Tuple[bool, bool, Optional[str]]:
        """Return (passed, pending, conclusion) for one check-run."""
        r = by_name.get(name)
        if not r:
            return False, True, None
        conc = r.get("conclusion")
        if conc is None:
            return False, True, None
        return str(conc) in ok, False, str(conc)

    results: List[CheckResult] = []
    for name in policy.required_checks:
        passed, pending, conc = status_of(name)
        if pending:
            details: List[str] = ["⏳ Still running…"]
        elif passed:
            details = []
        else:
            details = [f"**Error:** Check concluded with `{conc}`."]
        results.append(CheckResult(name, "🔎", passed, details, pending))

    if include_self:
        results.append(CheckResult("therock-pr-bot", "🤖", True, []))
    return results


def maybe_comment_precommit_failure(
    owner: str,
    repo: str,
    pr_number: int,
    token: str,
    policy: Policy,
    check_runs: List[Dict[str, Any]],
) -> None:
    """Post the configured pre-commit failure help comment, if applicable.

    Does nothing unless `policy.precommit_failure_comment` is set and the
    `pre-commit` check-run concluded in a failed state.
    """
    if not policy.precommit_failure_comment:
        return

    precommit_run = None
    for r in check_runs:
        if r.get("name") == "pre-commit":
            precommit_run = r
            break
    if not precommit_run:
        return

    conc = precommit_run.get("conclusion")
    if conc not in ("failure", "cancelled", "timed_out", "action_required"):
        return

    marker = "<!-- therock-pr-bot-precommit-failed -->"
    msg = (
        f"{marker}\n"
        f"### {policy.precommit_failure_comment.title}\n\n"
        f"{policy.precommit_failure_comment.body}"
    )
    upsert_comment(owner, repo, pr_number, token, marker, msg)


def add_label(owner: str, repo: str, pr_number: int, token: str, label: str) -> None:
    """Attach an EXISTING repository label to this PR.

    This intentionally does NOT create or manage repository label definitions.
    The `Not ready to Review` label must be created ONCE by a maintainer with
    triage access (repo → Issues → Labels → New label). If the label is missing,
    we log a clear setup hint instead of creating it automatically.
    """
    if not CAN_MUTATE_PR:
        print(f"ℹ️  Skipping add label '{label}' — no write access (fork PR).")
        return
    try:
        gh_post(
            f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/labels",
            token,
            {"labels": [label]},
        )
    except RuntimeError as exc:
        print(
            f"⚠️  Could not add label '{label}': {exc}\n"
            f"    If the label does not exist, a maintainer with triage access "
            f"must create it ONCE in {owner}/{repo} (Issues → Labels).",
            file=sys.stderr,
        )


def remove_label(owner: str, repo: str, pr_number: int, token: str, label: str) -> None:
    """Detach the label from THIS PR only.

    NOTE: This does NOT delete the repository's label definition — it only
    removes the label association from the current pull request.
    """
    if not CAN_MUTATE_PR:
        print(f"ℹ️  Skipping remove label '{label}' — no write access (fork PR).")
        return
    encoded = urllib.parse.quote(label, safe="")
    r = requests.delete(
        f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/labels/{encoded}",
        headers=gh_headers(token),
        timeout=30,
    )
    if r.status_code not in (200, 204, 404):
        print(
            f"⚠️  Could not remove label '{label}': {r.status_code}: {r.text}",
            file=sys.stderr,
        )


def is_bump_pr(policy: Policy, author_login: str) -> bool:
    """True if the PR author is one of the configured bump bots.

    Matches case-insensitively and ignores the GitHub App '[bot]' suffix
    (e.g. 'assistant-librarian[bot]' == 'assistant-librarian').
    """

    def norm(s: str) -> str:
        s = s.strip().lower()
        if s.endswith("[bot]"):
            s = s[: -len("[bot]")]
        return s

    target = norm(author_login)
    if not target:
        return False
    return target in {norm(a) for a in policy.bump_bot_authors}


def pr_wants_skip(body: str) -> bool:
    """True if the PR description opts out of the bot via the skip tag.

    Matches `@skip-pr-bot` as a whole word, case-insensitively, anywhere in the
    (comment-stripped) description.
    """
    text = _strip_markdown_comments(body or "")
    return (
        re.search(rf"(?<!\w){re.escape(SKIP_TAG)}(?!\w)", text, re.IGNORECASE)
        is not None
    )


def build_bump_pr_results(policy: Policy) -> List[CheckResult]:
    """All-pass table rows for an automated dependency bump PR."""
    bump_note = "Bump PR — check auto-approved (automated dependency update)"
    rows: List[CheckResult] = [
        CheckResult("PR Description", "📝", True, [], note=bump_note),
        CheckResult("Draft PR", "🚫", True, [], note=bump_note),
        CheckResult("Forbidden Files", "⛔", True, [], note=bump_note),
        CheckResult("Unit Test", "🧪", True, [], note=bump_note),
        CheckResult("Feature Flag", "🚩", True, [], note=bump_note),
        CheckResult("Code Coverage", "📊", True, [], note=bump_note),
    ]
    for name in policy.required_checks:
        rows.append(CheckResult(name, "🔎", True, [], note=bump_note))
    rows.append(CheckResult("therock-pr-bot", "🤖", True, [], note=bump_note))
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point: evaluate all policies and report results on the PR.

    Reads PR context from the environment, runs every policy check, posts the
    combined results table, manages the `Not ready to Review` label, and waits
    for required CI checks (e.g. pre-commit / CodeQL) to conclude. Returns 0
    when everything passes and 1 on any policy or required-check failure.
    """
    parser = argparse.ArgumentParser(
        description="TheRock PR Bot policy check (pre-review gate)"
    )
    parser.add_argument(
        "--policy",
        default=str(THIS_SCRIPT_DIR / "policy.yml"),
        help="Path to policy.yml (defaults to the file next to this script)",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=900,
        help="Max time to wait for required checks",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=15,
        help="Polling interval while waiting for checks",
    )
    args = parser.parse_args(argv)

    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    owner = os.environ.get("OWNER")
    repo = os.environ.get("REPO")
    pr_number_s = os.environ.get("PR_NUMBER")
    sha = os.environ.get("SHA")

    missing_env = [
        k
        for k, v in [
            ("GH_TOKEN/GITHUB_TOKEN", token),
            ("OWNER", owner),
            ("REPO", repo),
            ("PR_NUMBER", pr_number_s),
            ("SHA", sha),
        ]
        if not v
    ]
    if missing_env:
        raise RuntimeError(f"Missing required environment: {', '.join(missing_env)}")

    pr_number = int(pr_number_s)  # type: ignore[arg-type]

    # Resolve the policy path. A relative --policy is resolved FIRST against the
    # current working directory (the repo root, which is how the workflow passes
    # `tools/systems_pr_bot/policy.yml`); only if not found there do we fall back
    # to the directory next to THIS script. This prevents accidentally doubling
    # the path (e.g. `tools/systems_pr_bot/tools/systems_pr_bot/policy.yml`).
    policy_path = Path(args.policy)
    if not policy_path.is_absolute():
        cwd_candidate = (Path.cwd() / policy_path).resolve()
        if cwd_candidate.is_file():
            policy_path = cwd_candidate
        else:
            policy_path = (THIS_SCRIPT_DIR / policy_path).resolve()
    policy = load_policy(policy_path)

    pr = get_pr(owner=owner, repo=repo, pr_number=pr_number, token=token)  # type: ignore[arg-type]
    body = str(pr.get("body") or "")

    # --- PR description ---
    # If the author tagged the description with '@skip-pr-bot', do NOT run any
    # checks. This covers BOTH cases: the tag was present when the PR was
    # created, and the tag was added later via a description edit. In either
    # case we remove the "Not ready to Review" label and leave a short notice.
    if pr_wants_skip(body):
        skip_marker = "<!-- therock-pr-bot-skipped -->"
        skip_note = (
            f"{skip_marker}\n"
            f"✅ Author chose to skip pr bot run hence removing label "
            f"(`{SKIP_TAG}` found in the PR description)."
        )
        upsert_comment(owner, repo, pr_number, token, skip_marker, skip_note)  # type: ignore[arg-type]
        remove_label(owner, repo, pr_number, token, NOT_READY_LABEL)  # type: ignore[arg-type]
        print(f"✅ '{SKIP_TAG}' present — skipping all policy checks.")
        return 0

    # --- Special case: automated dependency "bump" PRs ---
    # If the author is a configured bump bot, bypass all policy checks.
    author = str((pr.get("user") or {}).get("login") or "")
    if is_bump_pr(policy, author):
        marker = "<!-- therock-pr-bot-policy-check -->"
        note = (
            f"🤖 **Bump PR detected** (author `@{author}`). All policy checks "
            "are auto-approved for automated dependency bumps."
        )
        upsert_comment(
            owner,
            repo,
            pr_number,
            token,  # type: ignore[arg-type]
            marker,
            note,
        )
        remove_label(owner, repo, pr_number, token, NOT_READY_LABEL)  # type: ignore[arg-type]
        update_comment_if_exists(
            owner,
            repo,
            pr_number,
            token,  # type: ignore[arg-type]
            "<!-- therock-pr-bot-fix-policies -->",
            "<!-- therock-pr-bot-fix-policies -->\n"
            "✅ Auto-approved — this is an automated dependency bump PR.",
        )
        print(f"🤖 Bump PR by @{author} — all checks auto-passed.")
        return 0

    pr_files = list(iter_pr_files(owner, repo, pr_number, token))  # type: ignore[arg-type]

    results: List[CheckResult] = []

    # Each check appends its failure messages to `check_errors`; an empty list
    # means the check passed. We reset it before every check.
    # NOTE: all policies are enforced for BOTH same-repo PRs and fork PRs.
    # `pull_request_target` gives us secret access (required for posting PR
    # comments) for forks, so there is no reason to skip the policy checks.
    check_errors: List[str] = []

    check_errors = []
    ensure_pr_description(policy, body, check_errors)
    results.append(CheckResult("PR Description", "📝", not check_errors, check_errors))

    # Only the JIRA/ISSUE ID reference rule of the description triggers the
    # "Not ready to Review" label — not the length or checklist rules.
    jira_issue_failed = any("must reference a JIRA ID" in e for e in check_errors)

    # Draft PR check is "Enabled soon" — logic kept in ensure_pr_not_draft but
    # not enforced yet (no check is performed).
    results.append(CheckResult("Draft PR", "🚫", passed=True, details=[], tbe=True))

    check_errors = []
    ensure_no_forbidden_files(policy, pr_files, check_errors)
    # Forbidden Files is WARNING-ONLY.
    #
    # NOTE on the two emojis (they are NOT a contradiction):
    #   • icon="⛔"  -> the row's IDENTITY emoji in the "Check" column
    #                   (always shown for the Forbidden Files row).
    #   • status ⚠️  -> the "Status" column value that build_policy_table_comment
    #                   renders because warn=True (see the `if r.warn:` branch).
    #
    # So the row always reads:  ⛔ Forbidden Files | ⚠️ Warning | <details>
    # passed=True guarantees it never turns the workflow red or adds a label;
    # warn=True just surfaces the offending file(s) as a warning.
    results.append(
        CheckResult(
            name="Forbidden Files",
            icon="⛔",
            passed=True,
            details=check_errors,
            warn=bool(check_errors),
        )
    )

    check_errors = []
    ensure_unit_tests(policy, pr_files, check_errors)
    ut_note = None
    ut_warn = bool(check_errors)
    if not check_errors and not pr_has_code_files(policy, pr_files):
        ut_note = "PR does not contain code files — Unit Test auto-passed"
    # Unit Test is WARNING-ONLY (same icon-vs-status distinction as above):
    #   • icon="🧪"  -> the row's identity emoji in the "Check" column.
    #   • status ⚠️  -> rendered in the "Status" column when warn=True.
    # passed=True means it never fails the workflow or adds a label; when a test
    # is missing we surface a ⚠️ Warning row (with details) instead.
    results.append(
        CheckResult(
            name="Unit Test",
            icon="🧪",
            passed=True,
            details=check_errors,
            warn=ut_warn,
            note=ut_note,
        )
    )

    # "Enabled soon" placeholders — logic to be implemented later.
    results.append(CheckResult("Feature Flag", "🚩", passed=True, details=[], tbe=True))
    results.append(
        CheckResult("Code Coverage", "📊", passed=True, details=[], tbe=True)
    )

    # Build the policy table; on failure we ALSO append the current
    # pre-commit / CodeQL rows so the table is always complete.
    # NOTE: warning-only rows (e.g. Unit Test) are excluded from the blocking
    # `errors` — they show a ⚠️ Warning but never fail the workflow.
    errors = [d for r in results for d in r.details if not r.warn]
    marker = "<!-- therock-pr-bot-policy-check -->"

    if errors:
        current_runs = get_check_runs(owner=owner, repo=repo, sha=sha, token=token)  # type: ignore[arg-type]
        combined = results + build_check_results(policy, current_runs)
        upsert_comment(
            owner,
            repo,
            pr_number,
            token,  # type: ignore[arg-type]
            marker,
            build_policy_table_comment(combined, marker),
        )

        # Add "Not ready to Review" ONLY when the JIRA/ISSUE ID reference is
        # missing from the description. All other failures (forbidden files,
        # Unit Test, pre-commit) do NOT add the label.
        if jira_issue_failed:
            add_label(owner, repo, pr_number, token, NOT_READY_LABEL)  # type: ignore[arg-type]
        else:
            remove_label(owner, repo, pr_number, token, NOT_READY_LABEL)  # type: ignore[arg-type]

        # Post/update a dedicated "fix policies" comment.
        failing_names = [r.name for r in results if not r.passed]
        fix_marker = "<!-- therock-pr-bot-fix-policies -->"
        fix_body = (
            f"{fix_marker}\n"
            "🚫 **Please fix the failed policies before requesting reviews.**\n\n"
            "The following policy checks failed:\n"
            + "\n".join(f"- ❌ {n}" for n in failing_names)
            + "\n\n"
            f"The **`{NOT_READY_LABEL}`** label has been added to this PR.\n"
            "Once all policies pass, the label will be removed automatically."
        )
        upsert_comment(owner, repo, pr_number, token, fix_marker, fix_body)  # type: ignore[arg-type]

        # --- Poll until pre-commit / CodeQL have a final conclusion so the
        # table updates from ⏳ Pending to a real ✅ Pass or ❌ Fail. ---
        ci_start = time.time()
        ok_set = {"success", "neutral", "skipped"}
        while True:
            poll_runs = get_check_runs(owner=owner, repo=repo, sha=sha, token=token)  # type: ignore[arg-type]
            by_name = {
                r.get("name"): r
                for r in poll_runs
                if isinstance(r, dict) and isinstance(r.get("name"), str)
            }
            # Check whether every required CI check has a conclusion yet.
            all_concluded = all(
                by_name.get(n) is not None and by_name[n].get("conclusion") is not None
                for n in policy.required_checks
            )
            if all_concluded:
                final_combined = results + build_check_results(policy, poll_runs)
                upsert_comment(
                    owner,
                    repo,
                    pr_number,
                    token,  # type: ignore[arg-type]
                    marker,
                    build_policy_table_comment(final_combined, marker),
                )
                break

            if time.time() - ci_start > args.timeout_seconds:
                print("⚠️  Timed out waiting for CI checks to conclude.")
                break

            time.sleep(args.poll_seconds)

        print("❌ Policy errors:\n")
        for e in errors:
            print(f"- {e}")
        return 1

    # No policy errors — show policy rows now; the required check rows
    # (pre-commit / CodeQL) are appended during the polling loop below.
    upsert_comment(
        owner,
        repo,
        pr_number,
        token,  # type: ignore[arg-type]
        marker,
        build_policy_table_comment(results, marker),
    )

    start = time.time()
    last: Dict[str, str] = {}

    while True:
        runs = get_check_runs(owner=owner, repo=repo, sha=sha, token=token)  # type: ignore[arg-type]
        missing, failing, conc_by_name = summarize_required_checks(policy, runs)
        last = conc_by_name

        if failing:
            final_results = results + build_check_results(policy, runs)
            upsert_comment(
                owner,
                repo,
                pr_number,
                token,
                marker,
                build_policy_table_comment(final_results, marker),
            )
            maybe_comment_precommit_failure(owner, repo, pr_number, token, policy, runs)  # type: ignore[arg-type]
            print("❌ Required checks failing:")
            for f in failing:
                print(f"- {f}")
            return 1

        # If any required checks are missing or still running, keep waiting.
        all_present = not missing
        all_ok = True
        ok = {"success", "neutral", "skipped"}
        by_name = {
            r.get("name"): r
            for r in runs
            if isinstance(r, dict) and isinstance(r.get("name"), str)
        }
        for name in policy.required_checks:
            r = by_name.get(name)
            if not r:
                all_ok = False
                continue
            conc = r.get("conclusion")
            if conc is None:
                all_ok = False
            elif str(conc) not in ok:
                all_ok = False

        if all_present and all_ok:
            final_results = results + build_check_results(
                policy,
                runs,
                include_self=True,
            )
            upsert_comment(
                owner,
                repo,
                pr_number,
                token,
                marker,
                build_policy_table_comment(final_results, marker, ready=True),
            )

            # Update the "fix policies" comment to reflect success.
            fix_marker = "<!-- therock-pr-bot-fix-policies -->"
            upsert_comment(
                owner,
                repo,
                pr_number,
                token,  # type: ignore[arg-type]
                fix_marker,
                f"{fix_marker}\n🎉 All checks passed! This PR is ready for review.",
            )

            # Mark any stale "blocked reviewer/assignee" gate comment as
            # resolved (edit in place — we never delete comments).
            update_comment_if_exists(
                owner,
                repo,
                pr_number,
                token,  # type: ignore[arg-type]
                "<!-- therock-pr-bot-review-gate -->",
                "<!-- therock-pr-bot-review-gate -->\n"
                "✅ All policy checks passed — this PR is ready for review.",
            )

            # All clean — remove the "Not ready to Review" label.
            remove_label(owner, repo, pr_number, token, NOT_READY_LABEL)  # type: ignore[arg-type]
            print("✅ All required checks passed.")
            return 0

        if time.time() - start > args.timeout_seconds:
            print("❌ Timed out waiting for required checks to complete.")
            print(json.dumps(last, indent=2))
            return 1

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    "Main block"
    raise SystemExit(main())
