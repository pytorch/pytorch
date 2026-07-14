#!/usr/bin/env python3

"""RADAR: AI risk-based auto-approval for low-risk maintainer PRs.

RADAR runs an AI risk assessment on a PR. When the assessed risk is low enough,
every changed file is inside a conservative allowlist, and the PR author has
write access, it posts an approval comment carrying a machine-readable marker
bound to the head commit SHA and signed with an HMAC, and adds the
``radar-approved`` label. ``trymerge`` treats a valid marker as satisfying the
human-approval requirement, so a maintainer can land the PR with
``@pytorchbot merge`` without a separate human review.

The trust anchor is the HMAC signature, not the comment author: the marker is
signed over ``(repo, pr, sha, score, tier)`` with a secret shared only by the
RADAR workflow and ``trymerge`` (``RADAR_HMAC_KEY``). A regular user -- even one
who can induce another workflow to reflect attacker-controlled text into a bot
comment -- cannot produce a valid signature, so a RADAR approval cannot be
forged. The marker is bound to the head SHA, so pushing any new commit
invalidates the approval and re-runs RADAR. When no key is configured, signing
and verification both fail closed.

This module is imported by ``trymerge`` for the parsing/detection helpers and
run as a CLI by the RADAR workflow to post approvals.
"""

from __future__ import annotations

import fnmatch
import hashlib
import hmac
import json
import os
import re
from argparse import ArgumentParser
from typing import NamedTuple

from github_utils import gh_delete_comment, gh_fetch_json_list, gh_post_pr_comment
from label_utils import gh_add_labels, gh_remove_label


RADAR_APPROVED_LABEL = "radar-approved"

# Identity used in the commit-message "Approved by:" trailer whenever a RADAR
# approval authorized the merge. Deliberately a distinct, greppable pseudo-login
# (NOT the unrelated real pytorch-bot account) so RADAR landings are auditable via
# `git log --grep "Approved by:.*pytorch-auto-radar"`. It is attributed whenever a
# valid RADAR approval was applied at merge time (alongside any human reviewers),
# which is a conservative superset of RADAR-only landings: it never misses a
# human-review-bypassing merge, and at worst also tags the rare case where a
# qualifying human approval was independently present. It is not a metamates
# member, so update_metamates_rule.py silently drops it and it can never leak into
# merge_rules.yaml.
RADAR_APPROVAL_LOGIN = "pytorch-auto-radar"

# Login(s) whose comments RADAR trusts. RADAR posts under the default
# GITHUB_TOKEN identity (github-actions[bot]), so keep this set as tight as the
# posting identity. This login check is only a cheap prefilter: the HMAC
# signature below is the real trust anchor, since another workflow could reflect
# attacker text into a comment under the same shared login.
RADAR_BOT_LOGINS = {"github-actions"}

# GitHub repo-permission levels that count as write access for RADAR purposes.
WRITE_ACCESS_PERMISSIONS = {"write", "admin", "maintain"}

# SIGNER/VERIFIER CONTRACT: the score/tier thresholds and the file allow/deny
# policy below are re-checked by trymerge, which imports is_low_risk /
# files_allowlisted from THIS module, so both sides evaluate byte-identical
# policy. The score ceiling and approved tiers are deliberately NOT
# env-overridable: a per-workflow override could let RADAR sign an approval that
# trymerge then rejects (or vice versa). Change policy only by editing these
# constants; both sides pick it up on next import. The only env-configured values
# RADAR shares with trymerge are the secret RADAR_HMAC_KEY and the tighten-only
# RADAR_DENYLIST_GLOBS (see below), both of which fail closed under divergence.
RADAR_APPROVED_TIERS = {"trivial"}
RADAR_MAX_SCORE = 10

# Signer-side only: additions+deletions ceiling. A change this large cannot be
# "trivial", and the count comes from `gh pr view --json additions,deletions`,
# which is never truncated (unlike the byte-capped diff shown to the AI). This is
# deliberately NOT re-checked in trymerge: adding fields to GH_GET_PR_INFO_QUERY
# would rekey every recorded gql_mocks fixture. It needs no verifier mirror
# because an oversized diff never gets a signed marker in the first place, and the
# marker is SHA-bound so any later change to the PR invalidates it.
RADAR_MAX_DIFF_LINES = 2000

# The same secret must be available to the RADAR workflow (to sign) and to
# trymerge (to verify). Absent key => fail closed on both sides.
RADAR_HMAC_KEY_ENV = "RADAR_HMAC_KEY"

# RADAR only auto-approves when *every* changed file matches one of these globs.
# This is a deterministic guardrail on top of the AI score that cannot be
# steered by prompt injection in the diff, and it keeps domain-specific reviewer
# requirements (FFT, Sparse, MPS, dispatch, ...) in force for any real source
# change. Keep this to genuinely low-blast-radius paths only: e.g. "*.pyi.in" is
# intentionally excluded because it would admit torch/_C/__init__.pyi.in, the
# core C-extension type surface gated by domain reviewers. fnmatch '*' spans '/',
# so "test/*" matches at any depth under test/. This list is a hardcoded constant
# (not env-overridable) so the RADAR signer and the trymerge verifier evaluate the
# same allowlist.
DEFAULT_ALLOWLIST_GLOBS = (
    "test/*",
    "docs/*",
    "benchmarks/*",
    "*.md",
)

# Files that sit under an allowlisted prefix but EXECUTE in CI, so their blast
# radius is far larger than a single test or doc page: pytest collection hooks,
# the test runner, global pytest config, the Sphinx docs-build config (conf.py
# is imported and run on every docs build), and Makefiles (they drive the build).
# RADAR must never auto-approve these even under test/ or docs/. Patterns are
# basename-anchored (e.g. conftest.py + */conftest.py) so nested files at any
# depth are caught without over-matching legitimate files like test/my_conftest.py.
DEFAULT_DENYLIST_GLOBS = (
    "conftest.py",
    "*/conftest.py",
    "run_test.py",
    "*/run_test.py",
    "pytest.ini",
    "*/pytest.ini",
    "conf.py",
    "*/conf.py",
    "Makefile",
    "*/Makefile",
)

# The marker is VISIBLE text, not an HTML comment: trymerge detects it from the
# GitHub GraphQL bodyText, which is rendered to plain text and strips HTML
# comments -- an invisible marker would never be seen. It carries the approved
# head SHA, the score/tier, and an HMAC over all of them so none can be tampered
# with after signing. Revealing the marker is safe: without RADAR_HMAC_KEY the
# signature cannot be forged for any other PR/SHA.
RADAR_MARKER_RE = re.compile(
    r"RADAR-APPROVED\s+sha=(?P<sha>[0-9a-f]{40})\s+"
    r"score=(?P<score>\d+)\s+tier=(?P<tier>[a-z]+)\s+"
    r"sig=(?P<sig>[0-9a-f]{64})"
)

# Signed revocation of a prior approval, bound to a head SHA. When RADAR re-runs
# on an UNCHANGED SHA and the new verdict no longer qualifies, deleting the old
# approval comment is best-effort and can fail. Posting a signed revocation makes
# withdrawal a POSITIVE, fail-closed signal instead: trymerge honors the newest
# RADAR decision for the current head, so a revocation invalidates an approval it
# could not delete. Like the approval marker, the HMAC (not the text) is the trust
# anchor, so reflected/forged revocation text cannot deny or grant anything.
RADAR_REVOKED_MARKER_RE = re.compile(
    r"RADAR-REVOKED\s+sha=(?P<sha>[0-9a-f]{40})\s+sig=(?P<sig>[0-9a-f]{64})"
)


class RadarMarker(NamedTuple):
    sha: str
    score: int
    tier: str
    sig: str


class RadarRevocation(NamedTuple):
    sha: str
    sig: str


def get_hmac_key() -> bytes | None:
    key = os.environ.get(RADAR_HMAC_KEY_ENV, "")
    return key.encode() if key else None


def compute_marker_sig(
    key: bytes,
    org: str,
    project: str,
    pr_num: int,
    sha: str,
    score: int,
    tier: str,
) -> str:
    # Canonical payload. Every field trymerge trusts must be covered here so it
    # cannot be altered after signing.
    payload = f"{org}/{project}#{pr_num}@{sha}:score={score}:tier={tier}".encode()
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def compute_revoked_sig(
    key: bytes, org: str, project: str, pr_num: int, sha: str
) -> str:
    # Distinct payload namespace (":REVOKED") so an approval signature can never
    # be replayed as a revocation or vice versa.
    payload = f"{org}/{project}#{pr_num}@{sha}:REVOKED".encode()
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def render_marker(marker: RadarMarker) -> str:
    # Plain visible text (no HTML comment) so it survives GitHub's bodyText
    # rendering; see RADAR_MARKER_RE.
    return (
        f"RADAR-APPROVED sha={marker.sha} score={marker.score} "
        f"tier={marker.tier} sig={marker.sig}"
    )


def render_revoked_marker(rev: RadarRevocation) -> str:
    return f"RADAR-REVOKED sha={rev.sha} sig={rev.sig}"


def _marker_from_match(m: re.Match[str]) -> RadarMarker:
    return RadarMarker(
        sha=m.group("sha"),
        score=int(m.group("score")),
        tier=m.group("tier"),
        sig=m.group("sig"),
    )


def parse_radar_marker(body: str) -> RadarMarker | None:
    m = RADAR_MARKER_RE.search(body)
    return _marker_from_match(m) if m is not None else None


def iter_radar_markers(body: str) -> list[RadarMarker]:
    # Return every marker in the body, not just the first: an approval comment
    # also contains AI-authored summary/reasoning text, and a marker-shaped
    # string there must not shadow the genuine (signed) marker at merge time.
    return [_marker_from_match(m) for m in RADAR_MARKER_RE.finditer(body)]


def iter_revoked_markers(body: str) -> list[RadarRevocation]:
    return [
        RadarRevocation(sha=m.group("sha"), sig=m.group("sig"))
        for m in RADAR_REVOKED_MARKER_RE.finditer(body)
    ]


def verify_marker_sig(marker: RadarMarker, org: str, project: str, pr_num: int) -> bool:
    key = get_hmac_key()
    if key is None:
        # No configured secret: an unsigned deployment must never honor a marker.
        return False
    expected = compute_marker_sig(
        key, org, project, pr_num, marker.sha, marker.score, marker.tier
    )
    return hmac.compare_digest(expected, marker.sig)


def verify_revoked_sig(
    rev: RadarRevocation, org: str, project: str, pr_num: int
) -> bool:
    key = get_hmac_key()
    if key is None:
        return False
    expected = compute_revoked_sig(key, org, project, pr_num, rev.sha)
    return hmac.compare_digest(expected, rev.sig)


def is_trusted_radar_login(login: str) -> bool:
    # GraphQL returns app logins with or without the "[bot]" suffix depending on
    # the surface, so normalize before comparing against the trusted set.
    return login.removesuffix("[bot]").lower() in RADAR_BOT_LOGINS


def is_low_risk(tier: str, score: int) -> bool:
    return tier.lower() in RADAR_APPROVED_TIERS and score <= RADAR_MAX_SCORE


def has_write_access(permission: str) -> bool:
    return permission.lower() in WRITE_ACCESS_PERMISSIONS


def _denylist_globs() -> tuple[str, ...]:
    # TIGHTEN-ONLY: RADAR_DENYLIST_GLOBS can only ADD to the hardcoded floor,
    # never remove from it. This keeps the policy fail-safe even if the RADAR
    # signer and the trymerge verifier are configured with different denylist
    # envs: a denylist set on only one side always makes that side stricter,
    # never more permissive.
    raw = os.environ.get("RADAR_DENYLIST_GLOBS", "").strip()
    extra = tuple(g.strip() for g in raw.split(",") if g.strip()) if raw else ()
    return DEFAULT_DENYLIST_GLOBS + extra


def files_allowlisted(files: list[str]) -> bool:
    # Empty file list fails closed: with nothing to reason about there is no
    # approval. Every file must match an allowlist glob AND no file may match a
    # denylist glob (denylist wins). The allowlist is a hardcoded constant (not
    # env-overridable) so the signer and verifier stay provably in sync; only the
    # tighten-only denylist is env-adjustable. fnmatch '*' spans '/'.
    if not files:
        return False
    deny = _denylist_globs()
    for f in files:
        if any(fnmatch.fnmatch(f, g) for g in deny):
            return False
        if not any(fnmatch.fnmatch(f, g) for g in DEFAULT_ALLOWLIST_GLOBS):
            return False
    return True


def build_approval_comment(
    marker: RadarMarker, summary: str, reasoning: str, run_url: str | None
) -> str:
    lines = [
        "## :radioactive: RADAR approves this PR",
        "",
        (
            f"RADAR assessed this change as **{marker.tier}** risk "
            f"(score {marker.score}/100, auto-approve threshold {RADAR_MAX_SCORE}), "
            "every changed file is in the RADAR allowlist, and the author has "
            "write access, so a human approval is not required to land it."
        ),
        "",
        f"> {summary}",
        "",
        "<details><summary>RADAR risk reasoning</summary>",
        "",
        reasoning,
        "",
        "</details>",
        "",
        (
            "A maintainer can now land this with `@pytorchbot merge`. Mandatory CI "
            "checks are still enforced, and pushing any new commit re-runs RADAR "
            "and invalidates this approval."
        ),
        "",
        # Machine-readable marker (see RADAR_MARKER_RE). Emitted as a plain
        # visible line -- not an HTML comment and not wrapped in markdown -- so it
        # survives GitHub's bodyText rendering verbatim. Visible on purpose; the
        # HMAC signature is what makes it unforgeable, not its invisibility.
        render_marker(marker),
    ]
    if run_url:
        lines += ["", f"<sub>Generated by [RADAR]({run_url}).</sub>"]
    return "\n".join(lines)


def _list_pr_comments(org: str, project: str, pr_num: int) -> list[dict]:
    # Paginate the full comment list (busy PyTorch PRs routinely exceed one page).
    # Best-effort; a listing failure never blocks the current decision.
    url = f"https://api.github.com/repos/{org}/{project}/issues/{pr_num}/comments"
    comments: list[dict] = []
    for page in range(1, 101):  # cap at 10k comments, mirroring get_comments
        try:
            batch = gh_fetch_json_list(url, params={"per_page": 100, "page": page})
        except Exception as e:
            print(f"Could not list comments page {page} (ok): {e}")
            break
        comments.extend(batch)
        if len(batch) < 100:
            break
    return comments


def _radar_comment_state(
    comments: list[dict], org: str, project: str, pr_num: int, head_sha: str
) -> tuple[bool, bool]:
    # Scan (no mutation) for whether a prior RADAR APPROVAL exists and whether a
    # valid signed revocation for the CURRENT head already stands. The caller uses
    # these to decide whether a fresh revocation must be posted before pruning, so
    # a standing withdrawal is guaranteed to exist at every instant (no window
    # where a stale approval is the only signal).
    had_approval = False
    has_head_revocation = False
    for c in comments:
        if not is_trusted_radar_login((c.get("user") or {}).get("login", "")):
            continue
        body = c.get("body", "") or ""
        if parse_radar_marker(body) is not None:
            had_approval = True
        if any(
            rev.sha == head_sha and verify_revoked_sig(rev, org, project, pr_num)
            for rev in iter_revoked_markers(body)
        ):
            has_head_revocation = True
    return had_approval, has_head_revocation


def _delete_radar_comments(
    comments: list[dict],
    org: str,
    project: str,
    pr_num: int,
    head_sha: str,
    keep_head_revocation: bool,
    dry_run: bool,
) -> None:
    # Remove superseded RADAR banners so the PR does not accumulate one per push.
    # When keep_head_revocation is set, a standing signed revocation for the
    # current head is preserved (it is the fail-closed withdrawal); everything
    # else -- approval banners and revocations for other SHAs -- is pruned.
    # Best-effort: delete failures never block the current decision.
    for c in comments:
        if not is_trusted_radar_login((c.get("user") or {}).get("login", "")):
            continue
        body = c.get("body", "") or ""
        has_approval = parse_radar_marker(body) is not None
        revs = iter_revoked_markers(body)
        if not (has_approval or revs):
            continue
        is_head_revocation = any(
            rev.sha == head_sha and verify_revoked_sig(rev, org, project, pr_num)
            for rev in revs
        )
        if keep_head_revocation and is_head_revocation and not has_approval:
            continue
        cid = c.get("id")
        if cid is None:
            continue
        if dry_run:
            print(f"Dryrun: would delete prior RADAR comment {cid}")
            continue
        try:
            gh_delete_comment(org, project, int(cid))
        except Exception as e:
            print(f"Could not delete prior RADAR comment {cid} (ok): {e}")


def _post_revocation(
    org: str, project: str, pr_num: int, sha: str, dry_run: bool
) -> None:
    # Post a signed revocation bound to the current head SHA. Callers guarantee a
    # key is present; without one no approval could ever have validated anyway.
    key = get_hmac_key()
    if key is None:
        return
    sig = compute_revoked_sig(key, org, project, pr_num, sha)
    marker = render_revoked_marker(RadarRevocation(sha=sha, sig=sig))
    body = "\n".join(
        [
            "## :radioactive: RADAR approval revoked",
            "",
            (
                "A previous RADAR auto-approval no longer applies to this PR: the "
                "current assessment does not qualify for auto-approval, so a human "
                "review is required to land it."
            ),
            "",
            marker,
        ]
    )
    gh_post_pr_comment(org, project, pr_num, body, dry_run=dry_run)


def _clear_stale_label(org: str, project: str, pr_num: int, dry_run: bool) -> None:
    # A previously-approved PR whose new head is riskier must lose the label.
    # gh_remove_label 404s when the label is absent, which is fine here.
    try:
        gh_remove_label(org, project, pr_num, RADAR_APPROVED_LABEL, dry_run)
    except Exception as e:
        print(f"Could not remove stale {RADAR_APPROVED_LABEL} label (ok): {e}")


def main() -> None:
    parser = ArgumentParser("RADAR risk-based auto-approval")
    parser.add_argument("--pr-num", type=int, required=True)
    parser.add_argument("--sha", type=str, required=True, help="full head commit SHA")
    parser.add_argument("--org", type=str, default="pytorch")
    parser.add_argument("--project", type=str, default="pytorch")
    parser.add_argument(
        "--verdict-file",
        type=str,
        required=True,
        help="path to the AI structured-output JSON (risk_score, risk_tier, ...)",
    )
    parser.add_argument(
        "--meta-file",
        type=str,
        required=True,
        help="path to `gh pr view --json ...files` metadata; used for the "
        "changed-files allowlist check",
    )
    parser.add_argument(
        "--author-permission",
        type=str,
        default="none",
        help="the PR author's repo permission (admin/maintain/write/read/none)",
    )
    parser.add_argument("--run-url", type=str, default=None)
    parser.add_argument(
        "--diff-truncated",
        action="store_true",
        help="set by the workflow when the diff shown to the AI was "
        "byte-truncated; forces non-approval regardless of the AI score",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with open(args.verdict_file) as f:
        verdict = json.load(f)
    score = int(verdict["risk_score"])
    tier = str(verdict["risk_tier"]).lower()
    summary = str(verdict.get("summary", ""))
    reasoning = str(verdict.get("reasoning", ""))

    with open(args.meta_file) as f:
        meta = json.load(f)
    files = [str(x["path"]) for x in meta.get("files", [])]
    # Fail closed if the line counts are absent: without them we cannot bound the
    # change size, so treat it as too large to auto-approve rather than defaulting
    # to zero (which would silently disable the size backstop).
    if meta.get("additions") is None or meta.get("deletions") is None:
        additions, deletions = RADAR_MAX_DIFF_LINES + 1, 0
    else:
        additions = int(meta["additions"])
        deletions = int(meta["deletions"])
    # `gh pr view --json files` caps its list at 100 entries, so on a >100-file PR
    # a non-allowlisted file could sit past the cap and the list would look fully
    # allowlisted. Fail closed unless GitHub's own changedFiles count (never
    # truncated) matches the list we actually see. trymerge re-checks the fully
    # paginated list, but this stops RADAR from POSTING a misleading approval.
    changed_total = meta.get("changedFiles")
    files_complete = changed_total is not None and len(files) == int(changed_total)

    low_risk = is_low_risk(tier, score)
    write_ok = has_write_access(args.author_permission)
    allowlist_ok = files_allowlisted(files)
    key_present = get_hmac_key() is not None
    # A truncated diff means the AI never saw the whole change, and an oversized
    # diff cannot be trivial; either forces non-approval regardless of the score.
    # The line count comes from meta.json, which is never truncated.
    size_ok = (not args.diff_truncated) and (
        additions + deletions
    ) <= RADAR_MAX_DIFF_LINES
    print(
        f"RADAR PR #{args.pr_num} @ {args.sha}: tier={tier} score={score} "
        f"low_risk={low_risk} author_permission={args.author_permission} "
        f"write_ok={write_ok} allowlist_ok={allowlist_ok} ({len(files)} files) "
        f"files_complete={files_complete} (changedFiles={changed_total}) "
        f"hmac_key_present={key_present} size_ok={size_ok} "
        f"diff_lines={additions + deletions} truncated={args.diff_truncated}"
    )

    if not (
        low_risk
        and write_ok
        and allowlist_ok
        and key_present
        and size_ok
        and files_complete
    ):
        print("RADAR does not approve; revoking any prior approval and clearing label.")
        comments = _list_pr_comments(args.org, args.project, args.pr_num)
        had_prior_approval, has_head_revocation = _radar_comment_state(
            comments, args.org, args.project, args.pr_num, args.sha
        )
        # Post a signed revocation for the current head BEFORE pruning, and only
        # when a prior approval exists but no standing revocation does yet. Posting
        # first (and keeping any current-head revocation during the prune below)
        # guarantees a fail-closed withdrawal is present at every instant, so
        # trymerge -- which honors the newest RADAR decision -- never sees a stale
        # approval unopposed even if the old approval comment fails to delete. A
        # write-access actor deleting the bot's revocation could still resurrect a
        # not-yet-deleted approval, but that is inherent to trusting write access
        # (they can already merge), and the SHA binding remains the primary guard.
        if had_prior_approval and not has_head_revocation and key_present:
            _post_revocation(
                args.org, args.project, args.pr_num, args.sha, args.dry_run
            )
        _delete_radar_comments(
            comments,
            args.org,
            args.project,
            args.pr_num,
            args.sha,
            keep_head_revocation=True,
            dry_run=args.dry_run,
        )
        _clear_stale_label(args.org, args.project, args.pr_num, args.dry_run)
        return

    key = get_hmac_key()
    assert key is not None  # guarded by key_present above
    sig = compute_marker_sig(
        key, args.org, args.project, args.pr_num, args.sha, score, tier
    )
    marker = RadarMarker(sha=args.sha, score=score, tier=tier, sig=sig)
    comment = build_approval_comment(marker, summary, reasoning, args.run_url)
    # Drop every prior RADAR banner (approvals and any revocation) so the fresh
    # approval is the sole, newest decision. The brief gap before the post below
    # is fail-closed (no signal -> trymerge denies), so the approval path needs no
    # keep-revocation handling.
    _delete_radar_comments(
        _list_pr_comments(args.org, args.project, args.pr_num),
        args.org,
        args.project,
        args.pr_num,
        args.sha,
        keep_head_revocation=False,
        dry_run=args.dry_run,
    )
    gh_post_pr_comment(
        args.org, args.project, args.pr_num, comment, dry_run=args.dry_run
    )
    gh_add_labels(
        args.org, args.project, args.pr_num, [RADAR_APPROVED_LABEL], args.dry_run
    )
    print("RADAR approved and posted.")


if __name__ == "__main__":
    main()
