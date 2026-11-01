#!/usr/bin/env python3
"""Turn a pi issue-triage plan into GitHub effects, and optionally apply them.

The plan comes from a model that read untrusted issue text, so it is treated
as a request: labels are filtered with the triaging-issues skill's rules,
comments are limited to templates.json, and only usage questions or expected
numerical behavior may close the issue. Labels are only ever added. Reproduction
requests redact download links using a fixed policy, never a model-written body.

Usage:
  apply_plan.py <owner/repo> <issue> <plan.json> [--apply] [--replay]

Without --apply nothing is written (dry run). --replay evaluates the plan
against the issue as it was before the triage bot acted (labels from before its
first label event) and compares it with the labels added since.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote, urlsplit

from labels import (
    DISTRIBUTED,
    is_forbidden,
    is_sub_queue,
    load_templates,
    load_valid_labels,
    PTD_TRIAGED,
    strip_redundant,
    TRIAGE_SKILL,
)


BOT_TRIAGED = "bot-triaged"
TRIAGE_REVIEW = "triage review"
TRIAGE_BOT = "github-actions[bot]"
LABELING_DECISIONS = {"label", "redirect_oncall", "triage_review"}
# Decisions that close the issue, and the template each must post.
CLOSING_TEMPLATES = {
    "close_question": "redirect_to_forum",
    "close_expected_behavior": "numerical_accuracy",
}


REDACTED_LINK = (
    "[Link removed - external file downloads are not permitted for security reasons]"
)
DOWNLOAD_SUFFIXES = (".zip", ".pt", ".pth", ".pkl", ".safetensors", ".onnx", ".bin")
STORAGE_HOSTS = (
    "drive.google.com",
    "dropbox.com",
    "dropboxusercontent.com",
    "onedrive.live.com",
    "1drv.ms",
    "mega.nz",
    "mega.co.nz",
    "wetransfer.com",
    "we.tl",
)


def gh_api(args: list[str], *, retry: bool = True) -> str:
    """Run `gh api`; disable retries for writes derived from a mutable snapshot."""
    attempts = 3 if retry else 1
    for attempt in range(1, attempts + 1):
        result = subprocess.run(
            ["gh", "api", *args], capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            return result.stdout
        print(f"gh api {args[0]} failed (attempt {attempt}): {result.stderr.strip()}")
        if attempt < attempts:
            time.sleep(2**attempt)
    raise RuntimeError(f"gh api {' '.join(args)} failed after {attempts} attempts")


def classify(labels: set[str], state: str, mutated: bool) -> str:
    """Name the triage outcome from the issue's resulting labels and state."""
    if any(label.startswith("oncall:") for label in labels):
        return "routed"
    if "triaged" in labels:
        return "triaged"
    if TRIAGE_REVIEW in labels:
        return "review"
    if state == "closed":
        return "closed"
    if mutated:
        return "waiting_on_reporter"
    return "no_action"


@dataclass
class Effects:
    """GitHub mutations derived from one plan."""

    add_labels: list[str] = field(default_factory=list)
    comment: str = ""
    comment_templates: list[str] = field(default_factory=list)
    close: bool = False
    redact_body: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def mutates(self) -> bool:
        return bool(self.add_labels or self.comment or self.close or self.redact_body)


def template_comment(
    keys: list[str],
    templates: dict[str, dict],
    bot_comments: list[str],
    notes: list[str],
) -> tuple[str, list[str], bool]:
    """Join the comments of the known `keys` the bot has not already posted.

    Also returns the included keys and whether any template was already posted, which
    counts as triage done: a rerun after a partial failure (comment posted,
    label call failed) still adds the bot marker.
    """
    unknown = [key for key in keys if key not in templates]
    if unknown:
        notes.append(f"dropped unknown templates {unknown}")
    known = [key for key in keys if key in templates]
    posted = [
        key
        for key in known
        if any(templates[key]["comment"] in body for body in bot_comments)
    ]
    if posted:
        notes.append(f"already posted {posted}")
    pending = [key for key in known if key not in posted]
    comment = "\n\n---\n\n".join(templates[key]["comment"] for key in pending)
    return comment, pending, bool(posted)


def fetch_issue(repo: str, number: int) -> tuple[dict, list[str]]:
    """The issue and the bodies of the triage bot's comments on it."""
    issue = json.loads(gh_api([f"repos/{repo}/issues/{number}"]))
    comments = json.loads(
        gh_api([f"repos/{repo}/issues/{number}/comments?per_page=100"])
    )
    return issue, [
        c["body"] or "" for c in comments if c["user"]["login"] == TRIAGE_BOT
    ]


def publish(summary: str) -> None:
    # The runner recognizes legacy ##[commands] anywhere in a log line.
    print(json.dumps({"summary": summary}).replace("##[", r"\u0023\u0023["))
    if path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(path, "a") as f:
            f.write(summary + "\n")


def plan_effects(
    plan: dict,
    existing_labels: set[str],
    valid_labels: set[str],
    templates: dict[str, dict],
    bot_comments: list[str],
) -> Effects:
    """Map a submitted plan to the effects the trusted step will apply.

    A template the bot already posted on the issue (in `bot_comments`) is not
    posted again.
    """
    effects = Effects()
    decision = plan["decision"]

    if any(label.startswith("oncall:") for label in existing_labels):
        effects.notes.append("issue already has an oncall: label; skipping")
        return effects
    if decision == "skip_already_routed":
        effects.notes.append("skip has no existing oncall label; flagging for review")
        effects.add_labels = [
            label
            for label in (TRIAGE_REVIEW, BOT_TRIAGED)
            if label not in existing_labels
        ]
        return effects

    requested = [
        label
        for label in plan.get("labels", [])
        if label not in (BOT_TRIAGED, PTD_TRIAGED)
    ]
    # Stage 2 routes to the parent queue only. The distributed triage picks the
    # sub-queue, and it stops early on an issue that already has one.
    sub_queues = [label for label in requested if is_sub_queue(label)]
    if sub_queues:
        effects.notes.append(f"mapped {sub_queues} to {DISTRIBUTED!r}")
        requested = [
            DISTRIBUTED if label in sub_queues else label for label in requested
        ]
    if decision == "transfer":
        effects.notes.append(
            f"transfer to {plan.get('transfer_repo', '?')} is not supported; flagging for review"
        )
        requested = [TRIAGE_REVIEW]

    forbidden = [label for label in requested if is_forbidden(label)]
    unknown = [
        label
        for label in requested
        if label not in forbidden and label not in valid_labels
    ]
    labels, redundant = strip_redundant(
        [
            label
            for label in requested
            if label not in forbidden and label not in unknown
        ]
    )
    if forbidden:
        effects.notes.append(f"dropped forbidden labels {forbidden}")
    if unknown:
        effects.notes.append(f"dropped unknown labels {unknown}")
    if redundant:
        effects.notes.append(f"dropped redundant labels {redundant}")
    if forbidden or (not labels and (unknown or decision in LABELING_DECISIONS)):
        labels.append(TRIAGE_REVIEW)
    if decision == "triage_review":
        labels.append(TRIAGE_REVIEW)
    keys = list(dict.fromkeys(plan.get("templates", [])))
    effects.close = decision in CLOSING_TEMPLATES
    if (decision == "close_expected_behavior" or "numerical_accuracy" in keys) and (
        decision != "close_expected_behavior"
        or {TRIAGE_REVIEW, "high priority"} & (existing_labels | set(labels))
    ):
        effects.close = False
        labels.append(TRIAGE_REVIEW)
        keys = [key for key in keys if key not in CLOSING_TEMPLATES.values()]
        effects.notes.append("numerical dismissal conflicts with review; kept open")
    if "triaged" in labels and TRIAGE_REVIEW in existing_labels | set(labels):
        effects.notes.append("dropped 'triaged': the issue needs human review")
        labels = [label for label in labels if label != "triaged"]

    effects.redact_body = decision == "request_reproduction"
    if effects.redact_body:
        labels = [label for label in labels if label != "triaged"]
    if decision == "request_info" and "request_more_info" not in keys:
        keys.insert(0, "request_more_info")
    repro_template = "request_self_contained_reproduction"
    if effects.redact_body and repro_template not in keys:
        keys.insert(0, repro_template)
    if effects.close and CLOSING_TEMPLATES[decision] not in keys:
        keys.insert(0, CLOSING_TEMPLATES[decision])
    effects.comment, effects.comment_templates, posted = template_comment(
        keys, templates, bot_comments, effects.notes
    )

    if (
        decision == "label"
        and not {TRIAGE_REVIEW, "needs reproduction"} & (existing_labels | set(labels))
        and not any(
            label.startswith("oncall:") and label != "oncall: pt2" for label in labels
        )
        and not {"request_more_info", repro_template} & set(keys)
    ):
        labels.append("triaged")

    effects.add_labels = [
        label for label in dict.fromkeys(labels) if label not in existing_labels
    ]
    if (effects.mutates or posted) and BOT_TRIAGED not in existing_labels:
        effects.add_labels.append(BOT_TRIAGED)
    return effects


def labels_before_triage(events: list[dict]) -> set[str]:
    """Labels the issue had before the triage bot's first label event.

    Issue templates apply labels such as `oncall: pt2` at creation; a replay
    must show those, not an unlabeled issue.
    """
    labels: set[str] = set()
    for event in events:
        if event["event"] not in ("labeled", "unlabeled"):
            continue
        if (event.get("actor") or {}).get("login") == TRIAGE_BOT:
            break
        name = event["label"]["name"]
        if event["event"] == "labeled":
            labels.add(name)
        else:
            labels.discard(name)
    return labels


def redact_external_links(body: str) -> str:
    """Replace download URLs without following them or accepting replacement prose."""

    def replace_url(match: re.Match[str]) -> str:
        text = match.group()
        url = text.rstrip(".,;:!?)]}")
        try:
            parsed = urlsplit(url)
        except ValueError:
            return text
        host = (parsed.hostname or "").lower().rstrip(".")
        path = unquote(parsed.path).lower()
        storage = any(
            host == item or host.endswith("." + item) for item in STORAGE_HOSTS
        )
        if storage or path.endswith(DOWNLOAD_SUFFIXES):
            return REDACTED_LINK + text[len(url) :]
        return text

    # Stop at Markdown/quote boundaries, allowing a parenthesized filename segment.
    url_pattern = (
        r"https?://[^\s<>\"'`\[\]()]+(?:\([^\s<>\"'`\[\]()]*\)[^\s<>\"'`\[\]()]*)*"
    )
    redacted = re.sub(url_pattern, replace_url, body, flags=re.IGNORECASE)
    # Preserve readable Markdown rather than leaving the notice in a URL position.
    notice = re.escape(REDACTED_LINK)
    redacted = re.sub(
        r"!?\[[^\[\]\n]*\]\(" + notice + r'(?:\s+"[^"\n]*")?\)', REDACTED_LINK, redacted
    )
    return redacted.replace("<" + REDACTED_LINK + ">", REDACTED_LINK)


def apply_effects(repo: str, issue: int, effects: Effects) -> None:
    """Write the effects to GitHub. Labels use the append endpoint, never SET."""
    if effects.redact_body:
        # Use the current body, not the model's input snapshot or proposed prose.
        current = json.loads(gh_api([f"repos/{repo}/issues/{issue}"]))["body"] or ""
        redacted = redact_external_links(current)
        if redacted != current:
            gh_api(
                [
                    "-X",
                    "PATCH",
                    f"repos/{repo}/issues/{issue}",
                    "-f",
                    f"body={redacted}",
                ],
                retry=False,
            )
    if effects.comment:
        gh_api(
            [
                "-X",
                "POST",
                f"repos/{repo}/issues/{issue}/comments",
                "-f",
                f"body={effects.comment}",
            ],
        )
    if effects.add_labels:
        gh_api(
            ["-X", "POST", f"repos/{repo}/issues/{issue}/labels"]
            + [
                arg
                for label in effects.add_labels
                for arg in ("-f", f"labels[]={label}")
            ],
        )
    if effects.close:
        gh_api(
            [
                "-X",
                "PATCH",
                f"repos/{repo}/issues/{issue}",
                "-f",
                "state=closed",
                "-f",
                "state_reason=not_planned",
            ],
        )


def summary_markdown(
    repo: str,
    issue: int,
    plan: dict,
    effects: Effects,
    outcome: str,
    applied: bool,
    reference: set[str] | None,
) -> str:
    lines = [
        f"### [{repo}#{issue}](https://github.com/{repo}/issues/{issue}) — `{plan['decision']}` → {outcome}"
        + ("" if applied else " (dry run)"),
        "",
        f"**Add labels:** {', '.join(f'`{label}`' for label in effects.add_labels) or '—'}",
        f"**Comment templates:** {', '.join(effects.comment_templates) or '—'} · **Close:** {effects.close}",
    ]
    if effects.redact_body:
        lines.append("**Body:** redact external download links using the fixed policy")
    if effects.notes:
        lines.append(f"**Notes:** {'; '.join(effects.notes)}")
    if reference is not None:
        planned = set(effects.add_labels) - {BOT_TRIAGED}
        lines.append(
            f"**vs labels added since:** matched {sorted(planned & reference) or '—'} · "
            f"missed {sorted(reference - planned) or '—'} · extra {sorted(planned - reference) or '—'}"
        )
    lines += ["", f"> {plan.get('reasoning', '').strip()}", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("repo")
    parser.add_argument("issue", type=int)
    parser.add_argument("plan", type=Path)
    parser.add_argument(
        "--apply", action="store_true", help="write effects (default: dry run)"
    )
    parser.add_argument(
        "--replay",
        action="store_true",
        help="plan was made on the issue as it was before the triage bot acted",
    )
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text())
    templates = load_templates(TRIAGE_SKILL)
    issue, bot_comments = fetch_issue(args.repo, args.issue)
    current = {label["name"] for label in issue["labels"]}

    if args.replay:
        pages = json.loads(
            gh_api(
                [
                    "--paginate",
                    "--slurp",
                    f"repos/{args.repo}/issues/{args.issue}/events?per_page=100",
                ]
            )
        )
        existing = labels_before_triage([event for page in pages for event in page])
    else:
        existing = current
    effects = plan_effects(
        plan,
        existing,
        load_valid_labels(),
        templates,
        [] if args.replay else bot_comments,
    )
    if args.apply:
        apply_effects(args.repo, args.issue, effects)

    labels_after = existing | set(effects.add_labels)
    state_after = (
        "closed" if effects.close else ("open" if args.replay else issue["state"])
    )
    # Only confirmed routing justifies reporting a skipped issue.
    if plan["decision"] == "skip_already_routed" and any(
        label.startswith("oncall:") for label in existing
    ):
        outcome = "skipped"
    else:
        outcome = classify(labels_after, state_after, effects.mutates)

    reference = current - existing - {BOT_TRIAGED} if args.replay else None
    summary = summary_markdown(
        args.repo, args.issue, plan, effects, outcome, args.apply, reference
    )
    publish(summary)
    if outcome == "no_action":
        print(f"::error::triage plan for #{args.issue} produced no action")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
