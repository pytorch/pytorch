#!/usr/bin/env python3
"""Turn a pi issue-triage plan into GitHub effects, and optionally apply them.

The plan comes from a model that read untrusted issue text, so it is treated
as a request: labels are filtered with the triaging-issues skill's rules,
comments are limited to templates.json and fixed related-issue references.
Only usage questions, expected numerical behavior, or covered reports in an
explicitly close-enabled umbrella policy may close the issue. Labels are only ever added. Reproduction
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
    DISTRIBUTED_MODULES,
    is_forbidden,
    load_templates,
    load_valid_labels,
    PARALLELISMS,
    strip_redundant,
    SUB_QUEUES,
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
    close_reason: str = "not_planned"
    umbrella_issue: int | None = None
    duplicate_issue_id: int | None = None
    umbrella_action: str = ""
    possible_duplicate: int | None = None
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


def fetch_issue(
    repo: str, number: int, *, all_comments: bool = False
) -> tuple[dict, list[str]]:
    """The issue and the bodies of the triage bot's comments on it."""
    issue = json.loads(gh_api([f"repos/{repo}/issues/{number}"]))
    endpoint = f"repos/{repo}/issues/{number}/comments?per_page=100"
    if all_comments:
        pages = json.loads(gh_api(["--paginate", "--slurp", endpoint]))
        comments = [comment for page in pages for comment in page]
    else:
        comments = json.loads(gh_api([endpoint]))
    return issue, [
        c["body"] or "" for c in comments if c["user"]["login"] == TRIAGE_BOT
    ]


def is_repo_issue(value: dict | None, repo: str, number: int) -> bool:
    return bool(
        value
        and value.get("number") == number
        and value.get("html_url") == f"https://github.com/{repo}/issues/{number}"
        and "pull_request" not in value
    )


def umbrella_policy(repo: str, annotation: object) -> dict | None:
    """Only a checked-in same-repository policy can authorize an umbrella action."""
    if (
        not isinstance(annotation, dict)
        or set(annotation) != {"issue", "match"}
        or type(annotation["issue"]) is not int
        or annotation["issue"] <= 0
        or annotation["match"] not in ("covered", "uncertain")
    ):
        return None
    catalog = json.loads((TRIAGE_SKILL / "umbrellas.json").read_text())
    return catalog.get(repo, {}).get(str(annotation["issue"]))


def fetch_umbrella_target(repo: str, source: int, plan: dict) -> dict | None:
    annotation = plan.get("umbrella")
    if (
        "possible_duplicate" in plan
        or not umbrella_policy(repo, annotation)
        or annotation["issue"] == source
    ):
        return None
    try:
        return json.loads(
            gh_api([f"repos/{repo}/issues/{annotation['issue']}"], retry=False)
        )
    except RuntimeError:
        return None


def fetch_possible_duplicate(repo: str, source: int, plan: dict) -> dict | None:
    number = plan.get("possible_duplicate")
    if type(number) is not int or number <= 0 or number == source or "umbrella" in plan:
        return None
    try:
        return json.loads(gh_api([f"repos/{repo}/issues/{number}"], retry=False))
    except RuntimeError:
        return None


def possible_duplicate_comment(
    repo: str,
    source: int,
    plan: dict,
    target: dict | None,
    bot_comments: list[str],
    effects: Effects,
) -> tuple[str, bool]:
    number = plan["possible_duplicate"]
    if "umbrella" in plan or plan["decision"] not in ("label", "triage_review"):
        effects.notes.append(
            "possible duplicate conflicts with the decision; kept open for review"
        )
        return "", False
    if (
        type(number) is not int
        or number <= 0
        or number == source
        or not isinstance(target, dict)
        or type(target.get("number")) is not int
        or not is_repo_issue(target, repo, number)
        or target.get("state") not in ("open", "closed")
        or umbrella_policy(repo, {"issue": number, "match": "covered"})
    ):
        effects.notes.append(
            "possible duplicate is not a valid non-umbrella issue; review"
        )
        return "", False
    effects.possible_duplicate = number
    marker = f"<!-- issue-triage-pi:possible-duplicate:{repo}#{number} -->"
    if any(marker in body.splitlines() for body in bot_comments):
        effects.notes.append(f"already suggested possible duplicate #{number}")
        return "", True
    reference = f"[{repo}#{number}](https://github.com/{repo}/issues/{number})"
    comment = (
        f"This looks similar to {reference} and may be the same underlying issue. "
        "Could you check whether that report covers your case?\n\n"
        "If it does, please add any new reproduction details there and close this issue "
        "so we can track the discussion in one place. If yours differs—or still "
        "reproduces after the linked issue’s fix—please explain the difference here."
    )
    return f"{comment}\n\n{marker}", False


def pt2_close_receipt(repo: str, number: int, target_id: int) -> str:
    return f"<!-- issue-triage-pi:umbrella-pt2-close:{repo}#{number}:{target_id} -->"


def pending_pt2_close(
    repo: str,
    plan: dict,
    labels: set[str],
    bot_comments: list[str],
    target: dict | None,
) -> bool:
    annotation = plan.get("umbrella")
    policy = umbrella_policy(repo, annotation)
    if not policy:
        return False
    requested = set(plan.get("labels", []))
    return bool(
        policy.get("action") == "close"
        and "possible_duplicate" not in plan
        and annotation["match"] == "covered"
        and plan["decision"] == "label"
        and "oncall: pt2" in requested
        and {label for label in labels | requested if label.startswith("oncall:")}
        == {"oncall: pt2"}
        and not {TRIAGE_REVIEW, "high priority", "needs reproduction"}
        & (labels | requested)
        and not {"request_more_info", "request_self_contained_reproduction"}
        & set(plan.get("templates", []))
        and target
        and type(target.get("id")) is int
        and target["id"] > 0
        and any(
            pt2_close_receipt(repo, annotation["issue"], target["id"])
            in body.splitlines()
            for body in bot_comments
        )
    )


def umbrella_comment(
    repo: str,
    source: int,
    plan: dict,
    target: dict | None,
    labels: set[str],
    bot_comments: list[str],
    effects: Effects,
    *,
    adding_pt2: bool = False,
) -> tuple[str, bool]:
    """Resolve scope asserted by the model under the trusted catalog's action."""
    annotation = plan["umbrella"]
    policy = umbrella_policy(repo, annotation)
    number = annotation.get("issue") if isinstance(annotation, dict) else None
    if type(number) is int and number > 0:
        effects.umbrella_issue = number
    effects.umbrella_action = "review"
    if plan["decision"] == "redirect_oncall" or any(
        label.startswith("oncall:") and label != "oncall: pt2" for label in labels
    ):
        effects.umbrella_action = "skip"
        effects.notes.append("umbrella reference skipped for oncall redirect")
        return "", False
    if (
        not policy
        or number == source
        or not is_repo_issue(target, repo, number)
        or type(target.get("id")) is not int
        or target["id"] <= 0
        or type(target.get("number")) is not int
        or target.get("state") != "open"
    ):
        effects.notes.append("umbrella target is not a configured open issue; review")
        return "", False
    if plan["decision"] in CLOSING_TEMPLATES or plan["decision"] == "transfer":
        effects.notes.append(
            "umbrella conflicts with the decision; kept open for review"
        )
        return "", False

    uncertain = annotation["match"] == "uncertain"
    blocked = bool({TRIAGE_REVIEW, "high priority"} & labels)
    pending_info = (
        "needs reproduction" in labels
        or plan["decision"] in ("request_info", "request_reproduction")
        or bool(
            {"request_more_info", "request_self_contained_reproduction"}
            & set(plan.get("templates", []))
        )
    )
    effects.close = policy.get("action") == "close" and not (
        uncertain or blocked or pending_info
    )
    effects.umbrella_action = "close" if effects.close else "link"
    if effects.close:
        effects.close_reason = "duplicate"
        effects.duplicate_issue_id = target["id"]
    if uncertain or TRIAGE_REVIEW in labels:
        effects.notes.append("umbrella coverage needs review; link only")
    elif blocked:
        effects.notes.append(
            "umbrella closure disabled by existing priority; link only"
        )
    url = f"https://github.com/{repo}/issues/{number}"
    reference = f"[{repo}#{number}]({url})"
    marker = f"<!-- issue-triage-pi:umbrella:{repo}#{number} -->"
    receipt = (
        pt2_close_receipt(repo, number, target["id"])
        if effects.close and plan["decision"] == "label" and adding_pt2
        else ""
    )
    posted = any(marker in body for body in bot_comments)
    if posted:
        effects.notes.append(f"already posted umbrella reference to #{number}")
        if receipt and not any(receipt in body.splitlines() for body in bot_comments):
            return receipt, True
        return "", True
    if effects.close:
        comment = (
            f"Duplicate of {reference}. This report is covered by the umbrella's scope."
        )
    else:
        comment = f"Related tracking issue: {reference}. Linking for context; this issue remains open for its own investigation."
    body = f"{comment}\n\n{marker}"
    if receipt:
        body += f"\n{receipt}"
    return body, False


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
    *,
    repo: str = "",
    issue: int = 0,
    source: dict | None = None,
    umbrella_target: dict | None = None,
    duplicate_target: dict | None = None,
) -> Effects:
    """Map a submitted plan to the effects the trusted step will apply.

    A template the bot already posted on the issue (in `bot_comments`) is not
    posted again.
    """
    effects = Effects()
    decision = plan["decision"]

    # Reopening ends the operation authorized by an old close receipt.
    if any(label.startswith("oncall:") for label in existing_labels) and not (
        "oncall: pt2" in valid_labels
        and (source or {}).get("state_reason") != "reopened"
        and pending_pt2_close(
            repo, plan, existing_labels, bot_comments, umbrella_target
        )
    ):
        effects.notes.append("issue already has an oncall: label; skipping")
        return effects
    if ("umbrella" in plan or "possible_duplicate" in plan) and (
        not is_repo_issue(source, repo, issue) or source.get("state") != "open"
    ):
        effects.notes.append("related-issue source is not an open issue; skipping")
        return effects
    if decision == "skip_already_routed":
        effects.notes.append("skip has no existing oncall label; flagging for review")
        effects.add_labels = [
            label
            for label in (TRIAGE_REVIEW, BOT_TRIAGED)
            if label not in existing_labels
        ]
        return effects

    # Completion markers are derived below, never taken from the plan.
    requested = [
        label
        for label in plan.get("labels", [])
        if label not in (BOT_TRIAGED, "triaged")
    ]
    # A complete distributed route has the parent queue and exactly one sub-queue.
    sub_queues = SUB_QUEUES.intersection(requested)
    if sub_queues:
        requested.append(DISTRIBUTED)
    if len(sub_queues) > 1:
        effects.notes.append(
            f"more than one sub-queue requested {sorted(sub_queues)}; flagging for review"
        )
        requested = [label for label in requested if label not in SUB_QUEUES]
        requested.append(TRIAGE_REVIEW)
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
    sub_queue = next((label for label in labels if label in SUB_QUEUES), None)
    if DISTRIBUTED in labels and sub_queue is None:
        effects.notes.append("no distributed sub-queue; flagging for review")
        labels.append(TRIAGE_REVIEW)
    if sub_queue == PARALLELISMS and not DISTRIBUTED_MODULES & (
        existing_labels | set(labels)
    ):
        effects.notes.append("no distributed module label; flagging for review")
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
    reference_body, reference_posted = "", False
    if "possible_duplicate" in plan:
        effects.close = False
        keys = [key for key in keys if key not in CLOSING_TEMPLATES.values()]
        reference_body, reference_posted = possible_duplicate_comment(
            repo, issue, plan, duplicate_target, bot_comments, effects
        )
        if effects.possible_duplicate is None:
            labels.append(TRIAGE_REVIEW)
    elif "umbrella" in plan:
        effects.close = False
        keys = [key for key in keys if key not in CLOSING_TEMPLATES.values()]
        reference_body, reference_posted = umbrella_comment(
            repo,
            issue,
            plan,
            umbrella_target,
            existing_labels | set(labels),
            bot_comments,
            effects,
            adding_pt2="oncall: pt2" in labels and "oncall: pt2" not in existing_labels,
        )
        annotation = plan["umbrella"]
        if effects.umbrella_action != "skip" and (
            effects.umbrella_action == "review"
            or isinstance(annotation, dict)
            and annotation.get("match") == "uncertain"
        ):
            labels.append(TRIAGE_REVIEW)

    effects.redact_body = decision == "request_reproduction"
    if decision == "request_info" and "request_more_info" not in keys:
        keys.insert(0, "request_more_info")
    repro_template = "request_self_contained_reproduction"
    if effects.redact_body and repro_template not in keys:
        keys.insert(0, repro_template)
    if (
        effects.close
        and decision in CLOSING_TEMPLATES
        and CLOSING_TEMPLATES[decision] not in keys
    ):
        keys.insert(0, CLOSING_TEMPLATES[decision])
    effects.comment, effects.comment_templates, posted = template_comment(
        keys, templates, bot_comments, effects.notes
    )

    if reference_body:
        effects.comment = "\n\n---\n\n".join(
            body for body in (effects.comment, reference_body) if body
        )
    posted = posted or reference_posted

    # PT2 and fully routed distributed issues can complete triage in this pass.
    completing_queues = {"oncall: pt2"}
    if sub_queue is not None:
        completing_queues |= {DISTRIBUTED, sub_queue}
    if (
        decision == "label"
        and not {TRIAGE_REVIEW, "needs reproduction"} & (existing_labels | set(labels))
        and all(
            label in completing_queues
            for label in labels
            if label.startswith("oncall:")
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


def current_reference_effects(repo: str, issue: int, plan: dict) -> Effects:
    current, comments = fetch_issue(repo, issue, all_comments=True)
    return plan_effects(
        plan,
        {label["name"] for label in current["labels"]},
        load_valid_labels(),
        load_templates(TRIAGE_SKILL),
        comments,
        repo=repo,
        issue=issue,
        source=current,
        umbrella_target=fetch_umbrella_target(repo, issue, plan),
        duplicate_target=fetch_possible_duplicate(repo, issue, plan),
    )


def post_labels(repo: str, issue: int, labels: list[str]) -> None:
    gh_api(
        ["-X", "POST", f"repos/{repo}/issues/{issue}/labels"]
        + [arg for label in labels for arg in ("-f", f"labels[]={label}")]
    )


def apply_effects(
    repo: str, issue: int, effects: Effects, *, plan: dict | None = None
) -> Effects:
    """Write only on the source issue; recheck related-issue plans before writes."""
    if plan is not None and ("umbrella" in plan or "possible_duplicate" in plan):
        effects = current_reference_effects(repo, issue, plan)
    elif effects.umbrella_action or effects.possible_duplicate is not None:
        raise ValueError(
            "related-issue effects require the plan for trusted revalidation"
        )
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
            retry=not bool(effects.umbrella_action or effects.possible_duplicate),
        )
    if effects.add_labels:
        post_labels(repo, issue, effects.add_labels)
    if effects.close and effects.umbrella_action:
        if plan is None:
            raise ValueError("umbrella closure requires a plan")
        checked = current_reference_effects(repo, issue, plan)
        effects.duplicate_issue_id = checked.duplicate_issue_id
        if not checked.close:
            effects.close = False
            effects.umbrella_action = checked.umbrella_action or "review"
            effects.notes.append("umbrella closure cancelled by final recheck")
            if (
                TRIAGE_REVIEW in checked.add_labels
                and TRIAGE_REVIEW not in effects.add_labels
            ):
                post_labels(repo, issue, [TRIAGE_REVIEW])
                effects.add_labels.append(TRIAGE_REVIEW)
    if effects.close:
        args = [
            "-X",
            "PATCH",
            f"repos/{repo}/issues/{issue}",
            "-f",
            "state=closed",
            "-f",
            f"state_reason={effects.close_reason}",
        ]
        if effects.umbrella_action:
            if effects.duplicate_issue_id is None:
                raise ValueError("umbrella closure requires the validated target ID")
            args.extend(["-F", f"duplicate_issue_id={effects.duplicate_issue_id}"])
        gh_api(args, retry=not bool(effects.umbrella_action))
    return effects


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
    if effects.umbrella_action:
        target = (
            f"{repo}#{effects.umbrella_issue}"
            if effects.umbrella_issue
            else "invalid target"
        )
        lines.append(f"**Umbrella:** {effects.umbrella_action} · **Target:** {target}")
    if effects.possible_duplicate is not None:
        lines.append(
            f"**Possible duplicate:** {repo}#{effects.possible_duplicate} (suggestion only)"
        )
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
    if "umbrella" in plan or "possible_duplicate" in plan:
        issue, bot_comments = fetch_issue(args.repo, args.issue, all_comments=True)
    else:
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
        repo=args.repo,
        issue=args.issue,
        source={**issue, "state": "open"} if args.replay else issue,
        umbrella_target=fetch_umbrella_target(args.repo, args.issue, plan),
        duplicate_target=fetch_possible_duplicate(args.repo, args.issue, plan),
    )
    if args.apply:
        effects = apply_effects(args.repo, args.issue, effects, plan=plan)

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
