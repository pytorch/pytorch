#!/usr/bin/env python3
"""Turn a pi distributed-triage plan into GitHub effects, and optionally apply them.

Second-level triage for issues routed to `oncall: distributed`. As with
apply_plan.py, the plan is a request: only labels from the distributed skill's
distributed-labels.json are added, an issue keeps at most one sub-queue label,
`triaged` is never paired with `triage review` or `needs reproduction`,
comments are limited to the skill's templates.json, and nothing is closed or
removed.

Usage:
  apply_distributed_plan.py <owner/repo> <issue> <plan.json> [--apply]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from apply_plan import (
    apply_effects,
    BOT_TRIAGED,
    Effects,
    fetch_issue,
    publish,
    summary_markdown,
    template_comment,
    TRIAGE_REVIEW,
)
from labels import (
    DISTRIBUTED,
    DISTRIBUTED_SKILL,
    is_sub_queue,
    load_labels,
    load_templates,
)


TRIAGED = "triaged"
# Distributed-bot marker; `bot-triaged` belongs to issue triage, which adds it
# before handing off.
PTD_TRIAGED = "ptd-bot-triaged"
NEEDS_REPRODUCTION = "needs reproduction"


def plan_effects(
    plan: dict,
    existing_labels: set[str],
    valid_labels: set[str],
    templates: dict[str, dict],
    bot_comments: list[str],
) -> Effects:
    """Map a submitted distributed-triage plan to the effects the apply step writes."""
    effects = Effects()
    decision = plan["decision"]
    if DISTRIBUTED not in existing_labels:
        effects.notes.append(f"issue is not labeled {DISTRIBUTED!r}; skipping")
        return effects

    requested = [
        label
        for label in plan.get("labels", [])
        if label not in (BOT_TRIAGED, PTD_TRIAGED)
    ]
    unknown = [label for label in requested if label not in valid_labels]
    if unknown:
        effects.notes.append(
            f"dropped labels outside distributed-labels.json {unknown}"
        )
    labels = [label for label in dict.fromkeys(requested) if label in valid_labels]

    # Exactly one sub-queue: keep an existing one, and refuse to guess between two.
    requested_queues = [label for label in labels if is_sub_queue(label)]
    if any(is_sub_queue(label) for label in existing_labels):
        if requested_queues:
            effects.notes.append(
                f"issue already has a sub-queue; dropped {requested_queues}"
            )
        labels = [label for label in labels if not is_sub_queue(label)]
    elif len(requested_queues) > 1:
        effects.notes.append(
            f"more than one sub-queue requested {requested_queues}; flagging for review"
        )
        labels = [label for label in labels if not is_sub_queue(label)] + [
            TRIAGE_REVIEW
        ]

    if decision == "high_priority" and TRIAGE_REVIEW not in labels:
        labels.append(TRIAGE_REVIEW)
    # A classification is only confident if a module label survived the allowlist.
    if decision == "classify" and not any(
        label.startswith("module: ") for label in existing_labels | set(labels)
    ):
        effects.notes.append("no allowlisted module label; flagging for review")
        labels.append(TRIAGE_REVIEW)
    resulting = existing_labels | set(labels)
    if TRIAGED in labels and resulting & {TRIAGE_REVIEW, NEEDS_REPRODUCTION}:
        effects.notes.append(
            f"dropped {TRIAGED!r}: the issue needs a human or a reproduction"
        )
        labels.remove(TRIAGED)

    effects.comment, posted = template_comment(
        list(dict.fromkeys(plan.get("templates", []))),
        templates,
        bot_comments,
        effects.notes,
    )

    effects.add_labels = [label for label in labels if label not in existing_labels]
    # The marker records that the distributed bot processed the issue, even when
    # nothing else changed, so the daily sweep skips it. High-priority issues get
    # none, so the sweep re-surfaces them.
    if decision != "high_priority" and PTD_TRIAGED not in existing_labels:
        effects.add_labels.append(PTD_TRIAGED)
    return effects


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("repo")
    parser.add_argument("issue", type=int)
    parser.add_argument("plan", type=Path)
    parser.add_argument(
        "--apply", action="store_true", help="write effects (default: dry run)"
    )
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text())
    issue, bot_comments = fetch_issue(args.repo, args.issue)
    existing = {label["name"] for label in issue["labels"]}

    effects = plan_effects(
        plan,
        existing,
        load_labels(DISTRIBUTED_SKILL / "distributed-labels.json"),
        load_templates(DISTRIBUTED_SKILL),
        bot_comments,
    )
    if args.apply:
        apply_effects(args.repo, args.issue, effects)

    labels_after = existing | set(effects.add_labels)
    if TRIAGED in labels_after:
        outcome = "triaged"
    elif TRIAGE_REVIEW in labels_after:
        outcome = "review"
    elif NEEDS_REPRODUCTION in labels_after:
        outcome = "waiting_on_reporter"
    elif any(is_sub_queue(label) for label in labels_after):
        outcome = "routed"
    else:
        outcome = "no_action"

    summary = summary_markdown(
        args.repo, args.issue, plan, effects, outcome, args.apply, None
    )
    publish(summary)
    if outcome == "no_action":
        print(f"::error::distributed triage plan for #{args.issue} produced no action")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
