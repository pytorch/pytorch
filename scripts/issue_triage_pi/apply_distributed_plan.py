#!/usr/bin/env python3
"""Turn a pi distributed-triage plan into GitHub effects, and optionally apply them.

Second-level triage for issues routed to `oncall: distributed`. As with
apply_plan.py, the plan is a request: only labels from the distributed skill's
distributed-labels.json are added, an issue keeps at most one sub-queue label,
`triaged` is not added when `triage review` or `needs reproduction` is present,
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
    PTD_TRIAGED,
)


TRIAGED = "triaged"
NEEDS_REPRODUCTION = "needs reproduction"
DECISION_STATUS = {
    "already_classified": TRIAGED,
    "route": TRIAGED,
    "classify": TRIAGED,
    "low_confidence": TRIAGE_REVIEW,
    "not_distributed": TRIAGE_REVIEW,
    "high_priority": TRIAGE_REVIEW,
    "needs_reproduction": NEEDS_REPRODUCTION,
}


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

    derived_labels = {BOT_TRIAGED, PTD_TRIAGED, *DECISION_STATUS.values()}
    requested = [
        label for label in plan.get("labels", []) if label not in derived_labels
    ]
    unknown = [label for label in requested if label not in valid_labels]
    if unknown:
        effects.notes.append(
            f"dropped labels outside distributed-labels.json {unknown}"
        )
    labels = [label for label in dict.fromkeys(requested) if label in valid_labels]

    # Exactly one sub-queue: keep an existing one, and refuse to guess between two.
    requested_queues = [label for label in labels if is_sub_queue(label)]
    if any(is_sub_queue(label) for label in existing_labels & valid_labels):
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

    # Status comes from the decision; model-suggested labels cannot override it.
    required_status = DECISION_STATUS[decision]
    if required_status not in labels:
        labels.append(required_status)
    # Performance describes the issue type, not a distributed module.
    if (
        decision == "classify"
        or (
            TRIAGED in labels
            and "oncall: distributed parallelisms" in existing_labels | set(labels)
        )
    ) and not any(
        label.startswith("module: ") and label != "module: performance"
        for label in (existing_labels | set(labels)) & valid_labels
    ):
        effects.notes.append("no allowlisted module label; flagging for review")
        labels.append(TRIAGE_REVIEW)
    resulting = existing_labels | set(labels)
    if TRIAGED in labels and resulting & {TRIAGE_REVIEW, NEEDS_REPRODUCTION}:
        effects.notes.append(
            f"dropped {TRIAGED!r}: the issue needs a human or a reproduction"
        )
        labels.remove(TRIAGED)
    if (decision == "route" or TRIAGED in labels) and not any(
        is_sub_queue(label) for label in resulting & valid_labels
    ):
        effects.notes.append("no allowlisted sub-queue; flagging for review")
        labels = [label for label in labels if label != TRIAGED] + [TRIAGE_REVIEW]

    template = {
        "not_distributed": "not_distributed",
        "needs_reproduction": "needs_distributed_reproduction",
    }.get(decision)
    keys = [template] if template else []
    dropped = [key for key in plan.get("templates", []) if key not in keys]
    if dropped:
        effects.notes.append(
            f"dropped templates inconsistent with {decision!r}: {dropped}"
        )
    effects.comment, effects.comment_templates, _ = template_comment(
        keys, templates, bot_comments, effects.notes
    )

    effects.add_labels = [
        label for label in dict.fromkeys(labels) if label not in existing_labels
    ]
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
    if TRIAGE_REVIEW in labels_after:
        outcome = "review"
    elif NEEDS_REPRODUCTION in labels_after:
        outcome = "waiting_on_reporter"
    elif TRIAGED in labels_after:
        outcome = "triaged"
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
