"""Behavior of the trusted plan-to-effects mapping for pi distributed triage."""

import json
import unittest
from pathlib import Path

from apply_distributed_plan import plan_effects
from labels import DISTRIBUTED_SKILL, load_labels, load_templates


TEMPLATES = load_templates(DISTRIBUTED_SKILL)
SCHEMA = json.loads(
    (
        Path(__file__).resolve().parents[2]
        / ".github/pi/schemas/distributed-triage-plan.json"
    ).read_text()
)["properties"]
VALID = load_labels(DISTRIBUTED_SKILL / "distributed-labels.json")
# Issue triage adds `bot-triaged` when it hands an issue off.
ROUTED = {"oncall: distributed", "bot-triaged"}
MARKER = "ptd-bot-triaged"


def effects(decision, labels=(), templates=(), existing=ROUTED, bot_comments=()):
    plan = {
        "decision": decision,
        "labels": list(labels),
        "templates": list(templates),
        "reasoning": "",
    }
    return plan_effects(plan, set(existing), VALID, TEMPLATES, list(bot_comments))


class DistributedPlanEffectsTest(unittest.TestCase):
    def test_classification_adds_sub_queue_and_modules(self):
        result = effects(
            "classify",
            ["oncall: distributed parallelisms", "module: dtensor", "triaged"],
            existing={"oncall: distributed"},
        )
        self.assertEqual(
            result.add_labels,
            [
                "oncall: distributed parallelisms",
                "module: dtensor",
                "triaged",
                MARKER,
            ],
        )

    def test_issue_outside_the_distributed_queue_is_left_alone(self):
        result = effects(
            "classify", ["module: fsdp", "triaged"], existing={"oncall: pt2"}
        )
        self.assertFalse(result.mutates)

    def test_labels_outside_the_distributed_allowlist_are_dropped(self):
        result = effects(
            "classify", ["module: fsdp", "high priority", "module: inductor"]
        )
        self.assertEqual(result.add_labels, ["module: fsdp", MARKER])

    def test_an_existing_sub_queue_is_kept(self):
        existing = ROUTED | {"oncall: distributed infra"}
        result = effects(
            "route", ["oncall: distributed checkpointing"], existing=existing
        )
        # Only the marker, so the daily sweep does not re-dispatch the issue.
        self.assertEqual(result.add_labels, [MARKER])

    def test_two_requested_sub_queues_go_to_review(self):
        result = effects(
            "route", ["oncall: distributed infra", "oncall: distributed checkpointing"]
        )
        self.assertEqual(result.add_labels, ["triage review", MARKER])

    def test_triaged_never_pairs_with_review_or_reproduction(self):
        review = effects(
            "low_confidence",
            ["oncall: distributed parallelisms", "triage review", "triaged"],
        )
        self.assertNotIn("triaged", review.add_labels)
        repro = effects(
            "needs_reproduction",
            ["needs reproduction", "triaged"],
            ["needs_distributed_reproduction"],
        )
        self.assertEqual(repro.add_labels, ["needs reproduction", MARKER])
        self.assertEqual(
            repro.comment, TEMPLATES["needs_distributed_reproduction"]["comment"]
        )

    def test_high_priority_goes_to_review_without_the_marker(self):
        result = effects(
            "high_priority", ["module: c10d"], existing={"oncall: distributed"}
        )
        self.assertEqual(result.add_labels, ["module: c10d", "triage review"])

    def test_a_template_the_bot_already_posted_is_not_repeated(self):
        posted = TEMPLATES["not_distributed"]["comment"]
        result = effects(
            "not_distributed",
            ["triage review"],
            ["not_distributed"],
            existing=ROUTED | {"triage review", MARKER},
            bot_comments=[posted],
        )
        self.assertFalse(result.mutates)

    def test_classification_without_an_allowlisted_module_goes_to_review(self):
        result = effects("classify", ["module: typo", "triaged"])
        self.assertEqual(result.add_labels, ["triage review", MARKER])

    def test_schema_matches_the_templates_and_the_high_priority_decision(self):
        self.assertEqual(set(SCHEMA["templates"]["items"]["enum"]), set(TEMPLATES))
        self.assertIn("high_priority", SCHEMA["decision"]["enum"])

    def test_nothing_is_ever_closed(self):
        self.assertFalse(
            effects("not_distributed", ["triage review"], ["not_distributed"]).close
        )


if __name__ == "__main__":
    unittest.main()
