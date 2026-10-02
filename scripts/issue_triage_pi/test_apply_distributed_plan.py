"""Behavior of the trusted plan-to-effects mapping for pi distributed triage."""

import unittest

from apply_distributed_plan import plan_effects
from labels import DISTRIBUTED_SKILL, load_labels, load_templates


TEMPLATES = load_templates(DISTRIBUTED_SKILL)
VALID = load_labels(DISTRIBUTED_SKILL / "distributed-labels.json")
ROUTED = {"oncall: distributed", "bot-triaged"}


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
                "bot-triaged",
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
        self.assertEqual(result.add_labels, ["module: fsdp"])

    def test_an_existing_sub_queue_is_kept(self):
        existing = ROUTED | {"oncall: distributed infra"}
        result = effects(
            "route", ["oncall: distributed checkpointing"], existing=existing
        )
        self.assertFalse(result.mutates)

    def test_two_requested_sub_queues_go_to_review(self):
        result = effects(
            "route", ["oncall: distributed infra", "oncall: distributed checkpointing"]
        )
        self.assertEqual(result.add_labels, ["triage review"])

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
        self.assertEqual(repro.add_labels, ["needs reproduction"])
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
            existing=ROUTED | {"triage review"},
            bot_comments=[posted],
        )
        self.assertFalse(result.mutates)

    def test_nothing_is_ever_closed(self):
        self.assertFalse(
            effects("not_distributed", ["triage review"], ["not_distributed"]).close
        )


if __name__ == "__main__":
    unittest.main()
