"""Behavior of the trusted plan-to-effects mapping for pi issue triage."""

import json
import unittest
from pathlib import Path

from apply_plan import (
    CLOSING_TEMPLATES,
    LABELING_DECISIONS,
    labels_before_triage,
    plan_effects,
)
from labels import load_templates, load_valid_labels, TRIAGE_SKILL


TEMPLATES = load_templates(TRIAGE_SKILL)
VALID = load_valid_labels()
SCHEMA = json.loads(
    (
        Path(__file__).resolve().parents[2]
        / ".github/pi/schemas/issue-triage-plan.json"
    ).read_text()
)["properties"]


def effects(decision, labels=(), templates=(), existing=(), bot_comments=(), **extra):
    plan = {
        "decision": decision,
        "labels": list(labels),
        "templates": list(templates),
        "reasoning": "",
        **extra,
    }
    return plan_effects(plan, set(existing), VALID, TEMPLATES, list(bot_comments))


class PlanEffectsTest(unittest.TestCase):
    def test_label_plan_adds_labels_and_bot_marker(self):
        result = effects("label", ["module: optimizer", "triaged"])
        self.assertEqual(
            result.add_labels, ["module: optimizer", "triaged", "bot-triaged"]
        )
        self.assertFalse(result.close)
        self.assertEqual(result.comment, "")

    def test_existing_oncall_label_wins_over_plan(self):
        result = effects("label", ["module: nn"], existing={"oncall: pt2"})
        self.assertFalse(result.mutates)

    def test_a_distributed_sub_queue_becomes_the_parent_queue(self):
        result = effects("redirect_oncall", ["oncall: distributed checkpointing"])
        self.assertEqual(result.add_labels, ["oncall: distributed", "bot-triaged"])

    def test_forbidden_labels_become_triage_review(self):
        result = effects("label", ["sev1", "ciflow/trunk", "module: nn"])
        self.assertEqual(
            result.add_labels, ["module: nn", "triage review", "bot-triaged"]
        )

    def test_only_unknown_labels_fall_back_to_triage_review(self):
        result = effects("label", ["module: made up"])
        self.assertEqual(result.add_labels, ["triage review", "bot-triaged"])

    def test_redundant_general_label_is_dropped(self):
        result = effects("label", ["module: rnn", "module: nn"])
        self.assertEqual(result.add_labels, ["module: rnn", "bot-triaged"])

    def test_labels_already_present_are_not_re_added(self):
        result = effects(
            "label", ["module: nn", "triaged"], existing={"module: nn", "bot-triaged"}
        )
        self.assertEqual(result.add_labels, ["triaged"])

    def test_only_close_question_closes_and_always_redirects(self):
        closed = effects("close_question")
        self.assertTrue(closed.close)
        self.assertEqual(closed.comment, TEMPLATES["redirect_to_forum"]["comment"])
        self.assertEqual(closed.add_labels, ["bot-triaged"])
        self.assertFalse(effects("label", ["module: nn"], ["redirect_to_forum"]).close)

    def test_request_info_posts_template_without_labels(self):
        result = effects("request_info", templates=["request_more_info"])
        self.assertEqual(result.comment, TEMPLATES["request_more_info"]["comment"])
        self.assertEqual(result.add_labels, ["bot-triaged"])

    def test_transfer_is_flagged_for_review(self):
        result = effects("transfer", ["module: vision"], transfer_repo="pytorch/vision")
        self.assertEqual(result.add_labels, ["triage review", "bot-triaged"])
        self.assertTrue(any("pytorch/vision" in note for note in result.notes))

    def test_a_template_the_bot_already_posted_is_not_repeated(self):
        posted = "Bot says: " + TEMPLATES["request_more_info"]["comment"]
        # A rerun after a partial failure (comment posted, label call failed)
        # still adds the marker, and does nothing once the marker is there.
        rerun = effects(
            "request_info", templates=["request_more_info"], bot_comments=[posted]
        )
        self.assertEqual((rerun.comment, rerun.add_labels), ("", ["bot-triaged"]))
        done = effects(
            "request_info",
            templates=["request_more_info"],
            existing={"bot-triaged"},
            bot_comments=[posted],
        )
        self.assertFalse(done.mutates)

    def test_expected_numerical_behavior_closes_with_its_own_template(self):
        result = effects("close_expected_behavior", ["module: edge cases"])
        self.assertTrue(result.close)
        self.assertEqual(result.comment, TEMPLATES["numerical_accuracy"]["comment"])

    def test_unknown_template_is_not_posted(self):
        result = effects("request_info", templates=["anything goes"])
        self.assertEqual(result.comment, "")
        self.assertFalse(result.mutates)


class PlanSchemaTest(unittest.TestCase):
    # The model can only submit what the schema allows, so a template or
    # decision renamed on one side silently disables that path.
    def test_schema_offers_exactly_the_templates_apply_can_post(self):
        self.assertEqual(set(SCHEMA["templates"]["items"]["enum"]), set(TEMPLATES))

    def test_schema_offers_every_decision_apply_acts_on(self):
        self.assertLessEqual(
            {
                *CLOSING_TEMPLATES,
                *LABELING_DECISIONS,
                "skip_already_routed",
                "transfer",
            },
            set(SCHEMA["decision"]["enum"]),
        )


class LabelsBeforeTriageTest(unittest.TestCase):
    def event(self, kind, label, actor="author"):
        return {"event": kind, "label": {"name": label}, "actor": {"login": actor}}

    def test_keeps_template_labels_and_stops_at_the_triage_bot(self):
        events = [
            self.event("labeled", "oncall: pt2"),
            self.event("labeled", "needs triage"),
            self.event("unlabeled", "needs triage"),
            {"event": "renamed", "actor": {"login": "author"}},
            self.event("labeled", "module: inductor", actor="github-actions[bot]"),
            self.event("labeled", "module: dynamo"),
        ]
        self.assertEqual(labels_before_triage(events), {"oncall: pt2"})


if __name__ == "__main__":
    unittest.main()
