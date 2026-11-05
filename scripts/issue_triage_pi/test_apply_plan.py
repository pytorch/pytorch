"""Behavior of the trusted plan-to-effects mapping for pi issue triage."""

import json
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from apply_plan import (
    CLOSING_TEMPLATES,
    LABELING_DECISIONS,
    labels_before_triage,
    main,
    plan_effects,
    publish,
    summary_markdown,
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


class PublishTest(unittest.TestCase):
    def test_untrusted_summary_is_json_logged_without_changing_markdown(self):
        summary = (
            "## Plan\n::warning::untrusted\n::stop-commands::token\r\n##[warning]legacy"
        )
        for write_summary in (False, True):
            with (
                self.subTest(write_summary=write_summary),
                tempfile.TemporaryDirectory() as temp,
            ):
                summary_file = Path(temp) / "summary.md"
                env = (
                    {"GITHUB_STEP_SUMMARY": str(summary_file)} if write_summary else {}
                )
                if write_summary:
                    summary_file.write_text("Earlier summary\n")
                with (
                    patch.dict("os.environ", env, clear=True),
                    patch("sys.stdout", new_callable=StringIO) as stdout,
                ):
                    publish(summary)
                self.assertEqual(len(stdout.getvalue().splitlines()), 1)
                self.assertNotIn("##[", stdout.getvalue())
                self.assertEqual(json.loads(stdout.getvalue()), {"summary": summary})
                if write_summary:
                    self.assertEqual(
                        summary_file.read_bytes(),
                        ("Earlier summary\n" + summary + "\n").encode(),
                    )
                else:
                    self.assertFalse(summary_file.exists())


class SummaryTest(unittest.TestCase):
    def test_summary_reports_only_comments_that_will_be_posted(self):
        for decision, requested, posted, expected in (
            ("close_question", [], [], "redirect_to_forum"),
            ("request_info", [], [], "request_more_info"),
            ("request_info", ["unknown"], [], "request_more_info"),
            (
                "request_info",
                ["request_more_info"],
                [TEMPLATES["request_more_info"]["comment"]],
                "—",
            ),
        ):
            with self.subTest(decision=decision, requested=requested, posted=posted):
                result = effects(decision, templates=requested, bot_comments=posted)
                summary = summary_markdown(
                    "pytorch/ciforge",
                    123,
                    {"decision": decision, "templates": requested},
                    result,
                    "test",
                    False,
                    None,
                )
                self.assertIn(f"**Comment templates:** {expected} ·", summary)


class PlanEffectsTest(unittest.TestCase):
    def test_label_plan_adds_labels_and_bot_marker(self):
        result = effects("label", ["module: optimizer", "triaged"])
        self.assertEqual(
            result.add_labels, ["module: optimizer", "triaged", "bot-triaged"]
        )
        self.assertFalse(result.close)
        self.assertEqual(result.comment, "")

    def test_complete_label_plan_gets_triaged_without_model_request(self):
        for labels in (
            ["module: mps", "module: performance", "module: memory format"],
            ["oncall: pt2", "module: aotdispatch"],
        ):
            with self.subTest(labels=labels):
                result = effects("label", labels)
                self.assertEqual(result.add_labels, [*labels, "triaged", "bot-triaged"])

    def test_other_decisions_do_not_get_automatic_completion(self):
        for decision in set(SCHEMA["decision"]["enum"]) - {"label"}:
            with self.subTest(decision=decision):
                result = effects(decision, ["module: nn"])
                self.assertNotIn("triaged", result.add_labels)

    def test_label_completion_respects_routing_and_pending_work(self):
        for labels, templates, existing in (
            (["oncall: distributed"], [], []),
            (["module: nn", "triage review"], [], []),
            (["module: nn"], [], ["triage review"]),
            (["module: nn"], [], ["needs reproduction"]),
            (["module: nn"], ["request_more_info"], []),
            (["module: nn"], ["request_self_contained_reproduction"], []),
        ):
            with self.subTest(labels=labels, templates=templates, existing=existing):
                result = effects("label", labels, templates, existing)
                self.assertNotIn("triaged", result.add_labels)

    def test_first_pass_cannot_add_the_distributed_completion_marker(self):
        result = effects("label", ["module: nn", "bot-triaged", "ptd-bot-triaged"])
        self.assertEqual(result.add_labels, ["module: nn", "triaged", "bot-triaged"])

    def test_high_priority_requires_review_not_triaged(self):
        result = effects("label", ["module: nn", "high priority", "triaged"])
        self.assertEqual(
            result.add_labels, ["module: nn", "triage review", "bot-triaged"]
        )

    def test_existing_oncall_label_wins_over_plan(self):
        result = effects("label", ["module: nn"], existing={"oncall: pt2"})
        self.assertFalse(result.mutates)

    def test_skip_requires_an_existing_oncall(self):
        self.assertFalse(
            effects("skip_already_routed", existing={"oncall: pt2"}).mutates
        )
        skipped = effects("skip_already_routed")
        self.assertEqual(skipped.add_labels, ["triage review", "bot-triaged"])
        self.assertFalse(skipped.close)
        self.assertEqual(skipped.comment, "")

    def test_unjustified_skip_reports_review(self):
        with tempfile.TemporaryDirectory() as temp:
            plan = Path(temp) / "plan.json"
            plan.write_text(json.dumps({"decision": "skip_already_routed"}))
            with (
                patch("sys.argv", ["apply_plan", "pytorch/ciforge", "123", str(plan)]),
                patch(
                    "apply_plan.fetch_issue",
                    return_value=({"labels": [], "state": "open"}, []),
                ),
                patch("apply_plan.publish") as publish,
            ):
                self.assertEqual(main(), 0)
            self.assertIn("→ review", publish.call_args.args[0])

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
        self.assertEqual(result.add_labels, ["module: rnn", "triaged", "bot-triaged"])

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
        for templates in ([], ["request_more_info"]):
            with self.subTest(templates=templates):
                result = effects("request_info", templates=templates)
                self.assertEqual(
                    result.comment, TEMPLATES["request_more_info"]["comment"]
                )
                self.assertEqual(result.add_labels, ["bot-triaged"])
                self.assertFalse(result.close)

    def test_transfer_is_flagged_for_review(self):
        result = effects("transfer", ["module: vision"], transfer_repo="pytorch/vision")
        self.assertEqual(result.add_labels, ["triage review", "bot-triaged"])
        self.assertTrue(any("pytorch/vision" in note for note in result.notes))

    def test_a_template_the_bot_already_posted_is_not_repeated(self):
        posted = "Bot says: " + TEMPLATES["request_more_info"]["comment"]
        # A rerun after a partial failure (comment posted, label call failed)
        # still adds the marker, and does nothing once the marker is there.
        for templates in ([], ["request_more_info"]):
            with self.subTest(templates=templates):
                rerun = effects(
                    "request_info", templates=templates, bot_comments=[posted]
                )
                self.assertEqual(
                    (rerun.comment, rerun.add_labels), ("", ["bot-triaged"])
                )
                done = effects(
                    "request_info",
                    templates=templates,
                    existing={"bot-triaged"},
                    bot_comments=[posted],
                )
                self.assertFalse(done.mutates)

    def test_expected_numerical_behavior_closes_with_its_own_template(self):
        result = effects("close_expected_behavior", ["module: edge cases"])
        self.assertTrue(result.close)
        self.assertEqual(result.comment, TEMPLATES["numerical_accuracy"]["comment"])

    def test_review_decision_requires_review_even_if_model_omits_label(self):
        result = effects("triage_review", ["module: sdpa", "triaged"])
        self.assertEqual(
            result.add_labels, ["module: sdpa", "triage review", "bot-triaged"]
        )
        self.assertFalse(result.close)

    def test_numerical_close_cannot_override_review_or_human_priority(self):
        for labels, existing in (
            (["module: sdpa", "triage review", "triaged"], []),
            (["module: sdpa"], ["triage review"]),
            (["module: sdpa"], ["high priority"]),
        ):
            with self.subTest(labels=labels, existing=existing):
                result = effects("close_expected_behavior", labels, existing=existing)
                self.assertFalse(result.close)
                self.assertEqual(result.comment, "")
                self.assertNotIn("triaged", result.add_labels)
                self.assertIn("triage review", set(result.add_labels) | set(existing))

    def test_numerical_template_requires_expected_behavior_closure(self):
        for decision in ("label", "triage_review", "request_info"):
            with self.subTest(decision=decision):
                result = effects(
                    decision, ["module: numerical-stability"], ["numerical_accuracy"]
                )
                self.assertFalse(result.close)
                expected = (
                    TEMPLATES["request_more_info"]["comment"]
                    if decision == "request_info"
                    else ""
                )
                self.assertEqual(result.comment, expected)
                self.assertIn("triage review", result.add_labels)
                self.assertNotIn("triaged", result.add_labels)

    def test_cancelled_numerical_close_drops_all_closing_comments(self):
        result = effects(
            "close_question", templates=["redirect_to_forum", "numerical_accuracy"]
        )
        self.assertFalse(result.close)
        self.assertEqual(result.comment, "")
        self.assertEqual(result.add_labels, ["triage review", "bot-triaged"])

    def test_unknown_template_is_not_posted(self):
        result = effects("request_info", templates=["anything goes"])
        self.assertEqual(result.comment, TEMPLATES["request_more_info"]["comment"])
        self.assertEqual(result.add_labels, ["bot-triaged"])
        self.assertIn("dropped unknown templates ['anything goes']", result.notes)


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
