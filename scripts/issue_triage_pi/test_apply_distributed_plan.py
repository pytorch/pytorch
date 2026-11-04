"""Behavior of the trusted plan-to-effects mapping for pi distributed triage."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from apply_distributed_plan import DECISION_STATUS, main, plan_effects
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
    def test_decision_requires_its_status_even_with_missing_or_conflicting_labels(self):
        for decision, required in (
            ("low_confidence", "triage review"),
            ("not_distributed", "triage review"),
            ("needs_reproduction", "needs reproduction"),
            ("high_priority", "triage review"),
        ):
            for labels in (
                [],
                ["oncall: distributed infra", "triaged"],
                ["triage-review"],
                ["triaged", "triage review", "needs reproduction"],
            ):
                with self.subTest(decision=decision, labels=labels):
                    result = effects(decision, labels)
                    self.assertEqual(
                        set(result.add_labels)
                        & {"triaged", "triage review", "needs reproduction"},
                        {required},
                    )
                    self.assertEqual(
                        MARKER in result.add_labels, decision != "high_priority"
                    )
                    self.assertFalse(result.close)

    def test_routing_ignores_model_status_but_preserves_existing_status(self):
        queue = "oncall: distributed infra"
        for status_label in ("triage review", "needs reproduction"):
            with self.subTest(status_label=status_label):
                routed = effects("route", [queue, status_label])
                self.assertEqual(routed.add_labels, [queue, "triaged", MARKER])
                self.assertEqual(routed.comment, "")
                held = effects("route", [queue], existing=ROUTED | {status_label})
                self.assertEqual(held.add_labels, [queue, MARKER])
                self.assertEqual(held.comment, "")

    def test_confident_decisions_complete_only_valid_classifications(self):
        for decision in ("route", "classify", "already_classified"):
            with self.subTest(decision=decision):
                complete = effects(
                    decision,
                    [
                        "oncall: distributed parallelisms",
                        "module: fsdp",
                        "needs reproduction",
                        "triage review",
                    ],
                )
                self.assertEqual(
                    set(complete.add_labels)
                    & {"triaged", "triage review", "needs reproduction"},
                    {"triaged"},
                )
                incomplete = effects(decision, ["oncall: distributed parallelisms"])
                self.assertNotIn("triaged", incomplete.add_labels)
                self.assertIn("triage review", incomplete.add_labels)
                missing_queue = effects(decision, ["module: fsdp"])
                self.assertNotIn("triaged", missing_queue.add_labels)
                self.assertIn("triage review", missing_queue.add_labels)

    def test_existing_triaged_does_not_hide_pending_outcome(self):
        for decision, outcome in (
            ("low_confidence", "review"),
            ("needs_reproduction", "waiting_on_reporter"),
        ):
            with self.subTest(decision=decision), tempfile.TemporaryDirectory() as temp:
                plan = Path(temp) / "plan.json"
                plan.write_text(json.dumps({"decision": decision, "labels": []}))
                issue = {"labels": [{"name": label} for label in ROUTED | {"triaged"}]}
                with (
                    patch(
                        "sys.argv",
                        [
                            "apply_distributed_plan",
                            "pytorch/ciforge",
                            "123",
                            str(plan),
                            "--apply",
                        ],
                    ),
                    patch(
                        "apply_distributed_plan.fetch_issue", return_value=(issue, [])
                    ),
                    patch("apply_distributed_plan.apply_effects") as apply,
                    patch("apply_distributed_plan.publish") as publish,
                ):
                    self.assertEqual(main(), 0)
                    self.assertEqual(apply.call_count, 1)
                    self.assertIn(f"→ {outcome}", publish.call_args.args[0])

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
        self.assertEqual(result.add_labels, ["module: fsdp", "triage review", MARKER])

    def test_an_existing_sub_queue_is_kept(self):
        existing = ROUTED | {"oncall: distributed infra"}
        result = effects(
            "route", ["oncall: distributed checkpointing"], existing=existing
        )
        self.assertEqual(result.add_labels, ["triaged", MARKER])

    def test_infra_and_checkpointing_routes_can_complete_without_modules(self):
        for queue in ("oncall: distributed infra", "oncall: distributed checkpointing"):
            with self.subTest(queue=queue):
                result = effects("route", [queue, "triaged"])
                self.assertEqual(result.add_labels, [queue, "triaged", MARKER])

    def test_existing_triaged_does_not_skip_second_level_routing_or_review(self):
        existing = ROUTED | {"triaged"}
        routed = effects("route", ["oncall: distributed infra"], existing=existing)
        self.assertEqual(routed.add_labels, ["oncall: distributed infra", MARKER])
        review = effects("low_confidence", ["triage review"], existing=existing)
        self.assertEqual(review.add_labels, ["triage review", MARKER])

    def test_completed_status_requires_a_valid_sub_queue(self):
        for decision in ("route", "classify", "already_classified"):
            with self.subTest(decision=decision):
                result = effects(
                    decision,
                    ["module: fsdp", "triaged"],
                    existing=ROUTED | {"oncall: distributed typo"},
                )
                self.assertEqual(
                    result.add_labels, ["module: fsdp", "triage review", MARKER]
                )

    def test_route_without_a_sub_queue_goes_to_review(self):
        result = effects("route", ["oncall: distributed typo"])
        self.assertEqual(result.add_labels, ["triage review", MARKER])

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
        for templates in ([], ["needs_distributed_reproduction"]):
            with self.subTest(templates=templates):
                repro = effects(
                    "needs_reproduction", ["needs reproduction", "triaged"], templates
                )
                self.assertEqual(repro.add_labels, ["needs reproduction", MARKER])
                self.assertEqual(
                    repro.comment,
                    TEMPLATES["needs_distributed_reproduction"]["comment"],
                )
                self.assertFalse(repro.close)

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

    def test_implicit_reproduction_request_is_not_repeated(self):
        posted = TEMPLATES["needs_distributed_reproduction"]["comment"]
        existing = ROUTED | {"needs reproduction"}
        rerun = effects("needs_reproduction", existing=existing, bot_comments=[posted])
        self.assertEqual(rerun.comment, "")
        self.assertEqual(rerun.comment_templates, [])
        self.assertEqual(rerun.add_labels, [MARKER])
        done = effects(
            "needs_reproduction", existing=existing | {MARKER}, bot_comments=[posted]
        )
        self.assertFalse(done.mutates)

    def test_decision_selects_the_comment_template(self):
        for decision, expected in (
            ("not_distributed", "not_distributed"),
            ("needs_reproduction", "needs_distributed_reproduction"),
            ("classify", None),
            ("route", None),
            ("already_classified", None),
            ("low_confidence", None),
            ("high_priority", None),
        ):
            for requested in ([], list(TEMPLATES)):
                with self.subTest(decision=decision, requested=requested):
                    result = effects(
                        decision,
                        ["oncall: distributed parallelisms", "module: dtensor"],
                        requested,
                    )
                    self.assertEqual(
                        result.comment,
                        TEMPLATES[expected]["comment"] if expected else "",
                    )
                    self.assertEqual(
                        result.comment_templates, [expected] if expected else []
                    )
                    self.assertFalse(result.close)

    def test_discarding_a_template_preserves_existing_review(self):
        result = effects(
            "classify",
            ["oncall: distributed parallelisms", "module: dtensor"],
            ["not_distributed"],
            existing=ROUTED | {"triage review"},
        )
        self.assertEqual(result.comment, "")
        self.assertEqual(
            result.add_labels,
            ["oncall: distributed parallelisms", "module: dtensor", MARKER],
        )

    def test_classification_without_a_distributed_module_goes_to_review(self):
        for existing_modules, requested_modules in (
            (set(), ["module: typo"]),
            ({"module: cuda"}, ["module: typo"]),
            ({"module: performance"}, []),
            (set(), ["module: performance"]),
        ):
            with self.subTest(existing=existing_modules, requested=requested_modules):
                result = effects(
                    "classify",
                    [*requested_modules, "triaged"],
                    existing=ROUTED
                    | {"oncall: distributed parallelisms"}
                    | existing_modules,
                )
                self.assertNotIn("triaged", result.add_labels)
                self.assertIn("triage review", result.add_labels)
                self.assertIn(MARKER, result.add_labels)

    def test_parallelisms_completion_requires_a_module_for_every_decision(self):
        for decision in ("route", "already_classified"):
            with self.subTest(decision=decision):
                result = effects(
                    decision,
                    ["oncall: distributed parallelisms", "triaged"],
                )
                self.assertNotIn("triaged", result.add_labels)
                self.assertIn("triage review", result.add_labels)

    def test_schema_matches_the_templates_and_the_high_priority_decision(self):
        self.assertEqual(set(SCHEMA["templates"]["items"]["enum"]), set(TEMPLATES))
        self.assertEqual(set(DECISION_STATUS), set(SCHEMA["decision"]["enum"]))

    def test_nothing_is_ever_closed(self):
        self.assertFalse(
            effects("not_distributed", ["triage review"], ["not_distributed"]).close
        )


if __name__ == "__main__":
    unittest.main()
