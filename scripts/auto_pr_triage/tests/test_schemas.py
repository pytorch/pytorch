from __future__ import annotations

import copy
import json
import unittest
from dataclasses import replace
from typing import Any

from schemas import (
    ActionPlan,
    AdditionalOwnerConcern,
    AddLabels,
    BypassIntakeMatch,
    IntakeFacts,
    json_schema,
    LLMInput,
    LLMResult,
    MAX_ADDITIONAL_OWNERS,
    MAX_CODEPATH_OWNERS,
    MISSING_ACTIONABLE_ISSUE_ACTIONS,
    OwnerMetadata,
    OwnershipResult,
    passes_intake,
    PathGroup,
    PlannerInput,
    PullRequestIdentity,
    RequestReviewers,
    RESULT_SCHEMA,
    ReviewerSnapshot,
)
from tests.stage_fixtures import (
    additional_owner_concern,
    bypass_match,
    concern,
    discarded_bypass_concern,
    intake_facts,
    intake_result,
    llm_input,
    llm_result,
    make_action_plan,
    make_ownership_result,
    REPOSITORY,
    uncovered_concern,
    WORKER_POLICY,
)


class IntakeFactsTest(unittest.TestCase):
    def test_passes_intake_is_derived_from_explicit_gate_facts(self) -> None:
        def should_run(
            *,
            target: bool = True,
            handled: bool = False,
            author: bool = False,
            issue: bool = False,
            activity: bool = False,
        ) -> bool:
            return passes_intake(
                is_open_non_draft_pr_against_main=target,
                is_already_handled=handled,
                author_has_triage_permission=author,
                has_actionable_linked_issue=issue,
                has_maintainer_activity=activity,
            )

        self.assertFalse(should_run())
        self.assertFalse(should_run(target=False, issue=True))
        self.assertTrue(should_run(author=True))
        self.assertTrue(should_run(issue=True))
        self.assertTrue(should_run(activity=True))
        self.assertTrue(should_run(author=True, issue=True))
        self.assertFalse(should_run(handled=True, author=True, issue=True))
        self.assertTrue(
            passes_intake(
                is_open_non_draft_pr_against_main=True,
                is_already_handled=False,
                author_has_triage_permission=False,
                has_actionable_linked_issue=False,
                has_maintainer_activity=False,
                has_supporter=True,
            )
        )
        self.assertTrue(
            passes_intake(
                is_open_non_draft_pr_against_main=True,
                is_already_handled=False,
                author_has_triage_permission=False,
                has_actionable_linked_issue=False,
                has_maintainer_activity=False,
                has_related_actionable_issue=True,
            )
        )

    def test_signals_are_only_valid_on_an_active_unhandled_pr(self) -> None:
        signals = (
            {"author_has_triage_permission": True},
            {"has_actionable_linked_issue": True},
            {"has_maintainer_activity": True},
            {"has_related_actionable_issue": True},
            {"has_supporter": True, "supporters": ["alice"]},
        )
        inactive_states = (
            {"is_already_handled": True},
            {"is_open_non_draft_pr_against_main": False},
        )
        for signal in signals:
            for state in inactive_states:
                facts = intake_facts(has_actionable_linked_issue=False).to_dict()
                facts.update(signal | state)
                with (
                    self.subTest(signal=signal, state=state),
                    self.assertRaisesRegex(RuntimeError, "inactive or handled PR"),
                ):
                    IntakeFacts.from_dict(facts)

        every_signal = intake_facts(has_actionable_linked_issue=False).to_dict()
        for signal in signals:
            every_signal.update(signal)
        facts = IntakeFacts.from_dict(every_signal)
        self.assertTrue(facts.passes_intake)
        self.assertTrue(facts.is_active)

    def test_invalid_intake_facts_are_rejected(self) -> None:
        cases = {
            "signals on an inactive or handled PR": [
                {
                    "has_supporter": True,
                    "supporters": ["a"],
                    "is_already_handled": True,
                },
            ],
            "has_supporter mismatches supporters": [
                {"has_supporter": True, "supporters": []},
                {"has_supporter": False, "supporters": ["alice"]},
            ],
            "supporters are not canonical logins": [
                {"has_supporter": True, "supporters": ["zed", "alice"]},
                {"has_supporter": True, "supporters": ["@alice"]},
                {"has_supporter": True, "supporters": ["alice", "ALICE"]},
                {"has_supporter": True, "supporters": ["a", "b", "c", "d"]},
            ],
            "actionable labelers without an issue": [
                {
                    "has_supporter": True,
                    "supporters": ["a"],
                    "actionable_labelers": ["b"],
                },
            ],
            "actionable_labelers are not canonical logins": [
                {
                    "has_related_actionable_issue": True,
                    "actionable_labelers": ["a", "b", "c", "d"],
                },
            ],
            "maintainer_requested_reviewers are not canonical logins": [
                {
                    "author_has_triage_permission": True,
                    "maintainer_requested_reviewers": ["@bob"],
                },
            ],
            "handoff reviewers without intake": [
                {"is_already_handled": True, "maintainer_requested_reviewers": ["bob"]},
            ],
            r"IntakeFacts\.is_already_handled \(int\) does not match bool": [
                {"is_already_handled": 0},
            ],
        }
        for message, overrides_list in cases.items():
            for overrides in overrides_list:
                facts = intake_facts(has_actionable_linked_issue=False).to_dict()
                facts.update(overrides)
                with (
                    self.subTest(overrides=overrides),
                    self.assertRaisesRegex(RuntimeError, message),
                ):
                    IntakeFacts.from_dict(facts)


class PullRequestIdentityTest(unittest.TestCase):
    def test_number_must_be_positive(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "number is not positive"):
            PullRequestIdentity(REPOSITORY, 0, "b" * 40, "a" * 40)


class LLMInputTest(unittest.TestCase):
    def test_codepath_artifact_covers_every_changed_path(self) -> None:
        prepared = llm_input(codepath_owners=["@owner", "@pytorch/team"])

        codepath = prepared.trusted_context.codepath_owners

        owners = ("@owner", "@pytorch/team")
        self.assertEqual(codepath.owners, owners)
        self.assertEqual(codepath.matched_path_groups, (PathGroup(owners, (0,)),))
        self.assertEqual(codepath.files_without_owners, ())

    def test_codepath_artifact_accepts_casefold_sorted_owners(self) -> None:
        prepared = llm_input(codepath_owners=["@albanD", "@Chillee"])

        self.assertEqual(
            prepared.trusted_context.codepath_owners.owners,
            ("@albanD", "@Chillee"),
        )

    def test_codepath_artifact_lists_files_without_owners(self) -> None:
        prepared = llm_input(codepath_owners=[])

        codepath = prepared.trusted_context.codepath_owners

        self.assertEqual(codepath.owners, ())
        self.assertEqual(codepath.matched_path_groups, ())
        self.assertEqual(codepath.files_without_owners, (0,))

    def test_codepath_artifact_rejects_incomplete_path_coverage(self) -> None:
        serialized = llm_input().to_dict()
        serialized["trusted_context"]["codepath_owners"]["owners"] = []
        serialized["trusted_context"]["codepath_owners"]["matched_path_groups"] = []

        with self.assertRaisesRegex(RuntimeError, "does not cover all changed paths"):
            LLMInput.from_dict(serialized)

    def test_codepath_artifact_rejects_owner_not_found_in_groups(self) -> None:
        serialized = llm_input().to_dict()
        serialized["trusted_context"]["codepath_owners"]["owners"].insert(0, "@extra")

        with self.assertRaisesRegex(RuntimeError, "do not match path groups"):
            LLMInput.from_dict(serialized)

    def test_llm_input_round_trip_preserves_analysis_artifacts(self) -> None:
        input_snapshot = llm_input()
        serialized = input_snapshot.to_dict()

        self.assertEqual(LLMInput.from_dict(serialized), input_snapshot)
        self.assertEqual(
            set(serialized["trusted_context"]),
            {
                "worker_policy",
                "codepath_owners",
                "extra_ownership_metadata",
                "diff_truncated_or_unavailable",
            },
        )

    def test_llm_input_rejects_unknown_fields(self) -> None:
        sections = ("trusted_context", "untrusted_context", "file")
        for name in sections:
            serialized = llm_input().to_dict()
            record = {
                "trusted_context": serialized["trusted_context"],
                "untrusted_context": serialized["untrusted_context"],
                "file": serialized["untrusted_context"]["files"][0],
            }[name]
            record["reviewer"] = "attacker"
            with (
                self.subTest(section=name),
                self.assertRaisesRegex(RuntimeError, "fields are invalid"),
            ):
                LLMInput.from_dict(serialized)


class LLMResultTest(unittest.TestCase):
    def test_result_schema_closes_every_object(self) -> None:
        def objects(schema: dict[str, Any]) -> list[dict[str, Any]]:
            if "anyOf" in schema:
                return [o for option in schema["anyOf"] for o in objects(option)]
            found = [schema] if schema["type"] == "object" else []
            children = schema.get("properties", {}).values()
            if schema["type"] == "array":
                children = [schema["items"]]
            return found + [o for child in children for o in objects(child)]

        found = objects(RESULT_SCHEMA)
        # The LLM result; each of the three concern records with its concern
        # and evidence; and the additional owner's bypass match with its evidence.
        self.assertEqual(len(found), 12)
        for schema in found:
            self.assertFalse(schema["additionalProperties"])
            self.assertEqual(set(schema["required"]), set(schema["properties"]))

    def test_rejects_fields_outside_the_schema(self) -> None:
        value = llm_result(additional_owners=["owner"])
        value["additional_owner_concerns"][0]["reviewer"] = "attacker"

        with self.assertRaisesRegex(RuntimeError, "unexpected \\['reviewer'\\]"):
            LLMResult.from_dict(value)

    def test_json_round_trip_keeps_all_three_concern_kinds(self) -> None:
        value = llm_result(
            additional_owners=["owner"],
            bypass=["owner"],
            codepath_owner_concerns=[
                {
                    "concern": concern(
                        file="torch/file.py",
                        description="Changes a codepath-owned API.",
                    ),
                    "codepath_owners": ["@pytorch/baseline"],
                    "reason": "The codepath owners review this API surface.",
                }
            ],
            uncovered_concerns=[
                {
                    "description": "Changes behavior no owner describes.",
                    "reason": "No metadata entry covers this contract.",
                    "files": ["torch/file.py"],
                }
            ],
        )
        parsed = LLMResult.from_dict(value)

        self.assertEqual(LLMResult.from_json(parsed.to_json()), parsed)
        self.assertEqual(parsed.to_dict(), value)
        self.assertEqual(
            [len(parsed.codepath_owner_concerns), len(parsed.uncovered_concerns)],
            [1, 1],
        )
        self.assertIsNotNone(parsed.additional_owner_concerns[0].bypass_intake_match)

    def test_worker_policy_example_parses(self) -> None:
        text = WORKER_POLICY
        start = text.index("Return only one JSON object:")
        block = text[start : text.index("Do not include Markdown")]
        example = "\n".join(line[4:] for line in block.splitlines()[1:])
        value = json.loads(example.replace("high | medium | low", "high"))

        parsed = LLMResult.from_dict(value)

        plain, bypass = parsed.additional_owner_concerns
        self.assertEqual(plain.confidence, "high")
        self.assertIsNone(plain.bypass_intake_match)
        self.assertIsInstance(bypass.bypass_intake_match, BypassIntakeMatch)
        self.assertEqual(len(parsed.codepath_owner_concerns), 1)

    def test_optional_bypass_match_accepts_null_or_its_record(self) -> None:
        value = llm_result(additional_owners=["owner"], bypass=["owner"])
        owner = value["additional_owner_concerns"][0]

        parsed = AdditionalOwnerConcern.from_dict(owner)
        self.assertEqual(
            parsed.bypass_intake_match,
            BypassIntakeMatch.from_dict(bypass_match(file="torch/file.py")),
        )
        without = AdditionalOwnerConcern.from_dict(
            {**owner, "bypass_intake_match": None}
        )
        self.assertIsNone(without.bypass_intake_match)
        message = "bypass_intake_match|BypassIntakeMatch"
        for invalid in ("yes", True, ["PRs that fix incorrect gradients."], {}):
            with (
                self.subTest(invalid=invalid),
                self.assertRaisesRegex(RuntimeError, message),
            ):
                AdditionalOwnerConcern.from_dict(
                    {**owner, "bypass_intake_match": invalid}
                )

    def test_json_schema_emits_nullable_records_only_for_optional_values(self) -> None:
        schema = json_schema(hint=BypassIntakeMatch | None)

        self.assertEqual(schema["anyOf"][0], json_schema(hint=BypassIntakeMatch))
        self.assertEqual(schema["anyOf"][1], {"type": "null"})
        with self.assertRaisesRegex(TypeError, "only optional values"):
            json_schema(hint=str | int)


def owners(*names: str) -> dict[str, tuple[str, ...]]:
    return {name: ("torch/file.py",) for name in names}


def concerns(*team_owner_ids: str) -> tuple[AdditionalOwnerConcern, ...]:
    return tuple(additional_owner_concern(owner) for owner in team_owner_ids)


class OwnershipResultTest(unittest.TestCase):
    def test_create_canonicalizes_and_round_trips_exact_shape(self) -> None:
        result = OwnershipResult.create(
            llm_run_status="succeeded",
            codepath_owners={
                "@pytorch/Zeta": ["torch/b.py", "torch/a.py"],
                "@alice": ["torch/a.py"],
            },
            additional_owner_concerns=concerns("zeta", "alpha"),
        )

        self.assertEqual(tuple(result.codepath_owners), ("@alice", "@pytorch/Zeta"))
        self.assertEqual(
            result.codepath_owners["@pytorch/Zeta"], ("torch/a.py", "torch/b.py")
        )
        self.assertEqual(result.additional_owners, ("alpha", "zeta"))
        self.assertEqual(
            set(result.to_dict()),
            {
                "llm_run_status",
                "codepath_owners",
                "codepath_owner_concerns",
                "additional_owner_concerns",
                "discarded_additional_owner_concerns",
                "uncovered_concerns",
            },
        )
        self.assertNotIn("\n", result.to_json())
        self.assertEqual(OwnershipResult.from_json(result.to_json()), result)

    def test_flags_are_derived_from_the_concern_lists(self) -> None:
        result = OwnershipResult.create(
            llm_run_status="succeeded",
            additional_owner_concerns=(
                additional_owner_concern("zeta", bypass=True),
                additional_owner_concern("alpha"),
                additional_owner_concern("beta", bypass=True),
            ),
        )
        self.assertEqual(result.bypass_intake_matches, ("beta", "zeta"))
        self.assertFalse(result.has_uncovered_concerns)
        self.assertFalse(result.has_discarded_bypass_intake_match)

        low = replace(additional_owner_concern("gamma"), confidence="low")
        flagged = OwnershipResult.create(
            llm_run_status="succeeded",
            discarded_additional_owner_concerns=(low,),
            uncovered_concerns=(uncovered_concern(),),
        )
        self.assertTrue(flagged.has_uncovered_concerns)
        self.assertFalse(flagged.has_discarded_bypass_intake_match)
        with_bypass = replace(
            flagged, discarded_additional_owner_concerns=(discarded_bypass_concern(),)
        )
        self.assertTrue(with_bypass.has_discarded_bypass_intake_match)
        self.assertEqual(OwnershipResult.from_json(with_bypass.to_json()), with_bypass)

    def test_rejects_invalid_ownership_state(self) -> None:
        cases = [
            ("unknown", {}, {}, "llm_run_status"),
            ("completed", {}, {}, "llm_run_status"),
            ("skipped", owners("@owner"), {}, "skipped LLM run"),
            ("skipped", {}, {"additional_owner_concerns": concerns("owner")}, ""),
            ("failed", {}, {"additional_owner_concerns": concerns("owner")}, ""),
            ("failed", {}, {"uncovered_concerns": (uncovered_concern(),)}, ""),
            (
                "failed",
                {},
                {"discarded_additional_owner_concerns": (discarded_bypass_concern(),)},
                "",
            ),
        ]
        for state, codepath, lists, message in cases:
            with (
                self.subTest(state=state, lists=list(lists)),
                self.assertRaisesRegex(
                    ValueError, message or "only a succeeded LLM run carries concerns"
                ),
            ):
                OwnershipResult(state, codepath, **lists)

    def test_rejects_noncanonical_or_untrusted_owners(self) -> None:
        low = replace(additional_owner_concern("owner"), confidence="low")
        cases = [
            (owners("not.valid"), (), "invalid codepath owner"),
            (owners("autograd"), (), "invalid codepath owner"),
            (owners("@bob", "@Alice"), (), "codepath owners are not canonical"),
            (owners("@Alice", "@alice"), (), "codepath owners are not canonical"),
            ({"@alice": ()}, (), "codepath owner has no files"),
            ({}, concerns("Bad-Team"), "invalid additional owner"),
            ({}, concerns("@owner"), "invalid additional owner"),
            ({}, concerns("zeta", "alpha"), "additional owners are not canonical"),
            ({}, concerns("alpha", "alpha"), "additional owners are not canonical"),
            ({}, (low,), "low-confidence owner"),
        ]
        for codepath, additional, message in cases:
            with (
                self.subTest(message=message, codepath=codepath),
                self.assertRaisesRegex(ValueError, message),
            ):
                OwnershipResult(
                    "succeeded", codepath, additional_owner_concerns=additional
                )

    def test_from_dict_requires_the_exact_unversioned_shape(self) -> None:
        valid = make_ownership_result(
            codepath_owners=["@owner"], additional_owners=["owner"]
        ).to_dict()
        cases = [
            {**valid, "unexpected": True},
            {k: v for k, v in valid.items() if k != "additional_owner_concerns"},
            {k: v for k, v in valid.items() if k != "uncovered_concerns"},
            {**valid, "schema_version": 1},
            {**valid, "codepath_owners": ["@owner"]},
            {**valid, "codepath_owners": {"@owner": "torch/file.py"}},
            {**valid, "additional_owner_concerns": "owner"},
            {**valid, "uncovered_concerns": "false"},
        ]
        for serialized in cases:
            with self.subTest(serialized=serialized), self.assertRaises(ValueError):
                OwnershipResult.from_dict(serialized)

        malformed = copy.deepcopy(valid)
        del malformed["additional_owner_concerns"][0]["concern"]["evidence"]
        with self.assertRaisesRegex(RuntimeError, "Concern fields are invalid"):
            OwnershipResult.from_dict(malformed)

    def test_from_json_rejects_invalid_json(self) -> None:
        with self.assertRaisesRegex(ValueError, "not valid JSON"):
            OwnershipResult.from_json("{not-json")

    def test_from_dict_rejects_noncanonical_owner_order(self) -> None:
        value = make_ownership_result(codepath_owners=["@alice", "@bob"]).to_dict()
        value["codepath_owners"] = dict(reversed(value["codepath_owners"].items()))

        with self.assertRaisesRegex(ValueError, "not canonical"):
            OwnershipResult.from_dict(value)

    def test_rejects_owner_count_limits(self) -> None:
        codepath = owners(
            *(f"@user{index:02}" for index in range(MAX_CODEPATH_OWNERS + 1))
        )
        additional = concerns(
            *(f"owner{index:02}" for index in range(MAX_ADDITIONAL_OWNERS + 1))
        )

        with self.assertRaisesRegex(ValueError, "too many codepath owners"):
            OwnershipResult("succeeded", codepath)
        with self.assertRaisesRegex(ValueError, "too many additional owners"):
            OwnershipResult("succeeded", {}, additional_owner_concerns=additional)


class ActionPlanTest(unittest.TestCase):
    def plan(self, **overrides: Any) -> ActionPlan:
        facts = overrides.pop("facts", intake_facts())
        return make_action_plan(facts, **overrides)

    def test_plan_rejects_effects_outside_its_bounds(self) -> None:
        cases = (
            ({"decision": "kept_open"}, "label outside its decision"),
            ({"labels": ("owner: compiler",)}, "label outside its decision"),
            ({"labels": ("bot-closed",)}, "label outside its decision"),
            ({"supporter_reviewers": ("stranger",)}, "unverified supporter"),
            ({"roster_reviewers": ("zed", "alice")}, "not canonical logins"),
            ({"decision": "missing_actionable_issue"}, "marks an admitted PR"),
            ({"decision": "close"}, "unknown decision"),
            ({"run_attempt": 0}, "invalid run attempt"),
            (
                {
                    "facts": intake_facts(has_actionable_linked_issue=False),
                    "run_attempt": 2,
                    "codepath_owners": (),
                    "additional_owners": (),
                    "decision": "missing_actionable_issue",
                },
                "marks a PR on a rerun",
            ),
            (
                {"facts": intake_facts(is_already_handled=True)},
                "inactive or handled PR",
            ),
            (
                {"bypass_intake_matches": ("nn",)},
                "bypass intake match is not an owner",
            ),
        )
        for overrides, message in cases:
            with (
                self.subTest(overrides=overrides),
                self.assertRaisesRegex(RuntimeError, message),
            ):
                self.plan(**overrides)

    def test_pr_that_fails_intake_is_routed_only_through_a_bypass(self) -> None:
        fails_intake = intake_facts(has_actionable_linked_issue=False)
        rejected = (
            (
                {
                    "codepath_owners": (),
                    "bypass_intake_matches": ("autograd",),
                    "decision": "missing_actionable_issue",
                },
                "marks an admitted PR",
            ),
            ({}, "routes a PR that was not admitted"),
            (
                {
                    "decision": "incomplete",
                    "roster_reviewers": ("alice",),
                    "labels": ("bot-triage-error",),
                },
                "routes a PR that was not admitted",
            ),
            (
                {"decision": "routed_untriaged", "labels": ()},
                "routes a PR that was not admitted",
            ),
        )
        for overrides, message in rejected:
            with (
                self.subTest(overrides=overrides),
                self.assertRaisesRegex(RuntimeError, message),
            ):
                self.plan(facts=fails_intake, **overrides)

        accepted = (
            {"decision": "incomplete", "labels": ("bot-triage-error",)},
            {"decision": "kept_open", "labels": ()},
            {
                "codepath_owners": (),
                "additional_owners": (),
                "decision": "missing_actionable_issue",
            },
            {"bypass_intake_matches": ("autograd",), "roster_reviewers": ("alice",)},
        )
        for overrides in accepted:
            with self.subTest(overrides=overrides):
                plan = self.plan(facts=fails_intake, **overrides)
                self.assertEqual(plan.decision, overrides.get("decision", "triage"))

    def test_actions_must_follow_the_fixed_order_and_shapes(self) -> None:
        fails_intake = intake_facts(has_actionable_linked_issue=False)
        labels = AddLabels(("triaged", "bot-triaged"))
        owners = RequestReviewers(("alice",), "owner_roster")
        supporter = RequestReviewers(("alice",), "supporter")
        with_alice = intake_facts(supporters=("alice",))
        many = tuple(sorted(f"user{index:02d}" for index in range(16)))
        cases = (
            (
                {
                    "facts": fails_intake,
                    "codepath_owners": (),
                    "additional_owners": (),
                    "decision": "missing_actionable_issue",
                    "actions": (labels,),
                },
                "not the fixed set",
            ),
            ({"actions": (labels, owners)}, "request reviewers, then add labels"),
            ({"actions": (labels, labels)}, "request reviewers, then add labels"),
            ({"actions": (owners, owners)}, "repeated or unordered"),
            ({"facts": with_alice, "actions": (owners, supporter)}, "unordered"),
            ({"facts": with_alice, "actions": (supporter, owners)}, "reviewer twice"),
            (
                {"actions": (RequestReviewers(("bob",), "actionable_labeler"),)},
                "unverified actionable_labeler",
            ),
            (
                {"actions": (RequestReviewers((), "owner_roster"),)},
                "not canonical logins",
            ),
            (
                {"actions": (RequestReviewers(many, "owner_roster"),)},
                "review request limit",
            ),
        )
        for overrides, message in cases:
            with (
                self.subTest(overrides=overrides),
                self.assertRaisesRegex(RuntimeError, message),
            ):
                self.plan(**overrides)

    def test_action_kind_is_fixed_per_record(self) -> None:
        for build in (
            lambda: AddLabels(("triaged",), kind="close"),
            lambda: RequestReviewers(("alice",), "owner_roster", kind="add_labels"),
        ):
            with self.assertRaisesRegex(RuntimeError, "wrong kind"):
                build()

    def test_plans_cannot_close_or_comment(self) -> None:
        plan = self.plan(
            facts=intake_facts(has_actionable_linked_issue=False),
            codepath_owners=(),
            additional_owners=(),
            decision="missing_actionable_issue",
        )
        for action in ({"kind": "close"}, {"kind": "comment", "template": "x"}):
            with (
                self.subTest(kind=action["kind"]),
                self.assertRaisesRegex(RuntimeError, "does not match"),
            ):
                ActionPlan.from_dict({**plan.to_dict(), "actions": [action]})

    def test_request_reason_must_be_known(self) -> None:
        for reason in ("handoff", "codepath_owner", "friends"):
            with (
                self.subTest(reason=reason),
                self.assertRaisesRegex(RuntimeError, "reason"),
            ):
                RequestReviewers(("alice",), reason)

    def test_json_round_trip_decodes_each_action_by_kind(self) -> None:
        routed = self.plan(
            facts=intake_facts(supporters=("alice",)),
            supporter_reviewers=("alice",),
            roster_reviewers=("bob",),
        )
        marked = self.plan(
            facts=intake_facts(has_actionable_linked_issue=False),
            codepath_owners=(),
            additional_owners=(),
            decision="missing_actionable_issue",
        )
        self.assertEqual(marked.actions, MISSING_ACTIONABLE_ISSUE_ACTIONS)
        for plan, kinds in (
            (routed, [RequestReviewers, RequestReviewers, AddLabels]),
            (marked, [AddLabels]),
        ):
            with self.subTest(decision=plan.decision):
                restored = ActionPlan.from_json(plan.to_json())
                self.assertEqual(restored, plan)
                self.assertEqual([type(action) for action in restored.actions], kinds)

        value = routed.to_dict()
        value["actions"][0]["kind"] = "delete_branch"
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            ActionPlan.from_dict(value)


class OwnerMetadataTest(unittest.TestCase):
    def test_description_and_optional_bypass_are_bounded_text(self) -> None:
        OwnerMetadata("Owns autograd.", None)
        OwnerMetadata("Owns autograd.", "PRs that fix gradient formulas.")
        for description, bypass in (
            ("  ", None),
            ("x" * 2_001, None),
            ("Owns autograd.", ""),
            ("Owns autograd.", "x" * 2_001),
        ):
            with (
                self.subTest(description=description[:10], bypass=bypass),
                self.assertRaisesRegex(RuntimeError, "entry is invalid"),
            ):
                OwnerMetadata(description, bypass)
        with self.assertRaisesRegex(RuntimeError, "does not match str"):
            OwnerMetadata("Owns autograd.", 1)  # type: ignore[arg-type]

    def test_trusted_context_reports_owners_with_bypass_criteria(self) -> None:
        trusted = llm_input(bypass={"owner": "PRs that fix gradients."})
        metadata = trusted.trusted_context.extra_ownership_metadata

        self.assertIsInstance(metadata["owner"], OwnerMetadata)
        self.assertEqual(
            trusted.trusted_context.teams_with_intake_bypass, frozenset({"owner"})
        )
        self.assertEqual(
            llm_input().trusted_context.teams_with_intake_bypass, frozenset()
        )


class ReviewerSnapshotTest(unittest.TestCase):
    def test_json_round_trip_keeps_rosters(self) -> None:
        snapshot = ReviewerSnapshot(
            existing_labels=("triaged",),
            missing_labels=("bot-closed",),
            requested_reviewers=("@alice",),
            submitted_reviewers=(),
            rosters={"autograd": ("@first", "@second")},
            errors={"rosters": "RuntimeError: unavailable"},
        )

        self.assertEqual(ReviewerSnapshot.from_json(snapshot.to_json()), snapshot)

    def test_rejects_malformed_snapshots(self) -> None:
        empty = ReviewerSnapshot((), (), None, None, None, {})
        cases = (
            ({**empty.to_dict(), "extra": 1}, "extra"),
            ({**empty.to_dict(), "round_robin": {}}, "round_robin"),
        )
        for value, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(RuntimeError, message),
            ):
                ReviewerSnapshot.from_dict(value)


class PlannerInputTest(unittest.TestCase):
    def test_rejects_inconsistent_stage_results(self) -> None:
        empty = ReviewerSnapshot((), (), None, None, None, {})
        active = intake_result()
        cases = (
            (make_ownership_result(llm_run_status="skipped"), 1, "intake decision"),
            (make_ownership_result(), 0, "invalid run attempt"),
        )
        for ownership, run_attempt, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                PlannerInput(active, ownership, empty, run_attempt)

    def test_json_round_trip(self) -> None:
        empty = ReviewerSnapshot((), (), None, None, None, {})
        value = PlannerInput(intake_result(), make_ownership_result(), empty, 2)

        self.assertEqual(PlannerInput.from_json(value.to_json()), value)


if __name__ == "__main__":
    unittest.main()
