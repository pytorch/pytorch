from __future__ import annotations

import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from apply_actions import ApplyOutcome
from plan_actions import (
    build_action_plan,
    EngagementReason,
    gather_planner_input,
    main as plan_main,
    new_reviewer_requests,
    OwnerChoice,
    ReviewerState,
    TeamOwnerReviewers,
)
from reviewer_state import stable_fallback_member
from schemas import (
    ActionPlan,
    AdditionalOwnerConcern,
    AddLabels,
    BOT_TRIAGE_ERROR_LABEL,
    CLOSE_ACTIONS,
    PlannerInput,
    RequestReviewers,
)
from tests.plan_fixtures import (
    close_args,
    configured,
    FakeGitHub,
    HEAD_SHA,
    mutations,
    ownership_config,
    plan_and_apply,
    plan_pr,
    printed_plan,
    printed_record,
    printed_reviewer_routing,
    run_apply,
    run_args,
    run_plan,
    run_without_owners,
    scenario_args,
    stage_results,
    WORKFLOW_SHA,
)


class StageInputTest(unittest.TestCase):
    def test_rejects_foreign_codepath_team_before_github_io(self) -> None:
        github = FakeGitHub()
        args = run_args(stage_results=stage_results(codepath_owners=("@other/team",)))

        with (
            configured(),
            self.assertRaisesRegex(ValueError, "foreign codepath owner team"),
        ):
            plan_pr(args=args, github=github)

        self.assertEqual((github.calls, github.live_reads), ([], []))

    def test_inactive_target_result_is_a_read_free_noop(self) -> None:
        github = FakeGitHub()
        args = run_args(
            stage_results=stage_results(
                is_open_non_draft_pr_against_main=False,
                has_actionable_linked_issue=False,
                llm_run_status="skipped",
                codepath_owners=(),
            )
        )

        self.assertEqual(
            plan_and_apply(args=args, github=github), ApplyOutcome("kept_open")
        )
        self.assertEqual(github.calls, [])

    def test_already_handled_result_is_a_read_free_noop(self) -> None:
        github = FakeGitHub()
        args = run_args(
            stage_results=stage_results(
                is_already_handled=True,
                has_actionable_linked_issue=False,
                llm_run_status="skipped",
                codepath_owners=(),
            )
        )

        self.assertEqual(
            plan_and_apply(args=args, github=github), ApplyOutcome("kept_open")
        )
        self.assertEqual(github.calls, [])

    def test_incomplete_result_without_codepath_owners_is_labeled(self) -> None:
        github = FakeGitHub()

        self.assertEqual(
            run_without_owners(github, llm_run_status="failed"),
            ApplyOutcome("incomplete"),
        )
        self.assertEqual(
            mutations(github),
            [
                (
                    "POST",
                    "repos/pytorch/ciforge/issues/123/labels",
                    {"labels": [BOT_TRIAGE_ERROR_LABEL]},
                )
            ],
        )

    def test_rejects_analysis_result_from_another_pull_request_or_run(self) -> None:
        for args in (
            run_args(pr=124),
            run_args(repository="pytorch/pytorch"),
            run_args(workflow_sha="d" * 40),
        ):
            github = FakeGitHub()
            with (
                self.subTest(args=args),
                self.assertRaisesRegex(ValueError, "another pull request or run"),
            ):
                plan_pr(args=args, github=github)
            self.assertEqual(github.calls, [])

    def test_rejects_ownership_result_that_contradicts_intake(self) -> None:
        cases = (
            stage_results(llm_run_status="skipped", codepath_owners=()),
            stage_results(
                has_actionable_linked_issue=False,
                llm_run_status="skipped",
                codepath_owners=(),
            ),
            stage_results(is_already_handled=True),
        )
        for results in cases:
            github = FakeGitHub()
            with (
                self.subTest(ownership=results[1].llm_run_status),
                self.assertRaisesRegex(ValueError, "does not match the intake"),
            ):
                plan_pr(args=run_args(stage_results=results), github=github)
            self.assertEqual((github.calls, github.live_reads), ([], []))

    def test_bot_author_login_is_valid(self) -> None:
        github = FakeGitHub()
        args = run_args(
            stage_results=stage_results(
                author_login="dependabot[bot]",
                is_open_non_draft_pr_against_main=False,
                llm_run_status="skipped",
                codepath_owners=(),
            ),
        )

        self.assertEqual(
            plan_and_apply(args=args, github=github), ApplyOutcome("kept_open")
        )
        self.assertEqual(github.calls, [])


class CodeownersComparisonTest(unittest.TestCase):
    def test_matching_native_requests_are_logged_without_being_requested(self) -> None:
        github = FakeGitHub()

        with mock.patch("builtins.print") as output:
            result = run_apply(github, native_codeowners_requests=True)

        self.assertEqual(result, ApplyOutcome("triaged"))
        self.assertEqual(
            printed_record(output=output, label="Auto PR Triage CODEOWNERS comparison")[
                "status"
            ],
            "match",
        )
        self.assertEqual(
            mutations(github),
            [
                (
                    "POST",
                    "repos/pytorch/ciforge/issues/123/labels",
                    {"labels": ["triaged", "bot-triaged"]},
                )
            ],
        )

    def test_different_native_requests_are_logged_as_mismatch(self) -> None:
        github = FakeGitHub(native_codeowners=[])

        with mock.patch("builtins.print") as output:
            result = run_apply(github, native_codeowners_requests=True)

        self.assertEqual(result, ApplyOutcome("triaged"))
        self.assertEqual(
            printed_record(output=output, label="Auto PR Triage CODEOWNERS comparison"),
            {
                "error": None,
                "expected": ["@codepath-owner"],
                "missing_from_github": ["@codepath-owner"],
                "observed": [],
                "oracle": "active_review_requests_as_code_owner",
                "status": "mismatch",
                "unexpected_from_github": [],
                "workflow_sha": WORKFLOW_SHA,
            },
        )
        self.assertEqual(
            mutations(github)[0][2], {"labels": ["triaged", "bot-triaged"]}
        )

    def test_inconclusive_comparison_does_not_block_semantic_addition(self) -> None:
        github = FakeGitHub(codeowner_error=RuntimeError("unavailable"))

        with mock.patch("builtins.print") as output:
            result = run_apply(
                github,
                additional_owners=("autograd",),
                native_codeowners_requests=True,
            )

        self.assertEqual(result, ApplyOutcome("triaged", 1, 1))
        self.assertEqual(
            printed_record(output=output, label="Auto PR Triage CODEOWNERS comparison")[
                "status"
            ],
            "inconclusive",
        )
        self.assertEqual(mutations(github)[0][2], {"reviewers": ["soulitzer"]})
        self.assertEqual(
            mutations(github)[1][2],
            {"labels": ["triaged", "bot-triaged", "owner: autograd"]},
        )

    def test_native_mode_never_requests_native_user_or_team(self) -> None:
        github = FakeGitHub()

        result = run_apply(
            github,
            codepath_owners=("@codepath-owner", "@pytorch/compiler"),
            native_codeowners_requests=True,
        )

        self.assertEqual(result, ApplyOutcome("triaged"))
        self.assertFalse(
            any(call[1].endswith("/requested_reviewers") for call in mutations(github))
        )

    def test_parser_only_owner_does_not_suppress_semantic_addition(self) -> None:
        github = FakeGitHub(native_codeowners=[])
        result = run_apply(
            github,
            codepath_owners=("@pytorch/autograd",),
            additional_owners=("autograd",),
            native_codeowners_requests=True,
        )

        self.assertEqual(result, ApplyOutcome("triaged", 1, 1))
        self.assertEqual(mutations(github)[0][2], {"reviewers": ["soulitzer"]})

    def test_native_team_handle_does_not_suppress_semantic_addition(self) -> None:
        github = FakeGitHub(native_codeowners=["@pytorch/autograd"])

        result = run_apply(
            github,
            codepath_owners=(),
            additional_owners=("autograd",),
            native_codeowners_requests=True,
        )

        self.assertEqual(result, ApplyOutcome("triaged", 1, 1))
        self.assertEqual(mutations(github)[0][2], {"reviewers": ["soulitzer"]})

    def test_comparison_excludes_the_author_from_expected_requests(self) -> None:
        github = FakeGitHub(native_codeowners=[])

        with mock.patch("builtins.print") as output:
            result = run_apply(
                github,
                codepath_owners=("@external-author",),
                native_codeowners_requests=True,
            )

        self.assertEqual(result, ApplyOutcome("triaged"))
        self.assertEqual(
            printed_record(output=output, label="Auto PR Triage CODEOWNERS comparison")[
                "status"
            ],
            "match",
        )


class PlanOnlyTest(unittest.TestCase):
    def test_close_plan_makes_no_writes(self) -> None:
        github = FakeGitHub(labels=["open source"])

        plan = plan_pr(args=close_args(), github=github)

        self.assertEqual((plan.decision, plan.actions), ("close", CLOSE_ACTIONS))
        self.assertEqual(mutations(github), [])
        self.assertEqual(github.pr["state"], "open")

    def test_close_plan_on_rerun_keeps_open(self) -> None:
        github = FakeGitHub(labels=["open source"])

        plan = plan_pr(args=close_args(run_attempt=2), github=github)

        self.assertEqual((plan.decision, plan.actions), ("kept_open", ()))
        self.assertEqual(mutations(github), [])

    def test_triage_plan_makes_no_writes(self) -> None:
        github = FakeGitHub()

        plan = run_plan(github)

        self.assertEqual(plan.decision, "triage")
        self.assertEqual(plan.actions, (AddLabels(("triaged", "bot-triaged")),))
        self.assertEqual(mutations(github), [])

    def test_uncovered_concerns_route_found_owners_without_triaging(self) -> None:
        github = FakeGitHub()
        with mock.patch("builtins.print") as output:
            result = run_apply(
                github,
                codepath_owners=(),
                additional_owners=("autograd",),
                has_uncovered_concerns=True,
            )

        self.assertEqual(result, ApplyOutcome("routed_untriaged", 1, 1))
        plan = printed_plan(output)
        self.assertEqual(plan["decision"], "routed_untriaged")
        self.assertTrue(plan["has_uncovered_concerns"])
        self.assertEqual(plan["planned_reviewer_requests"], ["@soulitzer"])
        self.assertEqual(
            mutations(github),
            [
                (
                    "POST",
                    "repos/pytorch/ciforge/pulls/123/requested_reviewers",
                    {"reviewers": ["soulitzer"]},
                ),
                (
                    "POST",
                    "repos/pytorch/ciforge/issues/123/labels",
                    {"labels": ["owner: autograd"]},
                ),
            ],
        )

    def test_review_activity_does_not_triage_uncovered_concerns(self) -> None:
        github = FakeGitHub(submitted_users=["reviewer"])
        with mock.patch("builtins.print") as output:
            result = run_apply(
                github,
                codepath_owners=(),
                additional_owners=("autograd",),
                has_uncovered_concerns=True,
            )

        self.assertEqual(result, ApplyOutcome("routed_untriaged", 1, 1))
        self.assertEqual(printed_plan(output)["decision"], "routed_untriaged")

    def test_uncovered_concern_ignores_unneeded_review_state_failure(self) -> None:
        class UnavailableReviewsGitHub(FakeGitHub):
            def graphql(
                self, *, query: str, variables: dict[str, Any]
            ) -> dict[str, Any]:
                if "reviews(first:" in query:
                    raise RuntimeError("submitted reviews unavailable")
                return super().graphql(query=query, variables=variables)

        for native_codeowners_requests, expected in (
            (True, ApplyOutcome("routed_untriaged")),
            (False, ApplyOutcome("routed_untriaged", 1)),
        ):
            github = UnavailableReviewsGitHub()
            with self.subTest(native_codeowners_requests=native_codeowners_requests):
                result = run_apply(
                    github,
                    has_uncovered_concerns=True,
                    native_codeowners_requests=native_codeowners_requests,
                )

            self.assertEqual(result, expected)
            self.assertFalse(
                any(
                    BOT_TRIAGE_ERROR_LABEL in payload["labels"]
                    for _, endpoint, payload in mutations(github)
                    if endpoint.endswith("/labels")
                )
            )

    def test_no_destination_stays_open_despite_review_activity(self) -> None:
        for has_uncovered_concerns in (False, True):
            github = FakeGitHub(submitted_users=["soulitzer"])
            with self.subTest(has_uncovered_concerns=has_uncovered_concerns):
                self.assertEqual(
                    run_without_owners(
                        github, has_uncovered_concerns=has_uncovered_concerns
                    ),
                    ApplyOutcome("kept_open"),
                )
            self.assertNotIn("submitted", github.live_reads)
            self.assertEqual(mutations(github), [])

    def test_plan_log_and_step_summary(self) -> None:
        github = FakeGitHub()
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / "summary.md"
            with mock.patch("builtins.print") as output:
                result = run_apply(
                    github,
                    additional_owners=("autograd",),
                    github_step_summary=summary,
                )
            summary_text = summary.read_text()
        plan = printed_plan(output)

        self.assertEqual(result.status, "triaged")
        self.assertNotIn("mode", plan)
        self.assertEqual(plan["analyzed_head_sha"], HEAD_SHA)
        self.assertNotIn("- Mode:", summary_text)
        self.assertIn("`soulitzer`", summary_text)
        self.assertNotIn("@soulitzer", summary_text)
        self.assertIn(
            "### Why this PR was admitted\n\n"
            "Admitted because it fixes an issue labeled `actionable`.\n",
            summary_text,
        )
        self.assertIn(
            "- **`soulitzer`** (new request): No previous `autograd` assignment was "
            "found, so its round-robin rotation began with `soulitzer`.\n"
            "  - Semantic owner `autograd`: autograd owns this changed behavior.\n",
            summary_text,
        )
        self.assertIn(
            "Planned effect: one deduplicated request for @soulitzer.",
            printed_reviewer_routing(output),
        )

    def test_step_summary_failure_does_not_block_planning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / "missing" / "summary.md"
            github = FakeGitHub()
            with mock.patch("builtins.print") as output:
                result = run_apply(github, github_step_summary=summary)

        self.assertEqual(result.status, "triaged")
        self.assertTrue(mutations(github))
        self.assertTrue(
            any(
                str(call.args[0]).startswith("Auto PR Triage step summary unavailable:")
                for call in output.call_args_list
            )
        )

    def test_plan_escapes_workflow_commands_in_llm_provenance(self) -> None:
        concern = AdditionalOwnerConcern.from_dict(
            {
                "concern": {
                    "description": "Autograd owns this changed behavior.",
                    "files": ["torch/semantic.py"],
                    "evidence": [
                        {
                            "file": "torch/semantic.py",
                            "diff_excerpt": "+\x1b[31m::notice:: changed behavior",
                            "relevance": "##[warning] This explains the relevant change.",
                        }
                    ],
                },
                "owner_id": "autograd",
                "rationale": [
                    "::warning:: This is provenance, not a workflow command.",
                    "##[error] This is provenance, not a workflow command.",
                    "The configured description covers this changed contract.",
                ],
                "confidence": "high",
                "bypass_intake_match": None,
            }
        )
        with mock.patch("builtins.print") as output:
            run_plan(
                FakeGitHub(),
                codepath_owners=(),
                additional_owners=("autograd",),
                additional_owner_concerns=(concern,),
            )

        prefix = "Auto PR Triage plan:\n"
        raw_plan = next(
            call.args[0]
            for call in output.call_args_list
            if call.args[0].startswith(prefix)
        )
        self.assertNotIn("::warning::", raw_plan)
        self.assertNotIn("##[error]", raw_plan)
        self.assertIn(
            "::warning::",
            printed_plan(output)["owner_choices"]["autograd"]["provenance"][
                "rationale"
            ][0],
        )
        routing = printed_reviewer_routing(output)
        self.assertNotIn("::notice::", routing)
        self.assertNotIn("##[warning]", routing)
        self.assertNotIn("\x1b", routing)
        self.assertIn(r"\u001b", routing)
        self.assertIn(r"\u003a\u003anotice\u003a\u003a", routing)
        self.assertIn(r"\u0023\u0023[warning]", routing)

    def test_pending_owner_is_planned_as_label_without_request(self) -> None:
        github = FakeGitHub(requested_users=["soulitzer"])

        with mock.patch("builtins.print") as output:
            plan = run_plan(
                github,
                codepath_owners=(),
                additional_owners=("autograd",),
            )

        self.assertEqual(plan.decision, "triage")
        self.assertEqual(
            plan.actions, (AddLabels(("triaged", "bot-triaged", "owner: autograd")),)
        )
        self.assertEqual(mutations(github), [])
        self.assertEqual(
            sum(
                endpoint.endswith("labels/owner%3A%20autograd")
                for method, endpoint, _ in github.calls
                if method == "GET"
            ),
            1,
        )
        printed = printed_plan(output)
        choice = printed["owner_choices"]["autograd"]
        self.assertEqual(choice["reviewer"], "@soulitzer")
        self.assertEqual(choice["state"], "pending")
        self.assertEqual(choice["provenance"]["source"], "semantic")
        self.assertEqual(printed["planned_reviewer_requests"], [])

    def test_missing_pending_owner_label_degrades_to_incomplete(self) -> None:
        github = FakeGitHub(
            requested_users=["soulitzer"],
            unavailable_labels=["owner: autograd"],
        )
        result = run_apply(
            github,
            codepath_owners=(),
            additional_owners=("autograd",),
        )

        self.assertEqual(result, ApplyOutcome("incomplete"))
        self.assertEqual(mutations(github)[0][2]["labels"][0], BOT_TRIAGE_ERROR_LABEL)

    def test_multiple_owners_deduplicate_the_proposed_reviewer(self) -> None:
        github = FakeGitHub()
        team_members = copy.deepcopy(ownership_config()["team_members"])
        team_members["members"]["compiler"] = ["@soulitzer"]

        with mock.patch("builtins.print") as output:
            plan = run_plan(
                github,
                codepath_owners=(),
                additional_owners=("autograd", "compiler"),
                team_members=team_members,
            )

        self.assertEqual(
            plan.actions[0], RequestReviewers(("soulitzer",), (), "owner_roster")
        )
        self.assertEqual(mutations(github), [])
        printed = printed_plan(output)
        self.assertEqual(printed["planned_reviewer_requests"], ["@soulitzer"])
        self.assertEqual(set(printed["owner_choices"]), {"autograd", "compiler"})
        self.assertEqual(
            {choice["reviewer"] for choice in printed["owner_choices"].values()},
            {"@soulitzer"},
        )
        routing = printed_reviewer_routing(output)
        self.assertEqual(routing.count("Reviewer @soulitzer"), 1)
        self.assertIn("Semantic owner `autograd`", routing)
        self.assertIn("Semantic owner `compiler`", routing)
        self.assertEqual(routing.count("Planned effect:"), 1)


class ApplyTriageTest(unittest.TestCase):
    def test_native_codepath_owner_is_not_requested_again(self) -> None:
        github = FakeGitHub()

        with mock.patch("builtins.print") as output:
            result = run_apply(github)

        self.assertEqual(result, ApplyOutcome("triaged"))
        self.assertEqual(
            mutations(github),
            [
                (
                    "POST",
                    "repos/pytorch/ciforge/issues/123/labels",
                    {
                        "labels": [
                            "triaged",
                            "bot-triaged",
                        ]
                    },
                )
            ],
        )
        self.assertEqual(
            github.live_reads,
            ["native_codeowners"],
        )
        choice = printed_plan(output)["owner_choices"]["@codepath-owner"]
        self.assertEqual(choice["state"], "native_codeowner")
        routing = printed_reviewer_routing(output)
        self.assertIn(
            "GitHub already has an active native CODEOWNERS request for @codepath-owner.",
            routing,
        )
        self.assertIn("Codepath owner `@codepath-owner`", routing)

    def test_direct_codepath_owner_does_not_load_roster_configuration(self) -> None:
        github = FakeGitHub(actionable_issue=True)
        args = run_args()

        error = AssertionError("roster configuration should not load")
        with (
            mock.patch("plan_actions.load_team_members", side_effect=error),
            mock.patch("apply_actions.load_team_members", side_effect=error),
            mock.patch(
                "plan_actions.NATIVE_CODEOWNERS_REQUESTS_CODEPATH_OWNERS", False
            ),
        ):
            result = plan_and_apply(args=args, github=github)

        self.assertEqual(result, ApplyOutcome("triaged", 1, 0))

    def test_failed_llm_run_preserves_codepath_owners(self) -> None:
        github = FakeGitHub()
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / "summary.md"
            with mock.patch("builtins.print") as output:
                result = run_apply(
                    github, llm_run_status="failed", github_step_summary=summary
                )
            summary_text = summary.read_text()

        self.assertEqual(result, ApplyOutcome("incomplete"))
        reason = "The LLM run failed; only codepath owners remain."
        self.assertEqual(printed_plan(output)["incomplete_reasons"], [reason])
        self.assertIn(f"### Why this run is incomplete\n\n- {reason}\n", summary_text)
        self.assertIn(
            f"Why this run is incomplete: {reason}", printed_reviewer_routing(output)
        )
        self.assertEqual(
            mutations(github)[0][2],
            {
                "labels": [
                    BOT_TRIAGE_ERROR_LABEL,
                ]
            },
        )

    def test_incomplete_analysis_resolves_internal_codepath_owner(self) -> None:
        github = FakeGitHub()

        result = run_apply(
            github,
            codepath_owners=("autograd",),
            llm_run_status="failed",
            native_codeowners_requests=False,
        )

        self.assertEqual(result, ApplyOutcome("incomplete", 1, 1))
        self.assertEqual(mutations(github)[0][2], {"reviewers": ["soulitzer"]})
        self.assertEqual(
            mutations(github)[1][2],
            {
                "labels": [
                    BOT_TRIAGE_ERROR_LABEL,
                    "owner: autograd",
                ]
            },
        )

    def test_fresh_round_robin_reviewer_is_requested_and_labeled(self) -> None:
        github = FakeGitHub()

        with mock.patch("builtins.print") as output:
            result = run_apply(
                github,
                codepath_owners=(),
                additional_owners=("autograd",),
            )

        self.assertEqual(result, ApplyOutcome("triaged", 1, 1))
        self.assertEqual(
            mutations(github),
            [
                (
                    "POST",
                    "repos/pytorch/ciforge/pulls/123/requested_reviewers",
                    {"reviewers": ["soulitzer"]},
                ),
                (
                    "POST",
                    "repos/pytorch/ciforge/issues/123/labels",
                    {
                        "labels": [
                            "triaged",
                            "bot-triaged",
                            "owner: autograd",
                        ]
                    },
                ),
            ],
        )
        self.assertEqual(
            sum(
                endpoint.endswith("labels/owner%3A%20autograd")
                for method, endpoint, _ in github.calls
                if method == "GET"
            ),
            1,
        )
        choice = printed_plan(output)["owner_choices"]["autograd"]
        self.assertEqual(choice["reviewer"], "@soulitzer")
        self.assertEqual(choice["selection_reason"], "round_robin_initial")
        concern = choice["provenance"]["concern"]
        self.assertEqual(choice["provenance"]["source"], "semantic")
        self.assertEqual(concern["files"], ["torch/semantic.py"])
        self.assertEqual(
            concern["evidence"],
            [
                {
                    "file": "torch/semantic.py",
                    "diff_excerpt": "+new semantic behavior",
                    "relevance": "This line implements the owned behavior.",
                }
            ],
        )
        self.assertEqual(concern["description"], "autograd owns this changed behavior.")

    def test_two_member_roster_bootstraps_first_member(self) -> None:
        github = FakeGitHub()
        team_members = copy.deepcopy(ownership_config()["team_members"])
        team_members["members"]["autograd"] = ["@first", "@second"]

        result = run_apply(
            github,
            codepath_owners=(),
            additional_owners=("autograd",),
            team_members=team_members,
        )

        self.assertEqual(result, ApplyOutcome("triaged", 1, 1))
        self.assertEqual(mutations(github)[0][2], {"reviewers": ["first"]})

    def test_two_member_roster_advances_from_history(self) -> None:
        github = FakeGitHub(
            round_robin_events=[
                {
                    "id": 91,
                    "event": "labeled",
                    "label": {"name": "owner: autograd"},
                    "issue": {"number": 7, "pull_request": {}},
                }
            ],
            round_robin_timelines={
                7: [
                    {
                        "id": 90,
                        "event": "review_requested",
                        "requested_reviewer": {"login": "first"},
                    },
                    {"id": 91, "event": "labeled"},
                ]
            },
        )
        team_members = copy.deepcopy(ownership_config()["team_members"])
        team_members["members"]["autograd"] = ["@first", "@second"]

        with mock.patch("builtins.print") as output:
            result = run_apply(
                github,
                codepath_owners=(),
                additional_owners=("autograd",),
                team_members=team_members,
            )

        self.assertEqual(result, ApplyOutcome("triaged", 1, 1))
        self.assertEqual(mutations(github)[0][2], {"reviewers": ["second"]})
        self.assertEqual(
            printed_plan(output)["owner_choices"]["autograd"]["selection_reason"],
            "round_robin_next",
        )
        self.assertIn(
            "@second was the next eligible member of `autograd`'s round-robin "
            "rotation after @first, who was assigned on #7.",
            printed_reviewer_routing(output),
        )

    def test_unassigned_owner_marker_uses_fallback_and_repairs_state(self) -> None:
        github = FakeGitHub(
            round_robin_events=[
                {
                    "id": 91,
                    "event": "labeled",
                    "label": {"name": "owner: autograd"},
                    "issue": {"number": 7, "pull_request": {}},
                }
            ],
            round_robin_timelines={7: [{"id": 91, "event": "labeled"}]},
        )
        team_members = copy.deepcopy(ownership_config()["team_members"])
        team_members["members"]["autograd"] = ["@first", "@second"]
        expected = stable_fallback_member(
            repo="pytorch/ciforge",
            current_number=123,
            owner="autograd",
            members=("@first", "@second"),
            ineligible_reviewers={"@external-author"},
        )

        with mock.patch("builtins.print") as output:
            result = run_apply(
                github,
                codepath_owners=(),
                additional_owners=("autograd",),
                team_members=team_members,
            )

        self.assertEqual(result, ApplyOutcome("triaged", 1, 1))
        self.assertEqual(mutations(github)[0][2], {"reviewers": [expected[1:]]})
        self.assertIn("owner: autograd", mutations(github)[1][2]["labels"])
        self.assertEqual(
            printed_plan(output)["owner_choices"]["autograd"]["selection_reason"],
            "stable_fallback",
        )
        self.assertIn("the stable fallback chose", printed_reviewer_routing(output))

    def test_two_member_roster_wraps_after_second_member(self) -> None:
        github = FakeGitHub(
            round_robin_events=[
                {
                    "id": 91,
                    "event": "labeled",
                    "label": {"name": "owner: autograd"},
                    "issue": {"number": 7, "pull_request": {}},
                }
            ],
            round_robin_timelines={
                7: [
                    {
                        "id": 90,
                        "event": "review_requested",
                        "requested_reviewer": {"login": "second"},
                    },
                    {"id": 91, "event": "labeled"},
                ]
            },
        )
        team_members = copy.deepcopy(ownership_config()["team_members"])
        team_members["members"]["autograd"] = ["@first", "@second"]

        result = run_apply(
            github,
            codepath_owners=(),
            additional_owners=("autograd",),
            team_members=team_members,
        )

        self.assertEqual(result, ApplyOutcome("triaged", 1, 1))
        self.assertEqual(mutations(github)[0][2], {"reviewers": ["first"]})

    def test_internal_codepath_owner_uses_roster_and_round_robin(self) -> None:
        github = FakeGitHub()

        with mock.patch("builtins.print") as output:
            result = run_apply(
                github,
                codepath_owners=("autograd",),
                native_codeowners_requests=False,
            )

        self.assertEqual(result, ApplyOutcome("triaged", 1, 1))
        self.assertEqual(
            mutations(github),
            [
                (
                    "POST",
                    "repos/pytorch/ciforge/pulls/123/requested_reviewers",
                    {"reviewers": ["soulitzer"]},
                ),
                (
                    "POST",
                    "repos/pytorch/ciforge/issues/123/labels",
                    {
                        "labels": [
                            "triaged",
                            "bot-triaged",
                            "owner: autograd",
                        ]
                    },
                ),
            ],
        )
        plan = printed_plan(output)
        self.assertEqual(plan["codepath_owners"], ["autograd"])
        choice = plan["owner_choices"]["autograd"]
        self.assertEqual(choice["reviewer"], "@soulitzer")
        self.assertEqual(choice["selection_reason"], "round_robin_initial")
        self.assertEqual(choice["provenance"]["source"], "codepath")
        self.assertEqual(choice["provenance"]["files"], ["torch/file.py"])

    def test_native_codepath_owner_and_additional_owner_share_reviewer(self) -> None:
        github = FakeGitHub(native_codeowners=[])

        with mock.patch("builtins.print") as output:
            result = run_apply(
                github,
                codepath_owners=("@soulitzer",),
                additional_owners=("autograd",),
            )

        self.assertEqual(result, ApplyOutcome("triaged", 1, 1))
        request = next(
            call
            for call in mutations(github)
            if call[1].endswith("/requested_reviewers")
        )
        self.assertEqual(request[2], {"reviewers": ["soulitzer"]})
        plan = printed_plan(output)
        self.assertEqual(plan["codepath_owners"], ["@soulitzer"])
        self.assertEqual(set(plan["owner_choices"]), {"autograd"})
        self.assertEqual(
            plan["owner_choices"]["autograd"]["selection_reason"],
            "round_robin_initial",
        )
        self.assertEqual(plan["planned_reviewer_requests"], ["@soulitzer"])

    def test_existing_codepath_owner_requests_are_preserved(self) -> None:
        github = FakeGitHub(
            requested_teams=["compiler"],
            submitted_users=["codepath-owner"],
        )

        with mock.patch("builtins.print") as output:
            result = run_apply(
                github,
                codepath_owners=("@codepath-owner", "@pytorch/compiler"),
                native_codeowners_requests=False,
            )

        self.assertEqual(result, ApplyOutcome("triaged", 0, 0))
        self.assertFalse(
            any(call[1].endswith("/requested_reviewers") for call in mutations(github))
        )
        choices = printed_plan(output)["owner_choices"]
        self.assertEqual(choices["@codepath-owner"]["state"], "submitted")
        self.assertEqual(choices["@pytorch/compiler"]["state"], "pending")

    def test_large_codepath_owner_set_allows_no_new_requests(self) -> None:
        reviewers = tuple(f"owner{index}" for index in range(16))
        github = FakeGitHub(requested_users=list(reviewers))

        result = run_apply(
            github,
            codepath_owners=tuple(f"@{reviewer}" for reviewer in reviewers),
        )

        self.assertEqual(result, ApplyOutcome("triaged", 0, 0, 0))
        self.assertFalse(
            any(call[1].endswith("/requested_reviewers") for call in mutations(github))
        )

    def test_codepath_owner_policy_does_not_request_the_author(self) -> None:
        github = FakeGitHub()

        result = run_apply(
            github,
            codepath_owners=("@external-author",),
        )

        self.assertEqual(result, ApplyOutcome("triaged", 0, 0))
        self.assertFalse(
            any(call[1].endswith("/requested_reviewers") for call in mutations(github))
        )

    def test_native_codepath_owners_do_not_consume_the_request_limit(self) -> None:
        github = FakeGitHub()
        codepath_owners = tuple(f"@owner{index}" for index in range(15))

        result = run_apply(
            github,
            codepath_owners=codepath_owners,
            additional_owners=("compiler",),
        )

        self.assertEqual(result, ApplyOutcome("triaged", 1, 1))
        self.assertEqual(mutations(github)[0][2], {"reviewers": ["reviewer"]})

    def test_pending_owner_member_gets_assignment_label(self) -> None:
        github = FakeGitHub(requested_users=["soulitzer"])

        result = run_apply(
            github,
            codepath_owners=(),
            additional_owners=("autograd",),
        )

        self.assertEqual(result, ApplyOutcome("triaged", 0, 1))
        label_post = next(
            call for call in mutations(github) if call[1].endswith("/labels")
        )
        self.assertEqual(
            label_post[2],
            {
                "labels": [
                    "triaged",
                    "bot-triaged",
                    "owner: autograd",
                ]
            },
        )

    def test_pending_and_new_additional_owners_get_assignment_labels(self) -> None:
        github = FakeGitHub(requested_users=["soulitzer"])

        result = run_apply(
            github,
            codepath_owners=(),
            additional_owners=("autograd", "compiler"),
        )

        self.assertEqual(result, ApplyOutcome("triaged", 1, 2))
        label_post = next(
            call for call in mutations(github) if call[1].endswith("/labels")
        )
        self.assertEqual(
            label_post[2],
            {
                "labels": [
                    "triaged",
                    "bot-triaged",
                    "owner: autograd",
                    "owner: compiler",
                ]
            },
        )

    def test_two_fresh_owners_request_two_reviewers(self) -> None:
        github = FakeGitHub()

        result = run_apply(
            github,
            codepath_owners=(),
            additional_owners=("autograd", "compiler"),
        )

        self.assertEqual(result, ApplyOutcome("triaged", 2, 2))
        self.assertEqual(
            mutations(github)[0][2],
            {"reviewers": ["reviewer", "soulitzer"]},
        )
        self.assertEqual(
            mutations(github)[1][2],
            {
                "labels": [
                    "triaged",
                    "bot-triaged",
                    "owner: autograd",
                    "owner: compiler",
                ]
            },
        )

    def test_two_owners_deduplicate_shared_rotation_member(self) -> None:
        github = FakeGitHub(
            round_robin_events=[
                {
                    "id": 91,
                    "event": "labeled",
                    "label": {"name": "owner: autograd"},
                    "issue": {"number": 7, "pull_request": {}},
                }
            ],
            round_robin_timelines={
                7: [
                    {
                        "id": 90,
                        "event": "review_requested",
                        "requested_reviewer": {"login": "soulitzer"},
                    },
                    {"id": 91, "event": "labeled"},
                ]
            },
        )
        team_members = copy.deepcopy(ownership_config()["team_members"])
        team_members["members"]["autograd"] = ["@soulitzer", "@izaitsevfb"]
        team_members["members"]["nn"] = ["@izaitsevfb"]

        result = run_apply(
            github,
            codepath_owners=(),
            additional_owners=("autograd", "nn"),
            team_members=team_members,
        )

        self.assertEqual(result, ApplyOutcome("triaged", 1, 2))
        self.assertEqual(
            mutations(github)[0][2],
            {"reviewers": ["izaitsevfb"]},
        )
        self.assertEqual(
            mutations(github)[1][2],
            {
                "labels": [
                    "triaged",
                    "bot-triaged",
                    "owner: autograd",
                    "owner: nn",
                ]
            },
        )

    def test_submitted_owner_member_needs_no_request_or_routing_label(self) -> None:
        github = FakeGitHub(submitted_users=["soulitzer"])

        result = run_apply(
            github,
            codepath_owners=(),
            additional_owners=("autograd",),
        )

        self.assertEqual(result, ApplyOutcome("triaged"))
        self.assertFalse(
            any(call[1].endswith("/requested_reviewers") for call in mutations(github))
        )
        self.assertFalse(
            any("owner: autograd" in str(call[2]) for call in mutations(github))
        )

    def test_unknown_additional_owner_marks_routing_incomplete(self) -> None:
        github = FakeGitHub()

        result = run_apply(
            github,
            codepath_owners=(),
            additional_owners=("unknown",),
        )

        self.assertEqual(result, ApplyOutcome("incomplete"))
        self.assertEqual(
            mutations(github)[0][2],
            {
                "labels": [
                    BOT_TRIAGE_ERROR_LABEL,
                ]
            },
        )

    def test_codepath_team_handle_does_not_suppress_additional_owner(self) -> None:
        github = FakeGitHub(actionable_issue=True)
        config = ownership_config()
        args = run_args(
            stage_results=stage_results(
                codepath_owners=("@pytorch/autograd",),
                additional_owners=("autograd",),
            )
        )

        with (
            configured(
                team_members=config["team_members"], native_codeowners_requests=False
            ),
            mock.patch("builtins.print") as output,
        ):
            result = plan_and_apply(args=args, github=github)
        self.assertEqual(result, ApplyOutcome("triaged", 1, 1, 1))
        self.assertEqual(mutations(github)[0][2], {"team_reviewers": ["autograd"]})
        self.assertEqual(mutations(github)[1][2], {"reviewers": ["soulitzer"]})
        choices = printed_plan(output)["owner_choices"]
        self.assertEqual(choices["autograd"]["provenance"]["source"], "semantic")
        self.assertEqual(
            choices["@pytorch/autograd"]["selection_reason"],
            "direct_codepath_owner",
        )
        self.assertEqual(
            choices["@pytorch/autograd"]["provenance"]["source"],
            "codepath",
        )

    def test_missing_live_reviewer_is_selected_at_plan_time(self) -> None:
        github = FakeGitHub()

        result = run_apply(
            github,
            codepath_owners=(),
            additional_owners=("autograd",),
        )
        self.assertEqual(result, ApplyOutcome("triaged", 1, 1))
        self.assertEqual(mutations(github)[0][2], {"reviewers": ["soulitzer"]})

    def test_missing_status_label_prevents_writes(self) -> None:
        github = FakeGitHub(unavailable_labels=["bot-triaged"])

        with self.assertRaisesRegex(RuntimeError, "label.*unavailable"):
            run_apply(
                github,
                codepath_owners=(),
                additional_owners=("autograd",),
            )

        self.assertEqual(mutations(github), [])

    def test_missing_routing_label_degrades_to_incomplete(self) -> None:
        github = FakeGitHub(unavailable_labels=["owner: autograd"])

        result = run_apply(
            github,
            codepath_owners=(),
            additional_owners=("autograd",),
        )

        self.assertEqual(result, ApplyOutcome("incomplete"))
        self.assertEqual(
            mutations(github)[0][2],
            {
                "labels": [
                    BOT_TRIAGE_ERROR_LABEL,
                ]
            },
        )

    def test_missing_error_label_prevents_fallback_write(self) -> None:
        github = FakeGitHub(unavailable_labels=[BOT_TRIAGE_ERROR_LABEL])

        with self.assertRaisesRegex(RuntimeError, "label.*unavailable"):
            run_apply(github, llm_run_status="failed")

        self.assertEqual(mutations(github), [])

    def test_internal_codepath_owner_requires_roster_and_label(self) -> None:
        cases = (
            ("unknown", [], "not configured"),
            ("autograd", ["owner: autograd"], "label.*unavailable"),
        )
        for owner, unavailable_labels, error in cases:
            github = FakeGitHub(unavailable_labels=unavailable_labels)
            with (
                self.subTest(owner=owner),
                self.assertRaisesRegex((ValueError, RuntimeError), error),
            ):
                run_apply(
                    github,
                    codepath_owners=(owner,),
                    native_codeowners_requests=False,
                )
            self.assertEqual(mutations(github), [])

    def test_unavailable_additional_owner_routing_preserves_codepath_handles(
        self,
    ) -> None:
        github = FakeGitHub(unavailable_labels=["owner: autograd"])

        with mock.patch("builtins.print") as output:
            result = run_apply(
                github,
                codepath_owners=("@codepath-owner",),
                additional_owners=("autograd",),
            )

        self.assertEqual(result, ApplyOutcome("incomplete"))
        self.assertEqual(printed_plan(output)["unresolved_owners"], ["autograd"])
        self.assertEqual(
            mutations(github),
            [
                (
                    "POST",
                    "repos/pytorch/ciforge/issues/123/labels",
                    {
                        "labels": [
                            BOT_TRIAGE_ERROR_LABEL,
                        ]
                    },
                ),
            ],
        )

    def test_unavailable_reviewer_state_preserves_codepath_handles(self) -> None:
        class UnavailableReviewersGitHub(FakeGitHub):
            def json(
                self,
                endpoint: str,
                *,
                method: str = "GET",
                payload: dict[str, Any] | None = None,
            ) -> Any:
                if endpoint.endswith("/requested_reviewers") and method == "GET":
                    raise RuntimeError("reviewer state unavailable")
                return super().json(endpoint, method=method, payload=payload)

        github = UnavailableReviewersGitHub()
        with mock.patch("builtins.print") as output:
            result = run_apply(
                github,
                codepath_owners=("@codepath-owner",),
                additional_owners=("autograd",),
            )

        self.assertEqual(result, ApplyOutcome("incomplete"))
        self.assertEqual(printed_plan(output)["unresolved_owners"], ["autograd"])
        self.assertEqual(
            mutations(github)[0][2],
            {
                "labels": [
                    BOT_TRIAGE_ERROR_LABEL,
                ]
            },
        )

    def test_unavailable_reviewer_state_does_not_drop_internal_codepath_owner(
        self,
    ) -> None:
        class UnavailableReviewersGitHub(FakeGitHub):
            def json(
                self,
                endpoint: str,
                *,
                method: str = "GET",
                payload: dict[str, Any] | None = None,
            ) -> Any:
                if endpoint.endswith("/requested_reviewers") and method == "GET":
                    raise RuntimeError("reviewer state unavailable")
                return super().json(endpoint, method=method, payload=payload)

        for codepath_owners in (("autograd",), ("@codepath-owner", "autograd")):
            github = UnavailableReviewersGitHub()
            with (
                self.subTest(codepath_owners=codepath_owners),
                self.assertRaisesRegex(RuntimeError, "reviewer state unavailable"),
            ):
                run_apply(
                    github,
                    codepath_owners=codepath_owners,
                    native_codeowners_requests=False,
                )
            self.assertEqual(mutations(github), [])

    def test_apply_deliberately_trusts_analysis_result(self) -> None:
        github = FakeGitHub(labels=["triaged", "bot-triaged"])
        github.pr.update(state="closed", draft=True, title="changed")
        github.pr["head"]["sha"] = "c" * 40
        github.pr["base"]["ref"] = "release"

        self.assertEqual(run_apply(github).status, "triaged")
        self.assertEqual(github.pr_fetches, 0)
        self.assertEqual(github.actionable_checks, 0)
        self.assertEqual(github.permission_checks, 0)

    def test_reviewer_state_is_fetched_once_without_race_recheck(self) -> None:
        github = FakeGitHub()

        self.assertEqual(
            run_apply(
                github,
                codepath_owners=(),
                additional_owners=("autograd",),
            ).status,
            "triaged",
        )
        self.assertEqual(github.live_reads.count("requested"), 1)
        self.assertEqual(github.live_reads.count("submitted"), 1)

    def test_no_owners_with_review_activity_stays_open_without_writes(self) -> None:
        cases = (
            FakeGitHub(requested_users=["soulitzer"], actionable_issue=True),
            FakeGitHub(
                submitted_users=["soulitzer"],
                submitted_user_state="DISMISSED",
                actionable_issue=True,
            ),
            FakeGitHub(submitted_users=["outsider"], actionable_issue=True),
            FakeGitHub(
                submitted_users=["soulitzer"],
                submitted_user_state="COMMENTED",
                actionable_issue=True,
            ),
        )
        for github in cases:
            with self.subTest(calls=github.calls):
                self.assertEqual(run_without_owners(github), ApplyOutcome("kept_open"))
            self.assertEqual(mutations(github), [])

    def test_author_cannot_be_selected_by_round_robin(self) -> None:
        github = FakeGitHub()
        config = ownership_config()
        config["team_members"]["members"]["autograd"] = ["@external-author"]
        args = run_args(
            stage_results=stage_results(
                codepath_owners=(),
                additional_owners=("autograd",),
            )
        )
        with (
            configured(
                team_members=config["team_members"], native_codeowners_requests=False
            ),
            mock.patch("builtins.print") as output,
        ):
            result = plan_and_apply(args=args, github=github)

        self.assertEqual(result, ApplyOutcome("incomplete"))
        self.assertEqual(
            mutations(github),
            [
                (
                    "POST",
                    "repos/pytorch/ciforge/issues/123/labels",
                    {"labels": [BOT_TRIAGE_ERROR_LABEL]},
                )
            ],
        )
        plan = printed_plan(output)
        self.assertEqual(plan["owner_choices"], {})
        self.assertEqual(plan["planned_reviewer_requests"], [])
        self.assertEqual(plan["unresolved_owners"], ["autograd"])
        self.assertEqual(
            plan["incomplete_reasons"], ["No eligible roster member for: autograd."]
        )

    def test_unresolved_owner_preserves_other_owner_selection(self) -> None:
        team_members = copy.deepcopy(ownership_config()["team_members"])
        team_members["members"]["autograd"] = ["@external-author"]
        for native_codeowners_requests in (True, False):
            github = FakeGitHub()
            with (
                self.subTest(native_codeowners_requests=native_codeowners_requests),
                tempfile.TemporaryDirectory() as directory,
                mock.patch("builtins.print") as output,
            ):
                summary = Path(directory) / "summary.md"
                result = run_apply(
                    github,
                    codepath_owners=(),
                    additional_owners=("autograd", "compiler"),
                    native_codeowners_requests=native_codeowners_requests,
                    team_members=team_members,
                    github_step_summary=summary,
                )

                self.assertEqual(result, ApplyOutcome("incomplete", 1, 1))
                plan = printed_plan(output)
                self.assertEqual(set(plan["owner_choices"]), {"compiler"})
                self.assertEqual(
                    plan["owner_choices"]["compiler"]["reviewer"], "@reviewer"
                )
                self.assertEqual(plan["planned_reviewer_requests"], ["@reviewer"])
                self.assertEqual(plan["unresolved_owners"], ["autograd"])
                self.assertIn("- Unresolved owners: `autograd`", summary.read_text())

            self.assertEqual(
                mutations(github),
                [
                    (
                        "POST",
                        "repos/pytorch/ciforge/pulls/123/requested_reviewers",
                        {"reviewers": ["reviewer"]},
                    ),
                    (
                        "POST",
                        "repos/pytorch/ciforge/issues/123/labels",
                        {"labels": [BOT_TRIAGE_ERROR_LABEL, "owner: compiler"]},
                    ),
                ],
            )

    def test_author_codepath_match_does_not_cover_additional_owner(self) -> None:
        github = FakeGitHub(actionable_issue=True)
        config = ownership_config()
        config["team_members"]["members"]["autograd"].append("@external-author")
        args = run_args(
            stage_results=stage_results(
                codepath_owners=("@external-author",),
                additional_owners=("autograd",),
            )
        )

        with configured(
            team_members=config["team_members"], native_codeowners_requests=False
        ):
            result = plan_and_apply(args=args, github=github)

        self.assertEqual(result, ApplyOutcome("triaged", 1, 1))
        self.assertEqual(mutations(github)[0][2], {"reviewers": ["soulitzer"]})

    def test_triage_trusts_analyzed_triage_facts(self) -> None:
        github = FakeGitHub()
        args = run_args()

        with configured(native_codeowners_requests=False):
            self.assertEqual(
                plan_and_apply(args=args, github=github), ApplyOutcome("triaged", 1, 0)
            )
        self.assertEqual(github.actionable_checks, 0)
        self.assertEqual(github.permission_checks, 0)

    def test_codepath_roster_member_suppresses_additional_owner(self) -> None:
        github = FakeGitHub()
        with mock.patch("builtins.print") as output:
            result = run_apply(
                github,
                codepath_owners=("@soulitzer",),
                additional_owners=("autograd",),
            )
        self.assertEqual(result, ApplyOutcome("triaged"))
        self.assertFalse(
            any("owner: autograd" in str(call[2]) for call in mutations(github))
        )
        routing = printed_reviewer_routing(output)
        self.assertIn(
            "@soulitzer covers `autograd` through codepath ownership; "
            "no `autograd` round-robin choice was needed.",
            routing,
        )
        self.assertIn("Semantic owner `autograd`", routing)
        self.assertLess(
            routing.index("Codepath owner `@soulitzer`"),
            routing.index("Semantic owner `autograd`"),
        )
        self.assertIn(
            "Planned effect: no new Auto PR Triage reviewer request is needed.",
            routing,
        )

    def test_ambiguous_reviewer_request_stops_before_labels(self) -> None:
        class TimeoutGitHub(FakeGitHub):
            def json(
                self,
                endpoint: str,
                *,
                method: str = "GET",
                payload: dict[str, Any] | None = None,
            ) -> Any:
                if endpoint.endswith("/requested_reviewers") and method == "POST":
                    raise subprocess.TimeoutExpired("gh api", 60)
                return super().json(endpoint, method=method, payload=payload)

        github = TimeoutGitHub()
        with self.assertRaisesRegex(
            RuntimeError, "owner_roster reviewer request failed"
        ):
            run_apply(
                github,
                codepath_owners=(),
                additional_owners=("compiler",),
            )
        self.assertFalse(any(call[1].endswith("/labels") for call in mutations(github)))

    def test_successful_user_request_is_not_refetched(self) -> None:
        class EmptyResponseGitHub(FakeGitHub):
            def json(
                self,
                endpoint: str,
                *,
                method: str = "GET",
                payload: dict[str, Any] | None = None,
            ) -> Any:
                if endpoint.endswith("/requested_reviewers") and method == "POST":
                    self.calls.append((method, endpoint, payload))
                    return {"requested_reviewers": []}
                return super().json(endpoint, method=method, payload=payload)

        github = EmptyResponseGitHub()
        result = run_apply(
            github,
            codepath_owners=(),
            additional_owners=("compiler",),
        )
        self.assertEqual(result, ApplyOutcome("triaged", 1, 1))
        self.assertEqual(github.live_reads.count("requested"), 1)
        self.assertTrue(any(call[1].endswith("/labels") for call in mutations(github)))

    def test_native_team_request_is_not_repeated(self) -> None:
        class EmptyResponseGitHub(FakeGitHub):
            def json(
                self,
                endpoint: str,
                *,
                method: str = "GET",
                payload: dict[str, Any] | None = None,
            ) -> Any:
                if endpoint.endswith("/requested_reviewers") and method == "POST":
                    self.calls.append((method, endpoint, payload))
                    return {"users": [], "teams": []}
                return super().json(endpoint, method=method, payload=payload)

        github = EmptyResponseGitHub()
        result = run_apply(
            github,
            codepath_owners=("@pytorch/compiler",),
        )
        self.assertEqual(result, ApplyOutcome("triaged"))
        self.assertEqual(github.live_reads.count("requested"), 0)
        self.assertTrue(any(call[1].endswith("/labels") for call in mutations(github)))


class EngagedReviewerApplyTest(unittest.TestCase):
    def test_supporters_and_labelers_are_requested_before_routing(self) -> None:
        github = FakeGitHub()
        with mock.patch("builtins.print") as output:
            result = run_apply(
                github,
                supporters=("external-author", "Maintainer"),
                has_related_actionable_issue=True,
                actionable_labelers=("labeler", "maintainer"),
                maintainer_requested_reviewers=("requested",),
            )

        self.assertEqual(result, ApplyOutcome("triaged"))
        requested = "repos/pytorch/ciforge/pulls/123/requested_reviewers"
        self.assertEqual(
            mutations(github)[:2],
            [
                ("POST", requested, {"reviewers": ["Maintainer"]}),
                ("POST", requested, {"reviewers": ["labeler"]}),
            ],
        )
        routing = printed_reviewer_routing(output)
        self.assertIn(
            "Why this reviewer: The PR description names @Maintainer as a verified "
            "supporter. @Maintainer labeled a linked or related issue `actionable`.",
            routing,
        )
        self.assertIn(
            "A triage-or-higher maintainer already requested @requested.", routing
        )

    def test_supporter_picked_by_round_robin_is_requested_once(self) -> None:
        github = FakeGitHub()
        with mock.patch("builtins.print") as output:
            plan = run_plan(
                github,
                codepath_owners=(),
                additional_owners=("autograd",),
                supporters=("soulitzer",),
            )

        requests = [a for a in plan.actions if isinstance(a, RequestReviewers)]
        self.assertEqual(requests, [RequestReviewers(("soulitzer",), (), "supporter")])
        self.assertIn(
            "Why this reviewer: The PR description names @soulitzer as a verified "
            "supporter. No previous `autograd` assignment was found, so its "
            "round-robin rotation began with @soulitzer.",
            printed_reviewer_routing(output),
        )

    def test_supporter_requests_are_planned_without_writes(self) -> None:
        github = FakeGitHub()
        with mock.patch("builtins.print") as output:
            plan = run_plan(github, supporters=("Maintainer",))

        self.assertEqual(plan.decision, "triage")
        self.assertEqual(
            plan.actions[0], RequestReviewers(("Maintainer",), (), "supporter")
        )
        self.assertEqual(mutations(github), [])
        self.assertIn(
            "Planned effect: one deduplicated request for @Maintainer.",
            printed_reviewer_routing(output),
        )

    def test_engaged_maintainers_who_already_review_are_not_requested_again(
        self,
    ) -> None:
        github = FakeGitHub(submitted_users=["Maintainer"], requested_users=["labeler"])
        with mock.patch("builtins.print") as output:
            plan = run_plan(
                github,
                supporters=("helper", "Maintainer"),
                has_related_actionable_issue=True,
                actionable_labelers=("labeler",),
            )

        self.assertEqual(
            plan.actions[0], RequestReviewers(("helper",), (), "supporter")
        )
        self.assertEqual(len(plan.actions), 2)  # the supporter request, then labels
        routing = printed_reviewer_routing(output)
        for login in ("labeler", "Maintainer"):
            self.assertIn(
                f"@{login} already reviewed or has a pending request, so no new "
                "request is needed.",
                routing,
            )

    def test_unreadable_reviewer_state_fails_instead_of_re_requesting(self) -> None:
        class UnavailableReviewersGitHub(FakeGitHub):
            def json(
                self,
                endpoint: str,
                *,
                method: str = "GET",
                payload: dict[str, Any] | None = None,
            ) -> Any:
                if endpoint.endswith("/requested_reviewers") and method == "GET":
                    raise RuntimeError("reviewer state unavailable")
                return super().json(endpoint, method=method, payload=payload)

        github = UnavailableReviewersGitHub()
        with (
            mock.patch("builtins.print"),
            self.assertRaisesRegex(RuntimeError, "reviewer state unavailable"),
        ):
            run_plan(github, codepath_owners=(), supporters=("Maintainer",))
        self.assertEqual(mutations(github), [])

    def test_unadmitted_pr_explains_why_it_was_not_admitted(self) -> None:
        github = FakeGitHub()
        with mock.patch("builtins.print") as output:
            run_plan(github, passes_intake=False, llm_run_status="failed")

        self.assertIn(
            "Why this PR was admitted: Not admitted: no intake fact or bypass "
            "intake match.",
            printed_reviewer_routing(output),
        )

    def test_handoff_reviewer_triages_uncovered_concerns(self) -> None:
        handoffs = (
            {"supporters": ("Maintainer",)},
            {"actionable_labelers": ("labeler",)},
            {"maintainer_requested_reviewers": ("requested",)},
        )
        for handoff in handoffs:
            github = FakeGitHub()
            with self.subTest(handoff=handoff), mock.patch("builtins.print") as output:
                result = run_apply(github, has_uncovered_concerns=True, **handoff)
                self.assertEqual(result, ApplyOutcome("triaged"))
                self.assertEqual(printed_plan(output)["decision"], "triage")

    def test_handoff_reviewer_triages_without_owners(self) -> None:
        github = FakeGitHub(native_codeowners=[])
        with mock.patch("builtins.print") as output:
            result = run_apply(
                github,
                codepath_owners=(),
                maintainer_requested_reviewers=("requested",),
            )

        self.assertEqual(result, ApplyOutcome("triaged"))
        self.assertEqual(printed_plan(output)["decision"], "triage")
        self.assertEqual(
            mutations(github)[-1],
            (
                "POST",
                "repos/pytorch/ciforge/issues/123/labels",
                {"labels": ["triaged", "bot-triaged"]},
            ),
        )


class BypassIntakeTest(unittest.TestCase):
    def test_pr_that_fails_intake_closes_without_a_bypass(self) -> None:
        for name, scenario in (
            ("completed_without_owners", {"codepath_owners": ()}),
            ("completed_without_bypass", {"additional_owners": ("autograd",)}),
        ):
            github = FakeGitHub()
            with self.subTest(name=name), mock.patch("builtins.print"):
                result = run_apply(github, passes_intake=False, **scenario)

            self.assertEqual(result, ApplyOutcome("closed"))
            self.assertEqual(github.pr["state"], "closed")
            self.assertNotIn(
                "repos/pytorch/ciforge/pulls/123/requested_reviewers",
                [endpoint for _, endpoint, _ in mutations(github)],
            )

    def test_bypass_admits_and_its_team_reviewer_is_a_handoff(self) -> None:
        github = FakeGitHub()
        with mock.patch("builtins.print") as output:
            result = run_apply(
                github,
                passes_intake=False,
                codepath_owners=(),
                additional_owners=("autograd",),
                bypass_intake_matches=("autograd",),
                has_uncovered_concerns=True,
            )

        self.assertEqual(result, ApplyOutcome("triaged", 1, 1))
        self.assertEqual(printed_plan(output)["decision"], "triage")
        routing = printed_reviewer_routing(output)
        self.assertIn(
            "Why this PR was admitted: Admitted because `autograd`'s bypass intake "
            '"PRs that fix incorrect gradients." matched: '
            "The change fixes a gradient formula the team asked to see.",
            routing,
        )
        self.assertIn(
            'Matches bypass intake "PRs that fix incorrect gradients.": '
            "The change fixes a gradient formula the team asked to see.",
            routing,
        )
        self.assertIn(
            "Bypass evidence:\n- `torch/semantic.py`: "
            "This line fixes the gradient the team asked to see.",
            routing,
        )
        self.assertEqual(
            mutations(github),
            [
                (
                    "POST",
                    "repos/pytorch/ciforge/pulls/123/requested_reviewers",
                    {"reviewers": ["soulitzer"]},
                ),
                (
                    "POST",
                    "repos/pytorch/ciforge/issues/123/labels",
                    {"labels": ["triaged", "bot-triaged", "owner: autograd"]},
                ),
            ],
        )

    def test_failed_analysis_keeps_a_pr_that_fails_intake_open(self) -> None:
        github = FakeGitHub()
        with mock.patch("builtins.print") as output:
            result = run_apply(github, passes_intake=False, llm_run_status="failed")

        self.assertEqual(result, ApplyOutcome("incomplete"))
        self.assertEqual(printed_plan(output)["decision"], "incomplete")
        self.assertEqual(github.pr["state"], "open")
        self.assertEqual(
            mutations(github),
            [
                (
                    "POST",
                    "repos/pytorch/ciforge/issues/123/labels",
                    {"labels": [BOT_TRIAGE_ERROR_LABEL]},
                )
            ],
        )

    def test_discarded_bypass_claim_keeps_the_pr_open_without_routing(self) -> None:
        github = FakeGitHub()
        plan = run_plan(
            github, passes_intake=False, has_discarded_bypass_intake_match=True
        )

        self.assertEqual((plan.decision, plan.actions), ("kept_open", ()))
        self.assertEqual(mutations(github), [])

    def test_owner_without_bypass_is_not_a_handoff_reviewer(self) -> None:
        github = FakeGitHub()
        with mock.patch("builtins.print"):
            result = run_apply(
                github,
                codepath_owners=(),
                additional_owners=("autograd",),
                has_uncovered_concerns=True,
            )

        self.assertEqual(result, ApplyOutcome("routed_untriaged", 1, 1))


class TeamOwnerReviewersTest(unittest.TestCase):
    def test_requests_labels_and_unresolved_owners_derive_from_choices(self) -> None:
        routing = TeamOwnerReviewers(
            team_owner_ids=("autograd", "compiler", "nn", "quantization"),
            choices={
                "autograd": OwnerChoice("@Bob", "selected", "round_robin_initial"),
                "compiler": OwnerChoice("@alice", "pending"),
                "nn": OwnerChoice("@carol", "submitted"),
            },
            incomplete_reasons=(),
        )

        self.assertEqual(routing.roster_users, ("Bob",))
        self.assertEqual(routing.owner_labels, ("owner: autograd", "owner: compiler"))
        self.assertEqual(routing.unresolved_owners, ("quantization",))

    def test_new_requests_drop_existing_reviewers_the_author_and_repeats(
        self,
    ) -> None:
        existing = ReviewerState(
            requested_reviewers=frozenset({"@pending", "@pytorch/compiler"}),
            submitted_reviewers=frozenset({"@reviewed"}),
        )
        supporter = EngagementReason("supporter", already_reviewing=False)
        requests = new_reviewer_requests(
            engaged_maintainers=[
                (login, supporter) for login in ("pending", "Fresh", "author")
            ],
            owner_candidates={
                "codepath_owner": (("reviewed", "fresh"), ("compiler", "nn")),
                "owner_roster": (("roster",), ()),
            },
            existing=existing,
            author_handle="@author",
            org="pytorch",
        )

        self.assertEqual(
            requests,
            (
                RequestReviewers(("Fresh",), (), "supporter"),
                RequestReviewers((), ("nn",), "codepath_owner"),
                RequestReviewers(("roster",), (), "owner_roster"),
            ),
        )

    def test_logged_choice_omits_fields_that_do_not_apply(self) -> None:
        self.assertEqual(
            OwnerChoice("@alice", "pending").to_log(),
            {"reviewer": "@alice", "state": "pending"},
        )


class ReplayTest(unittest.TestCase):
    def test_plan_replays_from_the_planner_input_alone(self) -> None:
        github = FakeGitHub(
            round_robin_events=[
                {
                    "id": 91,
                    "event": "labeled",
                    "label": {"name": "owner: autograd"},
                    "issue": {"number": 7, "pull_request": {}},
                }
            ],
            round_robin_timelines={7: [{"id": 91, "event": "labeled"}]},
        )
        args = scenario_args(
            github, codepath_owners=(), additional_owners=("autograd",)
        )
        intake, ownership = args.stage_results
        with configured(), mock.patch("builtins.print") as output:
            planner_input = gather_planner_input(
                args=args, intake=intake, ownership=ownership, github=github
            )
            plan, why = build_action_plan(planner_input)
            reads = len(github.calls)
            replayed = build_action_plan(
                PlannerInput.from_json(planner_input.to_json())
            )

        self.assertEqual(replayed, (plan, why))
        self.assertEqual(len(github.calls), reads)
        output.assert_not_called()  # deciding a plan logs nothing
        self.assertEqual(plan.decision, "triage")
        cursor = planner_input.reviewers.round_robin["autograd"]
        self.assertIsNone(cursor.last_assigned)

    def test_failed_reads_replay_to_the_same_plan(self) -> None:
        github = FakeGitHub(codeowner_error=RuntimeError("unavailable"))
        args = scenario_args(
            github, codepath_owners=(), additional_owners=("autograd",)
        )
        intake, ownership = args.stage_results
        with configured(), mock.patch("builtins.print"):
            planner_input = gather_planner_input(
                args=args, intake=intake, ownership=ownership, github=github
            )
            replayed = PlannerInput.from_json(planner_input.to_json())
            self.assertEqual(
                build_action_plan(replayed), build_action_plan(planner_input)
            )
        self.assertIn("native_codeowner_requests", replayed.reviewers.errors)


class PlanMainTest(unittest.TestCase):
    def test_main_reads_stage_files_and_publishes_the_plan(self) -> None:
        github = FakeGitHub(labels=["open source"])
        intake, ownership = stage_results(
            has_actionable_linked_issue=False, codepath_owners=()
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "intake.json").write_text(json.dumps(intake.to_dict()))
            (root / "ownership.json").write_text(json.dumps(ownership.to_dict()))
            github_output = root / "github-output"
            args = run_args(output_dir=root, github_output=github_output)
            with (
                mock.patch("plan_actions.parse_args", return_value=args),
                mock.patch("plan_actions.GitHubReader", return_value=github),
                mock.patch("builtins.print"),
            ):
                self.assertEqual(plan_main(), 0)
            written = ActionPlan.from_dict(json.loads((root / "plan.json").read_text()))
            planner_input = PlannerInput.from_json(
                (root / "planner_input.json").read_text()
            )
            output = github_output.read_text()
        replayed, _ = build_action_plan(planner_input)

        self.assertEqual((written.decision, written.actions), ("close", CLOSE_ACTIONS))
        self.assertEqual(output, f"action-plan-json={written.to_json()}\n")
        self.assertEqual(replayed, written)
        self.assertEqual(mutations(github), [])

    def test_main_fails_without_an_ownership_result(self) -> None:
        intake, _ = stage_results()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "intake.json").write_text(json.dumps(intake.to_dict()))
            github_output = root / "github-output"
            args = run_args(output_dir=root, github_output=github_output)
            with (
                mock.patch("plan_actions.parse_args", return_value=args),
                mock.patch("plan_actions.GitHubReader", return_value=FakeGitHub()),
                mock.patch("builtins.print"),
            ):
                self.assertEqual(plan_main(), 1)
            self.assertFalse((root / "plan.json").exists())
            self.assertFalse(github_output.exists())


if __name__ == "__main__":
    unittest.main()
