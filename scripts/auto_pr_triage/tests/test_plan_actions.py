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
    pick_roster_member,
    ReviewerState,
    TeamOwnerReviewers,
)
from schemas import (
    ActionPlan,
    AdditionalOwnerConcern,
    AddLabels,
    BOT_TRIAGE_ERROR_LABEL,
    MISSING_ACTIONABLE_ISSUE_ACTIONS,
    PlannerInput,
    RequestReviewers,
)
from tests.plan_fixtures import (
    configured,
    FakeGitHub,
    HEAD_SHA,
    mutations,
    ownership_config,
    plan_and_apply,
    plan_pr,
    printed_plan,
    printed_reviewer_routing,
    run_apply,
    run_args,
    run_plan,
    run_without_owners,
    scenario_args,
    stage_results,
    unadmitted_args,
)


class StageInputTest(unittest.TestCase):
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
    def test_codepath_owners_are_not_requested(self) -> None:
        github = FakeGitHub()

        result = run_apply(github)

        self.assertEqual(result, ApplyOutcome("triaged"))
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

    def test_codepath_users_and_teams_are_never_requested(self) -> None:
        github = FakeGitHub()

        result = run_apply(
            github, codepath_owners=("@codepath-owner", "@pytorch/compiler")
        )

        self.assertEqual(result, ApplyOutcome("triaged"))
        self.assertFalse(
            any(call[1].endswith("/requested_reviewers") for call in mutations(github))
        )

    def test_pending_team_request_does_not_suppress_semantic_addition(self) -> None:
        github = FakeGitHub(requested_teams=["autograd"])

        result = run_apply(github, codepath_owners=(), additional_owners=("autograd",))

        self.assertEqual(result, ApplyOutcome("triaged", 1))
        self.assertEqual(mutations(github)[0][2], {"reviewers": ["soulitzer"]})


class PlanOnlyTest(unittest.TestCase):
    def test_missing_issue_plan_makes_no_writes(self) -> None:
        github = FakeGitHub(labels=["open source"])

        plan = plan_pr(args=unadmitted_args(), github=github)

        self.assertEqual(
            (plan.decision, plan.actions),
            ("missing_actionable_issue", MISSING_ACTIONABLE_ISSUE_ACTIONS),
        )
        self.assertEqual(mutations(github), [])

    def test_missing_issue_plan_on_rerun_keeps_open(self) -> None:
        github = FakeGitHub(labels=["open source"])

        plan = plan_pr(args=unadmitted_args(run_attempt=2), github=github)

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

        self.assertEqual(result, ApplyOutcome("routed_untriaged", 1))
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

        self.assertEqual(result, ApplyOutcome("routed_untriaged", 1))
        self.assertEqual(printed_plan(output)["decision"], "routed_untriaged")

    def test_uncovered_concern_ignores_unneeded_review_state_failure(self) -> None:
        class UnavailableReviewsGitHub(FakeGitHub):
            def graphql(
                self, *, query: str, variables: dict[str, Any]
            ) -> dict[str, Any]:
                if "reviews(first:" in query:
                    raise RuntimeError("submitted reviews unavailable")
                return super().graphql(query=query, variables=variables)

        github = UnavailableReviewsGitHub()
        result = run_apply(github, has_uncovered_concerns=True)

        self.assertEqual(result, ApplyOutcome("routed_untriaged"))
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
        self.assertTrue(
            summary_text.startswith(
                "## Auto PR Triage: requests review from `soulitzer`\n"
            )
        )
        self.assertIn(
            "| Admission | Admitted because it fixes an issue labeled `actionable`. |\n",
            summary_text,
        )
        self.assertIn(
            "| `codepath-owner` | no new request | `codepath-owner` is a codepath owner in CODEOWNERS but has no pending request or review, so GitHub may not have requested them or someone may have removed the request.<br>Codepath owner `@codepath-owner` matched CODEOWNERS rules for: <code>torch/file.py</code>. |\n",  # noqa: B950
            summary_text,
        )
        self.assertIn(
            "| `soulitzer` | new request | `soulitzer` was picked at random from `autograd`'s roster, seeded by this PR so that reruns pick the same member.<br>[See why `autograd` owns this](#auto-pr-triage-why-autograd) |\n",  # noqa: B950
            summary_text,
        )
        self.assertIn(
            '<a id="auto-pr-triage-why-autograd"></a>\n\n'
            "### Why `autograd` owns this\n\n"
            "Semantic owner `autograd`: autograd owns this changed behavior.\n",
            summary_text,
        )
        self.assertIn(
            "- <code>torch/semantic.py</code>: This line implements the owned behavior.\n"
            "  ```diff\n"
            "  +new semantic behavior\n"
            "  ```\n",
            summary_text,
        )
        self.assertIn(
            "<details><summary>Auto PR Triage decision plan</summary>\n\n"
            "- Decision: `triage`\n",
            summary_text,
        )
        self.assertIn(
            "Planned effect: one deduplicated request for @soulitzer.",
            printed_reviewer_routing(output),
        )

    def test_step_summary_renders_paths_and_excerpts_inertly(self) -> None:
        concern = AdditionalOwnerConcern.from_dict(
            {
                "concern": {
                    "description": "Autograd owns this changed behavior.",
                    "files": ["torch/a|b_`x`.py"],
                    "evidence": [
                        {
                            "file": "torch/a|b_`x`.py",
                            "diff_excerpt": "+```\n+</details> | cell",
                            "relevance": "This explains the relevant change.",
                        }
                    ],
                },
                "owner_id": "autograd",
                "rationale": [
                    "The configured description covers this changed contract."
                ],
                "confidence": "high",
                "bypass_intake_match": None,
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / "summary.md"
            with mock.patch("builtins.print"):
                run_plan(
                    FakeGitHub(),
                    codepath_owners=(),
                    additional_owners=("autograd",),
                    additional_owner_concerns=(concern,),
                    github_step_summary=summary,
                )
            summary_text = summary.read_text()

        self.assertIn(
            "- <code>torch/a&#124;b&#95;&#96;x&#96;.py</code>: "
            "This explains the relevant change.\n"
            "  ````diff\n"
            "  +```\n"
            "  +</details> | cell\n"
            "  ````\n",
            summary_text,
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

    def test_pending_owner_is_covered_without_request(self) -> None:
        github = FakeGitHub(requested_users=["soulitzer"])

        with mock.patch("builtins.print") as output:
            plan = run_plan(
                github,
                codepath_owners=(),
                additional_owners=("autograd",),
            )

        self.assertEqual(plan.decision, "triage")
        self.assertEqual(plan.actions, (AddLabels(("triaged", "bot-triaged")),))
        self.assertEqual(mutations(github), [])
        printed = printed_plan(output)
        choice = printed["owner_choices"]["autograd"]
        self.assertEqual(choice["reviewer"], "@soulitzer")
        self.assertEqual(choice["state"], "pending")
        self.assertEqual(choice["provenance"]["source"], "semantic")
        self.assertEqual(printed["planned_reviewer_requests"], [])

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
            plan.actions[0], RequestReviewers(("soulitzer",), "owner_roster")
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
    def test_codepath_owner_explanation_reports_what_reviewer_state_shows(
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

        cases = (
            (
                FakeGitHub(requested_users=["codepath-owner"]),
                "codeowners_pending",
                "already has a pending request; no new request is needed.",
            ),
            (
                FakeGitHub(submitted_users=["codepath-owner"]),
                "codeowners_submitted",
                "already submitted a review; no new request is needed.",
            ),
            (
                FakeGitHub(),
                "codeowners_missing",
                "but has no pending request or review, so GitHub may not have "
                "requested them or someone may have removed the request.",
            ),
            (
                UnavailableReviewersGitHub(),
                "codeowners_unconfirmed",
                "GitHub requests codepath owners, but the reviewer state could not "
                "be read to confirm.",
            ),
        )
        for github, state, text in cases:
            with self.subTest(state=state), mock.patch("builtins.print") as output:
                result = run_apply(github)

                self.assertEqual(result, ApplyOutcome("triaged"))
                choice = printed_plan(output)["owner_choices"]["@codepath-owner"]
                self.assertEqual(choice["state"], state)
                routing = printed_reviewer_routing(output)
                self.assertIn(
                    "@codepath-owner is a codepath owner in CODEOWNERS", routing
                )
                self.assertIn(text, routing)
                self.assertIn(
                    "Codepath owner `@codepath-owner` matched CODEOWNERS rules for: "
                    "`torch/file.py`.",
                    routing,
                )

    def test_codepath_owner_alone_does_not_load_roster_configuration(self) -> None:
        github = FakeGitHub(actionable_issue=True)
        args = run_args()

        error = AssertionError("roster configuration should not load")
        with (
            mock.patch("plan_actions.load_team_members", side_effect=error),
            mock.patch("apply_actions.load_team_members", side_effect=error),
        ):
            result = plan_and_apply(args=args, github=github)

        self.assertEqual(result, ApplyOutcome("triaged"))

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
        self.assertIn(f"| Why this run is incomplete | {reason} |\n", summary_text)
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

    def test_fresh_roster_pick_is_requested_without_reading_history(self) -> None:
        github = FakeGitHub()

        with mock.patch("builtins.print") as output:
            result = run_apply(
                github,
                codepath_owners=(),
                additional_owners=("autograd",),
            )

        self.assertEqual(result, ApplyOutcome("triaged", 1))
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
                    {"labels": ["triaged", "bot-triaged"]},
                ),
            ],
        )
        reads = [endpoint for method, endpoint, _ in github.calls if method == "GET"]
        self.assertFalse(any("owner%3A" in endpoint for endpoint in reads))
        self.assertFalse(any("/issues/events" in endpoint for endpoint in reads))
        choice = printed_plan(output)["owner_choices"]["autograd"]
        self.assertEqual(choice["reviewer"], "@soulitzer")
        self.assertEqual(choice["state"], "selected")
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

    def test_two_member_roster_pick_is_seeded_by_the_pr(self) -> None:
        github = FakeGitHub()
        team_members = copy.deepcopy(ownership_config()["team_members"])
        team_members["members"]["autograd"] = ["@first", "@second"]
        expected = pick_roster_member(
            repository="pytorch/ciforge",
            number=123,
            owner="autograd",
            members=("@first", "@second"),
            author_handle="@external-author",
        )

        with mock.patch("builtins.print") as output:
            result = run_apply(
                github,
                codepath_owners=(),
                additional_owners=("autograd",),
                team_members=team_members,
            )

        self.assertEqual(result, ApplyOutcome("triaged", 1))
        self.assertEqual(mutations(github)[0][2], {"reviewers": [expected[1:]]})
        self.assertIn(
            f"{expected} was picked at random from `autograd`'s roster, seeded by "
            "this PR so that reruns pick the same member.",
            printed_reviewer_routing(output),
        )

    def test_existing_codepath_owner_requests_are_preserved(self) -> None:
        github = FakeGitHub(
            requested_teams=["compiler"],
            submitted_users=["codepath-owner"],
        )

        with mock.patch("builtins.print") as output:
            result = run_apply(
                github,
                codepath_owners=("@codepath-owner", "@pytorch/compiler"),
            )

        self.assertEqual(result, ApplyOutcome("triaged", 0))
        self.assertFalse(
            any(call[1].endswith("/requested_reviewers") for call in mutations(github))
        )
        choices = printed_plan(output)["owner_choices"]
        self.assertEqual(choices["@codepath-owner"]["state"], "codeowners_submitted")
        self.assertEqual(choices["@pytorch/compiler"]["state"], "codeowners_pending")

    def test_large_codepath_owner_set_allows_no_new_requests(self) -> None:
        reviewers = tuple(f"owner{index}" for index in range(16))
        github = FakeGitHub(requested_users=list(reviewers))

        result = run_apply(
            github,
            codepath_owners=tuple(f"@{reviewer}" for reviewer in reviewers),
        )

        self.assertEqual(result, ApplyOutcome("triaged"))
        self.assertFalse(
            any(call[1].endswith("/requested_reviewers") for call in mutations(github))
        )

    def test_codepath_owner_policy_does_not_request_the_author(self) -> None:
        github = FakeGitHub()

        result = run_apply(
            github,
            codepath_owners=("@external-author",),
        )

        self.assertEqual(result, ApplyOutcome("triaged", 0))
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

        self.assertEqual(result, ApplyOutcome("triaged", 1))
        self.assertEqual(mutations(github)[0][2], {"reviewers": ["extra"]})

    def test_pending_owner_and_new_owner_request_only_the_new_pick(self) -> None:
        github = FakeGitHub(requested_users=["soulitzer"])

        result = run_apply(
            github,
            codepath_owners=(),
            additional_owners=("autograd", "compiler"),
        )

        self.assertEqual(result, ApplyOutcome("triaged", 1))
        self.assertEqual(
            mutations(github),
            [
                (
                    "POST",
                    "repos/pytorch/ciforge/pulls/123/requested_reviewers",
                    {"reviewers": ["extra"]},
                ),
                (
                    "POST",
                    "repos/pytorch/ciforge/issues/123/labels",
                    {"labels": ["triaged", "bot-triaged"]},
                ),
            ],
        )

    def test_two_fresh_owners_request_two_reviewers(self) -> None:
        github = FakeGitHub()

        result = run_apply(
            github,
            codepath_owners=(),
            additional_owners=("autograd", "compiler"),
        )

        self.assertEqual(result, ApplyOutcome("triaged", 2))
        self.assertEqual(mutations(github)[0][2], {"reviewers": ["extra", "soulitzer"]})
        self.assertEqual(
            mutations(github)[1][2], {"labels": ["triaged", "bot-triaged"]}
        )

    def test_two_owners_deduplicate_shared_roster_member(self) -> None:
        github = FakeGitHub()
        team_members = copy.deepcopy(ownership_config()["team_members"])
        team_members["members"]["autograd"] = ["@soulitzer", "@izaitsevfb"]
        team_members["members"]["nn"] = ["@izaitsevfb"]

        result = run_apply(
            github,
            codepath_owners=(),
            additional_owners=("autograd", "nn"),
            team_members=team_members,
        )

        self.assertEqual(result, ApplyOutcome("triaged", 1))
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
        args = run_args(
            stage_results=stage_results(
                codepath_owners=("@pytorch/autograd",),
                additional_owners=("autograd",),
            )
        )

        with configured(), mock.patch("builtins.print") as output:
            result = plan_and_apply(args=args, github=github)
        self.assertEqual(result, ApplyOutcome("triaged", 1))
        self.assertEqual(mutations(github)[0][2], {"reviewers": ["soulitzer"]})
        choices = printed_plan(output)["owner_choices"]
        self.assertEqual(choices["autograd"]["provenance"]["source"], "semantic")
        self.assertEqual(choices["@pytorch/autograd"]["state"], "codeowners_missing")
        self.assertEqual(
            choices["@pytorch/autograd"]["provenance"]["source"], "codepath"
        )

    def test_missing_live_reviewer_is_selected_at_plan_time(self) -> None:
        github = FakeGitHub()

        result = run_apply(
            github,
            codepath_owners=(),
            additional_owners=("autograd",),
        )
        self.assertEqual(result, ApplyOutcome("triaged", 1))
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

    def test_missing_error_label_prevents_fallback_write(self) -> None:
        github = FakeGitHub(unavailable_labels=[BOT_TRIAGE_ERROR_LABEL])

        with self.assertRaisesRegex(RuntimeError, "label.*unavailable"):
            run_apply(github, llm_run_status="failed")

        self.assertEqual(mutations(github), [])

    def test_unavailable_additional_owner_routing_preserves_codepath_handles(
        self,
    ) -> None:
        github = FakeGitHub()
        team_members = {"members": {"compiler": ["@reviewer"]}}

        with mock.patch("builtins.print") as output:
            result = run_apply(
                github,
                codepath_owners=("@codepath-owner",),
                additional_owners=("autograd",),
                team_members=team_members,
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

    def test_author_cannot_be_picked_from_a_roster(self) -> None:
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
            configured(team_members=config["team_members"]),
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
        github = FakeGitHub()
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch("builtins.print") as output,
        ):
            summary = Path(directory) / "summary.md"
            result = run_apply(
                github,
                codepath_owners=(),
                additional_owners=("autograd", "compiler"),
                team_members=team_members,
                github_step_summary=summary,
            )

            self.assertEqual(result, ApplyOutcome("incomplete", 1))
            plan = printed_plan(output)
            self.assertEqual(set(plan["owner_choices"]), {"compiler"})
            self.assertEqual(plan["owner_choices"]["compiler"]["reviewer"], "@extra")
            self.assertEqual(plan["planned_reviewer_requests"], ["@extra"])
            self.assertEqual(plan["unresolved_owners"], ["autograd"])
            self.assertIn("- Unresolved owners: `autograd`", summary.read_text())

        self.assertEqual(
            mutations(github),
            [
                (
                    "POST",
                    "repos/pytorch/ciforge/pulls/123/requested_reviewers",
                    {"reviewers": ["extra"]},
                ),
                (
                    "POST",
                    "repos/pytorch/ciforge/issues/123/labels",
                    {"labels": [BOT_TRIAGE_ERROR_LABEL]},
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

        with configured(team_members=config["team_members"]):
            result = plan_and_apply(args=args, github=github)

        self.assertEqual(result, ApplyOutcome("triaged", 1))
        self.assertEqual(mutations(github)[0][2], {"reviewers": ["soulitzer"]})

    def test_triage_trusts_analyzed_triage_facts(self) -> None:
        github = FakeGitHub()
        args = run_args()

        with configured():
            self.assertEqual(
                plan_and_apply(args=args, github=github), ApplyOutcome("triaged")
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
        routing = printed_reviewer_routing(output)
        self.assertIn(
            "@soulitzer covers `autograd` through codepath ownership; "
            "no `autograd` roster pick was needed.",
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
        self.assertEqual(result, ApplyOutcome("triaged", 1))
        self.assertEqual(github.live_reads.count("requested"), 1)
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

    def test_supporter_picked_from_roster_is_requested_once(self) -> None:
        github = FakeGitHub()
        with mock.patch("builtins.print") as output:
            plan = run_plan(
                github,
                codepath_owners=(),
                additional_owners=("autograd",),
                supporters=("soulitzer",),
            )

        requests = [a for a in plan.actions if isinstance(a, RequestReviewers)]
        self.assertEqual(requests, [RequestReviewers(("soulitzer",), "supporter")])
        self.assertIn(
            "Why this reviewer: The PR description names @soulitzer as a verified "
            "supporter. @soulitzer was picked at random from `autograd`'s roster, "
            "seeded by this PR so that reruns pick the same member.",
            printed_reviewer_routing(output),
        )

    def test_supporter_requests_are_planned_without_writes(self) -> None:
        github = FakeGitHub()
        with mock.patch("builtins.print") as output:
            plan = run_plan(github, supporters=("Maintainer",))

        self.assertEqual(plan.decision, "triage")
        self.assertEqual(
            plan.actions[0], RequestReviewers(("Maintainer",), "supporter")
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

        self.assertEqual(plan.actions[0], RequestReviewers(("helper",), "supporter"))
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
        github = FakeGitHub()
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
    def test_pr_that_fails_intake_is_marked_without_a_bypass(self) -> None:
        for name, scenario in (
            ("completed_without_owners", {"codepath_owners": ()}),
            ("completed_without_bypass", {"additional_owners": ("autograd",)}),
        ):
            github = FakeGitHub()
            with self.subTest(name=name), mock.patch("builtins.print"):
                result = run_apply(github, passes_intake=False, **scenario)

            self.assertEqual(result, ApplyOutcome("missing_actionable_issue"))
            self.assertEqual(github.pr["state"], "open")
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

        self.assertEqual(result, ApplyOutcome("triaged", 1))
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
                    {"labels": ["triaged", "bot-triaged"]},
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

        self.assertEqual(result, ApplyOutcome("routed_untriaged", 1))


class TeamOwnerReviewersTest(unittest.TestCase):
    def test_requests_and_unresolved_owners_derive_from_choices(self) -> None:
        routing = TeamOwnerReviewers(
            team_owner_ids=("autograd", "compiler", "nn", "quantization"),
            choices={
                "autograd": OwnerChoice("@Bob", "selected"),
                "compiler": OwnerChoice("@alice", "pending"),
                "nn": OwnerChoice("@carol", "submitted"),
            },
            incomplete_reasons=(),
        )

        self.assertEqual(routing.roster_users, ("Bob",))
        self.assertEqual(routing.unresolved_owners, ("quantization",))

    def test_roster_pick_is_reproducible_and_skips_the_author(self) -> None:
        args = {
            "repository": "pytorch/ciforge",
            "number": 9,
            "owner": "autograd",
            "members": ("@first", "@Second", "@third"),
            "author_handle": "@second",
        }

        first = pick_roster_member(**args)

        self.assertEqual(first, pick_roster_member(**args))
        self.assertIn(first, {"@first", "@third"})

    def test_roster_pick_requires_a_member_other_than_the_author(self) -> None:
        self.assertIsNone(
            pick_roster_member(
                repository="pytorch/ciforge",
                number=9,
                owner="autograd",
                members=("@Author",),
                author_handle="@author",
            )
        )

    def test_roster_picks_spread_across_members(self) -> None:
        picks = [
            pick_roster_member(
                repository="pytorch/ciforge",
                number=number,
                owner="autograd",
                members=("@first", "@second", "@third"),
                author_handle="@external-author",
            )
            for number in range(300)
        ]

        for member in ("@first", "@second", "@third"):
            self.assertGreater(picks.count(member), 60)

    def test_new_requests_drop_existing_reviewers_the_author_and_repeats(
        self,
    ) -> None:
        existing = ReviewerState(
            requested_reviewers=frozenset({"@pending"}),
            submitted_reviewers=frozenset({"@reviewed"}),
        )
        supporter = EngagementReason("supporter", already_reviewing=False)
        requests = new_reviewer_requests(
            engaged_maintainers=[
                (login, supporter) for login in ("pending", "Fresh", "author")
            ],
            roster_users=("reviewed", "fresh", "roster"),
            existing=existing,
            author_handle="@author",
        )

        self.assertEqual(
            requests,
            (
                RequestReviewers(("Fresh",), "supporter"),
                RequestReviewers(("roster",), "owner_roster"),
            ),
        )

    def test_logged_choice_omits_fields_that_do_not_apply(self) -> None:
        self.assertEqual(
            OwnerChoice("@alice", "pending").to_log(),
            {"reviewer": "@alice", "state": "pending"},
        )


class ReplayTest(unittest.TestCase):
    def test_plan_replays_from_the_planner_input_alone(self) -> None:
        github = FakeGitHub()
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

    def test_failed_reads_replay_to_the_same_plan(self) -> None:
        github = FakeGitHub()
        args = scenario_args(
            github, codepath_owners=(), additional_owners=("autograd",)
        )
        intake, ownership = args.stage_results
        unavailable = RuntimeError("unavailable")
        with (
            mock.patch("plan_actions.load_team_members", side_effect=unavailable),
            mock.patch("builtins.print"),
        ):
            planner_input = gather_planner_input(
                args=args, intake=intake, ownership=ownership, github=github
            )
            replayed = PlannerInput.from_json(planner_input.to_json())
            self.assertEqual(
                build_action_plan(replayed), build_action_plan(planner_input)
            )
        self.assertIn("rosters", replayed.reviewers.errors)


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

        self.assertEqual(
            (written.decision, written.actions),
            ("missing_actionable_issue", MISSING_ACTIONABLE_ISSUE_ACTIONS),
        )
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
