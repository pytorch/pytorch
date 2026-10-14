from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from apply_actions import apply_action_plan, ApplyOutcome, GitHubClient, main
from schemas import ActionPlan
from tests.plan_fixtures import (
    configured,
    FakeGitHub,
    mutations,
    plan_facts,
    plan_pr,
    REPOSITORY,
    run_args,
    run_unadmitted,
    unadmitted_args,
)
from tests.stage_fixtures import make_action_plan


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = (
    "intake.json",
    "ownership.json",
    "planner_input.json",
    "plan.json",
    "result.json",
)


class GitHubClientTest(unittest.TestCase):
    def test_client_sends_mutation_payload_through_stdin(self) -> None:
        github = GitHubClient()
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout='[{"name":"triaged"}]', stderr=""
        )
        with mock.patch("apply_actions.subprocess.run", return_value=completed) as run:
            self.assertEqual(
                github.json(
                    "repos/pytorch/ciforge/issues/123/labels",
                    method="POST",
                    payload={"labels": ["triaged"]},
                ),
                [{"name": "triaged"}],
            )
        run.assert_called_once()
        self.assertEqual(
            run.call_args.args[0],
            [
                "gh",
                "api",
                "repos/pytorch/ciforge/issues/123/labels",
                "--method",
                "POST",
                "--input",
                "-",
            ],
        )
        self.assertEqual(run.call_args.kwargs["input"], '{"labels":["triaged"]}')


class MissingActionableIssueTest(unittest.TestCase):
    def test_unadmitted_pr_gets_the_missing_issue_labels(self) -> None:
        github = FakeGitHub(labels=["open source"])

        result = run_unadmitted(github)

        self.assertEqual(result, ApplyOutcome("missing_actionable_issue"))
        self.assertEqual(
            mutations(github),
            [
                (
                    "POST",
                    "repos/pytorch/ciforge/issues/123/labels",
                    {"labels": ["triaged", "bot-triaged", "missing actionable issue"]},
                ),
            ],
        )
        self.assertEqual(github.pr["state"], "open")
        self.assertEqual(github.live_reads, [])

    def test_missing_actionable_issue_on_retry_is_a_noop(self) -> None:
        github = FakeGitHub(labels=["open source"])

        self.assertEqual(
            run_unadmitted(github, run_attempt=2), ApplyOutcome("kept_open")
        )
        self.assertEqual(mutations(github), [])

    def test_apply_rejects_plan_from_an_earlier_attempt(self) -> None:
        github = FakeGitHub(labels=["open source"])
        args = unadmitted_args()
        args.action_plan_json = plan_pr(args=args, github=github).to_json()
        args.run_attempt = 2

        with self.assertRaisesRegex(ValueError, "from attempt 1; rerun all jobs"):
            apply_action_plan(args=args, github=github)
        self.assertEqual(mutations(github), [])

    def test_labels_do_not_refetch_triage_facts_or_pr_state(self) -> None:
        github = FakeGitHub(
            labels=["triaged", "bot-triaged"],
            actionable_issue=True,
            author_has_triage_permission=True,
            requested_users=["soulitzer"],
            submitted_users=["soulitzer"],
        )
        github.pr.update(draft=True, title="changed")
        github.pr["head"]["sha"] = "c" * 40
        github.pr["base"]["ref"] = "release"

        self.assertEqual(
            run_unadmitted(github), ApplyOutcome("missing_actionable_issue")
        )
        self.assertEqual(github.live_reads, [])
        self.assertEqual(github.pr_fetches, 0)
        self.assertEqual(github.actionable_checks, 0)
        self.assertEqual(github.permission_checks, 0)

    def test_missing_label_prevents_writes(self) -> None:
        github = FakeGitHub(
            labels=["open source"], unavailable_labels=["missing actionable issue"]
        )
        with self.assertRaisesRegex(RuntimeError, "required repository label"):
            run_unadmitted(github)
        self.assertEqual(mutations(github), [])

    def test_label_timeout_reports_ambiguous_state(self) -> None:
        class TimeoutGitHub(FakeGitHub):
            def json(
                self,
                endpoint: str,
                *,
                method: str = "GET",
                payload: dict[str, Any] | None = None,
            ) -> Any:
                if endpoint.endswith("/issues/123/labels") and method == "POST":
                    raise subprocess.TimeoutExpired("gh api", 60)
                return super().json(endpoint, method=method, payload=payload)

        github = TimeoutGitHub(labels=["open source"])
        with self.assertRaisesRegex(RuntimeError, "label request failed"):
            run_unadmitted(github)


class PlanExecutionBoundTest(unittest.TestCase):
    def plan(self, **overrides: Any) -> ActionPlan:
        facts = overrides.pop("facts", plan_facts())
        return make_action_plan(facts, **overrides)

    def test_apply_rejects_reviewer_outside_rosters(self) -> None:
        github = FakeGitHub()
        args = run_args(
            action_plan_json=self.plan(roster_reviewers=("stranger",)).to_json()
        )

        with (
            configured(),
            self.assertRaisesRegex(ValueError, "outside owner rosters"),
        ):
            apply_action_plan(args=args, github=github)
        self.assertEqual(github.calls, [])

    def test_apply_accepts_roster_member(self) -> None:
        github = FakeGitHub()
        plan = self.plan(roster_reviewers=("soulitzer",))
        args = run_args(action_plan_json=plan.to_json())

        with configured():
            outcome = apply_action_plan(args=args, github=github)

        self.assertEqual(outcome, ApplyOutcome("triaged", 1))
        self.assertEqual(mutations(github)[0][2], {"reviewers": ["soulitzer"]})

    def test_apply_rejects_a_codepath_owner_requested_from_a_roster(self) -> None:
        plan = self.plan(roster_reviewers=("codepath-owner",))
        args = run_args(action_plan_json=plan.to_json())

        with (
            configured(),
            self.assertRaisesRegex(ValueError, "outside owner rosters"),
        ):
            apply_action_plan(args=args, github=FakeGitHub())

    def test_actions_execute_in_planned_order(self) -> None:
        github = FakeGitHub()
        plan = self.plan(
            facts=plan_facts(supporters=("alice",)),
            supporter_reviewers=("alice",),
            roster_reviewers=("soulitzer",),
        )
        args = run_args(action_plan_json=plan.to_json())

        with configured():
            outcome = apply_action_plan(args=args, github=github)

        self.assertEqual(outcome, ApplyOutcome("triaged", 1))
        requested = "repos/pytorch/ciforge/pulls/123/requested_reviewers"
        self.assertEqual(
            mutations(github),
            [
                ("POST", requested, {"reviewers": ["alice"]}),
                ("POST", requested, {"reviewers": ["soulitzer"]}),
                (
                    "POST",
                    "repos/pytorch/ciforge/issues/123/labels",
                    {"labels": ["triaged", "bot-triaged"]},
                ),
            ],
        )

    def test_failed_action_before_a_close_stops_execution(self) -> None:
        class RequestFailureGitHub(FakeGitHub):
            def json(
                self,
                endpoint: str,
                *,
                method: str = "GET",
                payload: dict[str, Any] | None = None,
            ) -> Any:
                if endpoint.endswith("/requested_reviewers") and method == "POST":
                    self.calls.append((method, endpoint, payload))
                    raise RuntimeError("request failed")
                return super().json(endpoint, method=method, payload=payload)

        github = RequestFailureGitHub()
        plan = self.plan(
            facts=plan_facts(supporters=("alice",)),
            supporter_reviewers=("alice",),
            roster_reviewers=("soulitzer",),
        )
        args = run_args(action_plan_json=plan.to_json())

        with (
            configured(),
            self.assertRaisesRegex(RuntimeError, "supporter reviewer request failed"),
        ):
            apply_action_plan(args=args, github=github)
        self.assertEqual(len(mutations(github)), 1)

    def test_apply_rejects_plan_for_another_pull_request_or_run(self) -> None:
        plan_json = self.plan().to_json()
        for args in (
            run_args(pr=124, action_plan_json=plan_json),
            run_args(repository="pytorch/pytorch", action_plan_json=plan_json),
            run_args(workflow_sha="d" * 40, action_plan_json=plan_json),
        ):
            github = FakeGitHub()
            with (
                self.subTest(args=args),
                self.assertRaisesRegex(ValueError, "another pull request or run"),
            ):
                apply_action_plan(args=args, github=github)
            self.assertEqual(github.calls, [])


class ApplyMainTest(unittest.TestCase):
    def test_main_reports_each_apply_status(self) -> None:
        cases = (
            (ApplyOutcome("triaged", 2), "requested 2 owner reviewers"),
            (
                ApplyOutcome("incomplete", 2),
                "Applied incomplete Auto PR Triage",
            ),
            (
                ApplyOutcome("missing_actionable_issue"),
                "Labeled pytorch/ciforge#123 as missing an actionable issue",
            ),
            (ApplyOutcome("kept_open"), "did not qualify for an apply action"),
            (
                ApplyOutcome("routed_untriaged", 2),
                "Applied partial Auto PR Triage",
            ),
        )
        for outcome, message in cases:
            with self.subTest(status=outcome.status):
                args = argparse.Namespace(pr=123, repository=REPOSITORY)
                with (
                    mock.patch.dict("os.environ", {"GH_TOKEN": "token"}),
                    mock.patch("apply_actions.parse_args", return_value=args),
                    mock.patch("apply_actions.GitHubClient") as client,
                    mock.patch(
                        "apply_actions.apply_action_plan",
                        return_value=outcome,
                    ) as apply,
                    mock.patch("builtins.print") as output,
                ):
                    self.assertEqual(main(), 0)
                client.assert_called_once_with()
                apply.assert_called_once_with(args=args, github=client.return_value)
                self.assertIn(message, output.call_args.args[0])

    def test_main_reports_apply_failure(self) -> None:
        args = argparse.Namespace(pr=123, repository=REPOSITORY)
        with (
            mock.patch.dict("os.environ", {"GH_TOKEN": "token"}),
            mock.patch("apply_actions.parse_args", return_value=args),
            mock.patch("apply_actions.GitHubClient") as client,
            mock.patch(
                "apply_actions.apply_action_plan",
                side_effect=ValueError("plan is invalid"),
            ) as apply,
            mock.patch("builtins.print"),
        ):
            self.assertEqual(main(), 1)
        client.assert_called_once_with()
        apply.assert_called_once_with(args=args, github=client.return_value)

    def test_workflow_passes_current_apply_contract(self) -> None:
        action = (
            REPOSITORY_ROOT / ".github/actions/auto-pr-triage/action.yml"
        ).read_text()
        workflow = (
            REPOSITORY_ROOT / ".github/workflows/auto-pr-triage.yml"
        ).read_text()

        self.assertEqual(
            workflow.count("python3 scripts/auto_pr_triage/apply_actions.py"), 1
        )
        stages = (
            "assess_intake",
            "build_ownership_input",
            "validate_ownership",
            "plan_actions",
        )
        for stage in stages:
            self.assertEqual(
                action.count(f"scripts/auto_pr_triage/{stage}.py"), 1, stage
            )
            self.assertIn(f"id: {stage}\n", action)
            self.assertIn(f"steps.{stage}.outcome != 'success'", action)
        self.assertIn("  repository:\n", action)
        self.assertIn("  run-attempt:\n", action)
        self.assertNotIn("author-login", action)
        self.assertNotIn("author-login", workflow)
        self.assertIn("  action-plan-json:\n", action)
        self.assertIn(
            "value: ${{ steps.plan_actions.outputs.action-plan-json }}", action
        )
        self.assertIn("if: steps.assess_intake.outputs.active == 'true'", action)
        self.assertIn("if: steps.build_ownership_input.outcome == 'success'", action)
        self.assertIn(
            "steps.build_ownership_input.outcome == 'success' &&\n"
            "        steps.aws-credentials.outcome == 'success'",
            action,
        )
        self.assertIn("steps.assess_intake.outputs.active != 'true' ||", action)
        self.assertNotIn("outputs.admitted", action)
        self.assertNotIn("run-model", action)
        self.assertIn(
            "${{ steps.build_ownership_input.outputs.result-schema-json }}", action
        )
        self.assertIn("${{ steps.build_ownership_input.outputs.prompt-file }}", action)
        self.assertIn(
            "if: ${{ always() && steps.validate_ownership.outcome == 'success' }}",
            action,
        )
        for argument in ("--run-attempt", "--output-dir", "--github-output"):
            self.assertIn(argument, action)
        for removed in (
            "prepare_llm_input",
            "process_llm_output",
            "plan_triage",
            "run-llm",
            "analysis-result-json",
            "--author-login",
        ):
            self.assertNotIn(removed, action)
        self.assertIn("run-attempt: ${{ github.run_attempt }}", workflow)
        self.assertIn(
            "action-plan-json: ${{ steps.auto-pr-triage.outputs.action-plan-json }}",
            workflow,
        )
        self.assertIn("mode: ${{ steps.mode.outputs.mode }}", workflow)
        self.assertIn("should-run: ${{ steps.history.outputs.should-run }}", workflow)
        for argument in (
            "--repository",
            "--run-attempt",
            "--action-plan-json",
            "--workflow-sha",
        ):
            self.assertIn(argument, workflow)
        for removed in ("--mode", "--analysis-result-json", "--author-login"):
            self.assertNotIn(removed, workflow)
        self.assertIn("repository: ${{ github.repository }}", workflow)
        self.assertIn('--repository "$REPOSITORY"', workflow)
        self.assertNotIn("pytorch/ciforge", action)
        self.assertNotIn("pytorch/ciforge", workflow)
        mode_lines = [
            line
            for line in workflow.splitlines()
            if line.strip().startswith("AUTO_PR_TRIAGE_MODE:")
        ]
        self.assertEqual(len(mode_lines), 1)
        self.assertIn(
            mode_lines[0],
            {"  AUTO_PR_TRIAGE_MODE: shadow", "  AUTO_PR_TRIAGE_MODE: live"},
        )
        self.assertIn("env:\n  # Change only this value", workflow)
        self.assertEqual(workflow.count("needs.analyze.outputs.mode == 'live'"), 2)
        self.assertIn("needs.analyze.outputs.action-plan-json != ''", workflow)
        self.assertIn(
            "ACTION_PLAN_JSON: ${{ needs.analyze.outputs.action-plan-json }}",
            workflow,
        )
        self.assertIn(
            "  apply:\n    needs: analyze\n    if: >-\n"
            "      needs.analyze.result == 'success' &&\n"
            "      needs.analyze.outputs.mode == 'live' &&",
            workflow,
        )
        self.assertIn(
            "    permissions:\n      contents: read\n      pull-requests: write\n",
            workflow,
        )
        self.assertNotIn("should-apply", workflow)
        self.assertNotIn("should-apply", action)
        self.assertNotIn("expected-head-sha", workflow)
        self.assertNotIn("expected-head-sha", action)
        self.assertIn("types: [labeled, ready_for_review]", workflow)
        self.assertIn(
            "  analyze:\n    if: >-\n      github.repository_owner == 'pytorch' &&",
            workflow,
        )
        self.assertIn("github.event.pull_request.state == 'open'", workflow)
        # Intake checks the base ref, which also accepts ghstack base branches.
        self.assertNotIn("github.event.pull_request.base.ref", workflow)
        self.assertNotIn("expected-base-ref", action)
        self.assertIn("!github.event.pull_request.draft", workflow)
        self.assertIn("github.event.action == 'ready_for_review'", workflow)
        self.assertIn(
            "contains(github.event.pull_request.labels.*.name, 'open source')",
            workflow,
        )
        self.assertIn("READY_FOR_REVIEW_EVENT", workflow)
        self.assertIn("pageInfo { hasPreviousPage }", workflow)
        self.assertIn('"$READY_EVENT_COUNT" == "1"', workflow)
        for removed in (
            "action-basis",
            "analysis-complete",
            "analyzed-head-sha",
            "analyzed-pr-text-sha256",
            "baseline-reviewers-json",
            "confidence",
            "extra-reviewers-json",
            "extra-teams-json",
            "additional-reviewers-json",
            "required-reviewers-json",
            "routing-assessment",
            "validation-failed",
            "has-actionable-linked-issue",
            "pulls-triage-bot-do-not-close",
            "human-review-file",
            "bot-shadow",
            "bot-codeowners-shadow",
        ):
            self.assertNotIn(removed, action)
            self.assertNotIn(removed, workflow)
        self.assertNotIn("analysis-result-json", workflow)
        self.assertIn("WORKFLOW_SHA: ${{ github.sha }}", workflow)
        self.assertIn("  record-error:\n", workflow)
        self.assertIn("needs: [analyze, apply]", workflow)
        self.assertNotIn("  admit:\n", workflow)
        self.assertIn("needs.analyze.outputs.should-run == 'true'", workflow)
        self.assertIn("needs.analyze.result == 'failure'", workflow)
        self.assertIn("needs.apply.result == 'failure'", workflow)
        self.assertIn("needs.apply.result == 'skipped'", workflow)
        self.assertIn("labels[]=bot-triage-error", workflow)
        self.assertIn("Unable to add bot-triage-error", workflow)
        self.assertIn(
            "group: auto-pr-triage-${{ github.event.pull_request.number }}",
            workflow,
        )
        for artifact in ARTIFACTS:
            self.assertEqual(workflow.count(f"/{artifact}"), 2, artifact)
        self.assertEqual(action.count("continue-on-error: true"), 3)
        self.assertIn("always() &&", action)

    def test_workflow_admits_label_events_and_first_ready_event(self) -> None:
        workflow = (
            REPOSITORY_ROOT / ".github/workflows/auto-pr-triage.yml"
        ).read_text()
        lines = workflow.splitlines()
        step_index = lines.index(
            "      - name: Admit the label event or the first ready event after it"
        )
        run_index = lines.index("        run: |", step_index)
        script_lines = []
        for line in lines[run_index + 1 :]:
            if line.startswith("          "):
                script_lines.append(line[10:])
            elif not line:
                script_lines.append("")
            else:
                break
        script = "\n".join(script_lines)
        label = {"__typename": "LabeledEvent", "label": {"name": "open source"}}
        ready = {"__typename": "ReadyForReviewEvent"}
        cases = (
            ("labeled", [label], False, "true"),
            ("labeled", [label, label], False, "true"),
            ("labeled", [label], True, "true"),
            ("ready_for_review", [ready, label, ready], False, "true"),
            ("ready_for_review", [label, ready, ready], False, "false"),
            ("ready_for_review", [ready], False, "false"),
            ("ready_for_review", [label, ready], True, "false"),
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gh = root / "gh"
            history_path = root / "history.json"
            output_path = root / "github-output"
            gh.write_text('#!/bin/sh\nexec /bin/cat "$FAKE_HISTORY"\n')
            gh.chmod(0o755)
            for action, nodes, truncated, expected in cases:
                with self.subTest(action=action, nodes=nodes, truncated=truncated):
                    history_path.write_text(
                        json.dumps(
                            {
                                "data": {
                                    "repository": {
                                        "pullRequest": {
                                            "timelineItems": {
                                                "nodes": nodes,
                                                "pageInfo": {
                                                    "hasPreviousPage": truncated
                                                },
                                            }
                                        }
                                    }
                                }
                            }
                        )
                    )
                    output_path.write_text("")
                    environment = {
                        **os.environ,
                        "EVENT_ACTION": action,
                        "FAKE_HISTORY": str(history_path),
                        "GH_TOKEN": "token",
                        "GITHUB_OUTPUT": str(output_path),
                        "PATH": f"{root}:{os.environ['PATH']}",
                        "PR_NUMBER": "123",
                        "REPOSITORY": "pytorch/ciforge",
                    }
                    result = subprocess.run(
                        ["bash", "-eu", "-o", "pipefail", "-c", script],
                        capture_output=True,
                        text=True,
                        env=environment,
                        check=False,
                    )

                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(
                        output_path.read_text().splitlines(),
                        [f"should-run={expected}"],
                    )


if __name__ == "__main__":
    unittest.main()
