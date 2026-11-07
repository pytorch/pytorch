"""Security contract for the pi triage workflows and the pi-agent action.

Each test pins one property that keeps the model job locked down; a failing
test means a one-line edit weakened it.
"""

import re
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def load_jobs(name: str) -> dict:
    return yaml.safe_load((ROOT / ".github/workflows" / name).read_text())["jobs"]


JOBS = load_jobs("issue-triage-pi.yml")
PLAN = JOBS["plan"]
ACTION = yaml.safe_load((ROOT / ".github/actions/pi-agent/action.yml").read_text())
RUN_SH = (ROOT / ".github/actions/pi-agent/run.sh").read_text()
PINNED = re.compile(r"^[\w.-]+/[\w./-]+@[0-9a-f]{40}$")
READ_ONLY_TOOLS = {
    "read",
    "grep",
    "find",
    "ls",
    "get_issue",
    "get_issue_comments",
    "search_issues",
}
PLAN_PERMISSIONS = {"contents": "read", "issues": "read", "id-token": "write"}


def steps(job: dict) -> list[dict]:
    return job.get("steps", [])


def step(job: dict, predicate) -> dict:
    (match,) = [s for s in steps(job) if predicate(s)]
    return match


def all_steps() -> list[dict]:
    return [s for job in JOBS.values() for s in steps(job)] + ACTION["runs"]["steps"]


class TriageWorkflowContract(unittest.TestCase):
    def test_only_the_plan_job_runs_the_model(self):
        agents = [
            name
            for name, job in JOBS.items()
            for s in steps(job)
            if "pi-agent" in s.get("uses", "")
        ]
        self.assertEqual(agents, ["plan"])

    def test_handoff_requires_the_expected_workflow_and_event(self):
        condition = JOBS["prepare"]["if"]
        self.assertIn("github.event.workflow_run.conclusion == 'success'", condition)
        self.assertIn(
            "github.event.workflow.path == '.github/workflows/issue-triage.yml'",
            condition,
        )
        # Manual dispatch is supported. PR-triggered runs must not be accepted
        # as privileged handoffs.
        self.assertEqual(
            set(
                re.findall(
                    r"github\.event\.workflow_run\.event == '([^']+)'", condition
                )
            ),
            {"issues", "workflow_dispatch"},
        )
        self.assertIn(
            "(github.event.workflow_run.event == 'issues' || "
            "github.event.workflow_run.event == 'workflow_dispatch')",
            " ".join(condition.split()),
        )

    def test_model_job_blocks_egress_before_anything_runs(self):
        first = steps(PLAN)[0]
        self.assertTrue(first["uses"].startswith("step-security/harden-runner@"))
        self.assertEqual(first["with"]["egress-policy"], "block")
        self.assertTrue(first["with"]["disable-sudo"])
        self.assertEqual(
            set(first["with"]["allowed-endpoints"].split()),
            {
                "sts.us-east-1.amazonaws.com:443",
                "bedrock-runtime.us-east-1.amazonaws.com:443",
                "api.github.com:443",
                "github.com:443",
                "codeload.github.com:443",
                "objects.githubusercontent.com:443",
                "release-assets.githubusercontent.com:443",
                "registry.npmjs.org:443",
                "ossci-raw-job-status.s3.us-east-1.amazonaws.com:443",
                "ossci-raw-job-status.s3.amazonaws.com:443",
            },
        )

    def test_model_job_holds_no_write_scope(self):
        self.assertEqual(PLAN["permissions"], PLAN_PERMISSIONS)

    def test_model_gets_only_read_only_tools(self):
        agent = step(PLAN, lambda s: s.get("uses") == "./.github/actions/pi-agent")
        tools = set(agent["with"]["tools"].split(","))
        self.assertLessEqual(tools, READ_ONLY_TOOLS)
        self.assertIn("search_issues", tools)

    def test_search_source_is_bound_for_live_and_replay_runs(self):
        self.assertEqual(PLAN["env"]["TRIAGE_SOURCE_ISSUE"], "${{ matrix.issue }}")
        self.assertEqual(PLAN["env"]["TRIAGE_REPO"], "${{ github.repository }}")

    def test_metrics_are_best_effort_but_model_failures_block_apply(self):
        metrics = step(PLAN, lambda s: "upload-claude-usage" in s.get("uses", ""))
        self.assertIs(metrics.get("continue-on-error"), True)
        agent = step(PLAN, lambda s: s.get("uses") == "./.github/actions/pi-agent")
        self.assertFalse(agent.get("continue-on-error", False))

    def test_model_credentials_are_short_lived(self):
        aws = step(PLAN, lambda s: "configure-aws-credentials" in s.get("uses", ""))
        self.assertEqual(aws["with"]["role-duration-seconds"], 900)

    def test_write_jobs_have_no_model_credentials(self):
        apply = JOBS["apply"]
        self.assertNotIn("id-token", apply["permissions"])
        self.assertNotIn("environment", apply)

    def test_apply_serializes_by_issue_without_cancelling_writes(self):
        concurrency = JOBS["apply"]["concurrency"]
        self.assertIn("${{ needs.prepare.outputs.issues }}", concurrency["group"])
        self.assertIs(concurrency["cancel-in-progress"], False)

    def test_only_applied_triage_in_pytorch_writes_the_execution_log(self):
        # Replays and dry runs share the issue's S3 key; they must not overwrite
        # the log of the run that actually triaged it.
        log_prefix = PLAN["env"]["LOG_PREFIX"]
        self.assertIn("needs.prepare.outputs.mode == 'apply'", log_prefix)
        self.assertIn("github.repository == 'pytorch/pytorch'", log_prefix)
        upload = step(PLAN, lambda s: s.get("name") == "Upload execution log to S3")
        self.assertIn("env.LOG_PREFIX != ''", upload["if"])

    def test_every_action_is_pinned_to_a_commit(self):
        for s in all_steps():
            uses = s.get("uses", "")
            if uses and not uses.startswith("./"):
                self.assertRegex(uses, PINNED)

    def test_checkouts_do_not_persist_the_token(self):
        for s in all_steps():
            if s.get("uses", "").startswith("actions/checkout@"):
                self.assertIs(s.get("with", {}).get("persist-credentials"), False)

    def test_no_workflow_runs_claude_code(self):
        for s in all_steps():
            self.assertNotIn("claude-code-action", s.get("uses", ""))


class PiAgentActionContract(unittest.TestCase):
    def test_session_ignores_discovered_config_and_always_loads_tool_guard(self):
        for flag in (
            "--no-approve",
            "--no-extensions",
            "--no-skills",
            "--no-prompt-templates",
            "--no-context-files",
        ):
            self.assertIn(flag, RUN_SH)
        self.assertIn(
            'extensions=(-e "$PI_ACTION_PATH/extensions/tool-guard.ts")', RUN_SH
        )
        self.assertIn('export PI_ALLOWED_TOOLS="$tools"', RUN_SH)

    def test_install_is_lockfile_pinned_and_runs_no_package_scripts(self):
        (install,) = [
            step for step in ACTION["runs"]["steps"] if step.get("name") == "Install pi"
        ]
        self.assertIn("npm ci", install["run"])
        self.assertIn("--ignore-scripts", install["run"])


if __name__ == "__main__":
    unittest.main()
