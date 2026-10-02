"""Security contract for the pi triage workflows and the pi-agent action.

Each test pins one property that keeps the model job locked down; a failing
test means a one-line edit weakened it.
"""

import re
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = {
    name: yaml.safe_load((ROOT / ".github/workflows" / name).read_text())["jobs"]
    for name in ("issue-triage-pi.yml", "distributed-triage-pi.yml")
}
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


def steps(job: dict) -> list[dict]:
    return job.get("steps", [])


def all_steps() -> list[dict]:
    return [
        step
        for jobs in WORKFLOWS.values()
        for job in jobs.values()
        for step in steps(job)
    ] + ACTION["runs"]["steps"]


class TriageWorkflowContract(unittest.TestCase):
    def setUp(self):
        self.workflows = list(WORKFLOWS.items())

    def test_model_job_blocks_egress_before_anything_runs(self):
        for workflow, jobs in self.workflows:
            with self.subTest(workflow=workflow):
                first = steps(jobs["plan"])[0]
                self.assertTrue(
                    first["uses"].startswith("step-security/harden-runner@")
                )
                self.assertEqual(first["with"]["egress-policy"], "block")
                self.assertTrue(first["with"]["disable-sudo"])
                endpoints = set(first["with"]["allowed-endpoints"].split())
                self.assertEqual(
                    endpoints,
                    {
                        "sts.us-east-1.amazonaws.com:443",
                        "bedrock-runtime.us-east-1.amazonaws.com:443",
                        "api.github.com:443",
                        "github.com:443",
                        "codeload.github.com:443",
                        "objects.githubusercontent.com:443",
                        "registry.npmjs.org:443",
                        "ossci-raw-job-status.s3.us-east-1.amazonaws.com:443",
                        "ossci-raw-job-status.s3.amazonaws.com:443",
                    },
                )

    def test_model_job_holds_no_write_scope(self):
        for workflow, jobs in self.workflows:
            with self.subTest(workflow=workflow):
                self.assertEqual(
                    jobs["plan"]["permissions"],
                    {"contents": "read", "issues": "read", "id-token": "write"},
                )

    def test_write_job_has_no_model_credentials(self):
        for workflow, jobs in self.workflows:
            with self.subTest(workflow=workflow):
                apply = jobs["apply"]
                self.assertNotIn("id-token", apply["permissions"])
                self.assertNotIn("environment", apply)
                self.assertFalse(
                    any("pi-agent" in step.get("uses", "") for step in steps(apply))
                )

    def test_model_gets_only_read_only_tools(self):
        for workflow, jobs in self.workflows:
            with self.subTest(workflow=workflow):
                (agent,) = [
                    step
                    for step in steps(jobs["plan"])
                    if step.get("uses") == "./.github/actions/pi-agent"
                ]
                self.assertLessEqual(
                    set(agent["with"]["tools"].split(",")), READ_ONLY_TOOLS
                )

    def test_every_action_is_pinned_to_a_commit(self):
        for step in all_steps():
            uses = step.get("uses", "")
            if uses and not uses.startswith("./"):
                self.assertRegex(uses, PINNED)

    def test_checkouts_do_not_persist_the_token(self):
        for step in all_steps():
            if step.get("uses", "").startswith("actions/checkout@"):
                self.assertIs(step.get("with", {}).get("persist-credentials"), False)

    def test_only_applied_triage_writes_the_issue_execution_log(self):
        for workflow, jobs in self.workflows:
            with self.subTest(workflow=workflow):
                # Replays and dry runs share the issue's S3 key; they must not overwrite
                # the log of the run that actually triaged it.
                (upload,) = [
                    step
                    for step in steps(jobs["plan"])
                    if step.get("name") == "Upload execution log to S3"
                ]
                self.assertIn("needs.prepare.outputs.mode == 'apply'", upload["if"])
                self.assertIn("github.repository == 'pytorch/pytorch'", upload["if"])

    def test_model_credentials_are_short_lived(self):
        for workflow, jobs in self.workflows:
            with self.subTest(workflow=workflow):
                (aws,) = [
                    step
                    for step in steps(jobs["plan"])
                    if "configure-aws-credentials" in step.get("uses", "")
                ]
                self.assertEqual(aws["with"]["role-duration-seconds"], 900)

    def test_no_workflow_runs_claude_code(self):
        for step in all_steps():
            self.assertNotIn("claude-code-action", step.get("uses", ""))

    def test_manifest_job_runs_no_model_and_cannot_write_issues(self):
        record = WORKFLOWS["distributed-triage-pi.yml"]["record"]
        self.assertEqual(record["permissions"].get("issues"), "read")
        self.assertFalse(
            any("pi-agent" in step.get("uses", "") for step in steps(record))
        )


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
