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


PLAN = load_jobs("pi-triage-plan.yml")["plan"]
CALLERS = {
    name: load_jobs(name)
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
PLAN_PERMISSIONS = {"contents": "read", "issues": "read", "id-token": "write"}


def steps(job: dict) -> list[dict]:
    return job.get("steps", [])


def step(job: dict, predicate) -> dict:
    (match,) = [s for s in steps(job) if predicate(s)]
    return match


def all_steps() -> list[dict]:
    jobs = [PLAN] + [job for jobs in CALLERS.values() for job in jobs.values()]
    return [s for job in jobs for s in steps(job)] + ACTION["runs"]["steps"]


class TriageWorkflowContract(unittest.TestCase):
    def test_every_model_job_is_the_shared_plan_workflow(self):
        for workflow, jobs in CALLERS.items():
            with self.subTest(workflow=workflow):
                self.assertEqual(
                    jobs["plan"]["uses"], "./.github/workflows/pi-triage-plan.yml"
                )
                self.assertEqual(jobs["plan"]["permissions"], PLAN_PERMISSIONS)
                agents = [
                    s
                    for job in jobs.values()
                    for s in steps(job)
                    if "pi-agent" in s.get("uses", "")
                ]
                self.assertEqual(agents, [])

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
                "registry.npmjs.org:443",
                "ossci-raw-job-status.s3.us-east-1.amazonaws.com:443",
                "ossci-raw-job-status.s3.amazonaws.com:443",
            },
        )

    def test_model_job_holds_no_write_scope(self):
        self.assertEqual(PLAN["permissions"], PLAN_PERMISSIONS)

    def test_model_gets_only_read_only_tools(self):
        agent = step(PLAN, lambda s: s.get("uses") == "./.github/actions/pi-agent")
        self.assertLessEqual(set(agent["with"]["tools"].split(",")), READ_ONLY_TOOLS)

    def test_model_credentials_are_short_lived(self):
        aws = step(PLAN, lambda s: "configure-aws-credentials" in s.get("uses", ""))
        self.assertEqual(aws["with"]["role-duration-seconds"], 900)

    def test_write_jobs_have_no_model_credentials(self):
        for workflow, jobs in CALLERS.items():
            with self.subTest(workflow=workflow):
                apply = jobs["apply"]
                self.assertNotIn("id-token", apply["permissions"])
                self.assertNotIn("environment", apply)

    def test_only_applied_triage_in_pytorch_writes_the_execution_log(self):
        # Replays and dry runs share the issue's S3 key; they must not overwrite
        # the log of the run that actually triaged it.
        for workflow, jobs in CALLERS.items():
            with self.subTest(workflow=workflow):
                log_prefix = jobs["plan"]["with"]["log-prefix"]
                self.assertIn("needs.prepare.outputs.mode == 'apply'", log_prefix)
                self.assertIn("github.repository == 'pytorch/pytorch'", log_prefix)

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

    def test_manifest_job_runs_no_model_and_cannot_write_issues(self):
        record = CALLERS["distributed-triage-pi.yml"]["record"]
        self.assertEqual(record["permissions"].get("issues"), "read")
        self.assertFalse(any("pi-agent" in s.get("uses", "") for s in steps(record)))


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
