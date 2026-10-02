"""Security contract for the pi issue-triage workflow and the pi-agent action.

Each test pins one property that keeps the model job locked down; a failing
test means a one-line edit weakened it.
"""

import re
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = yaml.safe_load((ROOT / ".github/workflows/issue-triage-pi.yml").read_text())
ACTION = yaml.safe_load((ROOT / ".github/actions/pi-agent/action.yml").read_text())
RUN_SH = (ROOT / ".github/actions/pi-agent/run.sh").read_text()
JOBS = WORKFLOW["jobs"]
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
    return [step for job in JOBS.values() for step in steps(job)] + ACTION["runs"][
        "steps"
    ]


class TriageWorkflowContract(unittest.TestCase):
    def test_model_job_blocks_egress_before_anything_runs(self):
        first = steps(JOBS["plan"])[0]
        self.assertTrue(first["uses"].startswith("step-security/harden-runner@"))
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
            },
        )

    def test_model_job_holds_no_write_scope(self):
        self.assertEqual(
            JOBS["plan"]["permissions"],
            {"contents": "read", "issues": "read", "id-token": "write"},
        )

    def test_write_job_has_no_model_credentials(self):
        apply = JOBS["apply"]
        self.assertNotIn("id-token", apply["permissions"])
        self.assertNotIn("environment", apply)
        self.assertFalse(
            any("pi-agent" in step.get("uses", "") for step in steps(apply))
        )

    def test_model_gets_only_read_only_tools(self):
        (agent,) = [
            step
            for step in steps(JOBS["plan"])
            if step.get("uses") == "./.github/actions/pi-agent"
        ]
        self.assertLessEqual(set(agent["with"]["tools"].split(",")), READ_ONLY_TOOLS)

    def test_every_action_is_pinned_to_a_commit(self):
        for step in all_steps():
            uses = step.get("uses", "")
            if uses and not uses.startswith("./"):
                self.assertRegex(uses, PINNED)

    def test_checkouts_do_not_persist_the_token(self):
        for step in all_steps():
            if step.get("uses", "").startswith("actions/checkout@"):
                self.assertIs(step.get("with", {}).get("persist-credentials"), False)

    def test_model_credentials_are_short_lived(self):
        (aws,) = [
            step
            for step in steps(JOBS["plan"])
            if "configure-aws-credentials" in step.get("uses", "")
        ]
        self.assertEqual(aws["with"]["role-duration-seconds"], 900)


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
