"""Exercise the workflow's issue-input validation without calling GitHub."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = yaml.safe_load((ROOT / ".github/workflows/issue-triage-pi.yml").read_text())
CONFIG = next(
    step["run"]
    for step in WORKFLOW["jobs"]["prepare"]["steps"]
    if step.get("id") == "config"
)


class IssueInputTest(unittest.TestCase):
    def resolve(self, issues: str, mode: str, event: str = "workflow_dispatch"):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "output"
            output.touch()
            (root / "issue_number.txt").write_text(issues + "\n")
            env = {
                **os.environ,
                "EVENT": event,
                "ISSUES": issues,
                "MODE": mode,
                "REPO": "pytorch/ciforge",
                "GITHUB_REPOSITORY": "pytorch/ciforge",
                "GITHUB_OUTPUT": str(output),
                "MODEL": "test-model",
                "THINKING": "medium",
            }
            result = subprocess.run(
                [
                    "bash",
                    "-euo",
                    "pipefail",
                    "-c",
                    "gh() { printf 'v2.9.0\\n'; };\n" + CONFIG,
                ],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result, output.read_text()

    def test_apply_accepts_one_issue_and_rejects_batches(self):
        result, output = self.resolve("123", "apply")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("issues=[123]\n", output)
        result, output = self.resolve("123 456", "apply")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("exactly one issue", result.stdout)
        self.assertEqual(output, "")

    def test_read_only_modes_keep_batch_support(self):
        for mode in ("replay", "dry-run"):
            with self.subTest(mode=mode):
                result, output = self.resolve("123 456", mode)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("issues=[123,456]\n", output)

    def test_automatic_handoff_accepts_only_one_issue(self):
        result, output = self.resolve("123", "apply", "workflow_run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("issues=[123]\n", output)
        for issues in ("123 456", "not-an-issue"):
            with self.subTest(issues=issues):
                result, output = self.resolve(issues, "apply", "workflow_run")
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(output, "")


class HandoffLookupTest(unittest.TestCase):
    def test_api_failure_is_not_an_absent_handoff(self):
        scripts = []
        for name in ("claude-distributed-triage.yml", "distributed-triage-pi.yml"):
            workflow = ROOT / ".github/workflows" / name
            if workflow.exists():
                jobs = yaml.safe_load(workflow.read_text())["jobs"]
                scripts.extend(
                    step["run"]
                    for job in jobs.values()
                    for step in job.get("steps", [])
                    if ".artifacts[].name" in step.get("run", "")
                )
        self.assertEqual(len(scripts), 1)
        fake = """
        gh() {
          if [[ "$1" == "api" ]]; then
            printf '%s\\n' "$ARTIFACTS_RESULT"
            return "$API_RC"
          fi
          printf '123\\n' > issue_number.txt
        }
        """
        for api_rc, artifacts, succeeds, downloaded in (
            (1, "", False, False),
            (0, "", True, False),
            (0, "triage-completed-data", True, True),
        ):
            with self.subTest(api_rc=api_rc, artifacts=artifacts):
                with tempfile.TemporaryDirectory() as temp:
                    result = subprocess.run(
                        ["bash", "-euo", "pipefail", "-c", fake + scripts[0]],
                        cwd=temp,
                        env={
                            **os.environ,
                            "API_RC": str(api_rc),
                            "ARTIFACTS_RESULT": artifacts,
                            "RUN_ID": "123",
                            "GITHUB_REPOSITORY": "pytorch/ciforge",
                        },
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    self.assertEqual(result.returncode == 0, succeeds, result.stderr)
                    self.assertEqual(
                        (Path(temp) / "issue_number.txt").exists(), downloaded
                    )


if __name__ == "__main__":
    unittest.main()
