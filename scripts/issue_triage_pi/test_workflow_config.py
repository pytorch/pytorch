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


if __name__ == "__main__":
    unittest.main()
