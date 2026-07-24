import json
import os
import subprocess
import sys
from pathlib import Path

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


HOOK = (
    Path(__file__).resolve().parent.parent
    / ".claude/skills/triaging-issues/scripts/validate_issue_target.py"
)


def run_hook(tool_input: dict, **environment: str) -> subprocess.CompletedProcess[str]:
    """Run the target hook with an isolated workflow environment."""
    env = os.environ.copy()
    env.update(environment)
    return subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps({"tool_input": tool_input}),
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


class TestValidateIssueTarget(TestCase):
    def test_allows_the_workflow_target(self):
        result = run_hook(
            {"owner": "pytorch", "repo": "pytorch", "issue_number": 123},
            GITHUB_REPOSITORY="pytorch/pytorch",
            TRIAGE_ISSUE_NUMBER="123",
        )

        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stderr, "")

    @parametrize(
        "tool_input",
        [
            {"owner": "pytorch", "repo": "pytorch", "issue_number": 124},
            {"owner": "pytorch", "repo": "ciforge", "issue_number": 123},
            {"owner": "attacker", "repo": "pytorch", "issue_number": 123},
        ],
    )
    def test_blocks_a_different_target(self, tool_input):
        result = run_hook(
            tool_input,
            GITHUB_REPOSITORY="pytorch/pytorch",
            TRIAGE_ISSUE_NUMBER="123",
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "does not match triage target pytorch/pytorch#123",
            result.stderr,
        )

    @parametrize(
        "repository,issue_number",
        [
            ("", "123"),
            ("pytorch", "123"),
            ("pytorch/pytorch", ""),
            ("pytorch/pytorch", "abc"),
        ],
    )
    def test_blocks_when_the_trusted_target_is_invalid(self, repository, issue_number):
        result = run_hook(
            {"owner": "pytorch", "repo": "pytorch", "issue_number": 123},
            GITHUB_REPOSITORY=repository,
            TRIAGE_ISSUE_NUMBER=issue_number,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("Blocked issue mutation", result.stderr)

    def test_blocks_when_the_tool_omits_its_target(self):
        result = run_hook(
            {"body": "comment"},
            GITHUB_REPOSITORY="pytorch/pytorch",
            TRIAGE_ISSUE_NUMBER="123",
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("tool_input missing owner", result.stderr)


instantiate_parametrized_tests(TestValidateIssueTarget)


if __name__ == "__main__":
    run_tests()
