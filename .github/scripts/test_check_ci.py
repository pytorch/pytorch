# Owner(s): ["module: ci"]

import os
import subprocess
import tempfile
from pathlib import Path
from unittest import main, TestCase

import yaml


ROOT = Path(__file__).resolve().parents[2]
ACTION = ROOT / ".github/actions/check-ci/action.yml"


class TestCheckCI(TestCase):
    def run_gate(
        self,
        title: str,
        pr_number: str = "123",
        ref: str = "refs/pull/123/merge",
        failures: int = 0,
        error: int = 1,
    ) -> tuple[subprocess.CompletedProcess[str], list[str], list[str]]:
        script = yaml.safe_load(ACTION.read_text())["runs"]["steps"][0]["run"]
        with tempfile.TemporaryDirectory() as tmp:
            scratch = Path(tmp)
            calls = scratch / "calls"
            delays = scratch / "delays"
            gh = scratch / "gh"
            gh.write_text(
                "#!/bin/bash\n"
                'printf "%s\\n" "$*" >> "$CI_TEST_CALLS"\n'
                'printf "%s\\n" "$CI_TEST_TITLE"\n'
                'if [[ "$(wc -l < "$CI_TEST_CALLS")" -le "$CI_TEST_FAILURES" ]]; then\n'
                '  exit "$CI_TEST_ERROR"\n'
                "fi\n"
            )
            gh.chmod(0o755)
            sleep = scratch / "sleep"
            sleep.write_text('#!/bin/bash\nprintf "%s\\n" "$1" >> "$CI_TEST_DELAYS"\n')
            sleep.chmod(0o755)
            env = os.environ | {
                "PATH": f"{scratch}:{os.environ['PATH']}",
                "PR_NUMBER": pr_number,
                "GITHUB_REF": ref,
                "GITHUB_REPOSITORY": "pytorch/pytorch",
                "GH_TOKEN": "unused-test-token",
                "CI_TEST_TITLE": title,
                "CI_TEST_FAILURES": str(failures),
                "CI_TEST_ERROR": str(error),
                "CI_TEST_CALLS": str(calls),
                "CI_TEST_DELAYS": str(delays),
            }
            result = subprocess.run(
                ["bash", "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", script],
                env=env,
                capture_output=True,
                text=True,
                timeout=40,
            )
            return (
                result,
                calls.read_text().splitlines() if calls.exists() else [],
                delays.read_text().splitlines() if delays.exists() else [],
            )

    def test_title_prefix(self) -> None:
        for title, expected in (
            ("[no-ci] Work in progress", 1),
            ("[no-ci]", 1),
            ("Work in progress", 0),
            ("Support [no-ci] titles", 0),
            ("[NO-CI] Work in progress", 0),
            (" [no-ci] Work in progress", 0),
            ("[no ci] Work in progress", 0),
            ("", 0),
        ):
            with self.subTest(title=title):
                result, calls, delays = self.run_gate(title)
                self.assertEqual(result.returncode, expected, result.stderr)
                self.assertEqual(
                    calls, ["api repos/pytorch/pytorch/pulls/123 --jq .title"]
                )
                self.assertEqual(delays, [])
                if expected:
                    self.assertIn("Remove the prefix and push", result.stdout)

    def test_ciflow_pr(self) -> None:
        for workflow in ("pull", "trunk", "binaries_wheel", "inductor-perf-test"):
            with self.subTest(workflow=workflow):
                result, calls, _ = self.run_gate(
                    "[no-ci] Work", pr_number="", ref=f"refs/tags/ciflow/{workflow}/456"
                )
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertEqual(
                    calls, ["api repos/pytorch/pytorch/pulls/456 --jq .title"]
                )

    def test_event_pr_takes_precedence(self) -> None:
        result, calls, _ = self.run_gate("Work", ref="refs/tags/ciflow/trunk/456")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, ["api repos/pytorch/pytorch/pulls/123 --jq .title"])

    def test_non_pr_runs_do_not_query_github(self) -> None:
        for ref in (
            "refs/heads/main",
            "refs/heads/nightly",
            "refs/heads/release/2.14",
            "refs/heads/landchecks/test",
            "refs/heads/gh/user/1/head",
            "refs/tags/v2.14.0-rc1",
            "refs/tags/ciflow/trunk/" + "abcdef1234" * 4,
            "refs/tags/ciflow/trunk/not-a-pr",
            "refs/tags/ciflow/trunk/123/extra",
            "refs/heads/ciflow/trunk/123",
        ):
            with self.subTest(ref=ref):
                result, calls, delays = self.run_gate(
                    "[no-ci] Work", pr_number="", ref=ref
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(calls, [])
                self.assertEqual(delays, [])

    def test_api_failures_allow_ci(self) -> None:
        for error in (1, 22, 124):
            with self.subTest(error=error):
                result, calls, delays = self.run_gate(
                    "[no-ci] Work", failures=3, error=error
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(len(calls), 3)
                self.assertEqual(delays, ["1", "2"])
                self.assertIn("::warning::", result.stdout)
                self.assertIn("running CI", result.stdout)
                self.assertNotIn("::error::", result.stdout)

    def test_success_after_retry_honors_title(self) -> None:
        for title, expected in (("Work", 0), ("[no-ci] Work", 1)):
            with self.subTest(title=title):
                result, calls, delays = self.run_gate(title, failures=2)
                self.assertEqual(result.returncode, expected, result.stderr)
                self.assertEqual(len(calls), 3)
                self.assertEqual(delays, ["1", "2"])
                self.assertNotIn("::warning::", result.stdout)

    def test_rerun_reads_current_title(self) -> None:
        for pr_number, ref in (
            ("123", "refs/pull/123/merge"),
            ("", "refs/tags/ciflow/trunk/123"),
        ):
            with self.subTest(ref=ref):
                disabled, _, _ = self.run_gate("[no-ci] Work", pr_number, ref)
                enabled, calls, _ = self.run_gate("Work", pr_number, ref)
                disabled_again, _, _ = self.run_gate("[no-ci] Work", pr_number, ref)
                self.assertEqual(disabled.returncode, 1, disabled.stderr)
                self.assertEqual(enabled.returncode, 0, enabled.stderr)
                self.assertEqual(disabled_again.returncode, 1, disabled_again.stderr)
                self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    main()
