"""Offline action behavior: python3 -m unittest discover -s .github/actions/pi-agent/tests -v."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ACTION = Path(__file__).resolve().parents[1]
MODEL = json.loads((ACTION / "allowed-models.json").read_text())["models"][0]["id"]
FAKE_PI = r"""
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
with open(os.environ["FAKE_CALLS"], "a") as calls:
    calls.write(json.dumps(args) + "\n")
if "--version" in args:
    print("0.99.1")
elif "--list-models" in args:
    if os.environ["FAKE_KNOWN"] == "true" or Path(os.environ["FAKE_REFRESHED"]).exists():
        print("amazon-bedrock " + os.environ["PI_MODEL"])
elif args == ["update", "--models"]:
    Path(os.environ["FAKE_REFRESHED"]).touch()
elif "--mode" in args:
    if os.environ["FAKE_RESULT"] == "true" and os.environ.get("PI_RESULT_FILE"):
        Path(os.environ["PI_RESULT_FILE"]).write_text('{"verdict":"ready"}')
    sys.stdout.write(Path(os.environ["FAKE_EVENTS"]).read_text())
    sys.stdout.flush()
    print("fake pi diagnostic", file=sys.stderr)
    sys.exit(int(os.environ["FAKE_EXIT"]))
else:
    raise RuntimeError(f"unexpected fake pi invocation: {args}")
"""


def events(*, stop: str = "stop", tools: tuple[str, ...] = ("read",)) -> str:
    return (
        "\n".join(
            json.dumps(event)
            for event in [
                {"type": "session", "id": "session-test"},
                {
                    "type": "message_end",
                    "message": {
                        "role": "system",
                        "toolsAdded": [{"name": tool} for tool in tools],
                    },
                },
                {
                    "type": "tool_execution_start",
                    "toolName": "read",
                    "args": {"path": "README.md"},
                },
                {"type": "tool_execution_end", "toolName": "read", "isError": False},
                {
                    "type": "message_end",
                    "message": {
                        "role": "assistant",
                        "timestamp": 1700000000000,
                        "content": [{"type": "text", "text": "Finished."}],
                        "stopReason": stop,
                        "errorMessage": "provider unavailable"
                        if stop == "error"
                        else None,
                        "usage": {
                            "input": 11,
                            "output": 7,
                            "cacheRead": 3,
                            "cacheWrite": 2,
                            "cost": {"total": 0.012},
                        },
                    },
                },
            ]
        )
        + "\n"
    )


class TestRun(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.out = self.root / "run"
        bin_dir = self.root / "pi-install/node_modules/.bin"
        bin_dir.mkdir(parents=True)
        pi = bin_dir / "pi"
        pi.write_text(f"#!{sys.executable}\n" + FAKE_PI)
        pi.chmod(0o755)
        self.env = {
            **{
                key: value
                for key, value in os.environ.items()
                if not key.startswith("PI_")
            },
            "RUNNER_TEMP": str(self.root),
            "PI_ACTION_PATH": str(ACTION),
            "PI_OUTPUT_DIR": str(self.out),
            "PI_MODEL": MODEL,
            "PI_ALLOW_UNLISTED_MODEL": "false",
            "PI_PROMPT": "Review the checkout.",
            "PI_THINKING": "medium",
            "PI_TOOLS": "read,grep,find,ls",
            "PI_RESULT_SCHEMA": '{"type":"object"}',
            "PI_EXTENSIONS": "",
            "PI_SKILLS": "",
            "GITHUB_STEP_SUMMARY": str(self.root / "summary"),
            "GITHUB_OUTPUT": str(self.root / "output"),
            "FAKE_CALLS": str(self.root / "calls.jsonl"),
            "FAKE_EVENTS": str(self.root / "events.jsonl"),
            "FAKE_EXIT": "0",
            "FAKE_RESULT": "true",
            "FAKE_KNOWN": "true",
            "FAKE_REFRESHED": str(self.root / "refreshed"),
        }

    def run_action(self, stream: str, **env: str) -> subprocess.CompletedProcess[str]:
        Path(self.env["FAKE_EVENTS"]).write_text(stream)
        return subprocess.run(
            ["bash", str(ACTION / "run.sh")],
            cwd=self.root,
            env={**self.env, **env},
            text=True,
            capture_output=True,
            timeout=15,
        )

    def assert_failed(self, run: subprocess.CompletedProcess[str], reason: str) -> None:
        self.assertNotEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertIn(reason, run.stdout)
        self.assertIn(reason, (self.root / "summary").read_text())
        execution = json.loads((self.out / "execution.json").read_text())
        self.assertEqual(execution[-1]["subtype"], "error_during_execution")
        self.assertIn(reason, execution[-1]["result"])
        self.assertIn(
            f"execution-file={self.out}/execution.json",
            (self.root / "output").read_text(),
        )

    def test_success_preserves_outputs_and_usage(self) -> None:
        run = self.run_action(events())
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertIn("→ read", run.stdout)
        self.assertEqual((self.out / "final.txt").read_text(), "Finished.\n")
        execution = json.loads((self.out / "execution.json").read_text())
        self.assertEqual(execution[0]["session_id"], "session-test")
        self.assertEqual(execution[0]["tools"], ["read"])
        self.assertEqual(execution[-1]["subtype"], "success")
        self.assertEqual(execution[-1]["result"], "Finished.")
        self.assertEqual(execution[-1]["total_cost_usd"], 0.012)
        self.assertEqual(
            execution[-1]["usage"],
            {
                "input_tokens": 11,
                "output_tokens": 7,
                "cache_read_input_tokens": 3,
                "cache_creation_input_tokens": 2,
            },
        )
        output = (self.root / "output").read_text()
        self.assertIn("structured_output<<PI_EOF_", output)
        self.assertIn('{"verdict":"ready"}', output)

    def test_provider_error_with_exit_zero_fails(self) -> None:
        self.assert_failed(self.run_action(events(stop="error")), "model call failed")

    def test_missing_result_fails(self) -> None:
        self.assert_failed(
            self.run_action(events(), FAKE_RESULT="false"),
            "model finished without calling submit_result",
        )

    def test_nonzero_exit_fails_even_after_result(self) -> None:
        self.assert_failed(self.run_action(events(), FAKE_EXIT="9"), "pi exited with 9")

    def test_unexpected_declared_or_executed_tools_fail(self) -> None:
        for extra in [
            {
                "type": "message_end",
                "message": {"role": "system", "toolsAdded": [{"name": "bash"}]},
            },
            {"type": "tool_execution_end", "toolName": "bash", "isError": False},
        ]:
            with self.subTest(event=extra["type"]):
                self.assert_failed(
                    self.run_action(events() + json.dumps(extra) + "\n"),
                    "tools outside the allowlist were declared or used: bash",
                )

    def test_blocked_tool_does_not_fail_successful_run(self) -> None:
        blocked = {
            "type": "tool_execution_end",
            "toolName": "bash",
            "isError": True,
            "result": {
                "content": [
                    {"type": "text", "text": "tool bash is not allowed in this run"}
                ]
            },
        }
        run = self.run_action(events() + json.dumps(blocked) + "\n")
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertIn("blocked calls: 1", (self.root / "summary").read_text())

    def test_tool_whitespace_is_normalized_for_pi_and_audit(self) -> None:
        run = self.run_action(events(), PI_TOOLS=" read, grep ,find, ls , ")
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        calls = [
            json.loads(line)
            for line in (self.root / "calls.jsonl").read_text().splitlines()
        ]
        invocation = next(args for args in calls if "--mode" in args)
        self.assertEqual(
            invocation[invocation.index("--tools") + 1],
            "read,grep,find,ls,submit_result",
        )

    def test_malformed_events_fail_with_diagnostics_and_raw_stream(self) -> None:
        for malformed in ["not JSON\n", '{"type":\n', "42\n"]:
            with self.subTest(malformed=malformed):
                # Exceed pipe capacity: the live display must keep draining after bad input.
                stream = malformed + '{"type":"agent_start"}\n' * 10000 + events()
                run = self.run_action(stream)
                self.assert_failed(run, "invalid JSON events")
                self.assertEqual((self.out / "events.jsonl").read_text(), stream)
                self.assertIn(
                    "fake pi diagnostic", (self.out / "stderr.log").read_text()
                )

    def test_allowlisted_missing_model_does_not_refresh_catalog(self) -> None:
        run = self.run_action(events(), FAKE_KNOWN="false")
        self.assertNotEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertFalse((self.root / "refreshed").exists())
        self.assertIn("not in pi's model catalog", run.stdout)

    def test_explicit_model_probe_can_refresh_catalog(self) -> None:
        run = self.run_action(
            events(), FAKE_KNOWN="false", PI_ALLOW_UNLISTED_MODEL="true"
        )
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertTrue((self.root / "refreshed").exists())


class TestExtensions(unittest.TestCase):
    def test_extension_behavior(self) -> None:
        run = subprocess.run(
            [
                "node",
                "--experimental-strip-types",
                "--test",
                str(ACTION / "tests/tool-guard.mjs"),
                str(ACTION / "tests/submit-result.mjs"),
            ],
            text=True,
            capture_output=True,
            timeout=15,
        )
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)


if __name__ == "__main__":
    unittest.main()
