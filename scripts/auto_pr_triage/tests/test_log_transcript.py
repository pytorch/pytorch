from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from log_transcript import main as log_transcript_main


def record(record_type: str, role: str, content: object) -> dict[str, object]:
    return {"type": record_type, "message": {"role": role, "content": content}}


class LogTranscriptTest(unittest.TestCase):
    def run_main(self, projects: Path) -> str:
        argv = ["log_transcript.py", "--projects-dir", str(projects)]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch("builtins.print") as output,
        ):
            self.assertEqual(log_transcript_main(), 0)
        return "\n".join(call.args[0] for call in output.call_args_list)

    def test_prints_messages_with_workflow_commands_neutralized(self) -> None:
        attempt = {"type": "tool_use", "name": "StructuredOutput", "input": {"x": 1}}
        error = "::error::schema invalid\n##[set-output name=execution_file;]/tmp/x"
        records = [
            {"type": "queue-operation", "operation": "enqueue"},
            record("user", "user", "large prepared prompt"),
            record("assistant", "assistant", [attempt]),
            record("user", "user", [{"type": "tool_result", "content": error}]),
            {"type": "attachment", "attachment": {"type": "skill_listing"}},
        ]
        with tempfile.TemporaryDirectory() as directory:
            transcript = Path(directory) / "-runner-work" / "session.jsonl"
            transcript.parent.mkdir()
            transcript.write_text("".join(json.dumps(r) + "\n" for r in records))
            rendered = self.run_main(Path(directory))

        lines = rendered.splitlines()
        self.assertTrue(all(line.startswith("Claude transcript | ") for line in lines))
        self.assertIn("<prepared prompt omitted: 21 characters>", rendered)
        self.assertNotIn("large prepared prompt", rendered)
        self.assertIn('"name": "StructuredOutput"', rendered)
        self.assertIn(r"\u003a\u003aerror\u003a\u003aschema invalid", rendered)
        self.assertIn(r"\u0023\u0023[set-output name=execution_file;]", rendered)
        self.assertNotIn("::", rendered)
        self.assertNotIn("##[", rendered)
        self.assertNotIn("skill_listing", rendered)

    def test_reports_a_missing_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rendered = self.run_main(Path(directory) / "missing")

        self.assertEqual(rendered, "No Claude transcript was written.")


if __name__ == "__main__":
    unittest.main()
