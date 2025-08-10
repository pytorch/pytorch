# tests/test_cli.py
import io
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import MagicMock, patch

from cli.run import main


class TestArgparseCLI(unittest.TestCase):
    @patch("cli.build_cli.register_build.VllmBuildRunner")
    def test_cli_run_build_external(self, mock_runner_cls):
        mock_runner = MagicMock()
        mock_runner_cls.return_value = mock_runner

        test_args = ["cli.run", "build", "external", "vllm"]
        with patch.object(sys, "argv", test_args):
            with self.assertLogs(level="INFO") as caplog:
                # if argparse could exit on error, wrap in try/except SystemExit if needed
                main()

        # stdout print from your CLI plumbing
        # logs emitted inside your code (info/debug/error etc.)
        logs_text = "\n".join(caplog.output)
        self.assertIn("Running vllm build", logs_text)

    def test_build_help(self):
        test_args = ["cli.run", "build", "--help"]

        with patch.object(sys, "argv", test_args):
            stdout = io.StringIO()
            stderr = io.StringIO()

            # --help always raises SystemExit(0)
            with self.assertRaises(SystemExit) as cm:
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    main()

            self.assertEqual(cm.exception.code, 0)

            output = stdout.getvalue()
            self.assertIn("usage", output)
            self.assertIn(
                "external", output
            )  # assuming "external" is a subcommand of build


if __name__ == "__main__":
    unittest.main()
