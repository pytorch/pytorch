# tests/test_cli.py
import io
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from cli.run import main

from utils import create_temp_yaml


class TestArgparseCLI(unittest.TestCase):
    def test_cli_run_build_external(self):
        test_args = ["cli.run", "build", "external", "vllm"]

        with patch.object(sys, "argv", test_args):
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                main()

            output = stdout.getvalue()
            self.assertIn("vllm", output)  # or any output you expect
            print(output)

    def test_cli_with_fake_config_build_vllm(self):
        config_path = create_temp_yaml({"build_target": {"build": "vllm"}})
        target = "vllm"
        test_args = [
            "cli.run",
            "--config",
            config_path,
            "build",
            "external",
            target,
        ]

        with patch.object(sys, "argv", test_args):
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                try:
                    main()
                except SystemExit as e:
                    self.fail(f"Exited unexpectedly: {e}")

            output = stdout.getvalue()
            self.assertIn(f"Running external build for target: {target}", output)

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
