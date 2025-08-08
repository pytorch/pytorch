# tests/test_cli.py
import io
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import MagicMock, patch

from cli.run import main

from utils import create_temp_yaml


class TestArgparseCLI(unittest.TestCase):
    @patch("cli.build_cli.register_build.VllmBuildRunner")
    def test_cli_run_build_external(self, mock_runner_cls):
        mock_runner = MagicMock()
        mock_runner_cls.return_value = mock_runner

        test_args = ["cli.run", "build", "external", "vllm"]
        with patch.object(sys, "argv", test_args):
            stdout = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
                main()

        mock_runner_cls.assert_called_once_with(config_path=None)
        mock_runner.run.assert_called_once()

        output = stdout.getvalue()
        self.assertIn("Running external build for target: vllm", output)

    @patch("cli.build_cli.register_build.VllmBuildRunner")
    def test_cli_with_fake_config_build_vllm(self, mock_runner_cls):
        mock_runner = MagicMock()
        mock_runner_cls.return_value = mock_runner

        config_path = create_temp_yaml({"some": "config"})
        test_args = ["cli.run", "--config", config_path, "build", "external", "vllm"]

        with patch.object(sys, "argv", test_args):
            stdout = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
                try:
                    main()
                except SystemExit as e:
                    self.fail(f"Exited unexpectedly: {e}")

        mock_runner_cls.assert_called_once_with(config_path=config_path)
        mock_runner.run.assert_called_once()

        output = stdout.getvalue()
        self.assertIn("Running external build for target: vllm", output)

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
