# Owner(s): ["module: spin"]

import ast
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from click.testing import CliRunner
from spin.tests.testutil import spin

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_S390X,
    parametrize,
    run_tests,
    subtest,
    TestCase,
)


if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


PYTORCH_ROOT = Path(__file__).parent.parent.parent
CMDS_PATH = PYTORCH_ROOT / ".spin" / "cmds.py"

# .spin is not an importable package, so load the command file by path. It puts
# the repository root on sys.path, where the source tree would shadow an
# installed torch for every later import in this process; undo that.
_spec = importlib.util.spec_from_file_location("spin_cmds", CMDS_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load {CMDS_PATH}")
cmds = importlib.util.module_from_spec(_spec)
_sys_path = list(sys.path)
_spec.loader.exec_module(cmds)
sys.path[:] = _sys_path


def _registered_commands():
    with open(PYTORCH_ROOT / "pyproject.toml", "rb") as f:
        sections = tomllib.load(f)["tool"]["spin"]["commands"]
    entries = [entry for section in sections.values() for entry in section]
    prefix = ".spin/cmds.py:"
    return [
        getattr(cmds, entry.removeprefix(prefix)).name
        for entry in entries
        if entry.startswith(prefix)
    ]


REGISTERED_COMMANDS = _registered_commands()
PYTEST = "/env/bin/pytest"
NN = "test/test_nn.py"
PYTEST_ARGS = [NN, "-k", "Linear", "-x"]
OPTIONS_FIRST = ["-k", "Linear", NN]
RUN_TEST = [sys.executable, "test/run_test.py"]
SUITE = ["-i", "test_nn", "--dry-run"]


class TestSpin(TestCase):
    @unittest.skipIf(IS_S390X, "pyrefly doesn't support s390x")
    def test_autotype(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            here = Path(__file__).parent
            untyped_file_name = "autotype_test_untyped.py"
            source = here / untyped_file_name
            dest = Path(tmp_dir) / untyped_file_name
            shutil.copy(source, dest)

            cwd = os.getcwd()
            os.chdir(PYTORCH_ROOT)
            spin("pyrefly", "infer", dest)
            # Post test code
            os.chdir(cwd)

            with open(dest) as f:
                retyped_contents = f.read()
            retyped_ast = ast.parse(retyped_contents)

            typed_file_name = "autotype_test_typed.py"
            with open(here / typed_file_name) as f:
                typed_contents = f.read()
            typed_ast = ast.parse(typed_contents)

            self.assertEqual(
                ast.dump(typed_ast),
                ast.dump(retyped_ast),
            )


@instantiate_parametrized_tests
class TestSpinTestCommand(TestCase):
    """The command lines `spin test` builds and how it reports failures."""

    def invoke(self, args, runner_rc=0, probe_rc=0, probe_stderr="", pytest_exe=PYTEST):
        """Runs `spin test ARGS` with the test runner, the torch probe and the
        pytest lookup mocked; returns (result, runner, probe, which)."""
        runner = SimpleNamespace(returncode=runner_rc)
        probe = SimpleNamespace(returncode=probe_rc, stderr=probe_stderr)
        with (
            mock.patch.object(cmds.spin.util, "run", return_value=runner) as run,
            mock.patch.object(cmds.subprocess, "run", return_value=probe) as probe_run,
            mock.patch.object(cmds.shutil, "which", return_value=pytest_exe) as which,
        ):
            result = CliRunner().invoke(cmds.test, args)
        return result, run, probe_run, which

    def test_no_arguments_prints_help(self):
        result, run, _, _ = self.invoke([])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Usage:", result.output)
        run.assert_not_called()

    @parametrize(
        "args,expected",
        [
            subtest((PYTEST_ARGS, [PYTEST, *PYTEST_ARGS]), name="pytest"),
            subtest((OPTIONS_FIRST, [PYTEST, *OPTIONS_FIRST]), name="options_first"),
            subtest((["--ci", *SUITE], [*RUN_TEST, *SUITE]), name="ci"),
            subtest(([*SUITE, "--ci"], [*RUN_TEST, *SUITE]), name="ci_flag_last"),
            subtest((["--ci"], [*RUN_TEST, "--help"]), name="ci_bare"),
        ],
    )
    def test_forwards_arguments(self, args, expected):
        result, run, probe, _ = self.invoke(args)
        self.assertEqual(result.exit_code, 0, result.output)
        run.assert_called_once_with(expected, sys_exit=False)
        probe.assert_not_called()

    def test_pytest_is_looked_up_in_the_scripts_directory(self):
        # Not next to sys.executable: a conda environment on Windows keeps
        # python.exe at its root and the entry points in Scripts.
        scripts = "/env/Scripts"
        with mock.patch.object(cmds.sysconfig, "get_path", return_value=scripts) as get:
            _, _, _, which = self.invoke([NN])
        get.assert_called_once_with("scripts")
        which.assert_called_once_with("pytest", path=scripts)

    def test_missing_pytest(self):
        result, run, _, _ = self.invoke([NN], pytest_exe=None)
        self.assertEqual(result.exit_code, 1)
        self.assertIn("pytest is not installed", result.output)
        run.assert_not_called()

    @parametrize(
        "args",
        [subtest([NN], name="pytest"), subtest(["--ci", "-i", "test_nn"], name="ci")],
    )
    def test_failure_passes_the_exit_code_through(self, args):
        result, _, probe, _ = self.invoke(args, runner_rc=3)
        self.assertEqual(result.exit_code, 3)
        self.assertNotIn("torch is not importable", result.output)
        # The probe must not run from the checkout, where the source tree
        # would shadow an installed torch.
        probe.assert_called_once()
        self.assertEqual(probe.call_args.kwargs["cwd"], tempfile.gettempdir())

    def test_failure_without_torch_points_at_the_build(self):
        stderr = "Traceback (most recent call last):\nModuleNotFoundError: No module named 'torch'\n"
        result, *_ = self.invoke([NN], runner_rc=2, probe_rc=1, probe_stderr=stderr)
        self.assertEqual(result.exit_code, 1)
        self.assertIn("torch is not importable", result.output)
        self.assertIn("No module named 'torch'", result.output)
        self.assertNotIn("Traceback", result.output)

    def test_failure_without_torch_and_without_stderr(self):
        result, _, _, _ = self.invoke([NN], runner_rc=2, probe_rc=1)
        self.assertEqual(result.exit_code, 1)
        self.assertTrue(result.output.endswith("`spin install`.\n"), result.output)


@instantiate_parametrized_tests
class TestSpinCli(TestCase):
    """Runs the real spin against the repository's pyproject.toml."""

    def run_spin(self, *args):
        cmd = [sys.executable, "-m", "spin", *args]
        return subprocess.run(cmd, cwd=PYTORCH_ROOT, capture_output=True, text=True)

    def test_registered_commands_are_listed(self):
        p = self.run_spin("--help")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertTrue(REGISTERED_COMMANDS)
        for name in REGISTERED_COMMANDS:
            self.assertRegex(p.stdout, rf"(?m)^\s+{re.escape(name)}\s")

    @parametrize("name", REGISTERED_COMMANDS, name_fn=lambda n: n.replace("-", "_"))
    def test_command_has_help(self, name):
        p = self.run_spin(name, "--help")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("Usage:", p.stdout)

    def test_test_runs_pytest(self):
        # Collection imports torch, so this goes through the real pytest entry
        # point against the torch under test. No cache: the outer run owns it.
        args = ["test/spincli/test_spin.py", "--collect-only", "-q"]
        p = self.run_spin("test", *args, "-p", "no:cacheprovider")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("test_spin.py::", p.stdout)

    def test_test_passes_the_exit_code_through(self):
        # pytest exits with 4 on a usage error such as a missing file.
        p = self.run_spin("test", "test/spincli/missing.py", "-p", "no:cacheprovider")
        self.assertEqual(p.returncode, 4, p.stdout + p.stderr)
        self.assertNotIn("torch is not importable", p.stderr)


if __name__ == "__main__":
    run_tests()
