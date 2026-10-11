# mypy: ignore-errors
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

from tools.linter.adapters.public_api_linter import (
    LINTER_CODE,
    LintMessage,
    LintSeverity,
    make_message,
    path_to_module,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_ADAPTER = _REPO_ROOT / "tools" / "linter" / "adapters" / "public_api_linter.py"

# Runs the adapter's checks in a subprocess where the repo root is appended
# to sys.path instead of prepended: a not-yet-built source tree must not
# shadow an installed torch (CI runs tools tests with PYTHONPATH=repo root),
# and a failed `import torch` poisons a process for good.
_DRIVER = """
import importlib.util, json, os, sys

repo, scenario = sys.argv[1], sys.argv[2]
sys.path[:] = [p for p in sys.path if p not in (repo, "")] + [repo]
spec = importlib.util.spec_from_file_location(
    "public_api_linter",
    os.path.join(repo, "tools", "linter", "adapters", "public_api_linter.py"),
)
linter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(linter)

try:
    import torch
except ImportError:
    sys.exit(3)

if scenario == "probe":
    sys.exit(0)

import json as js_stdlib

messages = []
if scenario == "clean":
    messages = linter.run_checks(["torch/optim/adam.py"])
elif scenario == "reexport":
    torch.public_api_lint_probe = js_stdlib.dumps
    messages = linter.run_checks(["torch/optim/adam.py"])
elif scenario == "names":
    import torch.optim.adam as adam_mod

    saved_all = list(adam_mod.__all__)
    adam_mod.public_api_lint_probe = js_stdlib.dumps
    adam_mod.__all__ = saved_all + ["public_api_lint_probe"]
    messages = linter.run_checks(["torch/optim/adam.py"])
elif scenario == "import-fail":
    import torch.testing._internal.common_utils as common_utils

    # The import check is skipped on Windows/macOS/Jetson by the test's
    # decorators; emulate a supported platform so the check itself runs.
    common_utils.IS_MACOS = False
    common_utils.IS_WINDOWS = False
    common_utils.IS_JETSON = False
    probe = os.path.join(repo, "torch", "optim", "_public_api_lint_probe.py")
    with open(probe, "w") as f:
        f.write("import module_that_does_not_exist_xyz\\n")
    try:
        messages = linter.run_checks(["torch/optim/_public_api_lint_probe.py"])
    finally:
        os.remove(probe)
else:
    raise SystemExit(f"unknown scenario {scenario}")

print(json.dumps([m._asdict() for m in messages]))
"""

_deps_cache: dict[str, bool] = {}


def _run_driver(scenario: str) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_DRIVER), str(_REPO_ROOT), scenario],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout


def _deps_available() -> bool:
    if "available" not in _deps_cache:
        code, _ = _run_driver("probe")
        _deps_cache["available"] = code == 0
    return _deps_cache["available"]


class TestPathToModule(unittest.TestCase):
    def test_maps_package_files(self) -> None:
        self.assertEqual(path_to_module("torch/__init__.py"), "torch")
        self.assertEqual(path_to_module("torch/optim/__init__.py"), "torch.optim")
        self.assertEqual(path_to_module("torch/optim/adam.py"), "torch.optim.adam")

    def test_ignores_paths_outside_real_torch_packages(self) -> None:
        for path in (
            "test/test_torch.py",
            "tools/linter/adapters/foo.py",
            "torch/csrc/foo.py",  # no __init__.py chain
            "torch/optim/adam.pyi",
            "torch",
        ):
            with self.subTest(path):
                self.assertIsNone(path_to_module(path))


class TestLintMessage(unittest.TestCase):
    def test_serializes_to_lintrunner_json(self) -> None:
        message = make_message("torch/optim/adam.py", "[module-names]", "boom")
        parsed = json.loads(json.dumps(message._asdict()))
        self.assertEqual(parsed["code"], LINTER_CODE)
        self.assertEqual(parsed["severity"], "error")
        self.assertEqual(parsed["path"], "torch/optim/adam.py")
        self.assertEqual(parsed["description"], "boom")
        self.assertIsInstance(message, LintMessage)
        self.assertEqual(message.severity, LintSeverity.ERROR)


class TestMain(unittest.TestCase):
    def test_exits_cleanly_with_no_messages(self) -> None:
        # Whether or not this environment can check the tree (no torch at
        # all, or an installed wheel), a clean file produces no messages and
        # a zero exit code.
        proc = subprocess.run(
            [sys.executable, str(_ADAPTER), "--", "torch/__init__.py"],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "")


class TestRunChecks(unittest.TestCase):
    def setUp(self) -> None:
        if not _deps_available():
            self.skipTest("requires an importable torch")

    def _messages(self, scenario: str) -> list[dict]:
        code, stdout = _run_driver(scenario)
        self.assertEqual(code, 0, stdout)
        return json.loads(stdout)

    def test_clean_module_produces_no_messages(self) -> None:
        self.assertEqual(self._messages("clean"), [])

    def test_detects_new_reexported_callable(self) -> None:
        hits = [
            m
            for m in self._messages("reexport")
            if m["name"] == "[reexported-callables]"
            and "public_api_lint_probe" in (m["description"] or "")
        ]
        self.assertEqual(len(hits), 1)

    def test_detects_module_names_violation(self) -> None:
        hits = [
            m
            for m in self._messages("names")
            if m["name"] == "[module-names]" and m["path"] == "torch/optim/adam.py"
        ]
        self.assertEqual(len(hits), 1)

    def test_detects_unimportable_module(self) -> None:
        hits = [
            m
            for m in self._messages("import-fail")
            if m["name"] == "[failed-import]"
            and "_public_api_lint_probe" in (m["description"] or "")
        ]
        self.assertEqual(len(hits), 1)


if __name__ == "__main__":
    unittest.main()
