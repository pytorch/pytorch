# Owner(s): ["module: tests"]

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


# A source tree without a built extension cannot import common_utils.
# This hook does not need torch. CI, where torch imports, uses the real TestCase.
try:
    from torch.testing._internal.common_utils import run_tests, TestCase
except ImportError:
    from unittest import TestCase

    def run_tests() -> None:
        unittest.main()


_CHILD = textwrap.dedent(
    """\
    import os
    import unittest

    import pytest

    def _ran(name):
        with open(os.environ["RERUN_SKIP_MARKER"], "a", encoding="utf-8") as fh:
            fh.write(name + "\\n")

    @unittest.skip("unconditional")
    def test_unconditional_skip():
        _ran("unconditional")

    @unittest.skipIf(True, "missing capability")
    def test_skipif_true():
        _ran("skipif")

    @pytest.mark.skip(reason="unconditional pytest skip")
    def test_pytest_skip():
        _ran("pytest_skip")

    @pytest.mark.skipif(True, reason="missing capability")
    def test_pytest_skipif():
        _ran("pytest_skipif")

    def test_normal():
        _ran("normal")

    def test_listed_disabled():
        _ran("listed")
    """
)


def _outcome(output: str, name: str) -> str | None:
    for line in output.splitlines():
        if f"::{name}" not in line:
            continue
        for status in ("PASSED", "FAILED", "SKIPPED", "ERROR"):
            if status in line:
                return status
    return None


def _marks(path: str) -> list[str]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


class TestRerunDisabledUnittestSkip(TestCase):
    def test_rerun_collects_unconditional_skip_and_keeps_skipif(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        test_dir = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(dir=test_dir) as tmp:
            child = Path(tmp) / "test_rerun_skip_child.py"
            child.write_text(_CHILD, encoding="utf-8")
            disabled = Path(tmp) / "disabled.json"
            # conftest matches "name (module.Class)". For a module-level test the
            # class group is the pytest module node's name, which is the file name.
            base = child.name
            disabled.write_text(
                json.dumps(
                    {
                        f"test_listed_disabled (mod.{base})": [
                            "https://github.com/pytorch/pytorch/issues/1",
                            [],
                        ],
                        f"test_skipif_true (mod.{base})": [
                            "https://github.com/pytorch/pytorch/issues/1",
                            [],
                        ],
                    }
                ),
                encoding="utf-8",
            )
            marker_on = str(Path(tmp) / "ran-on.txt")
            marker_off = str(Path(tmp) / "ran-off.txt")

            def run(rerun: bool, marker: str) -> tuple[int, str]:
                env = os.environ.copy()
                env["DISABLED_TESTS_FILE"] = str(disabled)
                env["RERUN_SKIP_MARKER"] = marker
                env["PYTHONDONTWRITEBYTECODE"] = "1"
                if rerun:
                    env["PYTORCH_TEST_RERUN_DISABLED_TESTS"] = "1"
                else:
                    env.pop("PYTORCH_TEST_RERUN_DISABLED_TESTS", None)
                proc = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        str(child),
                        "-vv",
                        "--tb=short",
                        "-p",
                        "no:cacheprovider",
                    ],
                    cwd=repo,
                    env=env,
                    capture_output=True,
                    text=True,
                )
                return proc.returncode, proc.stdout + proc.stderr

            on_code, on_out = run(True, marker_on)
            off_code, off_out = run(False, marker_off)
            on_marks = _marks(marker_on)
            off_marks = _marks(marker_off)

        expected_on = ["unconditional", "pytest_skip", "listed"]
        self.assertEqual(on_code, 0, msg=on_out)
        self.assertEqual(on_marks, expected_on, msg=on_out)
        got_uncond = _outcome(on_out, "test_unconditional_skip")
        self.assertEqual(got_uncond, "PASSED", msg=on_out)
        self.assertEqual(_outcome(on_out, "test_pytest_skip"), "PASSED", msg=on_out)
        got_listed_on = _outcome(on_out, "test_listed_disabled")
        self.assertEqual(got_listed_on, "PASSED", msg=on_out)
        self.assertEqual(_outcome(on_out, "test_skipif_true"), "SKIPPED", msg=on_out)
        self.assertEqual(_outcome(on_out, "test_pytest_skipif"), None, msg=on_out)
        self.assertEqual(_outcome(on_out, "test_normal"), None, msg=on_out)
        self.assertEqual("skipif" in on_marks, False, msg=on_out)
        self.assertEqual("normal" in on_marks, False, msg=on_out)
        self.assertEqual("pytest_skipif" in on_marks, False, msg=on_out)

        expected_off = ["normal", "listed"]
        self.assertEqual(off_code, 0, msg=off_out)
        self.assertEqual(off_marks, expected_off, msg=off_out)
        got = _outcome(off_out, "test_unconditional_skip")
        self.assertEqual(got, "SKIPPED", msg=off_out)
        self.assertEqual(_outcome(off_out, "test_pytest_skip"), "SKIPPED", msg=off_out)
        self.assertEqual(_outcome(off_out, "test_skipif_true"), "SKIPPED", msg=off_out)
        got_markif = _outcome(off_out, "test_pytest_skipif")
        self.assertEqual(got_markif, "SKIPPED", msg=off_out)
        self.assertEqual(_outcome(off_out, "test_normal"), "PASSED", msg=off_out)
        got_listed = _outcome(off_out, "test_listed_disabled")
        self.assertEqual(got_listed, "PASSED", msg=off_out)

    def test_testcase_method_skip_cleared_and_skipif_stays(self) -> None:
        # Pytest's unittest items are methods. Prove the same helper the
        # collection hook calls will run a TestCase body and leave skipIf.
        test_dir = Path(__file__).resolve().parent
        script = textwrap.dedent(
            """\
            import os
            import sys
            import unittest

            sys.path.insert(0, os.environ["TEST_DIR"])
            import conftest

            class Example(unittest.TestCase):
                @unittest.skip("unconditional")
                def test_unconditional_skip(self):
                    Example.ran.append("unconditional")

                @unittest.skipIf(True, "missing capability")
                def test_skipif_true(self):
                    Example.ran.append("skipif")

            Example.ran = []

            class Item:
                def __init__(self, name):
                    self.cls = Example
                    self.name = name
                    self.originalname = name
                    self.obj = getattr(Example, name)

            item = Item("test_unconditional_skip")
            if not conftest._enable_unconditional_unittest_skip(item):
                raise SystemExit("expected unconditional skip to be enabled")
            skipif_item = Item("test_skipif_true")
            if conftest._enable_unconditional_unittest_skip(skipif_item):
                raise SystemExit("skipIf(True) must stay skipped")
            func = Example.test_unconditional_skip
            if not getattr(func, "__pt_unconditional_skip__", False):
                raise SystemExit("missing unconditional stamp")
            if getattr(func, "__unittest_skip__", False):
                raise SystemExit("unittest skip flag still set")

            result = unittest.TestResult()
            suite = unittest.defaultTestLoader.loadTestsFromTestCase(Example)
            suite.run(result)
            if Example.ran != ["unconditional"]:
                raise SystemExit(f"ran={Example.ran!r}")
            if len(result.skipped) != 1 or result.failures or result.errors:
                detail = (
                    f"skipped={result.skipped!r} "
                    f"failures={result.failures!r} errors={result.errors!r}"
                )
                raise SystemExit(detail)
            print("METHOD_OK")
            """
        )
        env = os.environ.copy()
        env["TEST_DIR"] = str(test_dir)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=test_dir,
            env=env,
            capture_output=True,
            text=True,
        )
        output = proc.stdout + proc.stderr
        self.assertEqual(proc.returncode, 0, msg=output)
        self.assertEqual("METHOD_OK" in proc.stdout, True, msg=output)


if __name__ == "__main__":
    run_tests()
