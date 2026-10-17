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
        env["PYTORCH_TEST_RERUN_DISABLED_TESTS"] = "1"
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

    def test_unittest_skip_wrapper_follows_rerun_mode(self) -> None:
        # Dynamo looks up unittest.case.skip by file and function name.
        # Importing the plugin must not replace that function unless rerun
        # mode is on. No Dynamo import: the file name is the check.
        test_dir = Path(__file__).resolve().parent
        script = textwrap.dedent(
            """\
            import os
            import sys
            import unittest
            import unittest.case

            sys.path.insert(0, os.environ["TEST_DIR"])
            import conftest

            want = os.environ.get("EXPECT_WRAPPER") == "1"
            case_skip = unittest.case.skip
            if unittest.skip is not case_skip:
                raise SystemExit("unittest.skip and unittest.case.skip differ")
            installed = case_skip is conftest._rerun_aware_unittest_skip
            original = case_skip is conftest._ORIG_UNITTEST_SKIP
            if installed != want or original == want:
                raise SystemExit(
                    f"want_wrapper={want} installed={installed} original={original}"
                )
            if not want:
                filename = case_skip.__code__.co_filename
                if os.path.basename(filename) != "case.py" or case_skip.__name__ != "skip":
                    raise SystemExit(
                        f"normal run saw {case_skip.__name__} from {filename}"
                    )
            print("WRAPPER_OK")
            """
        )
        env_base = os.environ.copy()
        env_base["TEST_DIR"] = str(test_dir)
        env_base["PYTHONDONTWRITEBYTECODE"] = "1"

        def run(wrapper_on: bool) -> tuple[int, str]:
            env = env_base.copy()
            env.pop("PYTORCH_TEST_RERUN_DISABLED_TESTS", None)
            env["EXPECT_WRAPPER"] = "1" if wrapper_on else "0"
            if wrapper_on:
                env["PYTORCH_TEST_RERUN_DISABLED_TESTS"] = "1"
            proc = subprocess.run(
                [sys.executable, "-c", script],
                cwd=test_dir,
                env=env,
                capture_output=True,
                text=True,
            )
            return proc.returncode, proc.stdout + proc.stderr

        off_code, off_out = run(False)
        on_code, on_out = run(True)
        self.assertEqual(off_code, 0, msg=off_out)
        self.assertEqual("WRAPPER_OK" in off_out, True, msg=off_out)
        self.assertEqual(on_code, 0, msg=on_out)
        self.assertEqual("WRAPPER_OK" in on_out, True, msg=on_out)

    def test_class_level_skip_runs_every_method(self) -> None:
        # Same helper the collection hook calls. A pytest TestCase import
        # pulls in torch, which this tree may not have built, so run the
        # class here. Rerun mode must keep and execute every method. Off
        # mode must leave the class skipped.
        test_dir = Path(__file__).resolve().parent
        script = textwrap.dedent(
            """\
            import os
            import sys
            import unittest

            rerun = os.environ.get("PYTORCH_TEST_RERUN_DISABLED_TESTS") == "1"
            sys.path.insert(0, os.environ["TEST_DIR"])
            import conftest

            @unittest.skip("class-level known failure")
            class TestSkippedClass(unittest.TestCase):
                def test_first_method(self):
                    TestSkippedClass.ran.append("first")

                def test_second_method(self):
                    TestSkippedClass.ran.append("second")

            TestSkippedClass.ran = []

            class Item:
                def __init__(self, name):
                    self.cls = TestSkippedClass
                    self.name = name
                    self.originalname = name
                    self.obj = getattr(TestSkippedClass, name)

            enabled = []
            for name in ("test_first_method", "test_second_method"):
                item = Item(name)
                if conftest._enable_unconditional_unittest_skip(item):
                    enabled.append(name)
                    reason = getattr(item, conftest._CODE_SKIP_REASON_ATTR, None)
                    if reason != "class-level known failure":
                        raise SystemExit(f"{name} reason={reason!r}")
            if rerun and enabled != ["test_first_method", "test_second_method"]:
                raise SystemExit(f"dropped methods: {enabled!r}")
            if not rerun and enabled:
                raise SystemExit(f"off mode enabled {enabled!r}")

            result = unittest.TestResult()
            suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestSkippedClass)
            suite.run(result)
            if rerun:
                if TestSkippedClass.ran != ["first", "second"]:
                    raise SystemExit(f"ran={TestSkippedClass.ran!r}")
                if result.skipped or result.failures or result.errors:
                    detail = (
                        f"skipped={result.skipped!r} "
                        f"failures={result.failures!r} errors={result.errors!r}"
                    )
                    raise SystemExit(detail)
            else:
                if TestSkippedClass.ran:
                    raise SystemExit(f"off mode ran {TestSkippedClass.ran!r}")
                if len(result.skipped) != 2 or result.failures or result.errors:
                    detail = (
                        f"skipped={result.skipped!r} "
                        f"failures={result.failures!r} errors={result.errors!r}"
                    )
                    raise SystemExit(detail)
            print("CLASS_OK")
            """
        )

        def run(rerun: bool) -> tuple[int, str]:
            env = os.environ.copy()
            env["TEST_DIR"] = str(test_dir)
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            if rerun:
                env["PYTORCH_TEST_RERUN_DISABLED_TESTS"] = "1"
            else:
                env.pop("PYTORCH_TEST_RERUN_DISABLED_TESTS", None)
            proc = subprocess.run(
                [sys.executable, "-c", script],
                cwd=test_dir,
                env=env,
                capture_output=True,
                text=True,
            )
            return proc.returncode, proc.stdout + proc.stderr

        on_code, on_out = run(True)
        off_code, off_out = run(False)
        self.assertEqual(on_code, 0, msg=on_out)
        self.assertEqual("CLASS_OK" in on_out, True, msg=on_out)
        self.assertEqual(off_code, 0, msg=off_out)
        self.assertEqual("CLASS_OK" in off_out, True, msg=off_out)


if __name__ == "__main__":
    run_tests()
