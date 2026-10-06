# Owner(s): ["module: tests"]

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


# A source tree without a built extension cannot import common_utils.
# This hook does not need torch. CI, where torch imports, uses the real TestCase.
try:
    from torch.testing._internal.common_utils import run_tests, TestCase
except ImportError:
    from unittest import TestCase

    def run_tests() -> None:
        unittest.main()


_REPO = Path(__file__).resolve().parents[1]
_HELPER = _REPO / "torch" / "testing" / "_internal" / "rerun_code_skip.py"


def _load_helper():
    spec = importlib.util.spec_from_file_location("rerun_code_skip", _HELPER)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load {_HELPER}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_CHILD = textwrap.dedent(
    """\
    import importlib.util
    import os
    import unittest

    def _load():
        spec = importlib.util.spec_from_file_location(
            "rerun_code_skip", os.environ["RERUN_CODE_SKIP_PY"]
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    helper = _load()
    rerun_code_skip = helper.rerun_code_skip
    rocm_message_is_known_bug = helper.rocm_message_is_known_bug

    def _ran(name):
        with open(os.environ["RERUN_SKIP_MARKER"], "a", encoding="utf-8") as fh:
            fh.write(name + "\\n")

    def _known_failure(predicate, reason):
        def deco(fn):
            if predicate:
                return rerun_code_skip(reason)(fn)
            return fn
        return deco

    def _skip_if_rocm(msg):
        # Same split as skipIfRocm: stamp only a known-bug message, and only
        # when the ROCm predicate is already true.
        def deco(fn):
            if os.environ.get("FAKE_ROCM") != "1":
                return fn
            reason = "skipIfRocm: " + msg
            if rocm_message_is_known_bug(msg):
                return rerun_code_skip(reason)(fn)
            return unittest.skipIf(True, reason)(fn)
        return deco

    _WINDOWS = os.environ.get("FAKE_WINDOWS") == "1"

    @_known_failure(_WINDOWS, "skipIfWindows: test doesn't currently work on the Windows stack")
    def test_known_failure():
        _ran("known")

    @_skip_if_rocm("https://github.com/pytorch/pytorch/issues/180006")
    def test_rocm_issue():
        _ran("issue")

    @_skip_if_rocm("ROCm may have different numerical behavior")
    def test_rocm_numerical():
        _ran("numerical")

    @_skip_if_rocm("PTX test")
    def test_rocm_ptx():
        _ran("ptx")

    @_skip_if_rocm("not supported by hipBLAS")
    def test_rocm_hipblas():
        _ran("hipblas")

    @unittest.skipIf(True, "ROCm version less than (9, 0) required")
    def test_version_floor():
        _ran("floor")
    """
)


def _outcome(output: str, name: str) -> str | None:
    for line in output.splitlines():
        if name not in line:
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


class TestRerunDisabledCodeSkip(TestCase):
    def test_message_split_matches_real_call_sites(self) -> None:
        helper = _load_helper()
        bug = helper.rocm_message_is_known_bug
        self.assertEqual(bug("https://github.com/pytorch/pytorch/issues/180006"), True)
        self.assertEqual(bug("test doesn't currently work on the ROCm stack"), True)
        self.assertEqual(bug("ROCm may have different numerical behavior"), True)
        self.assertEqual(bug("PTX test"), False)
        self.assertEqual(bug("PTX atan codegen is CUDA-specific"), False)
        self.assertEqual(bug("not supported by hipBLAS"), False)
        self.assertEqual(bug("NVIDIA-only API"), False)
        self.assertEqual(
            bug("expandable_segments mode is not supported on ROCm"), False
        )
        self.assertEqual(
            helper.has_rerun_skip_stamp(
                type("F", (), {"__pt_rerun_code_skip__": True})()
            ),
            True,
        )
        plain = type("F", (), {})()
        self.assertEqual(helper.has_rerun_skip_stamp(plain), False)

    def test_known_failure_and_rocm_message_split(self) -> None:
        test_dir = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(dir=test_dir) as tmp:
            child = Path(tmp) / "test_rerun_code_skip_child.py"
            child.write_text(_CHILD, encoding="utf-8")
            disabled = Path(tmp) / "disabled.json"
            base = child.name
            # Opposite-set and missing-API skips are listed so rerun mode keeps
            # them and we can see that the skip itself is not cleared.
            disabled.write_text(
                json.dumps(
                    {
                        f"test_rocm_ptx (mod.{base})": [
                            "https://github.com/pytorch/pytorch/issues/1",
                            [],
                        ],
                        f"test_rocm_hipblas (mod.{base})": [
                            "https://github.com/pytorch/pytorch/issues/1",
                            [],
                        ],
                        f"test_version_floor (mod.{base})": [
                            "https://github.com/pytorch/pytorch/issues/1",
                            [],
                        ],
                    }
                ),
                encoding="utf-8",
            )

            def run(
                rerun: bool, windows: bool, rocm: bool
            ) -> tuple[int, str, list[str]]:
                marker = str(
                    Path(tmp) / f"ran-{int(rerun)}-{int(windows)}-{int(rocm)}.txt"
                )
                env = os.environ.copy()
                env["DISABLED_TESTS_FILE"] = str(disabled)
                env["RERUN_SKIP_MARKER"] = marker
                env["RERUN_CODE_SKIP_PY"] = str(_HELPER)
                env["PYTHONDONTWRITEBYTECODE"] = "1"
                env["FAKE_WINDOWS"] = "1" if windows else "0"
                env["FAKE_ROCM"] = "1" if rocm else "0"
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
                    cwd=_REPO,
                    env=env,
                    capture_output=True,
                    text=True,
                )
                output = proc.stdout + proc.stderr
                return proc.returncode, output, _marks(marker)

            on_code, on_out, on_marks = run(True, True, True)
            off_code, off_out, off_marks = run(False, True, True)
            false_code, false_out, false_marks = run(True, False, False)
            plain_code, plain_out, plain_marks = run(False, False, False)

        self.assertEqual(on_code, 0, msg=on_out)
        self.assertEqual(_outcome(on_out, "test_known_failure"), "PASSED", msg=on_out)
        self.assertEqual(_outcome(on_out, "test_rocm_issue"), "PASSED", msg=on_out)
        self.assertEqual(_outcome(on_out, "test_rocm_numerical"), "PASSED", msg=on_out)
        self.assertEqual(_outcome(on_out, "test_rocm_ptx"), "SKIPPED", msg=on_out)
        self.assertEqual(_outcome(on_out, "test_rocm_hipblas"), "SKIPPED", msg=on_out)
        self.assertEqual(_outcome(on_out, "test_version_floor"), "SKIPPED", msg=on_out)
        self.assertEqual("known" in on_marks, True, msg=on_out)
        self.assertEqual("issue" in on_marks, True, msg=on_out)
        self.assertEqual("numerical" in on_marks, True, msg=on_out)
        self.assertEqual("ptx" in on_marks, False, msg=on_out)
        self.assertEqual("hipblas" in on_marks, False, msg=on_out)
        self.assertEqual("floor" in on_marks, False, msg=on_out)

        self.assertEqual(off_code, 0, msg=off_out)
        self.assertEqual(
            _outcome(off_out, "test_known_failure"), "SKIPPED", msg=off_out
        )
        self.assertEqual(_outcome(off_out, "test_rocm_issue"), "SKIPPED", msg=off_out)
        self.assertEqual(_outcome(off_out, "test_rocm_ptx"), "SKIPPED", msg=off_out)
        self.assertEqual("known" in off_marks, False, msg=off_out)
        self.assertEqual("issue" in off_marks, False, msg=off_out)

        # Predicate false: the decorator does not skip, so rerun mode has
        # nothing to bypass and drops the test. Normal mode runs the body.
        self.assertEqual(false_code, 0, msg=false_out)
        self.assertEqual(_outcome(false_out, "test_known_failure"), None, msg=false_out)
        self.assertEqual(_outcome(false_out, "test_rocm_issue"), None, msg=false_out)
        self.assertEqual("known" in false_marks, False, msg=false_out)
        self.assertEqual(plain_code, 0, msg=plain_out)
        self.assertEqual(
            _outcome(plain_out, "test_known_failure"), "PASSED", msg=plain_out
        )
        self.assertEqual(
            _outcome(plain_out, "test_rocm_issue"), "PASSED", msg=plain_out
        )
        self.assertEqual(_outcome(plain_out, "test_rocm_ptx"), "PASSED", msg=plain_out)
        self.assertEqual("known" in plain_marks, True, msg=plain_out)
        self.assertEqual("ptx" in plain_marks, True, msg=plain_out)

    def test_device_instantiation_stamp_runs_only_matching_device(self) -> None:
        # Pytest collection of a TestCase imports torch. This machine has no
        # build, so call the same helper the collection hook uses.
        test_dir = Path(__file__).resolve().parent
        script = textwrap.dedent(
            """\
            import functools
            import importlib.util
            import os
            import sys
            import unittest

            sys.path.insert(0, os.environ["TEST_DIR"])
            import conftest

            spec = importlib.util.spec_from_file_location(
                "rerun_code_skip", os.environ["RERUN_CODE_SKIP_PY"]
            )
            helper = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(helper)

            ran = []

            def body(self):
                ran.append(self.device_type)

            @functools.wraps(body)
            def wrapper(self):
                if getattr(self, "device_type", None) == "mps":
                    raise unittest.SkipTest("test doesn't work on MPS backend")
                return body(self)

            stamped = helper.stamp_rerun_device_skip(wrapper, "mps")

            class OnMPS:
                device_type = "mps"
                test_device = stamped

            class OnCPU:
                device_type = "cpu"
                test_device = stamped

            class Item:
                def __init__(self, cls):
                    self.cls = cls
                    self.name = "test_device"
                    self.originalname = "test_device"
                    self.obj = cls.test_device

            mps_item = Item(OnMPS)
            if not conftest._enable_unconditional_unittest_skip(mps_item):
                raise SystemExit("mps instantiation should run")
            cpu_item = Item(OnCPU)
            if conftest._enable_unconditional_unittest_skip(cpu_item):
                raise SystemExit("cpu instantiation must stay skipped")
            OnMPS.test_device(OnMPS())
            if ran != ["mps"]:
                raise SystemExit(f"ran={ran!r}")
            if OnCPU.test_device is not stamped:
                raise SystemExit("cpu method was replaced")
            print("DEVICE_OK")
            """
        )
        env = os.environ.copy()
        env["TEST_DIR"] = str(test_dir)
        env["RERUN_CODE_SKIP_PY"] = str(_HELPER)
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
        self.assertEqual("DEVICE_OK" in proc.stdout, True, msg=output)

    def test_process_report_records_code_skip_reason(self) -> None:
        sys.path.insert(0, str(_REPO))
        from tools.stats.check_disabled_tests import process_report

        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.xml"
            suite = ET.Element("testsuite")

            def add(name: str, kind: str, message: str = "") -> None:
                case = ET.SubElement(
                    suite,
                    "testcase",
                    name=name,
                    classname="pkg.Cls",
                    file="test_mod.py",
                )
                if kind == "skip":
                    ET.SubElement(case, "skipped", message=message)
                elif kind == "fail":
                    ET.SubElement(case, "failure", message=message)

            add("test_code_skip", "skip", "skipIfRocm: PTX test")
            add("test_pass", "pass")
            add("test_fail", "fail", "assertion failed")
            add(
                "test_accounted",
                "skip",
                json.dumps({"num_green": 1, "num_red": 2}),
            )
            ET.ElementTree(suite).write(report, encoding="unicode")
            stats = process_report(report)

        code = stats["test_code_skip;pkg.Cls;test_mod.py"]
        passed = stats["test_pass;pkg.Cls;test_mod.py"]
        failed = stats["test_fail;pkg.Cls;test_mod.py"]
        accounted = stats["test_accounted;pkg.Cls;test_mod.py"]
        self.assertEqual(code.get("code_skip"), "skipIfRocm: PTX test")
        self.assertEqual(code.get("num_green"), 0)
        self.assertEqual(code.get("num_red"), 0)
        self.assertEqual(passed.get("num_green"), 1)
        self.assertEqual(passed.get("num_red"), 0)
        self.assertEqual("code_skip" in passed, False)
        self.assertEqual(failed.get("num_green"), 0)
        self.assertEqual(failed.get("num_red"), 1)
        self.assertEqual(accounted.get("num_green"), 1)
        self.assertEqual(accounted.get("num_red"), 2)
        self.assertEqual("code_skip" in accounted, False)


if __name__ == "__main__":
    run_tests()
