from __future__ import annotations

import ast
import io
import itertools
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, TYPE_CHECKING
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from tools.testing import do_target_determination_for_s3
from tools.testing.discover_tests import BLOCKLISTED_PATTERNS, TESTS
from tools.testing.do_target_determination_for_s3 import (
    import_results,
    PREVIOUS_FAILURES_KEY,
    td_results_json,
)
from tools.testing.llm_td import select_tests, selection
from tools.testing.llm_td.selection import (
    apply_selection,
    DRIVER_MAPPINGS,
    needs_run_all,
    normalize_test_name,
    Selection,
    split_tests,
)
from tools.testing.target_determination import gen_artifact
from tools.testing.target_determination.heuristics import previously_failed_in_pr
from tools.testing.target_determination.heuristics.interface import TestPrioritizations
from tools.testing.target_determination.heuristics.previously_failed_in_pr import (
    get_all_previous_failures,
)
from tools.testing.target_determination.heuristics.utils import (
    td_enabled_job,
    td_enabled_workflow,
)
from tools.testing.test_run import TestRun


if TYPE_CHECKING:
    from collections.abc import Collection


CANDIDATES = {
    "test_nn",
    "test_ops",
    "dynamo/test_misc",
    "cpython/v3_13/test_set",
    "inductor/test_torchinductor",
}


class TestSelection(unittest.TestCase):
    def test_normalize(self) -> None:
        for name in [
            "test_nn",
            "test/test_nn.py",
            "./test/test_nn.py",
            "test_nn [12m]",
            "test_nn.py::TestNN",
        ]:
            self.assertEqual(normalize_test_name(name, CANDIDATES), "test_nn")
        self.assertEqual(
            normalize_test_name("dynamo.test_misc", CANDIDATES), "dynamo/test_misc"
        )
        self.assertIsNone(normalize_test_name("test_nonexistent", CANDIDATES))
        # A file a driver runs, or a misspelling a driver's regex matches, names no candidate.
        candidates = CANDIDATES | {"test_jit", "test_cpp_extensions_aot_ninja"}
        for name in ("jit/test_freezing", "test/jit/test_freezing.py", "cpp_extensions_jit"):  # fmt: skip
            self.assertIsNone(normalize_test_name(name, candidates), name)
            sel = Selection.from_model_output({"tests": [{"file": name}]}, candidates, [])  # fmt: skip
            self.assertTrue(sel.run_all and sel.error, name)

    def test_from_model_output(self) -> None:
        out = {
            "run_all": False,
            "reason": "r",
            "sweeps": ["dynamo_wrapped"],
            "dirs": ["test/dynamo/"],
            "tests": [{"file": "test/test_ops.py", "confidence": "high"}],
        }
        sel = Selection.from_model_output(
            out, CANDIDATES, ["torch/_dynamo/x.py", "test/test_nn.py"]
        )
        self.assertEqual(sel.files, {"test_ops", "test_nn", "dynamo/test_misc"})
        self.assertEqual(sel.sweeps, {"dynamo_wrapped"})
        self.assertFalse(sel.run_all)
        self.assertTrue(sel.covers("inductor/test_torchinductor", "dynamo_wrapped"))
        self.assertFalse(sel.covers("inductor/test_torchinductor", "default"))
        self.assertTrue(sel.covers("test_ops", "default"))
        # The inductor config compiles every test of the files it runs with --inductor.
        bare = Selection.from_model_output({"tests": []}, CANDIDATES | {"test_torch"}, [])  # fmt: skip
        self.assertTrue(bare.covers("test_torch", "inductor"))
        self.assertFalse(bare.covers("test_torch", "default"))
        self.assertFalse(bare.covers("inductor/test_torchinductor", "inductor"))
        self.assertIsNone(sel.error)
        self.assertEqual(Selection.from_json(sel.to_json()), sel)
        failed = Selection.from_model_output(None, CANDIDATES, [])
        self.assertIsNotNone(failed.error)
        self.assertEqual(Selection.from_json(failed.to_json()), failed)

    def test_always_covered_matches_test_sh(self) -> None:
        sh = (REPO_ROOT / ".ci/pytorch/test.sh").read_text()
        lines = sh.splitlines()
        # The top-level if/elif chain whose *inductor* branch calls test_inductor_shard.
        end = next(i for i, line in enumerate(lines) if line.startswith("  test_inductor_shard "))  # fmt: skip
        start = max(i for i in range(end) if lines[i].startswith("if [["))
        dispatch = re.compile(r'(?:el)?if \[\[ "?\$\{?TEST_CONFIG\}?"? ==? (\S+) \]\]; then$')  # fmt: skip
        branches = [m.group(1) for line in lines[start:end] if (m := dispatch.match(line))]  # fmt: skip
        globs = tuple(b.strip("*") for b in branches if b.startswith("*"))
        self.assertEqual(globs, selection.TEST_SH_GLOBS)
        self.assertFalse([b for b in branches if not b.startswith("*") and "inductor" in b])  # fmt: skip
        funcs = {"test_inductor_shard": "inductor", "test_inductor_cpp_wrapper_shard": "inductor_cpp_wrapper"}  # fmt: skip
        runs: dict[str, set[str]] = {}
        for func, body in re.findall(r"^(\w+)\(\) \{\n(.*?)^\}", sh, re.MULTILINE | re.DOTALL):  # fmt: skip
            for include in re.findall(r"run_test\.py --inductor \\\n\s+--include ([^\\\n]+)", body):  # fmt: skip
                runs.setdefault(funcs.get(func, func), set()).update(include.split())
        self.assertEqual(runs, selection.ALWAYS_COVERED)
        sel = Selection.from_model_output({"tests": []}, CANDIDATES | {"test_torch"}, [])  # fmt: skip
        for config, covered in [("inductor_amx", True), ("inductor_avx2", True), ("inductor_cpp_wrapper", True), ("inductor_core", False), ("inductor_huggingface", False), ("default", False)]:  # fmt: skip
            self.assertEqual(sel.covers("test_torch", config), covered, config)
        self.assertFalse(sel.covers("test_ops", "inductor_cpp_wrapper"))

    def test_files_the_model_never_saw_are_covered(self) -> None:
        sel = Selection.from_model_output({"tests": []}, CANDIDATES, [])
        self.assertEqual(sel.candidates, CANDIDATES)
        for f in ("test_openreg", "cpp/test_jit"):
            self.assertTrue(sel.covers(f, "default"), f)
            self.assertFalse(Selection().covers(f, "default"), f)
        self.assertFalse(sel.covers("test_nn", "default"))
        self.assertEqual(Selection.from_json(sel.to_json()), sel)

    def test_model_run_all_is_kept(self) -> None:
        out = {"run_all": True, "reason": "", "sweeps": [], "dirs": [], "tests": []}
        sel = Selection.from_model_output(out, CANDIDATES, ["torch/a.py"])
        self.assertTrue(sel.run_all)
        self.assertTrue(sel.covers("test_nn", "default"))
        self.assertTrue(Selection.from_json(sel.to_json()).covers("test_nn", "default"))

    def select_in_tree(
        self,
        sources: dict[str, str],
        candidates: set[str],
        changed: list[str],
        deleted: tuple[str, ...] = (),
    ) -> Selection:
        """from_model_output with an empty answer and a grep over `sources`, a test/ tree."""
        with tempfile.TemporaryDirectory() as d:
            for name, text in sources.items():
                (Path(d) / name).parent.mkdir(parents=True, exist_ok=True)
                (Path(d) / name).write_text(text)

            def grep(pattern: str) -> list[str]:
                cmd = ["grep", "-rlE", pattern, "."]
                found = subprocess.run(cmd, cwd=d, capture_output=True, text=True).stdout.split()  # fmt: skip
                return [f"test/{f.removeprefix('./')}" for f in found]

            return Selection.from_model_output({"tests": []}, candidates, changed, grep, deleted)  # fmt: skip

    def test_importers_are_selected(self) -> None:
        sources = {
            "dynamo/test_misc.py": "from inductor.test_torchinductor import check_model\n",
            "test_ops.py": "    import test_torchinductor\n",
            "test_nn.py": "from test_torchinductor_utils import x\nfrom torch._inductor.test_case import TestCase\n",
            "helper.py": "import test_torchinductor\n",
            "cpython/v3_13/test_set.py": "from helper import *\n",
            "jit/test_rel.py": "    from . import helper\n",
            "test_jit.py": "from jit.test_rel import *\n",
            "test_jit_legacy.py": "from test_jit import *\n",
        }  # fmt: skip
        candidates = CANDIDATES | {"test_jit", "test_jit_legacy"}
        sel = self.select_in_tree(sources, candidates, ["test/inductor/test_torchinductor.py"])  # fmt: skip
        expected = {"inductor/test_torchinductor", "dynamo/test_misc", "test_ops", "cpython/v3_13/test_set", "test_jit", "test_jit_legacy"}  # fmt: skip
        self.assertEqual(sel.files, expected)

    def test_import_forms(self) -> None:
        sources = {
            "inductor/ext/triton/device.py": "",
            "inductor/test_ext.py": "from ext.triton.device import X\n",
            "onnx/defs/model.py": "",
            "onnx/test_onnx.py": "from defs import word, model\n",
            "onnx/test_paren.py": "from defs import (\n    model,\n)\n",
            "pkg/__init__.py": "",
            "dynamo/test_misc.py": "import pkg.util\n",
            "test_autograd.py": "",
            "test_nn.py": 'test_autograd = load_test_module("test_autograd")\n',
        }
        candidates = CANDIDATES | {"inductor/test_ext", "onnx/test_onnx", "onnx/test_paren", "test_autograd"}  # fmt: skip
        for changed, expected in [
            ("test/inductor/ext/triton/device.py", {"inductor/test_ext"}),
            ("test/onnx/defs/model.py", {"onnx/test_onnx", "onnx/test_paren"}),
            ("test/pkg/__init__.py", {"dynamo/test_misc"}),
            ("test/test_autograd.py", {"test_autograd", "test_nn"}),
        ]:
            sel = self.select_in_tree(sources, candidates, [changed])
            self.assertEqual(sel.files, expected, changed)

    def test_data_file_readers_are_selected(self) -> None:
        sources = {
            "test_public_bindings.py": 'open("allowlist_for_publicAPI.json")\n',
            "test_ops.py": 'load("cpp_extensions/my_ext.cpp")\n',
            "test_nn.py": "class TestNN(TestCase):\n",
            "dynamo/test_misc.py": "from test_nn import TestNN\n",
            "test_tensorboard.py": 'class TestTensorBoardSummary(TestCase):\n    path = "TestTensorBoard." + name\n',
            "distributed/launcher/test_run.py": 'path("bin/test_script.py")\n',
            "distributed/launcher/bin/test_script.py": "",
            "distributed/launcher/bin/unused.py": "",
            "README.md": "notes.txt\n",
            "run_test.py": 'open("run_only.json")\n',
            "cpp/api/CMakeLists.txt": "cpp_data.bin\n",
        }  # fmt: skip
        drivers = {"test_cpp_extensions_aot_ninja", "test_cpp_extensions_aot_no_ninja", "test_autoload_enable", "test_autoload_disable"}  # fmt: skip
        candidates = CANDIDATES | drivers | {"test_public_bindings", "test_tensorboard", "distributed/launcher/test_run"}  # fmt: skip
        for changed, expected in [
            ("test/allowlist_for_publicAPI.json", {"test_public_bindings"}),
            # The drivers build every extension, and test_ops loads this one by its path.
            ("test/cpp_extensions/my_ext.cpp", {"test_ops"} | drivers),
            ("test/cpp_extensions/ext.cpp", drivers),
            ("test/cpp_data.bin", set()),
            ("test/expect/TestNNCPU.test_x-y.expect", {"test_nn", "dynamo/test_misc"}),
            # test_tensorboard.py builds this path for TestTensorBoardSummary.
            ("test/expect/TestTensorBoard.test_x.expect", {"test_tensorboard"}),
            # test_run.py runs this file by its path.
            ("test/distributed/launcher/bin/test_script.py", {"distributed/launcher/test_run"}),
            ("test/cpp/api/any.cpp", set()),
            ("test/README.md", set()),
            ("test/cpython/v3_13/test_set.diff", set()),
        ]:  # fmt: skip
            sel = self.select_in_tree(sources, candidates, [changed])
            self.assertEqual((sel.files, sel.run_all), (expected, False), changed)
        # Named only by a README, by run_test.py, which no test imports, or by
        # another edited data file that nothing reads.
        for changed in ("test/expect/TestGone.test_x.expect", "test/distributed/launcher/bin/unused.py", "test/notes.txt", "test/run_only.json"):  # fmt: skip
            sel = self.select_in_tree(sources, candidates, ["test/allowlist_for_publicAPI.json", changed])  # fmt: skip
            self.assertEqual((sel.run_all, sel.reason, sel.error), (True, f"nothing found that reads {changed}", None))  # fmt: skip
        sel = self.select_in_tree(sources | {"allowlist_for_publicAPI.json": "b.json\n"}, candidates, ["test/allowlist_for_publicAPI.json", "test/b.json"])  # fmt: skip
        self.assertEqual((sel.run_all, sel.reason), (True, "nothing found that reads test/b.json"))  # fmt: skip
        sel = self.select_in_tree(sources, CANDIDATES, ["test/cpp_extensions/ext.cpp"])
        self.assertEqual((sel.run_all, sel.reason), (True, "nothing found that reads test/cpp_extensions/ext.cpp"))  # fmt: skip
        # A deleted file, or the old path of a moved one, needs no reader.
        gone = ["test/gone.json", "test/distributed/launcher/bin/unused.py"]
        sel = self.select_in_tree(sources, candidates, gone, deleted=tuple(gone))
        self.assertEqual((sel.files, sel.run_all), (set(), False))
        # Except an expect file, which the tests of its class still assert against.
        gone = ["test/expect/TestNNCPU.test_x-y.expect"]
        sel = self.select_in_tree(sources, candidates, gone, deleted=tuple(gone))
        self.assertEqual((sel.files, sel.run_all), ({"test_nn", "dynamo/test_misc"}, False))  # fmt: skip
        many = [f"test/data{i}.json" for i in range(selection.MAX_DATA_FILES + 1)]
        sel = self.select_in_tree(sources, candidates, many)
        reason = f"more than {selection.MAX_DATA_FILES} edited files under test/ are not tests"
        self.assertEqual((sel.run_all, sel.reason, sel.error), (True, reason, None))

    def test_endless_import_chain_runs_all(self) -> None:
        calls: list[str] = []

        def grep(pattern: str) -> list[str]:
            calls.append(pattern)
            return [f"test/m{len(calls)}.py"]

        sel = Selection.from_model_output({"tests": []}, CANDIDATES, ["test/test_nn.py"], grep)  # fmt: skip
        self.assertEqual((sel.run_all, sel.error), (True, None))
        self.assertEqual(len(calls), selection.MAX_IMPORT_HOPS)

    def test_failure_list_edits_select_their_tests(self) -> None:
        sources = {
            "test_nn.py": "class TestNN(TestCase):\n",
            "cpython/v3_13/test_set.py": "    class TestSet(unittest.TestCase):\n",
            "inductor/test_torchinductor.py": "class TestSetCopy:\n",
        }
        for changed, files, sweeps in [
            ("test/dynamo_expected_failures/TestNNCPU.test_x", {"test_nn"}, set()),
            ("test/dynamo_skips/TestNN.test_x_cuda", {"test_nn"}, set()),
            ("test/inductor_skips/TestNNCUDA.test_x", {"test_nn"}, set()),
            ("test/dynamo_skips/CPython313-test_set-TestSet.test_x", {"cpython/v3_13/test_set"}, set()),
            ("test/dynamo_skips/TestGone.test_x", set(), {"dynamo_wrapped"}),
            ("test/compiled_autograd_skips/TestNN.test_x", {"test_nn"}, set()),
            ("test/compiled_autograd_skips/TestGone.test_x", set(), {"dynamo_wrapped"}),
            ("test/dynamo_skips/.gitkeep", set(), set()),
            # Not checked at import, and it names no test.
            ("test/compiled_autograd_skips/TestNN", set(), set()),
            # dynamo_test_failures.py rejects these names at import.
            ("test/dynamo_skips/TestNN_test_x", set(), {"dynamo_wrapped"}),
            ("test/dynamo_expected_failures/TestNN.test_x.y", set(), {"dynamo_wrapped"}),
            ("test/dynamo_skips/TestNN.x-TestNN.test_y", set(), {"dynamo_wrapped"}),
        ]:  # fmt: skip
            sel = self.select_in_tree(sources, CANDIDATES, [changed])
            self.assertFalse(sel.run_all, changed)
            self.assertEqual((sel.files, sel.sweeps), (files, sweeps), changed)
        for changed in ("test/inductor_expected_failures/TestGone.test_x", "test/inductor_skips/TestGone.test_x", "test/inductor_skips/TestNN.test_x.y"):  # fmt: skip
            sel = self.select_in_tree(sources, CANDIDATES, [changed])
            self.assertEqual((sel.run_all, sel.error), (True, None), changed)
        n = selection.MAX_FAILURE_LIST_CLASSES + 1
        many_sources = sources | {"test_nn.py": "".join(f"class TestNN{i}(TestCase):\n" for i in range(n))}  # fmt: skip
        many = [f"test/dynamo_skips/TestNN{i}.test_x" for i in range(n)]
        sel = self.select_in_tree(many_sources, CANDIDATES, many)
        self.assertEqual((sel.run_all, sel.files, sel.sweeps), (False, set(), {"dynamo_wrapped"}))  # fmt: skip
        # Without a grep, the classes cannot be looked up.
        sel = Selection.from_model_output({"tests": []}, CANDIDATES, ["test/dynamo_skips/TestNN.test_x"])  # fmt: skip
        self.assertEqual(sel.sweeps, {"dynamo_wrapped"})
        sel = Selection.from_model_output({"tests": []}, CANDIDATES, ["test/inductor_skips/TestNN.test_x"])  # fmt: skip
        self.assertTrue(sel.run_all)
        sel = Selection.from_model_output({"tests": []}, CANDIDATES, ["test/compiled_autograd_skips/TestNN.test_x"])  # fmt: skip
        self.assertEqual(sel.sweeps, {"dynamo_wrapped"})
        sel = Selection.from_model_output({"tests": []}, CANDIDATES, ["test/dynamo_skips/TestNN_test_x"])  # fmt: skip
        self.assertEqual(sel.sweeps, {"dynamo_wrapped"})

    def test_fallbacks_run_all(self) -> None:
        sel = Selection.from_model_output(None, CANDIDATES, ["torch/a.py"])
        self.assertEqual((sel.run_all, sel.error), (True, "no model output"))
        out = {"run_all": False, "reason": "", "sweeps": [], "dirs": [], "tests": []}
        sel = Selection.from_model_output(out, CANDIDATES, ["test/run_test.py"])
        self.assertEqual((sel.run_all, sel.error), (True, None))
        self.assertFalse(
            Selection.from_model_output(out, CANDIDATES, ["docs/a.md"]).run_all
        )

    def test_load_malformed_file(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "sel.json"
            stderr = io.StringIO()
            with mock.patch("sys.stderr", stderr):
                self.assertIsNone(Selection.load(p))
            self.assertEqual(stderr.getvalue(), "")
            for sel in (Selection(candidates={"test_nn"}), Selection(run_all=True), Selection(run_all=True, error="e")):  # fmt: skip
                p.write_text(json.dumps(sel.to_json()))
                self.assertEqual(Selection.load(p), sel)
            valid = Selection(candidates={"test_nn"}).to_json()
            # Without candidates, a selection would skip every file it does not name.
            bad = [{**valid, "version": 2}, {**valid, "version": True}, {**valid, "files": "test_nn"}, {**valid, "files": [1]}, {**valid, "run_all": "false"}, {**valid, "enforce": "false"}, {**valid, "reason": 1}, {**valid, "candidates": [1]}, {**valid, "candidates": None}, {**valid, "error": 1}]  # fmt: skip
            bad += [{k: v for k, v in valid.items() if k != key} for key in ("version", "candidates", "enforce", "error")]  # fmt: skip
            for text in (
                "{",
                '{"sweeps": []}',
                "[]",
                "[" * 100_000,
                *map(json.dumps, bad),
            ):
                p.write_text(text)
                stderr = io.StringIO()
                with mock.patch("sys.stderr", stderr):
                    self.assertIsNone(Selection.load(p), text[:100])
                self.assertTrue(stderr.getvalue().startswith("::warning::"), text[:100])
            sel = Selection({"test_nn"}, candidates={"test_nn", "test_ops"})
            p.write_text(json.dumps(sel.to_json()))
            self.assertEqual(Selection.load(p), sel)

    def test_previous_failures_skip_bad_entries_like_td(self) -> None:
        def scored() -> set[str]:
            heuristic = previously_failed_in_pr.PreviouslyFailedInPR()
            tp = heuristic.get_prediction_confidence(["test_nn", "test_ops"])
            return {str(x) for x in tp.get_all_tests() if tp.get_score(x) > 0}

        with (
            tempfile.TemporaryDirectory() as d,
            mock.patch.object(previously_failed_in_pr, "REPO_ROOT", Path(d)),
            mock.patch("sys.stdout"),
            mock.patch("sys.stderr"),
        ):
            folder = Path(d) / ".additional_ci_files"
            folder.mkdir()
            self.assertEqual(get_all_previous_failures(), set())
            (folder / "previous_failures.json").write_text(
                '{"test/test_nn.py::TestNN::test_a": true}'
            )
            (folder / "previous_failures_additional.json").write_text('[1, "test_ops"]')
            self.assertEqual(get_all_previous_failures(), {"test_nn", "test_ops"})
            self.assertEqual(scored(), {"test_nn", "test_ops"})
            for name in ("previous_failures.json", "previous_failures_additional.json"):
                for text in ("{", "1", '[["x"]]', "[[1]]", "[" * 100_000):
                    (folder / name).write_text(text)
                    self.assertIsNone(get_all_previous_failures(), (name, text[:10]))
                    # TD's determinator logs the error and skips the heuristic.
                    with self.assertRaises(
                        previously_failed_in_pr.PREVIOUS_FAILURES_ERRORS
                    ):
                        scored()
                (folder / name).unlink()
                (folder / name).mkdir()
                self.assertIsNone(get_all_previous_failures(), (name, "a directory"))
                (folder / name).rmdir()
                (folder / name).write_text("[]")

    def test_td_enabled(self) -> None:
        # run_test.py's --enable-td default, which the eval harness shares.
        for workflow in ("pull", "trunk", "rocm-mi300", "periodic-rocm-mi200"):
            self.assertTrue(td_enabled_workflow(workflow), workflow)
        for workflow in ("periodic", "inductor", "slow", "pull-rocm"):
            self.assertFalse(td_enabled_workflow(workflow), workflow)

        def job(env: str, macos=False, no_td=False, tests=" ") -> bool:
            return td_enabled_job(env, macos=macos, no_td=no_td, tests_to_include=tests)

        self.assertTrue(job("linux-jammy-py3.10"))
        self.assertFalse(job("linux-xpu"))
        self.assertFalse(job("linux-onnx"))
        self.assertFalse(job("linux", macos=True))
        self.assertFalse(job("linux", no_td=True))
        self.assertFalse(job("linux", tests="test_nn"))

    def test_td_results_round_trip(self) -> None:
        tp = TestPrioritizations(["test_nn", "test_ops"], {TestRun("test_nn"): 1})
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "td_results.json"
            for failed in ({"test_ops", "elsewhere", "test_nn"}, set(), None):
                td_results = td_results_json(tp, failed)
                if failed is not None:
                    self.assertEqual(td_results[PREVIOUS_FAILURES_KEY], sorted(failed))
                p.write_text(json.dumps(td_results))
                with mock.patch("sys.stderr"):
                    loaded, loaded_failed = import_results(p)
                self.assertEqual(loaded_failed, failed)
                scores = [(str(x), loaded.get_score(x)) for x in loaded.get_all_tests()]
                self.assertEqual(scores, [("test_nn", 1), ("test_ops", 0)])
        # get_score is an exact lookup: a part of a scored file scores 0.
        self.assertEqual(tp.get_score(TestRun("test_nn", included=["TestNN"])), 0.0)

    def test_td_job_writes_the_previous_failures(self) -> None:
        tp = TestPrioritizations(["test_nn", "test_ops"], {TestRun("test_nn"): 1})
        heuristics = mock.Mock(**{"get_aggregated_priorities.return_value": tp})
        fetches: dict[str, Any] = dict.fromkeys(
            [
                "get_test_times",
                "get_test_class_times",
                "get_test_file_ratings",
                "get_test_class_ratings",
                "get_td_heuristic_historial_edited_files_json",
                "get_td_heuristic_profiling_json",
                "copy_additional_previous_failures",
            ],
            mock.DEFAULT,
        )
        lastfailed = '{"test/test_ops.py::TestOps::test_a": true}'
        # A failed or skipped download leaves no cache, as does a PR without
        # one yet, so only the step's outcome makes the failures known.
        for outcome, cache, failed in (
            ("success", lastfailed, {"test_ops"}),
            ("success", None, set()),
            ("success", "{", None),
            ("failure", None, None),
            ("failure", lastfailed, None),
            (None, None, None),
        ):
            with tempfile.TemporaryDirectory() as d:
                folder = Path(d) / ".additional_ci_files"
                folder.mkdir()

                def copy_pytest_cache(cache: str | None = cache) -> None:
                    if cache is not None:
                        (folder / "previous_failures.json").write_text(cache)

                env = {"CI": "false"}
                if outcome is not None:
                    env["PYTEST_CACHE_DOWNLOAD_OUTCOME"] = outcome
                with (
                    mock.patch.multiple(
                        do_target_determination_for_s3,
                        REPO_ROOT=Path(d),
                        get_test_prioritizations=mock.Mock(return_value=heuristics),
                        copy_pytest_cache=copy_pytest_cache,
                        **fetches,
                    ),
                    mock.patch.object(previously_failed_in_pr, "REPO_ROOT", Path(d)),
                    mock.patch.dict(os.environ, env),
                    mock.patch("sys.stdout"),
                    mock.patch("sys.stderr"),
                ):
                    if outcome is None:
                        os.environ.pop("PYTEST_CACHE_DOWNLOAD_OUTCOME", None)
                    do_target_determination_for_s3.main()
                    _, loaded = import_results(Path(d) / "td_results.json")
                self.assertEqual(loaded, failed, (outcome, cache))

    def test_import_results(self) -> None:
        with tempfile.TemporaryDirectory() as d, mock.patch("sys.stdout"):
            p = Path(d) / "td_results.json"
            stderr = io.StringIO()
            with mock.patch("sys.stderr", stderr):
                tp, failed = import_results(p)
            self.assertEqual(
                (tp.get_all_tests(), failed, stderr.getvalue()), ([], None, "")
            )
            tp = TestPrioritizations(["test_nn", "test_ops"], {TestRun("test_nn"): 1})
            # An older writer stores only TD's ranking.
            for value in ("missing", "test_nn", [1], None):
                td_results = tp.to_json()
                if value != "missing":
                    td_results[PREVIOUS_FAILURES_KEY] = value
                p.write_text(json.dumps(td_results))
                stderr = io.StringIO()
                with mock.patch("sys.stderr", stderr):
                    self.assertIsNone(import_results(p)[1], value)
                message = (
                    "Previous failures not recorded in"
                    if value == "missing"
                    else "Ignoring the previous failures in"
                )
                self.assertIn(message, stderr.getvalue())
            td_results = {
                **tp.to_json(),
                PREVIOUS_FAILURES_KEY: ["test_nn", "elsewhere"],
            }
            p.write_text(json.dumps(td_results))
            loaded, failed = import_results(p)
            self.assertEqual(failed, {"test_nn", "elsewhere"})
            scores = [(str(x), loaded.get_score(x)) for x in loaded.get_all_tests()]
            self.assertEqual(scores, [("test_nn", 1), ("test_ops", 0)])

    def test_run_all_patterns(self) -> None:
        for f in [
            ".ci/docker/common/install_cuda.sh",
            ".ci/docker/requirements-ci.txt",
            ".ci/docker/build.sh",
            ".ci/docker/ci_commit_pins/triton.txt",
            ".ci/docker/ubuntu/Dockerfile",
            ".ci/pytorch/test.sh",
            ".ci/pytorch/multigpu-test.sh",
            ".ci/pytorch/win-test.sh",
            ".ci/pytorch/macos-test.sh",
            ".ci/pytorch/macos-common.sh",
            ".ci/pytorch/common.sh",
            ".ci/pytorch/common-build.sh",
            ".ci/pytorch/common_utils.sh",
            ".ci/pytorch/torch_trace.sh",
            ".ci/pytorch/win-test-helpers/setup_pytorch_env.bat",
            ".ci/pytorch/win-test-helpers/test_python_shard.bat",
            ".ci/pytorch/win-test-helpers/installation-helpers/activate_miniconda3.bat",
            ".github/workflows/_linux-test.yml",
            ".github/workflows/_win-test.yml",
            ".github/workflows/_mac-test.yml",
            ".github/workflows/_rocm-test.yml",
            ".github/workflows/_rocm-test-no-dind.yml",
            ".github/workflows/_xpu-test.yml",
            ".github/workflows/_s390x-test.yml",
            ".github/workflows/_tpu-test.yml",
            ".github/actions/setup-rocm/action.yml",
            ".github/actions/setup-win/action.yml",
            ".github/actions/setup-xpu/action.yml",
            ".github/actions/filter-test-configs/action.yml",
            ".github/actions/download-td-artifacts/action.yml",
            ".github/scripts/filter_test_configs.py",
            ".github/ci_commit_pins/vision.txt",
            ".github/ci_commit_pins/audio.txt",
            ".github/ci_commit_pins/spmd_types.txt",
            "torch/testing/_internal/torchci/plugin.py",
            "test/run_test.py",
            "test/conftest.py",
            "test/torch_np/conftest.py",
            "test/slow_tests.json",
            "torch/testing/_internal/dynamo_test_failures.py",
            "test/pytest_shard_custom.py",
            "pytest.ini",
            "torch/testing/_internal/common_utils.py",
            "torch/testing/_internal/common_device_type.py",
            "torch/testing/_internal/common_dtype.py",
            "torch/testing/_internal/common_cuda.py",
            "torch/testing/_comparison.py",
            "scripts/onnx/test.sh",
            ".ci/docker/triton_version.txt",
            ".ci/docker/triton_xpu_version.txt",
            ".ci/pytorch/numba-cuda-13.patch",
            ".ci/pytorch/win-test-helpers/test_python_jit_legacy.bat",
            "tools/testing/discover_tests.py",
            "tools/stats/import_test_stats.py",
            "tools/stats/upload_metrics.py",
        ]:
            self.assertTrue(needs_run_all([f]), f)
            self.assertTrue((REPO_ROOT / f).exists(), f)
        for f in [
            "docs/source/index.rst",
            ".github/workflows/lint.yml",
            ".ci/pytorch/build.sh",
            ".github/actions/setup-linux/action.yml",
            ".github/ci_commit_pins/vllm.txt",
        ]:
            self.assertFalse(needs_run_all([f]), f)

    def test_malformed_output_runs_all(self) -> None:
        malformed: list[Any] = [
            [],
            {},
            {"tests": "test_nn"},
            {"tests": ["test_nn"]},
            {"tests": [{"confidence": "high"}]},
            {"tests": [{"file": 3}]},
            {"tests": [], "dirs": "dynamo"},
            {"tests": [], "dirs": [1]},
            {"tests": [], "sweeps": None},
            {"tests": [], "sweeps": ["dynamo-wrapped"]},
            {"tests": [], "run_all": "false"},
            {"tests": [], "dirs": [""]},
            {"tests": [], "dirs": ["test"]},
            {"tests": [], "dirs": ["test/"]},
            {"tests": [], "dirs": ["dynamo_foo"]},
            {"tests": [], "dirs": ["dynamo/cpython"]},
            {"tests": [{"file": "nope"}]},
            {"tests": [{"file": "test_nn (12m)"}, {"file": "nn/test_conv"}]},
            {"tests": [{"file": "nope"}, {"file": "test_ops"}]},
        ]
        for out in malformed:
            sel = Selection.from_model_output(out, CANDIDATES, ["test/test_nn.py"])
            self.assertTrue(sel.run_all, out)
            self.assertIn("malformed", sel.reason)
            self.assertRegex(str(sel.error), "^malformed model output: ")

    def test_dirs(self) -> None:
        candidates = CANDIDATES | {"dynamo_foo/test_a"}
        for d in ["dynamo", "dynamo/", "dynamo/*", "dynamo/**", " test/dynamo/* "]:
            sel = Selection.from_model_output({"tests": [], "dirs": [d]}, candidates, [])  # fmt: skip
            self.assertEqual(sel.files, {"dynamo/test_misc"}, d)
        sel = Selection.from_model_output({"tests": [], "dirs": ["cpython"]}, candidates, [])  # fmt: skip
        self.assertEqual(sel.files, {"cpython/v3_13/test_set"})

    def test_edited_tests_select_their_driver(self) -> None:
        aot, autoload = {"test_cpp_extensions_aot_ninja", "test_cpp_extensions_aot_no_ninja"}, {"test_autoload_enable", "test_autoload_disable"}  # fmt: skip
        candidates = CANDIDATES | {"test_jit"} | aot | autoload
        for changed, expected in [
            ("test/jit/test_freezing.py", {"test_jit"}),
            ("test/cpp_extensions/extension.cpp", aot | autoload),
            ("test/test_cpp_extensions_aot.py", aot),
            ("test/test_autoload.py", autoload),
            ("test/test_jit_fuser_te.py", set()),
        ]:  # fmt: skip
            sel = Selection.from_model_output({"tests": []}, candidates, [changed])
            self.assertEqual(sel.files, expected, changed)

    def test_drivers_match_the_test_list(self) -> None:
        for test in DRIVER_MAPPINGS:
            self.assertIn(test, TESTS)
        # Directories whose files no candidate runs. torch_openreg/tests runs
        # through test_openreg, which is never a candidate and so is always
        # covered; edits there also select the cpp_extensions drivers.
        undriven = {"custom_backend", "mobile", "onnx_caffe2"}
        for pattern in BLOCKLISTED_PATTERNS:
            path = f"test/{pattern}/test_x.py"
            driven = any(re.search(r, path) for rs in DRIVER_MAPPINGS.values() for r in rs)  # fmt: skip
            self.assertEqual(driven, pattern not in undriven, pattern)


class TestSplit(unittest.TestCase):
    def split(
        self,
        sel: Selection,
        scores: dict[TestRun, float],
        times: dict[str, float],
        prev: Collection[str] = (),
        td: Collection[str] = (),
        share: float | None = None,
    ) -> tuple[list[str], list[str], selection.Split]:
        tp = TestPrioritizations(times, scores)
        ranked = tp.get_all_tests()
        split = sel.split(
            ranked,
            tp.get_score,
            times,
            "default",
            previously_failed=prev,
            td_include=td,
            share=share,
        )
        return [str(x) for x in split.include], [str(x) for x in split.exclude], split

    def test_hedge_is_greedy_in_rank_order(self) -> None:
        scores = {"a": 1, "b": 0.9, "c": 0.8, "d": 0.7, "untimed": 0.6, "neg": -0.5}
        times: dict[str, float] = {"a": 10, "b": 150, "c": 52, "d": 48, "untimed": 0}
        times |= {"zero": 1, "neg": 1, "rest": 738}
        # The budget is 20% of 1000s = 200s. b leaves 50s, so c (52s, fits
        # alone) is skipped and d (48s) still fits, leaving 2s that the
        # unscored and negatively scored 1s files may not use. With 19% d does
        # not fit, with 21% c does.
        ranked = {TestRun(f): s for f, s in scores.items()}
        include, exclude, split = self.split(Selection(files={"a"}), ranked, times)
        self.assertEqual(include, ["a", "b", "d"])
        self.assertEqual(exclude, ["c", "untimed", "rest", "zero", "neg"])
        self.assertEqual(split.covered, {"a"})
        self.assertEqual((split.hedged, split.budget), ({"b", "d"}, 200))
        for share, hedged in ((0.19, {"b"}), (0.21, {"b", "c"})):
            split = self.split(Selection(files={"a"}), ranked, times, share=share)[2]
            self.assertEqual(split.hedged, hedged, share)

    def test_nothing_chosen_keeps_td_include(self) -> None:
        # A targeted call such as `run_test.py --include test_cuda`: one file
        # is all of the call's time, so it never fits the 20% hedge.
        scores, times = {TestRun("test_cuda"): 1.0}, {"test_cuda": 100.0}
        for files, include in (({"a"}, []), ({"test_cuda"}, ["test_cuda"])):
            sel = Selection(files=files)
            got, _, split = self.split(sel, scores, times, td={"test_cuda"})
            self.assertEqual((got, split.chose_nothing), (include, not include))

    def test_partial_runs_of_a_file_are_charged_once(self) -> None:
        f_a, f_rest = TestRun("f", included=["A"]), TestRun("f", excluded=["A"])
        g_b = TestRun("g", included=["B"])
        scores = {f_a: 1, g_b: 1, f_rest: 0.6, TestRun("h"): 0.5}
        times: dict[str, float] = {"f": 40, "g": 30, "h": 130, "r": 800}
        # f's two scored runs cost 40s once, and g's positive class pulls in
        # its unscored rest, so h (130s) fits the 200s budget exactly.
        include, exclude, split = self.split(Selection(), scores, times)
        self.assertEqual(split.hedged, {"f", "g", "h"})
        self.assertEqual(exclude, ["r"])
        self.assertEqual(include, ["f, A", "g, B", "f, not (A)", "h", "g, not (B)"])

    def test_previous_failures_in_td_always_run_outside_the_budget(self) -> None:
        scores = {TestRun("s"): 1, TestRun("p"): 0.5, TestRun("q"): 0.4}
        times: dict[str, float] = {"a": 10, "s": 150, "p": 100, "q": 60, "big": 600}
        times |= {"untimed": 0, "rest": 80}
        sel = Selection(files={"a"})
        include, exclude, split = self.split(sel, scores, times)
        self.assertEqual((include, split.hedged), (["s", "a"], {"s"}))
        # p always runs and leaves the whole 200s budget to s, so q (60s)
        # still does not fit next to s.
        td = {"s", "p", "a"}
        include, exclude, split = self.split(sel, scores, times, {"p", "a"}, td)
        self.assertEqual(include, ["s", "p", "a"])
        self.assertEqual((split.previous_failures_added, split.hedged), ({"p"}, {"s"}))
        # big (three times the budget, unscored) and untimed also run because
        # TD runs them, and the hedge still gets the whole budget.
        scores[TestRun("tiny")] = 0.1
        times |= {"tiny": 1, "rest": 79}
        prev, td = {"p", "big", "untimed", "a", "elsewhere"}, {"p", "big", "untimed"}
        include, exclude, split = self.split(sel, scores, times, prev, td)
        self.assertEqual(include, ["s", "p", "tiny", "a", "big", "untimed"])
        self.assertEqual(exclude, ["q", "rest"])
        self.assertEqual((split.covered, split.budget), ({"a"}, 200))
        self.assertEqual(split.previous_failures_added, {"p", "big", "untimed"})
        self.assertEqual(split.hedged, {"s", "tiny"})

    def test_only_a_previous_failure_chosen_is_a_choice(self) -> None:
        # p is all of TD's include and over the budget, so only its previous
        # failure runs it.
        times = {"p": 500.0, "r": 500.0}
        include, exclude, split = self.split(
            Selection(), {TestRun("p"): 1}, times, prev={"p"}, td={"p"}
        )
        self.assertEqual((include, exclude), (["p"], ["r"]))
        self.assertEqual((split.previous_failures_added, split.hedged), ({"p"}, set()))
        self.assertFalse(split.chose_nothing)

    def test_previous_failures_outside_td_are_hedged_like_other_files(self) -> None:
        scores = {TestRun("s"): 1, TestRun("big"): 0.8, TestRun("p"): 0.5}
        times: dict[str, float] = {"s": 150, "big": 600, "p": 40, "rest": 210}
        # TD runs only s, so big (over budget) does not run and p is hedged
        # after s like any other scored file.
        include, exclude, split = self.split(
            Selection(), scores, times, prev={"big", "p"}, td={"s"}
        )
        self.assertEqual((include, exclude), (["s", "p"], ["big", "rest"]))
        self.assertEqual((split.previous_failures_added, split.hedged), (set(), {"s", "p"}))  # fmt: skip

    def test_run_all_and_sweeps_take_everything(self) -> None:
        for sel in (Selection(run_all=True), Selection(sweeps={"default"})):
            include, exclude, split = self.split(sel, {}, {"a": 1, "b": 1})
            self.assertEqual((include, exclude, split.hedged), (["a", "b"], [], set()))


class TestApplySelection(unittest.TestCase):
    TIMES: dict[str, float] = {"a": 10, "p": 20, "s": 170, "big": 600, "rest": 200}

    def apply(
        self,
        sel: Selection,
        td_enabled: bool = True,
        pr_number: int | None = 1,
        prev: Collection[str] | None = (),
        times: dict[str, float] | None = None,
        stderr: io.StringIO | None = None,
    ) -> selection.AppliedSelection:
        times = self.TIMES if times is None else times
        tp = TestPrioritizations(list(self.TIMES), {TestRun("p"): 1, TestRun("s"): 0.5})
        include, exclude = tp.get_top_per_tests(25)
        with (
            mock.patch.object(selection, "get_pr_number", return_value=pr_number),
            mock.patch("sys.stderr", io.StringIO() if stderr is None else stderr),
        ):
            return apply_selection(
                sel,
                include,
                exclude,
                tp.get_score,
                times,
                test_config="default",
                td_enabled=td_enabled,
                previously_failed=prev,
            )

    def metric(self, enforced: bool) -> dict[str, object]:
        # p previously failed, and s is hedged within the 200s budget.
        return {
            "list_id": "7916c4091b31",  # sha256 of the sorted test list
            "enforced": enforced,
            "td_enabled": True,
            "run_all": False,
            "selector_error": None,
            "num_tests": 5,
            "num_covered": 1,
            "num_td_include": 2,
            "num_hedged": 1,
            "num_previous_failures_added": 1,
            "kept_td": False,
            "previous_failures_known": True,
            "time_total": 1000,
            "time_covered": 10,
            "time_hedged": 170,
            "time_previous_failures_added": 20,
        }

    def test_gate(self) -> None:
        cases = itertools.product((False, True), (False, True), (None, 1), (None, "x"))  # fmt: skip
        for td_enabled, enforce, pr_number, error in cases:
            sel = Selection(files={"a"}, enforce=enforce, error=error)
            applied = self.apply(sel, td_enabled, pr_number)
            expected = td_enabled and enforce and pr_number is not None and not error
            case = (td_enabled, enforce, pr_number, error)
            self.assertEqual(applied.enforced, expected, case)
            self.assertEqual(applied.selection_metric["enforced"], expected)
            self.assertFalse(applied.kept_td)
            # No previous failures is known, not unknown.
            self.assertTrue(applied.previous_failures_known)
            self.assertTrue(applied.failure_metric(TestRun("a"))["previous_failures_known"])  # fmt: skip

    def test_not_enforced_measures_the_selection(self) -> None:
        applied = self.apply(Selection(files={"a"}), prev={"p", "elsewhere"})
        self.assertFalse(applied.enforced)
        self.assertEqual(applied.td_include, {"p", "s"})
        self.assertEqual(applied.selection_metric, self.metric(enforced=False))
        self.assertEqual(
            applied.failure_metric(TestRun("s")),
            {
                "failure": "s",
                "covered": False,
                "enforced": False,
                "td_enabled": True,
                "in_td_include": True,
                "hedged": True,
                "previous_failure_added": False,
                "kept_td": False,
                "previous_failures_known": True,
                "selector_error": None,
            },
        )

    def test_enforced_swaps_in_the_selection(self) -> None:
        sel = Selection(files={"a"}, enforce=True)
        applied = self.apply(sel, prev={"p", "elsewhere"})
        self.assertEqual([str(x) for x in applied.split.include], ["p", "s", "a"])
        self.assertEqual([str(x) for x in applied.split.exclude], ["big", "rest"])
        self.assertEqual(applied.selection_metric, self.metric(enforced=True))

    def test_enforced_log_line(self) -> None:
        stderr = io.StringIO()
        self.apply(Selection(files={"a"}, enforce=True), prev={"p"}, stderr=stderr)
        self.assertEqual(
            stderr.getvalue(),
            "LLM selection: running 3 of 5 test files (1 covered; 1 previous failures added; 1 hedged within 200s)\n",
        )

    def test_unknown_previous_failures_fall_back_to_td(self) -> None:
        # td_results.json from a writer that does not store them.
        stderr = io.StringIO()
        sel = Selection(files={"a"}, enforce=True)
        applied = self.apply(sel, prev=None, stderr=stderr)
        self.assertFalse(applied.enforced)
        self.assertEqual(
            stderr.getvalue(),
            "LLM selection: previous failures not recorded in the TD results, using TD\n",
        )
        # Without them p is hedged too, as the shadow metrics record.
        unknown = {
            "num_previous_failures_added": 0,
            "time_previous_failures_added": 0,
            "num_hedged": 2,
            "time_hedged": 190,
            "previous_failures_known": False,
            "kept_td": True,
        }
        self.assertEqual(applied.selection_metric, self.metric(False) | unknown)
        failure = applied.failure_metric(TestRun("p"))
        keys = ("previous_failure_added", "kept_td", "previous_failures_known")
        self.assertEqual([failure[k] for k in keys], [False, True, False])
        shadow = self.apply(Selection(files={"a"}), prev=None)
        self.assertEqual((shadow.enforced, shadow.kept_td), (False, False))

    def test_no_test_times_falls_back_to_td(self) -> None:
        stderr = io.StringIO()
        sel = Selection(files={"a"}, enforce=True)
        applied = self.apply(sel, times={}, stderr=stderr)
        self.assertFalse(applied.enforced)
        self.assertEqual(stderr.getvalue(), "LLM selection: no test times, using TD\n")
        self.assertEqual(applied.selection_metric["time_total"], 0)
        self.assertTrue(applied.kept_td)
        # test-times.json is there but has none of the call's files.
        applied = self.apply(sel, times={"elsewhere": 100.0})
        self.assertEqual((applied.enforced, applied.kept_td), (False, True))

    def test_nothing_chosen_is_reported_as_td(self) -> None:
        # Neither scored file fits the 200s hedge and the selection names none.
        stderr = io.StringIO()
        times = {"p": 500.0, "s": 500.0}
        applied = self.apply(Selection(enforce=True), times=times, stderr=stderr)
        self.assertFalse(applied.enforced)
        self.assertEqual(stderr.getvalue(), "LLM selection: chose no files, using TD\n")
        self.assertTrue(applied.kept_td)
        failure = applied.failure_metric(TestRun("p"))
        self.assertEqual((failure["enforced"], failure["kept_td"]), (False, True))
        self.assertFalse(self.apply(Selection(), times=times).kept_td)

    def test_fallback_order(self) -> None:
        # No test times, then unknown previous failures, then choosing nothing.
        nothing = {"p": 500.0, "s": 500.0}
        for sel, times, prev, reason in (
            (Selection(files={"a"}, enforce=True), {}, None, "no test times"),
            (Selection(enforce=True), {}, (), "no test times"),
            (Selection(enforce=True), nothing, None, "previous failures not recorded in the TD results"),
        ):  # fmt: skip
            stderr = io.StringIO()
            applied = self.apply(sel, prev=prev, times=times, stderr=stderr)
            self.assertEqual((applied.enforced, applied.kept_td), (False, True))
            self.assertEqual(stderr.getvalue(), f"LLM selection: {reason}, using TD\n")

    def test_failure_metric(self) -> None:
        # big failed before but TD does not run it and it does not fit the
        # budget, so it does not run.
        applied = self.apply(Selection(files={"a"}, enforce=True), prev={"p", "big"})
        self.assertEqual([str(x) for x in applied.split.exclude], ["big", "rest"])
        expected = {
            "a": (True, False, False, False),
            "p": (False, True, False, True),
            "s": (False, True, True, False),
            "big": (False, False, False, False),
        }
        for f, (covered, in_td, hedged, prev) in expected.items():
            self.assertEqual(
                applied.failure_metric(TestRun(f)),
                {
                    "failure": f,
                    "covered": covered,
                    "enforced": True,
                    "td_enabled": True,
                    "in_td_include": in_td,
                    "hedged": hedged,
                    "previous_failure_added": prev,
                    "kept_td": False,
                    "previous_failures_known": True,
                    "selector_error": None,
                },
            )


def import_test_dashboard() -> Any:
    # upload_stats_lib imports boto3, which these tests never reach.
    boto3 = sys.modules.get("boto3") or mock.MagicMock()
    with mock.patch.dict(sys.modules, {"boto3": boto3}):
        from tools.stats import test_dashboard
    return test_dashboard


class TestSplitTests(unittest.TestCase):
    """run_test.py's wiring of the selection."""

    def split_with(
        self,
        sel: Selection | None,
        tp: TestPrioritizations | None = None,
        times: dict[str, float] | None = None,
        prev: Collection[str] | None = (),
        td_enabled: bool = True,
        artifacts: list[tuple[list, list, dict]] | None = None,
    ) -> tuple[selection.CallSplit, str]:
        """split_tests' result and its stderr; the artifacts it records go to
        `artifacts`."""
        tp = tp or TestPrioritizations(["a", "p"], {TestRun("p"): 1})
        include, exclude = tp.get_top_per_tests(25 if td_enabled else 100)
        stderr = io.StringIO()

        recorded = [] if artifacts is None else artifacts

        def gen(included: list, excluded: list, **kwargs: object) -> None:
            recorded.append((included, excluded, kwargs))

        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.object(selection, "get_pr_number", return_value=1),
            mock.patch("sys.stderr", stderr),
        ):
            selection_file = Path(tmp) / "sel.json"
            if sel is not None:
                selection_file.write_text(json.dumps(sel.to_json()))
            split = split_tests(
                test_prioritizations=tp,
                test_times={"a": 1, "p": 1} if times is None else times,
                selection_file=selection_file,
                test_config="default",
                td_enabled=td_enabled,
                previously_failed=prev,
                gen_artifact=None if artifacts is None else gen,
            )
        self.assertEqual((split.td_include, split.td_exclude), (include, exclude))
        return split, stderr.getvalue()

    def test_wiring(self) -> None:
        def to_json(runs: list[TestRun]) -> list[dict]:
            return [x.to_json() for x in runs]

        enforced_sel = Selection(files={"a"}, enforce=True, candidates={"a"})
        shadow_sel = Selection(files={"a"}, candidates={"a"})
        for sel, enforced in ((None, False), (shadow_sel, False), (enforced_sel, True)):
            artifacts: list[tuple[list, list, dict]] = []
            split, stderr = self.split_with(sel, artifacts=artifacts)
            self.assertEqual(split.enforced, enforced)
            include = ["p", "a"] if enforced else ["p"]
            self.assertEqual([str(x) for x in split.include], include)
            self.assertEqual([str(x) for x in split.exclude], [] if enforced else ["a"])
            self.assertEqual([str(x) for x in split.td_exclude], ["a"])
            self.assertEqual((split.llm is None), sel is None)
            self.assertNotIn("::warning::", stderr)
            self.assertEqual(split.source, "llm_selection" if enforced else "td")
            self.assertEqual("Running 25% of tests based on TD" in stderr, not enforced)
            source, td_excluded = split.source, to_json(split.td_exclude)
            recorded = (to_json(split.include), to_json(split.exclude), {"source": source, "td_excluded": td_excluded})  # fmt: skip
            self.assertEqual(artifacts, [recorded])

    def test_unknown_previous_failures_keep_td(self) -> None:
        sel = Selection(files={"a"}, enforce=True, candidates={"a"})
        split, stderr = self.split_with(sel, prev=None)
        self.assertFalse(split.enforced)
        self.assertEqual([str(x) for x in split.include], ["p"])
        self.assertIn("previous failures not recorded", stderr)

    def test_previous_failure_in_td_include_runs_when_enforced(self) -> None:
        # TD ranks class A of f first and the rest of f last, and f is over
        # the hedge budget, so only its previous failure runs it. r1 fits the
        # budget but TD does not score it.
        f_a = TestRun("f", included=["A"])
        files = ["a", "f", "r1", "r2", "r3"]
        tp = TestPrioritizations(files, {f_a: 1})
        self.assertEqual([str(x) for x in tp.get_top_per_tests(25)[0]], ["f, A", "a"])
        sel = Selection(files={"a"}, enforce=True, candidates=set(files))
        times: dict[str, float] = {"a": 1, "f": 300, "r1": 150, "r2": 233, "r3": 233}
        for prev, runs in (({"f"}, ["f, A", "a", "f, not (A)"]), (set(), ["a"])):
            split, _ = self.split_with(sel, tp, times, prev)
            self.assertTrue(split.enforced)
            self.assertEqual([str(x) for x in split.include], runs)

    def test_metrics_tell_a_failed_selector_from_a_run_all(self) -> None:
        # select_tests clears enforce when the selector failed, and a file
        # that sets both is not enforced either.
        failed = Selection(run_all=True, reason="selector failed: x", error="x")
        failed_enforce = Selection(run_all=True, reason="selector failed: x", enforce=True, error="x")  # fmt: skip
        deliberate = Selection(run_all=True, reason="infra", enforce=True)
        cases = ((failed, "x", False), (failed_enforce, "x", False), (deliberate, None, True))  # fmt: skip
        for sel, error, enforced in cases:
            split, stderr = self.split_with(sel)
            if split.llm is None:
                raise AssertionError("no selection read")
            self.assertEqual(split.enforced, enforced)
            self.assertFalse(split.llm.kept_td)
            self.assertEqual("based on TD" in stderr, not enforced)
            self.assertEqual(split.llm.selection_metric["selector_error"], error)
            failure = split.llm.failure_metric(TestRun("a"))
            self.assertEqual(failure["selector_error"], error)

    def test_td_disabled(self) -> None:
        for sel in [None, Selection(files={"a"}, enforce=True, candidates={"a"})]:
            split, stderr = self.split_with(sel, td_enabled=False)
            self.assertFalse(split.enforced)
            self.assertEqual([str(x) for x in split.include], ["p", "a"])
            self.assertIn("Running all tests", stderr)
            self.assertNotIn("based on TD", stderr)

    def test_run_test_main_wiring(self) -> None:
        # main() passes the TD results, the selection file and, in CI,
        # gen_ci_artifact to split_tests, and runs the split it returns; the
        # tests above cover split_tests. emit_metric records its caller's
        # file, module and function, which queries of the metrics filter on,
        # so main() emits them, in CI and only when there is a selection.
        # Locals are matched by where they are bound, expressions by their AST.
        tree = ast.parse((REPO_ROOT / "test" / "run_test.py").read_text())
        funcs = [f for f in tree.body if isinstance(f, ast.FunctionDef)]
        (main,) = [f for f in funcs if f.name == "main"]
        # Each call with the conjuncts of the if tests it runs under and its
        # enclosing for loops.
        calls: dict[str, list[tuple[set[str], list[ast.For], ast.Call]]] = {}
        bound: dict[str, list[list[str]]] = {}
        values: dict[str, list[ast.expr]] = {}
        assigned: dict[str, list[tuple[set[str], str]]] = {}

        def conjuncts(test: ast.expr) -> set[str]:
            if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
                return set().union(*(conjuncts(v) for v in test.values))
            return {ast.unparse(test)}

        def visit(node: ast.AST, guards: set[str], loops: list[ast.For]) -> None:
            if isinstance(node, ast.Call):
                calls.setdefault(ast.unparse(node.func), []).append((guards, loops, node))  # fmt: skip
            target, value = None, None
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                (target,), value = node.targets, node.value
            elif isinstance(node, ast.AnnAssign):
                target, value = node.target, node.value
            if target is not None and value is not None:
                elts = target.elts if isinstance(target, ast.Tuple) else [target]
                names = [ast.unparse(e) for e in elts]
                if isinstance(value, ast.Call):
                    bound.setdefault(ast.unparse(value.func), []).append(names)
                else:
                    values.setdefault(names[0], []).append(value)
                    assigned.setdefault(names[0], []).append((guards, ast.unparse(value)))  # fmt: skip
            if isinstance(node, ast.If):
                visit(node.test, guards, loops)
                for child in node.body:
                    visit(child, guards | conjuncts(node.test), loops)
                for child in node.orelse:
                    visit(child, guards | {f"not ({ast.unparse(node.test)})"}, loops)
                return
            if isinstance(node, ast.For):
                loops = [*loops, node]
            for child in ast.iter_child_nodes(node):
                visit(child, guards, loops)

        def src(node: ast.expr) -> str:
            """`node`'s source, through a local bound once in main() to a non-call."""
            if isinstance(node, ast.Name) and len(values.get(node.id, [])) == 1:
                node = values[node.id][0]
            return ast.unparse(node)

        def canon(expr: str) -> str:
            return ast.unparse(ast.parse(expr, mode="eval"))

        visit(main, set(), [])
        # A non-blank TESTS_TO_INCLUDE turns TD off, even with an explicit --enable-td.
        ((include_env,),) = bound[canon('os.environ.get("TESTS_TO_INCLUDE", "").strip')]  # fmt: skip
        self.assertIn(({include_env}, "False"), assigned["options.enable_td"])
        ((tp, prev),) = bound["import_results"]
        ((file_times,),) = bound["load_test_file_times"]
        ((split,),) = bound["split_tests"]
        ((_, _, call),) = calls["split_tests"]
        self.assertEqual(call.args, [])
        kwargs = {str(k.arg): src(k.value) for k in call.keywords}
        expected = {
            "test_prioritizations": tp,
            "test_times": file_times,
            "selection_file": "REPO_ROOT / ADDITIONAL_CI_FILES_FOLDER / LLM_TD_SELECTION_FILE",
            "test_config": "TEST_CONFIG",
            "td_enabled": "options.enable_td",
            "previously_failed": prev,
            "gen_artifact": "gen_ci_artifact if IS_CI else None",
        }
        self.assertEqual(kwargs, {k: canon(v) for k, v in expected.items()})
        global_metrics = [(src(c.args[0]), g) for g, _, c in calls["add_global_metric"]]  # fmt: skip
        self.assertIn(("'call_id'", set()), global_metrics)
        batches = [src(c.args[1]) for _, _, c in calls["TestBatch"]]
        self.assertEqual(batches, [f"{split}.include", f"{split}.exclude"])
        ((test_batch,), _) = bound["TestBatch"]
        metrics = [(c.args[0].value, (g, loops, c.args[1])) for g, loops, c in calls["emit_metric"] if isinstance(c.args[0], ast.Constant)]  # fmt: skip
        emitted = dict(metrics)
        self.assertEqual(len(emitted), len(metrics), "a metric is emitted twice")
        guard, _, fields = emitted["llm_td_selection"]
        self.assertLessEqual({"IS_CI", f"{split}.llm is not None"}, guard)
        self.assertEqual(src(fields), f"{split}.llm.selection_metric")
        guard, loops, fields = emitted["llm_td_failure"]
        self.assertLessEqual({"IS_CI", f"{split}.llm is not None"}, guard)
        failures = loops[-1]
        self.assertEqual(src(failures.iter), f"{test_batch}.failures")
        target = failures.target
        test = target.elts[0] if isinstance(target, ast.Tuple) else target
        self.assertEqual(src(fields), f"{split}.llm.failure_metric({src(test)})")
        guard, loops, fields = emitted["td_test_failure_stats_v2"]
        self.assertLessEqual({"IS_CI"}, guard)
        self.assertEqual(loops[-1], failures)
        if not isinstance(fields, ast.Dict):
            raise AssertionError(f"not a dict: {ast.unparse(fields)}")
        by_key = {src(k): src(v) for k, v in zip(fields.keys, fields.values) if k}
        self.assertEqual(by_key["'llm_enforced'"], f"{split}.enforced")

    def test_run_test_enable_td_default(self) -> None:
        # parse_args's --enable-td default, with its locals inlined.
        tree = ast.parse((REPO_ROOT / "test" / "run_test.py").read_text())
        (parse_args,) = [f for f in tree.body if isinstance(f, ast.FunctionDef) and f.name == "parse_args"]  # fmt: skip
        local = {ast.unparse(n.targets[0]): n.value for n in parse_args.body if isinstance(n, ast.Assign)}  # fmt: skip
        calls = [c for c in ast.walk(parse_args) if isinstance(c, ast.Call) and c.args and ast.unparse(c.args[0]) == "'--enable-td'"]  # fmt: skip
        ((default,),) = [[k.value for k in c.keywords if k.arg == "default"] for c in calls]  # fmt: skip

        class Inline(ast.NodeTransformer):
            def visit_Name(self, node: ast.Name) -> ast.expr:
                return local.get(node.id, node)

        expected = 'IS_CI and get_pr_number() is not None and td_enabled_job(BUILD_ENVIRONMENT, macos=IS_MACOS, no_td=strtobool(os.environ.get("NO_TD", "False")), tests_to_include=os.environ.get("TESTS_TO_INCLUDE", "")) and td_enabled_workflow(os.environ.get("GITHUB_WORKFLOW", "slow"))'
        self.assertEqual(ast.unparse(Inline().visit(default)), ast.unparse(ast.parse(expected)))  # fmt: skip


class TestDashboardExclusions(unittest.TestCase):
    def exclusions(
        self, artifacts: dict[str, dict], jobs: dict[str, str]
    ) -> tuple[dict, list[str]]:
        """get_td_exclusions over `artifacts`, and the artifacts whose jobs it looked up."""
        test_dashboard = import_test_dashboard()
        looked_up: list[str] = []

        def unzip(_: object) -> None:
            for name, content in artifacts.items():
                Path(name).write_text(json.dumps(content))

        def get_job_name(path: Path) -> str:
            looked_up.append(path.name)
            return jobs.get(path.name, "linux / test (default, 1, 2, x)")

        cwd = os.getcwd()
        try:
            with mock.patch.multiple(
                test_dashboard,
                download_s3_artifacts=mock.Mock(return_value=["x"]),
                unzip=unzip,
                get_job_id=lambda path: path,
                get_job_name=get_job_name,
            ):
                exclusions = test_dashboard.get_td_exclusions(1, 1)
        finally:
            os.chdir(cwd)
        return json.loads(json.dumps(exclusions)), sorted(looked_up)

    def test_exclusions_are_grouped_by_source(self) -> None:
        llm = gen_artifact.LLM_SELECTION_SOURCE
        with tempfile.TemporaryDirectory() as d:
            reports = Path(d) / "test" / "test-reports"
            reports.mkdir(parents=True)
            with mock.patch.object(gen_artifact, "REPO_ROOT", Path(d)):
                excluded = [{"test_file": "c"}]
                td_excluded = [{"test_file": "d"}]
                gen_artifact.gen_ci_artifact(
                    [], excluded, source=llm, td_excluded=td_excluded
                )
            (written,) = reports.glob("td_exclusions-*.json")
            llm_artifact = json.loads(written.read_text())

        b = [{"test_file": "b"}]
        artifacts = {
            "td_exclusions-1.json": {"included": [], "excluded": [{"test_file": "a"}]},
            "td_exclusions-2.json": {"included": [{"test_file": "c"}], "excluded": b, "source": "td"},
            "td_exclusions-3.json": llm_artifact,
            # An enforced call that excluded nothing.
            "td_exclusions-4.json": {**llm_artifact, "excluded": [], "td_excluded": []},
            "td_exclusions-5.json": {"included": [], "excluded": [], "source": "td"},
            # A call that excluded nothing but ran a file another call excluded.
            "td_exclusions-6.json": {"included": [{"test_file": "a"}], "excluded": [], "source": "td"},
        }  # fmt: skip
        jobs = {
            "td_exclusions-4.json": "linux / test (slow, 1, 1, x)",
            "td_exclusions-5.json": "linux / test (crossref, 1, 1, x)",
        }
        exclusions, _ = self.exclusions(artifacts, jobs)
        self.assertEqual(
            exclusions,
            {
                "td": {"linux": {"default": ["a", "b", "d"]}},
                # What a config with an enforced call excluded and none of its calls ran.
                "llm_selection": {"linux": {"default": ["b"], "slow": []}},
            },
        )
        # Without an enforced call, the calls that excluded nothing are not
        # looked up, and an artifact without a source is TD's.
        td_only = {k: artifacts[k] for k in (f"td_exclusions-{i}.json" for i in (1, 2, 5, 6))}  # fmt: skip
        exclusions, looked_up = self.exclusions(td_only, jobs)
        self.assertEqual(exclusions, {"td": {"linux": {"default": ["a", "b"]}}})
        self.assertEqual(looked_up, ["td_exclusions-1.json", "td_exclusions-2.json"])

    def test_a_file_split_by_class_ran_only_if_it_ran_whole(self) -> None:
        llm = gen_artifact.LLM_SELECTION_SOURCE
        a_in, a_out = {"test_file": "a", "included": ["A"]}, {"test_file": "a", "excluded": ["A"]}  # fmt: skip
        b_in, b_out = {"test_file": "b", "included": ["B"]}, {"test_file": "b", "excluded": ["B"]}  # fmt: skip
        artifacts = {
            "td_exclusions-1.json": {"included": [a_in, b_in], "excluded": [a_out, b_out], "source": llm},
            "td_exclusions-2.json": {"included": [{"test_file": "b"}], "excluded": [], "source": "td"},
        }  # fmt: skip
        exclusions, _ = self.exclusions(artifacts, {})
        self.assertEqual(exclusions["llm_selection"], {"linux": {"default": ["a"]}})

    def test_both_sources_are_uploaded(self) -> None:
        dash = import_test_dashboard()
        td, llm = {"linux": {"default": ["a"]}}, {"linux": {"default": ["c"]}}
        for exclusions, expected in (
            ({}, [mock.call(1, 2, "additional_info/td_exclusions", [{}])]),
            ({"td": td}, [mock.call(1, 2, "additional_info/td_exclusions", [td])]),
            (
                {"td": td, "llm_selection": llm},
                [
                    mock.call(1, 2, "additional_info/td_exclusions", [td]),
                    mock.call(1, 2, "additional_info/llm_td_exclusions", [llm]),
                ],
            ),
        ):
            with (
                mock.patch.object(dash, "get_td_exclusions", return_value=exclusions),
                mock.patch.object(dash, "upload_workflow_stats_to_s3") as upload,
            ):
                dash.upload_additional_info(1, 2)
            self.assertEqual(upload.call_args_list, expected)


class TestPrompt(unittest.TestCase):
    def test_truncate_diff_keeps_every_file(self) -> None:
        diff = "\n".join(
            f"diff --git a/f{i} b/f{i}\n" + "+x\n" * 5000 for i in range(4)
        )
        short = select_tests.truncate_diff(diff, budget=8000)
        self.assertLessEqual(len(short), 8000)
        for i in range(4):
            self.assertIn(f"diff --git a/f{i} b/f{i}", short)

    def test_truncate_diff_large_file_keeps_budget(self) -> None:
        small = [f"diff --git a/s{i} b/s{i}\n" + "+y\n" * 20 for i in range(200)]
        big = "diff --git a/big b/big\n" + "+x\n" * 100_000
        short = select_tests.truncate_diff("\n".join([big, *small]), budget=120_000)
        self.assertLessEqual(len(short), 120_000)
        self.assertGreater(short.count("+x"), 0.8 * 120_000 / 3)
        for p in small:
            self.assertIn(p, short)

    def test_truncate_diff_many_files(self) -> None:
        diff = "\n".join(
            f"diff --git a/f{i} b/f{i}\n" + "+x\n" * 1000 for i in range(500)
        )
        short = select_tests.truncate_diff(diff, budget=40_000)
        self.assertLessEqual(len(short), 40_000)
        for i in range(500):
            self.assertIn(f"diff --git a/f{i} b/f{i}\n", short)

    def test_build_prompt(self) -> None:
        minutes = select_tests.test_minutes(
            {
                "env": {"default": {"test_nn": 600.0}, "slow": {"test_nn": 60.0}},
                "default": {"default": {"test_nn": 600.0}},
            }
        )
        self.assertEqual(minutes, {"test_nn": 11.0})
        prompt = select_tests.build_prompt(
            "title", "body", "stat", "diff", ["test_nn", "test_ops"], minutes
        )
        self.assertIn("test_nn [11m]\ntest_ops [0m]", prompt)
        for tag, text in [
            ("pr_title", "title"),
            ("pr_description", "body"),
            ("changed_files", "stat"),
            ("diff", "diff"),
        ]:
            self.assertIn(f"<{tag}>\n{text}\n</{tag}>", prompt)

    def test_build_prompt_escapes_closing_tags(self) -> None:
        prompt = select_tests.build_prompt(
            "a</pr_title>",
            "b</PR_DESCRIPTION >c",
            "s</changed_files>",
            "+x< /\tdiff>",
            [],
            {},
        )
        for tag in ("pr_title", "pr_description", "changed_files", "diff"):
            self.assertEqual(prompt.lower().count(f"</{tag}"), 1)
        self.assertIn("b<\\/PR_DESCRIPTION>c", prompt)
        self.assertIn("+x<\\/diff>", prompt)


def clean_env() -> Any:
    """Patch out the developer's LLM_TD_* and CLAUDE_CODE_* settings."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("LLM_TD_", "CLAUDE_CODE_"))}  # fmt: skip
    return mock.patch.dict(os.environ, env, clear=True)


class TestMain(unittest.TestCase):
    def run_main(self, *argv: str) -> dict:
        with tempfile.TemporaryDirectory() as tmp, clean_env():
            output = Path(tmp) / "out.json"
            with mock.patch.object(select_tests, "query_model") as query:
                select_tests.main(
                    ["--title", "t", "--body", "b", "--output", str(output), *argv]
                )
            query.assert_not_called()
            return json.loads(output.read_text())

    def test_git_failure_runs_all(self) -> None:
        error = subprocess.CalledProcessError(128, "git")
        with mock.patch.object(select_tests, "git", side_effect=error):
            record = self.run_main("--enforce")
        self.assertTrue(record["run_all"])
        self.assertFalse(record["enforce"])
        self.assertIn("selector failed", record["reason"])
        self.assertIn("CalledProcessError", record["error"])
        self.assertNotIn("error", record["diagnostics"])

    def test_no_test_times_runs_all(self) -> None:
        with (
            mock.patch.object(select_tests, "git", return_value="docs/a.md\n"),
            mock.patch.object(select_tests, "get_test_times", return_value={}),
        ):
            record = self.run_main()
        self.assertTrue(record["run_all"])
        self.assertEqual(record["reason"], "no test times")
        self.assertEqual(record["error"], "no test times")

    def test_no_changed_files_runs_all(self) -> None:
        with (
            mock.patch.object(select_tests, "git", return_value=""),
            mock.patch.object(select_tests, "get_test_times") as times,
        ):
            record = self.run_main()
        times.assert_not_called()
        self.assertTrue(record["run_all"])
        self.assertEqual(record["reason"], "no changed files")
        self.assertIsNone(record["error"])

    def test_bad_effort_env_is_rejected(self) -> None:
        stderr = io.StringIO()
        with (
            mock.patch.dict(os.environ, {"LLM_TD_EFFORT": "bogus"}),
            mock.patch.object(select_tests, "select") as select,
            mock.patch("sys.stderr", stderr),
            self.assertRaises(SystemExit),
        ):
            select_tests.main(["--print-prompt"])
        select.assert_not_called()
        self.assertIn("LLM_TD_EFFORT='bogus' is not one of", stderr.getvalue())

    def test_too_many_changed_files_runs_all(self) -> None:
        changed = "".join(f"torch/f{i}.py\n" for i in range(select_tests.MAX_CHANGED_FILES + 1))  # fmt: skip
        with (
            mock.patch.object(select_tests, "git", return_value=changed),
            mock.patch.object(select_tests, "get_test_times") as times,
        ):
            record = self.run_main()
        times.assert_not_called()
        self.assertTrue(record["run_all"])
        self.assertIsNone(record["error"])

    def test_query_model_runs_outside_the_checkout(self) -> None:
        run = mock.Mock(return_value=subprocess.CompletedProcess([], 0, "{}", ""))
        with mock.patch.object(select_tests.subprocess, "run", run):
            select_tests.query_model("p", "m", "medium", checkout="/wt")
        cmd = run.call_args.args[0]
        self.assertEqual(cmd[cmd.index("--add-dir") + 1], "/wt")
        self.assertNotEqual(run.call_args.kwargs["cwd"], "/wt")

    def test_query_model_failures_run_all(self) -> None:
        def completed(stdout: str) -> subprocess.CompletedProcess:
            return subprocess.CompletedProcess([], 0, stdout, "err")

        timeout = subprocess.TimeoutExpired(["claude", "-p", "<prompt>"], 600)
        for run, error in [
            (completed("not json"), "not jsonerr"),
            (completed(json.dumps({"is_error": True, "result": "overloaded"})), "overloaded"),
            (completed(json.dumps({"result": "", "total_cost_usd": 0.5})), "no model output"),
            (mock.Mock(side_effect=timeout), "model call timed out after 600s"),
            (mock.Mock(side_effect=FileNotFoundError("claude")), "FileNotFoundError: claude"),
        ]:  # fmt: skip
            run = run if isinstance(run, mock.Mock) else mock.Mock(return_value=run)
            with (
                tempfile.TemporaryDirectory() as tmp,
                clean_env(),
                mock.patch.object(select_tests, "git", return_value="docs/a.md\n"),
                mock.patch.object(select_tests, "get_test_times", return_value={"env": {"default": {"test_nn": 1.0}}}),
                mock.patch.object(select_tests.subprocess, "run", run),
            ):  # fmt: skip
                output = Path(tmp) / "out.json"
                select_tests.main(["--title", "t", "--output", str(output)])
                record = json.loads(output.read_text())
            self.assertTrue(record["run_all"], error)
            self.assertEqual(record["error"], error)
            self.assertEqual(
                run.call_args.kwargs["timeout"], select_tests.QUERY_TIMEOUT
            )
            cmd = run.call_args.args[0]
            self.assertEqual(cmd[:2], ["claude", "-p"])
            self.assertEqual(cmd[cmd.index("--tools") + 1], "")
            self.assertEqual(json.loads(cmd[cmd.index("--json-schema") + 1]), select_tests.SCHEMA)  # fmt: skip

    def test_infra_change_runs_all(self) -> None:
        with (
            mock.patch.object(select_tests, "git", return_value="test/run_test.py\n"),
            mock.patch.object(select_tests, "get_test_times") as times,
        ):
            record = self.run_main()
        times.assert_not_called()
        self.assertTrue(record["run_all"])
        self.assertIsNone(record["error"])

    def test_only_a_deliberate_run_all_is_enforced(self) -> None:
        def run(changed: str, answer: dict, times: dict, *argv: str) -> dict:
            with tempfile.TemporaryDirectory() as tmp, clean_env():
                output = Path(tmp) / "out.json"
                with (
                    mock.patch.object(select_tests, "git", return_value=changed),
                    mock.patch.object(
                        select_tests, "get_test_times", return_value=times
                    ),
                    mock.patch.object(select_tests, "query_model", return_value=answer),
                ):
                    select_tests.main(["--title", "t", "--output", str(output), *argv])
                return json.loads(output.read_text())

        times = {"env": {"default": {"test_nn": 1.0}}}
        model_run_all = {"structured_output": {"run_all": True, "reason": "", "sweeps": [], "dirs": [], "tests": []}}  # fmt: skip
        normal = {"structured_output": {"run_all": False, "reason": "", "sweeps": [], "dirs": [], "tests": [{"file": "test_nn", "confidence": "high"}]}}  # fmt: skip
        error = {"is_error": True, "result": "overloaded"}
        malformed = {"structured_output": {"run_all": False, "tests": "test_nn"}}
        for changed, answer, t, enforced, argv in [
            ("test/run_test.py\n", error, times, True, ["--enforce"]),
            ("docs/a.md\n", model_run_all, times, True, ["--enforce"]),
            ("docs/a.md\n", normal, times, True, ["--enforce"]),
            ("docs/a.md\n", normal, times, False, []),
            ("docs/a.md\n", error, times, False, ["--enforce"]),
            ("docs/a.md\n", malformed, times, False, ["--enforce"]),
            ("docs/a.md\n", model_run_all, {}, False, ["--enforce"]),
            ("", model_run_all, times, True, ["--enforce"]),
        ]:
            record = run(changed, answer, t, *argv)
            self.assertEqual(record["run_all"], answer is not normal)
            self.assertEqual(record["enforce"], enforced, (changed, answer, t, argv))

    def run_in_repo(self, answer: dict) -> tuple[dict, str]:
        env = {k: v for k, v in os.environ.items() if not k.startswith(("GIT_", "LLM_TD_", "CLAUDE_CODE_"))}  # fmt: skip
        env |= {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}  # fmt: skip

        def git(*args: str) -> str:
            return subprocess.check_output(["git", *args], cwd=repo, env=env, text=True)

        with tempfile.TemporaryDirectory() as repo:
            git("init", "-q")
            git("config", "user.email", "a@b.c")
            git("config", "user.name", "a")
            (Path(repo) / "test" / "nn").mkdir(parents=True)
            (Path(repo) / "test" / "test_foo.py").write_text("x = 1\n" * 50)
            (Path(repo) / "test" / "test_bar.py").write_text(
                'import test_foo\nrun("test_ select nothing.py")\n'
            )
            (Path(repo) / "test" / "unused.json").write_text("")
            git("add", "-A")
            git("commit", "-qm", "base")
            git("mv", "test/test_foo.py", "test/nn/test_foo.py")
            git("rm", "-q", "test/unused.json")
            (Path(repo) / "test" / "test_ select nothing.py").write_text("")
            (Path(repo) / "test" / "jit").mkdir()
            (Path(repo) / "test" / "jit" / "test_new.py").write_text("")
            git("add", "-A")
            git("commit", "-qm", "move")
            output = Path(repo) / "out.json"
            times = {"env": {"default": {"test_foo": 1.0}}}
            with (
                mock.patch.object(select_tests, "git", git),
                mock.patch.object(select_tests, "REPO_ROOT", Path(repo)),
                mock.patch.dict(os.environ, env, clear=True),
                mock.patch.object(
                    select_tests, "TESTS", ["test_bar", "test_foo", "test_jit"]
                ),
                mock.patch.object(select_tests, "get_test_times", return_value=times),
                mock.patch.object(
                    select_tests, "query_model", return_value=answer
                ) as query,
            ):
                select_tests.main(
                    ["--title", "t", "--base", "HEAD~1", "--output", str(output)]
                )
            record = json.loads(output.read_text())
        return record, query.call_args.args[0].split("# Candidate test files")[1]

    def test_moved_test_file_is_selected(self) -> None:
        answer = {"structured_output": {"run_all": False, "reason": "", "sweeps": [], "dirs": [], "tests": []}}  # fmt: skip
        record, candidates = self.run_in_repo(answer)
        self.assertIsNone(record["error"])
        self.assertFalse(record["run_all"])
        self.assertEqual(
            set(record["files"]), {"test_foo", "nn/test_foo", "test_bar", "test_jit"}
        )
        self.assertNotIn("select nothing", candidates)
        self.assertNotIn("jit/test_new", candidates)

    def test_unmatched_tests_are_an_error(self) -> None:
        tests = [{"file": "test_jit", "confidence": "high"}, {"file": "nope", "confidence": "low"}]  # fmt: skip
        answer = {"structured_output": {"run_all": False, "reason": "", "sweeps": [], "dirs": [], "tests": tests}}  # fmt: skip
        record, _ = self.run_in_repo(answer)
        self.assertIn("match no test file: ['nope']", record["error"])
        self.assertTrue(record["run_all"])

    def test_malformed_output_is_an_error(self) -> None:
        answer = {"structured_output": {"run_all": False, "reason": "", "sweeps": [], "dirs": [], "tests": ["test_foo"]}}  # fmt: skip
        record, _ = self.run_in_repo(answer)
        self.assertIn("malformed", record["error"])
        self.assertTrue(record["run_all"])


if __name__ == "__main__":
    unittest.main()
