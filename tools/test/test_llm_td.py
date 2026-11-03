from __future__ import annotations

import ast
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from typing import Any, TYPE_CHECKING
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from tools.testing import do_target_determination_for_s3
from tools.testing.discover_tests import BLOCKLISTED_PATTERNS, TESTS
from tools.testing.do_target_determination_for_s3 import td_results_json
from tools.testing.llm_td import resolve_selection, select_tests, selection
from tools.testing.llm_td.selection import (
    apply_selection,
    DRIVER_MAPPINGS,
    load_previous_failures,
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
    PREVIOUS_FAILURES_KEY,
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
        self.assertIsNone(normalize_test_name("jit/test_freezing", CANDIDATES))
        candidates = CANDIDATES | {"test_jit"}
        for name in ("jit/test_freezing", "test/jit/test_freezing.py", "jit.test_freezing"):  # fmt: skip
            self.assertEqual(normalize_test_name(name, candidates), "test_jit")

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
        self.assertEqual(Selection.from_json(sel.to_json()), sel)

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
            self.assertEqual((sel.run_all, sel.reason), (True, f"nothing found that reads {changed}"))  # fmt: skip
        sel = self.select_in_tree(sources | {"allowlist_for_publicAPI.json": "b.json\n"}, candidates, ["test/allowlist_for_publicAPI.json", "test/b.json"])  # fmt: skip
        self.assertEqual((sel.run_all, sel.reason), (True, "nothing found that reads test/b.json"))  # fmt: skip
        sel = self.select_in_tree(sources, CANDIDATES, ["test/cpp_extensions/ext.cpp"])
        self.assertEqual((sel.run_all, sel.reason), (True, "nothing found that reads test/cpp_extensions/ext.cpp"))  # fmt: skip
        # A deleted file, or the old path of a moved one, needs no reader.
        gone = ["test/gone.json", "test/distributed/launcher/bin/unused.py"]
        sel = self.select_in_tree(sources, candidates, gone, deleted=tuple(gone))
        self.assertEqual((sel.files, sel.run_all), (set(), False))
        many = [f"test/data{i}.json" for i in range(selection.MAX_DATA_FILES + 1)]
        self.assertTrue(self.select_in_tree(sources, candidates, many).run_all)

    def test_endless_import_chain_runs_all(self) -> None:
        calls: list[str] = []

        def grep(pattern: str) -> list[str]:
            calls.append(pattern)
            return [f"test/m{len(calls)}.py"]

        sel = Selection.from_model_output({"tests": []}, CANDIDATES, ["test/test_nn.py"], grep)  # fmt: skip
        self.assertTrue(sel.run_all)
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
            self.assertTrue(self.select_in_tree(sources, CANDIDATES, [changed]).run_all)
        many = [f"test/dynamo_skips/TestNN{i}.test_x" for i in range(selection.MAX_FAILURE_LIST_CLASSES + 1)]  # fmt: skip
        self.assertEqual(self.select_in_tree(sources, CANDIDATES, many).sweeps, {"dynamo_wrapped"})  # fmt: skip
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
        self.assertTrue(
            Selection.from_model_output(None, CANDIDATES, ["torch/a.py"]).run_all
        )
        out = {"run_all": False, "reason": "", "sweeps": [], "dirs": [], "tests": []}
        self.assertTrue(
            Selection.from_model_output(out, CANDIDATES, ["test/run_test.py"]).run_all
        )
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
            for sel in (Selection(candidates={"test_nn"}), Selection(run_all=True)):
                p.write_text(json.dumps(sel.to_json()))
                self.assertEqual(Selection.load(p), sel)
            valid = Selection(candidates={"test_nn"}).to_json()
            # Without candidates, a selection would skip every file it does not name.
            bad = [{**valid, "version": 2}, {**valid, "version": True}, {**valid, "files": "test_nn"}, {**valid, "files": [1]}, {**valid, "run_all": "false"}, {**valid, "enforce": "false"}, {**valid, "reason": 1}, {**valid, "candidates": [1]}, {**valid, "candidates": None}]  # fmt: skip
            bad += [{k: v for k, v in valid.items() if k != key} for key in ("version", "candidates", "enforce")]  # fmt: skip
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
            tp = previously_failed_in_pr.PreviouslyFailedInPR().get_prediction_confidence(["test_nn", "test_ops"])  # fmt: skip
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
                    with self.assertRaises(Exception):
                        scored()
                (folder / name).write_text("[]")

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
                    self.assertEqual(load_previous_failures(p), failed)
                loaded = TestPrioritizations.from_json(json.loads(p.read_text()))
                self.assertEqual([(str(x), loaded.get_score(x)) for x in loaded.get_all_tests()], [("test_nn", 1), ("test_ops", 0)])  # fmt: skip

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
                "copy_pytest_cache",
                "copy_additional_previous_failures",
            ],
            mock.DEFAULT,
        )
        with (
            tempfile.TemporaryDirectory() as d,
            mock.patch.multiple(
                do_target_determination_for_s3,
                REPO_ROOT=Path(d),
                get_test_prioritizations=mock.Mock(return_value=heuristics),
                get_all_previous_failures=mock.Mock(return_value={"test_ops"}),
                **fetches,
            ),
            mock.patch.dict(os.environ, {"CI": "false"}),
            mock.patch("sys.stdout"),
        ):
            do_target_determination_for_s3.main()
            td_results = Path(d) / "td_results.json"
            with mock.patch("sys.stderr"):
                self.assertEqual(load_previous_failures(td_results), {"test_ops"})

    def test_load_previous_failures(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "td_results.json"
            stderr = io.StringIO()
            with mock.patch("sys.stderr", stderr):
                self.assertIsNone(load_previous_failures(p))
            self.assertEqual(stderr.getvalue(), "")
            tp = TestPrioritizations(["test_nn", "test_ops"], {TestRun("test_nn"): 1})
            # An older writer stores only TD's ranking.
            for text in (json.dumps(tp.to_json()), "{", "[]", "[" * 100_000, *(json.dumps({**tp.to_json(), "previous_failures": v}) for v in ("test_nn", [1], None))):  # fmt: skip
                p.write_text(text)
                stderr = io.StringIO()
                with mock.patch("sys.stderr", stderr):
                    self.assertIsNone(load_previous_failures(p), text[:100])
                self.assertTrue(stderr.getvalue().startswith("Ignoring the previous failures"), text[:100])  # fmt: skip
            td_results = {**tp.to_json(), "previous_failures": ["test_nn", "elsewhere"]}
            p.write_text(json.dumps(td_results))
            self.assertEqual(load_previous_failures(p), {"test_nn", "elsewhere"})
            loaded = TestPrioritizations.from_json(td_results)
            self.assertEqual([(str(x), loaded.get_score(x)) for x in loaded.get_all_tests()], [("test_nn", 1), ("test_ops", 0)])  # fmt: skip

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
        out = {"tests": [{"file": "jit/test_freezing", "confidence": "high"}]}
        sel = Selection.from_model_output(out, candidates, [])
        self.assertEqual((sel.files, sel.run_all), ({"test_jit"}, False))

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
    ) -> tuple[list[str], list[str], selection.Split]:
        tp = TestPrioritizations(times, scores)
        split = sel.split(tp.get_all_tests(), tp.get_score, times, "default", previously_failed=prev, td_include=td)  # fmt: skip
        return [str(x) for x in split.include], [str(x) for x in split.exclude], split

    def test_hedge_is_greedy_in_rank_order(self) -> None:
        scores = {"a": 1, "b": 0.9, "c": 0.8, "d": 0.7, "untimed": 0.6, "neg": -0.5}
        times: dict[str, float] = {"a": 10, "b": 150, "c": 52, "d": 48, "untimed": 0, "zero": 1, "neg": 1, "rest": 738}  # fmt: skip
        # The budget is 20% of 1000s = 200s. b leaves 50s, so c (52s, fits
        # alone) is skipped and d (48s) still fits, leaving 2s that the
        # unscored and negatively scored 1s files may not use. With 19% d does
        # not fit, with 21% c does.
        include, exclude, split = self.split(
            Selection(files={"a"}), {TestRun(f): s for f, s in scores.items()}, times
        )
        self.assertEqual(include, ["a", "b", "d"])
        self.assertEqual(exclude, ["c", "untimed", "rest", "zero", "neg"])
        self.assertEqual((split.covered, split.hedged, split.budget), ({"a"}, {"b", "d"}, 200))  # fmt: skip

    def test_nothing_chosen_keeps_td_include(self) -> None:
        # A targeted call such as `run_test.py --include test_cuda`: one file
        # is all of the call's time, so it never fits the 20% hedge.
        include, exclude, split = self.split(Selection(files={"a"}), {TestRun("test_cuda"): 1}, {"test_cuda": 100}, td={"test_cuda"})  # fmt: skip
        self.assertEqual((include, exclude, split.kept_td), ([], ["test_cuda"], True))
        include, exclude, split = self.split(Selection(files={"test_cuda"}), {TestRun("test_cuda"): 1}, {"test_cuda": 100}, td={"test_cuda"})  # fmt: skip
        self.assertEqual((include, split.kept_td), (["test_cuda"], False))

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
        times: dict[str, float] = {"a": 10, "s": 150, "p": 100, "q": 60, "big": 600, "untimed": 0, "rest": 80}  # fmt: skip
        include, exclude, split = self.split(Selection(files={"a"}), scores, times)
        self.assertEqual((include, split.hedged), (["s", "a"], {"s"}))
        # p always runs and leaves the whole 200s budget to s, so q (60s)
        # still does not fit next to s.
        include, exclude, split = self.split(
            Selection(files={"a"}), scores, times, prev={"p", "a"}, td={"s", "p", "a"}
        )
        self.assertEqual((include, split.previously_failed, split.hedged), (["s", "p", "a"], {"p"}, {"s"}))  # fmt: skip
        # big (three times the budget, unscored) and untimed also run because
        # TD runs them, and the hedge still gets the whole budget.
        scores[TestRun("tiny")] = 0.1
        times |= {"tiny": 1, "rest": 79}
        include, exclude, split = self.split(
            Selection(files={"a"}), scores, times, prev={"p", "big", "untimed", "a", "elsewhere"}, td={"p", "big", "untimed"}
        )  # fmt: skip
        self.assertEqual(include, ["s", "p", "tiny", "a", "big", "untimed"])
        self.assertEqual(exclude, ["q", "rest"])
        self.assertEqual((split.covered, split.previously_failed, split.hedged, split.budget), ({"a"}, {"p", "big", "untimed"}, {"s", "tiny"}, 200))  # fmt: skip

    def test_only_a_previous_failure_chosen_is_not_kept_td(self) -> None:
        # p is all of TD's include and over the budget, so only its previous
        # failure runs it.
        include, exclude, split = self.split(Selection(), {TestRun("p"): 1}, {"p": 500, "r": 500}, prev={"p"}, td={"p"})  # fmt: skip
        self.assertEqual((include, exclude), (["p"], ["r"]))
        self.assertEqual((split.previously_failed, split.hedged, split.kept_td), ({"p"}, set(), False))  # fmt: skip

    def test_previous_failures_outside_td_are_hedged_like_other_files(self) -> None:
        scores = {TestRun("s"): 1, TestRun("big"): 0.8, TestRun("p"): 0.5}
        times: dict[str, float] = {"s": 150, "big": 600, "p": 40, "rest": 210}
        # TD runs only s, so big (over budget) does not run and p is hedged
        # after s like any other scored file.
        include, exclude, split = self.split(
            Selection(), scores, times, prev={"big", "p"}, td={"s"}
        )
        self.assertEqual((include, exclude), (["s", "p"], ["big", "rest"]))
        self.assertEqual((split.previously_failed, split.hedged), (set(), {"s", "p"}))

    def test_run_all_and_sweeps_take_everything(self) -> None:
        for sel in (Selection(run_all=True), Selection(sweeps={"default"})):
            include, exclude, split = self.split(sel, {}, {"a": 1, "b": 1})
            self.assertEqual((include, exclude, split.hedged), (["a", "b"], [], set()))


# sha256 of the call's sorted test files, "a big p rest s".
CALL_ID = "6eab3c031b50"


class TestApplySelection(unittest.TestCase):
    TIMES: dict[str, float] = {"a": 10, "p": 20, "s": 170, "big": 600, "rest": 200}

    def apply(
        self,
        sel: Selection,
        td_enabled: bool = True,
        pr_number: int | None = 1,
        prev: Collection[str] | None = (),
        times: dict[str, float] | None = None,
        tests: list[str] | None = None,
        stderr: io.StringIO | None = None,
    ) -> selection.AppliedSelection:
        times = self.TIMES if times is None else times
        tests = list(self.TIMES) if tests is None else tests
        tp = TestPrioritizations(tests, {TestRun("p"): 1, TestRun("s"): 0.5})
        include, exclude = tp.get_top_per_tests(25)
        with (
            mock.patch.object(selection, "get_pr_number", return_value=pr_number),
            mock.patch("sys.stderr", io.StringIO() if stderr is None else stderr),
        ):
            return apply_selection(sel, include, exclude, tp.get_score, times, test_config="default", td_enabled=td_enabled, previously_failed=prev)  # fmt: skip

    def metric(self, enforced: bool) -> dict[str, object]:
        # p previously failed, and s is hedged within the 200s budget.
        return {
            "call_id": CALL_ID,
            "enforced": enforced,
            "td_enabled": True,
            "run_all": False,
            "num_tests": 5,
            "num_covered": 1,
            "num_td_include": 2,
            "num_hedged": 1,
            "num_previously_failed": 1,
            "kept_td": False,
            "previous_failures_known": True,
            "time_total": 1000,
            "time_covered": 10,
            "time_hedged": 170,
            "time_previously_failed": 20,
        }

    def test_gate(self) -> None:
        for td_enabled in (False, True):
            for enforce in (False, True):
                for pr_number in (None, 1):
                    sel = Selection(files={"a"}, enforce=enforce)
                    applied = self.apply(sel, td_enabled, pr_number)
                    expected = td_enabled and enforce and pr_number is not None
                    self.assertEqual(applied.enforced, expected, (td_enabled, enforce, pr_number))  # fmt: skip
                    self.assertEqual(applied.selection_metric["enforced"], expected)
                    include = ["p", "s", "a"] if expected else ["p", "s"]
                    self.assertEqual([str(x) for x in applied.include], include)
                    self.assertEqual(len(applied.include + applied.exclude), 5)

    def test_not_enforced_keeps_td_split_and_measures_the_selection(self) -> None:
        applied = self.apply(Selection(files={"a"}), prev={"p", "elsewhere"})
        self.assertEqual([str(x) for x in applied.include], ["p", "s"])
        self.assertEqual(applied.td_include, {"p", "s"})
        self.assertEqual(applied.selection_metric, self.metric(enforced=False))
        self.assertEqual(
            applied.failure_metric(TestRun("s")),
            {
                "failure": "s",
                "call_id": CALL_ID,
                "covered": False,
                "enforced": False,
                "td_enabled": True,
                "in_td_include": True,
                "hedged": True,
                "previously_failed": False,
                "kept_td": False,
            },
        )

    def test_enforced_swaps_in_the_selection(self) -> None:
        applied = self.apply(Selection(files={"a"}, enforce=True), prev={"p", "elsewhere"})  # fmt: skip
        self.assertEqual([str(x) for x in applied.include], ["p", "s", "a"])
        self.assertEqual([str(x) for x in applied.exclude], ["big", "rest"])
        self.assertEqual(applied.selection_metric, self.metric(enforced=True))

    def test_enforced_log_line(self) -> None:
        stderr = io.StringIO()
        self.apply(Selection(files={"a"}, enforce=True), prev={"p"}, stderr=stderr)
        self.assertEqual(
            stderr.getvalue(),
            "LLM selection: running 3 of 5 test files (1 selected; 1 previously failed; 1 hedged within 200s)\n",
        )

    def test_unknown_previous_failures_fall_back_to_td(self) -> None:
        # td_results.json from a writer that does not store them.
        stderr = io.StringIO()
        applied = self.apply(Selection(files={"a"}, enforce=True), prev=None, stderr=stderr)  # fmt: skip
        self.assertFalse(applied.enforced)
        self.assertEqual([str(x) for x in applied.include], ["p", "s"])
        self.assertEqual(stderr.getvalue(), "LLM selection: no previous failures in the TD results, using TD\n")  # fmt: skip
        # Without them p is hedged too, as the shadow metrics record.
        unknown = {"num_previously_failed": 0, "time_previously_failed": 0, "num_hedged": 2, "time_hedged": 190, "previous_failures_known": False}  # fmt: skip
        self.assertEqual(
            applied.selection_metric, self.metric(enforced=False) | unknown
        )
        self.assertFalse(applied.failure_metric(TestRun("p"))["previously_failed"])
        self.assertFalse(self.apply(Selection(files={"a"}), prev=None).enforced)

    def test_no_test_times_falls_back_to_td(self) -> None:
        applied = self.apply(Selection(files={"a"}, enforce=True), times={})
        self.assertFalse(applied.enforced)
        self.assertEqual([str(x) for x in applied.include], ["p", "s"])
        self.assertEqual(applied.selection_metric["time_total"], 0)

    def test_nothing_chosen_is_reported_as_td(self) -> None:
        # Neither scored file fits the 200s hedge and the selection names none.
        stderr = io.StringIO()
        applied = self.apply(Selection(enforce=True), times={"p": 500, "s": 500}, stderr=stderr)  # fmt: skip
        self.assertFalse(applied.enforced)
        self.assertEqual([str(x) for x in applied.include], ["p", "s"])
        self.assertEqual(stderr.getvalue(), "LLM selection: chose no files, using TD\n")
        self.assertEqual((applied.selection_metric["enforced"], applied.selection_metric["kept_td"]), (False, True))  # fmt: skip
        self.assertEqual({k: applied.failure_metric(TestRun("p"))[k] for k in ("enforced", "kept_td")}, {"enforced": False, "kept_td": True})  # fmt: skip

    def test_call_id_tells_apart_the_calls_of_a_job(self) -> None:
        first, again = self.apply(Selection()), self.apply(Selection(), times={})
        other = self.apply(Selection(), tests=["p", "s", "x"])
        self.assertEqual(first.selection_metric["call_id"], CALL_ID)
        self.assertEqual(again.selection_metric["call_id"], CALL_ID)
        self.assertNotEqual(other.selection_metric["call_id"], CALL_ID)
        self.assertEqual(other.failure_metric(TestRun("p"))["call_id"], other.selection_metric["call_id"])  # fmt: skip

    def test_failure_metric(self) -> None:
        # big failed before but TD does not run it and it does not fit the
        # budget, so it does not run.
        applied = self.apply(Selection(files={"a"}, enforce=True), prev={"p", "big"})
        self.assertEqual([str(x) for x in applied.exclude], ["big", "rest"])
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
                    "call_id": CALL_ID,
                    "covered": covered,
                    "enforced": True,
                    "td_enabled": True,
                    "in_td_include": in_td,
                    "hedged": hedged,
                    "previously_failed": prev,
                    "kept_td": False,
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

    def run_split(self, sel: Selection | None, emit: mock.Mock | None) -> tuple[selection.TestSplit, str]:  # fmt: skip
        tp = TestPrioritizations(["a", "p"], {TestRun("p"): 1})
        include, exclude = tp.get_top_per_tests(25)
        stderr = io.StringIO()
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.object(selection, "get_pr_number", return_value=1),
            mock.patch("sys.stderr", stderr),
        ):
            selection_file, td_results = Path(tmp) / "sel.json", Path(tmp) / "td.json"
            if sel is not None:
                selection_file.write_text(json.dumps(sel.to_json()))
            td_results.write_text(json.dumps(td_results_json(tp, set())))
            split = split_tests(include, exclude, tp.get_score, {"a": 1, "p": 1}, selection_file=selection_file, td_results=td_results, test_config="default", td_enabled=True, percent=25, emit=emit)  # fmt: skip
        return split, stderr.getvalue()

    def test_wiring(self) -> None:
        for sel, enforced in [(None, False), (Selection(files={"a"}, candidates={"a"}), False), (Selection(files={"a"}, enforce=True, candidates={"a"}), True)]:  # fmt: skip
            emit = mock.Mock()
            split, stderr = self.run_split(sel, emit)
            self.assertEqual(split.enforced, enforced)
            self.assertEqual([str(x) for x in split.include], ["p", "a"] if enforced else ["p"])  # fmt: skip
            self.assertEqual(split.source, gen_artifact.LLM_SELECTION_SOURCE if enforced else "td")  # fmt: skip
            self.assertNotIn("::warning::", stderr)
            self.assertEqual("Running 25% of tests based on TD" in stderr, not enforced)
            split.report_failure(TestRun("p"))
            calls = []
            if split.llm is not None:
                selected = mock.call("llm_td_shadow_selection", split.llm.selection_metric)  # fmt: skip
                shadow = mock.call("llm_td_shadow_failure", split.llm.failure_metric(TestRun("p")))  # fmt: skip
                calls = [selected, shadow]
            self.assertEqual(emit.call_args_list, calls)

    def test_td_failure_metric_is_emitted_by_run_test_main(self) -> None:
        # emit_metric records its caller's file, module and function, which
        # queries of td_test_failure_stats_v2 filter on.
        tree = ast.parse((REPO_ROOT / "test" / "run_test.py").read_text())
        callers = [
            f.name
            for f in ast.walk(tree)
            if isinstance(f, ast.FunctionDef)
            for c in ast.walk(f)
            if isinstance(c, ast.Call)
            and getattr(c.func, "id", None) == "emit_metric"
            and isinstance(c.args[0], ast.Constant)
            and c.args[0].value == "td_test_failure_stats_v2"
        ]
        self.assertEqual(callers, ["main"])

    def test_no_metrics_outside_ci(self) -> None:
        split, _ = self.run_split(
            Selection(files={"a"}, enforce=True, candidates={"a"}), emit=None
        )
        self.assertTrue(split.enforced)
        split.report_failure(TestRun("p"))


class TestDashboardExclusions(unittest.TestCase):
    def test_exclusions_are_grouped_by_source(self) -> None:
        test_dashboard = import_test_dashboard()
        with tempfile.TemporaryDirectory() as d:
            reports = Path(d) / "test" / "test-reports"
            reports.mkdir(parents=True)
            with mock.patch.object(gen_artifact, "REPO_ROOT", Path(d)):
                gen_artifact.gen_ci_artifact([], [{"test_file": "c"}], gen_artifact.LLM_SELECTION_SOURCE)  # fmt: skip
            (written,) = reports.glob("td_exclusions-*.json")
            llm_artifact = json.loads(written.read_text())

        artifacts = {
            "td_exclusions-1.json": {"included": [], "excluded": [{"test_file": "a"}]},
            "td_exclusions-2.json": {
                "included": [],
                "excluded": [{"test_file": "b"}],
                "source": "td",
            },  # fmt: skip
            "td_exclusions-3.json": llm_artifact,
        }

        def unzip(_: object) -> None:
            for name, content in artifacts.items():
                Path(name).write_text(json.dumps(content))

        cwd = os.getcwd()
        try:
            with (
                mock.patch.object(
                    test_dashboard, "download_s3_artifacts", return_value=["x"]
                ),  # fmt: skip
                mock.patch.object(test_dashboard, "unzip", unzip),
                mock.patch.object(test_dashboard, "get_job_id", return_value=1),
                mock.patch.object(
                    test_dashboard,
                    "get_job_name",
                    return_value="linux / test (default, 1, 2, x)",
                ),  # fmt: skip
            ):
                exclusions = test_dashboard.get_td_exclusions(1, 1)
        finally:
            os.chdir(cwd)
        self.assertEqual(
            json.loads(json.dumps(exclusions)),
            {
                "all": {"linux": {"default": ["a", "b", "c"]}},
                "llm_selection": {"linux": {"default": ["c"]}},
            },  # fmt: skip
        )

    def test_both_sources_are_uploaded(self) -> None:
        dash = import_test_dashboard()
        td, llm = {"linux": {"default": ["a"]}}, {"linux": {"default": ["c"]}}
        for exclusions, expected in (
            ({}, [mock.call(1, 2, "additional_info/td_exclusions", [{}])]),
            ({"all": td}, [mock.call(1, 2, "additional_info/td_exclusions", [td])]),
            (
                {"all": td, "llm_selection": llm},
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

    def test_no_test_times_runs_all(self) -> None:
        with (
            mock.patch.object(select_tests, "git", return_value="docs/a.md\n"),
            mock.patch.object(select_tests, "get_test_times", return_value={}),
        ):
            record = self.run_main()
        self.assertTrue(record["run_all"])
        self.assertEqual(record["reason"], "no test times")

    def test_no_changed_files_runs_all(self) -> None:
        with (
            mock.patch.object(select_tests, "git", return_value=""),
            mock.patch.object(select_tests, "get_test_times") as times,
        ):
            record = self.run_main()
        times.assert_not_called()
        self.assertTrue(record["run_all"])
        self.assertEqual(record["reason"], "no changed files")

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
            (completed(json.dumps({"result": "", "total_cost_usd": 0.5})), "no structured output"),
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
            self.assertEqual(record["diagnostics"]["error"], error)
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
            ("", model_run_all, times, False, ["--enforce"]),
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
        self.assertIsNone(record["diagnostics"]["error"])
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
        self.assertIn("match no test file: ['nope']", record["diagnostics"]["error"])
        self.assertTrue(record["run_all"])

    def test_malformed_output_is_an_error(self) -> None:
        answer = {"structured_output": {"run_all": False, "reason": "", "sweeps": [], "dirs": [], "tests": ["test_foo"]}}  # fmt: skip
        record, _ = self.run_in_repo(answer)
        self.assertIn("malformed", record["diagnostics"]["error"])
        self.assertTrue(record["run_all"])


SHA = "a" * 40
REPO = "pytorch/pytorch"
ARTIFACT = f"llm_td_selection_7_{SHA}"


def selector_run(id: int, **kw: object) -> dict:
    """An llm-td.yml run that passes every provenance check; higher ids are newer."""
    run = {
        "id": id,
        "created_at": f"2026-11-0{id}T00:00:00Z",
        "html_url": f"https://github.com/{REPO}/actions/runs/{id}",
        "path": ".github/workflows/llm-td.yml",
        "event": "pull_request_target",
        "head_sha": SHA,
        "head_repository": {"id": 1},
        "repository": {"id": 1},
        "status": "completed",
        "conclusion": "success",
    }
    return {**run, **kw}


def pending_run(id: int, status: str = "in_progress") -> dict:
    return selector_run(id, status=status, conclusion=None)


class TestResolveSelection(unittest.TestCase):
    # What the last resolve() slept, downloaded and printed.
    sleeps: list[float]
    fetched: list[str]
    logs: list[str]

    def resolve(
        self,
        *polls: list[dict],
        labels: tuple[str, ...] = ("ci-llm-td",),
        head_repo: str = REPO,
        selected_enforce: bool = True,
        **env: str,
    ) -> dict | None:
        """Run main() against a fake API that lists the next of `polls` on each
        poll (repeating the last), and return the selection it writes."""
        polls_left = list(polls)
        now = [0.0]
        self.sleeps, self.fetched, self.logs = [], [], []
        pr = {"head": {"repo": {"full_name": head_repo}}, "labels": [{"name": n} for n in labels]}  # fmt: skip

        def get(path: str) -> dict:
            if path == "pulls/7":
                return pr
            if path.startswith("actions/workflows/llm-td.yml/runs?"):
                self.assertIn(f"head_sha={SHA}", path)
                runs = polls_left.pop(0) if len(polls_left) > 1 else polls_left[0]
                return {"workflow_runs": runs}
            run_id = path.split("/")[2]
            self.assertEqual(path, f"actions/runs/{run_id}/artifacts?name={ARTIFACT}")
            url = f"zip/{run_id}"
            return {"artifacts": [{"name": ARTIFACT, "expired": False, "archive_download_url": url}]}  # fmt: skip

        def fetch(url: str) -> bytes:
            self.fetched.append(url)
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as z:
                sel = Selection({"test_nn"}, enforce=selected_enforce, candidates={"test_nn"})  # fmt: skip
                z.writestr("llm_td_selection.json", json.dumps(sel.to_json()))
            return buf.getvalue()

        def sleep(t: float) -> None:
            self.sleeps.append(t)
            now[0] += t

        env = {"GITHUB_REPOSITORY": REPO, "HEAD_SHA": SHA, "PR_NUMBER": "7", "TD_ENABLED": "true", **env}  # fmt: skip
        time = resolve_selection.time
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.dict(os.environ, env),
            mock.patch.object(resolve_selection, "get", get),
            mock.patch.object(resolve_selection, "fetch", fetch),
            mock.patch.object(time, "monotonic", lambda: now[0]),
            mock.patch.object(time, "sleep", sleep),
            mock.patch("builtins.print", self.logs.append),
        ):
            output = Path(tmp) / "out.json"
            resolve_selection.main(output)
            record = json.loads(output.read_text()) if output.exists() else None
        return record

    def selected(self, *polls: list[dict], **kw: Any) -> dict:
        record = self.resolve(*polls, **kw)
        if record is None:
            raise AssertionError(f"no selection written; logs: {self.logs}")
        return record

    def test_enforce_takes_the_newest_run_not_the_first_listed(self) -> None:
        record = self.selected([selector_run(1), selector_run(2)])
        self.assertEqual(self.fetched, ["zip/2"])
        self.assertTrue(record["enforce"])
        self.assertEqual(record["files"], ["test_nn"])
        self.assertEqual(self.sleeps, [])

    def test_untrusted_runs_are_ignored(self) -> None:
        for bad in [
            {"path": ".github/workflows/pull.yml"},
            {"event": "pull_request"},
            {"head_sha": "b" * 40},
            {"head_repository": {"id": 2}},
            {"head_repository": None},
        ]:
            with self.subTest(bad=bad):
                untrusted = selector_run(2, **bad)
                self.resolve([selector_run(1), untrusted])
                self.assertEqual(self.fetched, ["zip/1"])
                self.assertIsNone(self.resolve([untrusted]))
                self.assertEqual(self.fetched, [])

    def test_waits_for_a_pending_run(self) -> None:
        for status in ["queued", "in_progress", "waiting", "requested", "pending"]:
            with self.subTest(status=status):
                old, new = selector_run(1), pending_run(2, status)
                record = self.selected([old, new], [old, new], [old, selector_run(2)])
                self.assertEqual(self.fetched, ["zip/2"])
                self.assertTrue(record["enforce"])
                self.assertEqual(self.sleeps, [15, 30])

    def test_skipped_runs_are_ignored(self) -> None:
        skipped = selector_run(2, conclusion="skipped")
        self.resolve([selector_run(1), skipped])
        self.assertEqual(self.fetched, ["zip/1"])

    def test_enforce_gives_up_at_the_deadline(self) -> None:
        record = self.resolve([selector_run(1), pending_run(2)])
        self.assertIsNone(record)
        self.assertEqual(self.fetched, [])
        self.assertEqual(sum(self.sleeps), resolve_selection.WAIT_SECONDS)
        self.assertIn("Gave up", self.logs[-1])

    def test_enforce_does_not_fall_back_to_an_older_run(self) -> None:
        for conclusion in ["failure", "cancelled", "timed_out"]:
            with self.subTest(conclusion=conclusion):
                failed = selector_run(2, conclusion=conclusion)
                record = self.resolve([selector_run(1), failed])
                self.assertIsNone(record)
                self.assertEqual(self.fetched, [])

    def test_shadow_unless_labeled_and_td_enabled(self) -> None:
        for labels, td_enabled in [
            ((), "true"),
            (("ci-llm-td", "ci-no-td"), "true"),
            (("ci-llm-td",), "false"),
        ]:
            with self.subTest(labels=labels, td_enabled=td_enabled):
                runs = [selector_run(1), pending_run(2, "queued")]
                record = self.selected(runs, labels=labels, TD_ENABLED=td_enabled)
                self.assertFalse(record["enforce"])
                self.assertEqual(self.fetched, ["zip/1"])
                self.assertEqual(self.sleeps, [])

    def test_failed_selection_is_not_enforced(self) -> None:
        # select_tests writes enforce=False when the model call failed.
        record = self.selected([selector_run(1)], selected_enforce=False)
        self.assertFalse(record["enforce"])

    def test_pr_number(self) -> None:
        runs = [selector_run(1)]
        tag = "refs/tags/ciflow/trunk/7"
        self.assertTrue(self.selected(runs, PR_NUMBER="", GITHUB_REF=tag)["enforce"])
        branch = "refs/heads/main"
        self.assertIsNone(self.resolve(runs, PR_NUMBER="", GITHUB_REF=branch))
        self.assertIsNone(self.resolve(runs, HEAD_SHA=""))

    def test_fork_pr_gets_nothing(self) -> None:
        self.assertIsNone(self.resolve([selector_run(1)], head_repo="someone/pytorch"))
        self.assertEqual(self.fetched, [])


if __name__ == "__main__":
    unittest.main()
