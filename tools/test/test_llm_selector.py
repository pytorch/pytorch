from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from tools.testing.llm_selector.selection import needs_run_all, Selection


sys.path.remove(str(REPO_ROOT))


class TestSelection(unittest.TestCase):
    def test_load_malformed_file(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "sel.json"
            stderr = io.StringIO()
            with mock.patch("sys.stderr", stderr):
                self.assertIsNone(Selection.load(p))
            self.assertEqual(stderr.getvalue(), "")
            for sel in (
                Selection(candidates={"test_nn"}),
                Selection(run_all=True),
                Selection(run_all=True, error="e"),
                Selection(
                    {"test_nn"},
                    {"dynamo_wrapped", "crossref"},
                    reason="r",
                    enforce=True,
                    candidates={"test_nn", "test_ops"},
                ),
            ):
                p.write_text(json.dumps(sel.to_json()))
                self.assertEqual(Selection.load(p), sel)
            valid = Selection(candidates={"test_nn"}).to_json()
            # Without candidates, a selection would skip every file it does not name.
            bad = [
                {**valid, "version": 2},
                {**valid, "version": True},
                {**valid, "files": "test_nn"},
                {**valid, "files": [1]},
                {**valid, "run_all": "false"},
                {**valid, "enforce": "false"},
                {**valid, "reason": 1},
                {**valid, "candidates": [1]},
                {**valid, "candidates": None},
                {**valid, "error": 1},
                {**valid, "sweeps": ["default"]},
                {**valid, "files": ["test_ops"]},
                {**valid, "error": "e"},
                {**valid, "extra": 1},
            ]
            # run_all and no candidates leave only the type checks to reject these.
            r = Selection(run_all=True).to_json()
            bad += [{**r, "files": "test_nn"}, {**r, "files": [1]}, {**r, "error": 1}]
            bad += [
                {k: v for k, v in valid.items() if k != key}
                for key in ("version", "files", "sweeps", "run_all", "reason")
                + ("enforce", "candidates", "error")
            ]
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

    def test_covers(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be run_all"):
            Selection()
        cands = {"test_nn", "test_ops", "test_torch"}
        sel = Selection({"test_nn"}, {"dynamo_wrapped"}, candidates=cands)
        # The model was not shown test_openreg, so it always runs.
        for f, config, covered in [
            ("test_openreg", "default", True),
            ("test_nn", "default", True),
            ("test_ops", "default", False),
            ("test_ops", "dynamo_wrapped", True),
            ("test_ops", "crossref", False),
        ]:
            self.assertEqual(sel.covers(f, config), covered, (f, config))
        self.assertTrue(Selection(run_all=True).covers("test_ops", "default"))
        # The first of TEST_SH_GLOBS a config contains decides its ALWAYS_COVERED.
        for config, torch, ops in [
            ("inductor", True, True),
            ("inductor_amx", True, True),
            ("inductor_cpp_wrapper", True, False),
            ("inductor_core", False, False),
            ("inductor_huggingface", False, False),
        ]:
            got = (sel.covers("test_torch", config), sel.covers("test_ops", config))
            self.assertEqual(got, (torch, ops), config)

    def test_run_all_patterns(self) -> None:
        for f in [
            ".ci/docker/common/install_cuda.sh",
            ".ci/docker/requirements-ci.txt",
            ".ci/docker/requirements-docs.txt",
            ".ci/docker/java/jni.h",
            ".ci/docker/build.sh",
            ".ci/docker/ci_commit_pins/triton.txt",
            ".ci/docker/ubuntu/Dockerfile",
            ".ci/docker/ubuntu-rocm/Dockerfile",
            ".ci/docker/ubuntu-xpu/Dockerfile",
            ".ci/pytorch/test.sh",
            ".ci/pytorch/multigpu-test.sh",
            ".ci/pytorch/win-test.sh",
            ".ci/pytorch/macos-test.sh",
            ".ci/pytorch/macos-common.sh",
            ".ci/pytorch/build.sh",
            ".ci/pytorch/common.sh",
            ".ci/pytorch/common-build.sh",
            ".ci/pytorch/common_utils.sh",
            ".ci/pytorch/torch_trace.sh",
            ".ci/pytorch/win-build.sh",
            ".ci/pytorch/macos-build.sh",
            ".ci/pytorch/rocm_utils.sh",
            ".ci/pytorch/win-test-helpers/build_pytorch.bat",
            ".github/ci_commit_pins/torch_tpu.txt",
            "tools/stats/export_test_times.py",
            ".ci/pytorch/win-test-helpers/setup_pytorch_env.bat",
            ".ci/pytorch/win-test-helpers/test_python_shard.bat",
            ".ci/pytorch/win-test-helpers/installation-helpers/activate_miniconda3.bat",
            ".ci/pytorch/win-test-helpers/installation-helpers/install_magma.bat",
            ".ci/pytorch/win-test-helpers/installation-helpers/install_sccache.bat",
            ".github/workflows/_linux-test.yml",
            ".github/workflows/_win-test.yml",
            ".github/workflows/_mac-test.yml",
            ".github/workflows/_rocm-test.yml",
            ".github/workflows/_rocm-test-no-dind.yml",
            ".github/workflows/_xpu-test.yml",
            ".github/workflows/_s390x-test.yml",
            ".github/workflows/_tpu-test.yml",
            ".github/workflows/_linux-build.yml",
            ".github/workflows/_win-build.yml",
            ".github/workflows/_mac-build.yml",
            ".github/workflows/_s390x-build.yml",
            ".github/actions/setup-rocm/action.yml",
            ".github/actions/setup-win/action.yml",
            ".github/actions/setup-xpu/action.yml",
            ".github/actions/filter-test-configs/action.yml",
            ".github/actions/download-td-artifacts/action.yml",
            ".github/actions/download-build-artifacts/action.yml",
            ".github/scripts/filter_test_configs.py",
            ".github/scripts/parse_ref.py",
            ".github/scripts/pytest_cache.py",
            ".github/scripts/pytest_caching_utils.py",
            ".github/scripts/file_io_utils.py",
            ".github/actions/pytest-cache-download/action.yml",
            ".github/actions/pytest-cache-upload/action.yml",
            ".github/workflows/pull.yml",
            ".github/workflows/trunk.yml",
            ".github/workflows/periodic.yml",
            ".github/workflows/slow.yml",
            ".github/workflows/inductor.yml",
            ".github/workflows/inductor-periodic.yml",
            ".github/actions/reuse-old-whl/action.yml",
            ".github/workflows/target_determination.yml",
            ".github/workflows/llm_td_retrieval.yml",
            "tools/native_aot/build_stage2.py",
            ".ci/pytorch/generate_ck_config_h.cmake",
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
            "torch/testing/__init__.py",
            "torch/testing/_creation.py",
            "torch/testing/_utils.py",
            "scripts/onnx/test.sh",
            ".ci/docker/triton_version.txt",
            ".ci/docker/triton_xpu_version.txt",
            ".ci/pytorch/numba-cuda-13.patch",
            ".ci/pytorch/win-test-helpers/test_python_jit_legacy.bat",
            "tools/testing/discover_tests.py",
            "tools/stats/import_test_stats.py",
            "tools/stats/upload_metrics.py",
            "tools/stats/upload_test_stats.py",
            "tools/stats/upload_stats_lib.py",
            "tools/stats/test_dashboard.py",
            "test/cpp_extensions/open_registration_extension/torch_openreg/pyproject.toml",
        ]:
            self.assertTrue(needs_run_all([f]), f)
            self.assertTrue(
                (REPO_ROOT / f).exists(),
                f"{f} is gone: update selection.RUN_ALL_PATTERNS",
            )
        # pytest reads the first of these it finds from the test's directory up.
        for f in [
            "conftest.py",
            "test/pytest.ini",
            "test/.pytest.ini",
            "test/dynamo/pyproject.toml",
            "test/tox.ini",
            "test/setup.cfg",
        ]:
            self.assertTrue(needs_run_all([f]), f)
        # Every caller of a test workflow sets the test matrix of its jobs.
        for wf in (REPO_ROOT / ".github/workflows").glob("*.yml"):
            called = [
                line.split("uses: ./", 1)[1].split()[0]
                for line in wf.read_text().splitlines()
                if "uses: ./.github/workflows/_" in line
            ]
            if any("-test" in c and needs_run_all([c]) for c in called):
                self.assertTrue(needs_run_all([f".github/workflows/{wf.name}"]), wf)
        for f in [
            "docs/source/index.rst",
            "test/data/setup.cfg.in",
            ".github/workflows/lint.yml",
            ".github/workflows/_binary-build-linux.yml",
            ".github/workflows/docs-build.yml",
            ".github/actions/setup-linux/action.yml",
            # A cross-build image; no test job runs in it.
            ".ci/docker/ubuntu-cross-riscv/Dockerfile",
            ".github/ci_commit_pins/vllm.txt",
        ]:
            self.assertFalse(needs_run_all([f]), f)
        self.assertTrue(needs_run_all(["docs/source/index.rst", "test/run_test.py"]))
        self.assertFalse(needs_run_all([]))


if __name__ == "__main__":
    unittest.main()
