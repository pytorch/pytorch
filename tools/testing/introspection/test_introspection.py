# Owner(s): ["module: ci"]
"""Tests for the test-introspection collector.

Fast tests cover the platform/descriptor logic. The collector smoke tests import a
real (small) test file in a subprocess and are marked slow.
"""

import os
import pathlib
import subprocess
import sys
import tempfile

from tools.testing.introspection import collector, platforms

from torch.testing._internal.common_utils import run_tests, slowTest, TestCase


INDUCTOR = "test/inductor/test_torchinductor.py"


class TestPlatforms(TestCase):
    def test_registry_get(self):
        self.assertEqual(platforms.get("linux-cpu").device_type, "cpu")
        with self.assertRaises(KeyError):
            platforms.get("does-not-exist")

    def test_cuda_caps_only_overrides_underivable_flags(self):
        # Every other PLATFORM_SUPPORTS_* must be absent so common_cuda computes it
        # for the simulated capability. A hardcoded copy silently diverges: an
        # earlier one claimed FP8_GROUPED_GEMM was `sm >= (9, 0)` when the real
        # predicate is `SM90OrLater and not SM100OrLater`, so B200 results were wrong.
        underivable = {
            "PLATFORM_SUPPORTS_FUSED_SDPA",
            "PLATFORM_SUPPORTS_CK_SDPA",
            "PLATFORM_SUPPORTS_GREEN_CONTEXT",
            "PLATFORM_SUPPORTS_WORKQUEUE_CONFIG",
        }
        for name in (
            "linux-cuda-sm80",
            "linux-cuda-sm86",
            "linux-cuda-sm90",
            "linux-cuda-sm100",
        ):
            self.assertEqual(set(platforms.get(name).caps), underivable, name)

    def test_registry_covers_the_ciflow_runner_capabilities(self):
        self.assertEqual(platforms.get("linux-cuda-sm86").cuda_capability, (8, 6))
        self.assertEqual(platforms.get("linux-cuda-sm100").cuda_capability, (10, 0))

    def test_subprocess_env_hides_accelerators(self):
        env = platforms.get("linux-rocm").subprocess_env()
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(env["PYTORCH_TEST_WITH_ROCM"], "1")


class TestCollector(TestCase):
    @slowTest
    def test_worker_does_not_shadow_torch_from_cwd(self):
        # In CI torch is wheel-installed (site-packages) while the repo tree still has
        # a torch/ source dir. The worker must import the installed torch, not the repo
        # source. Reproduce by running the worker from a cwd containing a poison torch/
        # package; it must still import the real torch (worker runs by path, so cwd is
        # not on sys.path[0]).
        with tempfile.TemporaryDirectory() as td:
            (pathlib.Path(td) / "torch").mkdir()
            (pathlib.Path(td) / "torch" / "__init__.py").write_text(
                "raise RuntimeError('poison torch on path')\n"
            )
            env = dict(os.environ)
            env.update(platforms.get_job("linux-cpu").subprocess_env())
            proc = subprocess.run(
                [
                    sys.executable,
                    collector._COLLECTOR,
                    "linux-cpu/default",
                    "enumerate",
                    "test/test_bundled_inputs.py",
                ],
                cwd=td,
                env=env,
                capture_output=True,
                text=True,
                timeout=300,
            )
            self.assertNotIn("poison torch", proc.stderr)
            self.assertTrue(
                any(
                    line.startswith(collector._SENTINEL)
                    for line in proc.stdout.splitlines()
                ),
                msg=f"worker failed:\n{proc.stderr[-2000:]}",
            )

    @slowTest
    def test_device_gating(self):
        # GPU classes appear only on the cuda platform, not on cpu.
        cpu = collector.enumerate_tests(
            INDUCTOR, platforms.get_job("linux-cpu"), use_cache=False
        )
        cuda = collector.enumerate_tests(
            INDUCTOR, platforms.get_job("linux-cuda-sm80"), use_cache=False
        )
        self.assertNotIn("GPUTests", cpu)
        self.assertIn("GPUTests", cuda)
        self.assertIn("CpuTests", cpu)

    @slowTest
    def test_status_consistency(self):
        # ran union skipped must equal the enumerated set.
        job = platforms.get_job("linux-cpu")
        enum = collector.enumerate_tests(INDUCTOR, job, use_cache=False)
        enumerated = {f"{c}::{m}" for c, ms in enum.items() for m in ms}
        st = collector.status(INDUCTOR, job, use_cache=False)
        observed = set(st["ran"]) | {k for k, _ in st["skipped"]}
        self.assertEqual(enumerated, observed)


if __name__ == "__main__":
    run_tests()
