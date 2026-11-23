# Owner(s): ["module: autograd"]

import os
import pathlib
import subprocess
import sys
import unittest

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    HardwareClassification,
    IS_WINDOWS,
    run_tests,
    slowTest,
    TemporaryFileName,
    TestCase,
)

PYTORCH_COLLECT_COVERAGE = bool(os.environ.get("PYTORCH_COLLECT_COVERAGE"))


# This is a very simple smoke test for the functional autograd benchmarking script.
class _FunctionalAutogradBenchmarkHarness:
    def _test_runner(self, model, gpu):
        # Note about windows:
        # The temporary file is exclusively open by this process and the child process
        # is not allowed to open it again. As this is a simple smoke test, we choose for now
        # not to run this on windows and keep the code here simple.
        benchmark_script = str(
            pathlib.Path(__file__).resolve().parent.parent
            / "benchmarks"
            / "functional_autograd_benchmark"
            / "functional_autograd_benchmark.py"
        )
        with TemporaryFileName() as out_file:
            cmd = [sys.executable, benchmark_script]
            # Only run the warmup
            cmd += ["--num-iters", "0"]
            # Only run the vjp task (fastest one)
            cmd += ["--task-filter", "vjp"]
            # Only run the specified model
            cmd += ["--model-filter", model]
            # Output file
            cmd += ["--output", out_file]
            # -1 runs on cpu, any other value is the accelerator device index
            cmd += ["--gpu", str(gpu)]

            res = subprocess.run(cmd, check=False)

            self.assertTrue(res.returncode == 0)
            # Check that something was written to the file
            self.assertTrue(os.stat(out_file).st_size > 0)


class TestFunctionalAutogradBenchmark(_FunctionalAutogradBenchmarkHarness, TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    @unittest.skipIf(
        PYTORCH_COLLECT_COVERAGE,
        "Can deadlocks with gcov, see https://github.com/pytorch/pytorch/issues/49656",
    )
    def test_fast_tasks(self, device):
        fast_tasks = [
            "resnet18",
            "ppl_simple_reg",
            "ppl_robust_reg",
            "wav2letter",
            "transformer",
            "multiheadattn",
        ]

        benchmark_device = torch.device(device)
        gpu = -1 if benchmark_device.type == "cpu" else benchmark_device.index or 0
        for task in fast_tasks:
            self._test_runner(task, gpu)


class TestFunctionalAutogradBenchmarkSlow(_FunctionalAutogradBenchmarkHarness, TestCase):
    # The slow tasks only run on cpu: the CI accelerators don't have enough
    # memory for them, so the coverage they need is cpu coverage.
    hw_classification = HardwareClassification.CPU

    @slowTest
    @unittest.skipIf(
        IS_WINDOWS,
        "NamedTemporaryFile on windows does not have all the features we need.",
    )
    def test_slow_tasks(self):
        slow_tasks = ["fcn_resnet", "detr"]
        # deepspeech is voluntarily excluded as it takes too long to run without
        # proper tuning of the number of threads it should use.

        for task in slow_tasks:
            # Disable GPU for slow test as the CI GPU don't have enough memory
            self._test_runner(task, -1)


instantiate_device_type_tests(TestFunctionalAutogradBenchmark, globals(), allow_xpu=True, allow_mps=True)

if __name__ == "__main__":
    run_tests()
