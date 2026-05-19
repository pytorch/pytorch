import os
import unittest
from types import SimpleNamespace

import torch

from .common import BenchmarkRunner, parse_args, run
from .torchbench import setup_torchbench_cwd, TorchBenchmarkRunner


try:
    # fbcode only
    from aiplatform.utils.sanitizer_status import is_asan_or_tsan
except ImportError:

    def is_asan_or_tsan():
        return False


class TestDynamoBenchmark(unittest.TestCase):
    def test_adam_accuracy_collects_single_iteration(self) -> None:
        runner = BenchmarkRunner()
        runner.args = SimpleNamespace(iterations=4, accuracy=True, training=True)
        param = torch.nn.Parameter(torch.ones(()))
        runner.optimizer = torch.optim.Adam([param])

        calls = 0

        def model_iter_fn(mod, inputs, collect_outputs=True):
            nonlocal calls
            calls += 1
            return calls

        self.assertEqual(runner.run_n_iterations(None, None, model_iter_fn), 1)
        self.assertEqual(calls, 1)

    def test_non_adam_accuracy_collects_configured_iterations(self) -> None:
        runner = BenchmarkRunner()
        runner.args = SimpleNamespace(iterations=4, accuracy=True, training=True)
        param = torch.nn.Parameter(torch.ones(()))
        runner.optimizer = torch.optim.SGD([param], lr=0.1)

        calls = 0

        def model_iter_fn(mod, inputs, collect_outputs=True):
            nonlocal calls
            calls += 1
            return calls

        self.assertEqual(runner.run_n_iterations(None, None, model_iter_fn), 4)
        self.assertEqual(calls, 4)

    def test_accuracy_snapshot_clones_tensors(self) -> None:
        runner = BenchmarkRunner()
        tensor = torch.ones(2)

        snapshot = runner.clone_tensors_for_accuracy({"tensor": tensor})
        tensor.add_(1)

        self.assertTrue(torch.equal(snapshot["tensor"], torch.ones(2)))

    def test_torchbench_adam_accuracy_snapshots_before_step(self) -> None:
        runner = TorchBenchmarkRunner()
        runner.args = SimpleNamespace(accuracy=True, training=True)
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        runner.optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

        result = runner.forward_and_backward_pass(
            model, (torch.ones(1, 1),), collect_outputs=True
        )
        weight_snapshot = result[3]["weight"]

        self.assertTrue(torch.equal(weight_snapshot, torch.ones_like(weight_snapshot)))
        self.assertFalse(torch.equal(model.weight, weight_snapshot))

    @unittest.skipIf(is_asan_or_tsan(), "ASAN/TSAN not supported")
    def test_benchmark_infra_runs(self) -> None:
        """
        Basic smoke test that TorchBench runs.

        This test is mainly meant to check that our setup in fbcode
        doesn't break.

        If you see a failure here related to missing CPP headers, then
        you likely need to update the resources list in:
            //caffe2:inductor
        """
        original_dir = setup_torchbench_cwd()
        try:
            args = parse_args(
                [
                    "-dcpu",
                    "--inductor",
                    "--training",
                    "--performance",
                    "--only=BERT_pytorch",
                    "-n1",
                    "--batch-size=1",
                ]
            )
            run(TorchBenchmarkRunner(), args, original_dir)
        finally:
            os.chdir(original_dir)
