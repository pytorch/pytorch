import os
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import torch

from . import common
from .common import parse_args, run, speedup_experiment, timed
from .torchbench import setup_torchbench_cwd, TorchBenchmarkRunner


try:
    # fbcode only
    from aiplatform.utils.sanitizer_status import is_asan_or_tsan
except ImportError:

    def is_asan_or_tsan():
        return False


class TestDynamoBenchmark(unittest.TestCase):
    def test_timed_does_not_retain_outputs_when_result_unused(self) -> None:
        refs: list[weakref.ref[object]] = []

        class Output:
            pass

        def model_iter_fn(model, example_inputs, collect_outputs=True):
            if refs:
                self.assertIsNone(refs[-1]())
            output = Output()
            refs.append(weakref.ref(output))
            return output

        timed(None, model_iter_fn, (), times=3, return_result=False)

        self.assertIsNone(refs[-1]())

    def test_speedup_experiment_uses_separate_eager_model(self) -> None:
        class StatefulModel(torch.nn.Module):
            def __init__(self, label):
                super().__init__()
                self.label = label
                self.counter = 0

            def forward(self, x):
                self.counter += 1
                return x + self.counter

        args = SimpleNamespace(
            repeat=1,
            iterations_per_run=1,
            randomize_input=False,
            xla_tolerance=1e-2,
            trace_on_xla=False,
            export_aot_inductor=False,
            export_nativert=False,
            torchscript_jit_trace=False,
            aot_precompile=False,
            collect_outputs=False,
            export_profiler_trace=False,
            dump_raw_metrics=False,
            baseline=None,
            _print_latency_ms=False,
        )
        eager_model = StatefulModel("eager")
        dynamo_model = StatefulModel("dynamo")
        seen_labels = []

        def model_iter_fn(model, example_inputs, collect_outputs=True):
            seen_labels.append(model.label)
            return model(*example_inputs)

        with (
            patch.object(common, "write_outputs"),
            patch.object(common, "output_signpost", return_value=0),
            patch.object(common, "output_filename", "speedup_test.csv"),
            patch.object(common, "current_device", "cpu"),
            patch.object(common, "current_name", "stateful_model"),
            patch.object(common, "current_batch_size", 1),
        ):
            speedup_experiment(
                args,
                model_iter_fn,
                dynamo_model,
                (torch.ones(1),),
                hf_llm=False,
                eager_model=eager_model,
            )

        self.assertEqual(seen_labels, ["eager", "dynamo"])
        self.assertEqual(eager_model.counter, 1)
        self.assertEqual(dynamo_model.counter, 1)

    def test_call_model_generate_uses_passed_model(self) -> None:
        class ModelWithGenerate:
            def __init__(self, label):
                self.label = label

            def generate(self, **kwargs):
                return self.label

        self.assertEqual(
            common.call_model_generate(ModelWithGenerate("passed"), {}),
            "passed",
        )

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
