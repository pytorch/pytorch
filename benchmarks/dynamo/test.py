import importlib
import importlib.util
import io
import os
import sys
import types
import unittest
import weakref
from contextlib import nullcontext, redirect_stdout
from unittest import mock

import torch
from torch.testing._internal.common_utils import run_tests, TestCase

from . import common
from .common import parse_args, run
from .torchbench import setup_torchbench_cwd, TorchBenchmarkRunner


try:
    # fbcode only
    from aiplatform.utils.sanitizer_status import is_asan_or_tsan
except ImportError:

    def is_asan_or_tsan():
        return False


class TestDynamoBenchmark(TestCase):
    def test_eager_warmup_does_not_retain_compiled_model(self) -> None:
        live_copies = weakref.WeakSet()

        class TrackingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def __deepcopy__(self, memo):
                model_copy = type(self)()
                model_copy.load_state_dict(self.state_dict())
                memo[id(self)] = model_copy
                live_copies.add(model_copy)
                return model_copy

            def forward(self, inputs):
                return self.linear(inputs)

        class TrackingRunner(common.BenchmarkRunner):
            def __init__(self):
                super().__init__()
                self.copy_counts = []

            def pick_grad(self, name, is_training):
                return nullcontext()

        runner = TrackingRunner()
        runner.args = parse_args(
            ["-dcpu", "--backend=eager", "--performance", "--inference", "-n1"]
        )

        def model_iter_fn(model, inputs, collect_outputs=True):
            runner.copy_counts.append(len(live_copies))
            return model(inputs)

        runner.model_iter_fn = model_iter_fn
        experiment = mock.Mock(return_value="done")
        experiment.func = common.speedup_experiment

        with mock.patch("benchmarks.dynamo.common.current_device", "cpu"):
            runner.run_performance_test(
                "model",
                TrackingModel(),
                torch.ones(1, 2),
                lambda fn: fn,
                experiment,
            )

        self.assertTrue(runner.copy_counts)
        self.assertEqual(max(runner.copy_counts), 1)

    def test_timm_auto_install_uses_no_deps(self) -> None:
        module_name = "benchmarks.dynamo._timm_models_install_test"
        module_path = os.path.join(os.path.dirname(__file__), "timm_models.py")
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None:
            self.fail("could not load timm_models.py spec")
        if spec.loader is None:
            self.fail("could not load timm_models.py loader")
        module = importlib.util.module_from_spec(spec)

        fake_timm = types.ModuleType("timm")
        fake_timm.__version__ = "1.0.0"
        fake_timm_data = types.ModuleType("timm.data")
        fake_timm_data.resolve_data_config = lambda *args, **kwargs: {}
        fake_timm_models = types.ModuleType("timm.models")
        fake_timm_models.create_model = lambda *args, **kwargs: None
        fake_timm_models.list_models = lambda *args, **kwargs: []
        fake_timm.data = fake_timm_data
        fake_timm.models = fake_timm_models

        original_import_module = importlib.import_module

        def import_module(name, package=None):
            if name == "timm":
                raise ModuleNotFoundError("No module named 'timm'")
            return original_import_module(name, package)

        with (
            mock.patch.dict(
                sys.modules,
                {
                    "timm": fake_timm,
                    "timm.data": fake_timm_data,
                    "timm.models": fake_timm_models,
                },
            ),
            mock.patch("importlib.import_module", side_effect=import_module),
            mock.patch("subprocess.check_call") as check_call,
        ):
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            finally:
                sys.modules.pop(module_name, None)

        check_call.assert_called_once_with(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "git+https://github.com/rwightman/pytorch-image-models",
            ]
        )

    def test_prepare_repro_installs_timm_without_deps(self) -> None:
        from . import perf_cli

        def read_pin(name):
            return "abc123" if name == "timm.txt" else "unused"

        output = io.StringIO()
        args = types.SimpleNamespace(suite="timm", no_repro=True)
        with (
            mock.patch.object(perf_cli, "read_pin", side_effect=read_pin),
            redirect_stdout(output),
        ):
            perf_cli.cmd_prepare_repro(args)

        self.assertIn(
            "pip install --no-deps "
            "git+https://github.com/huggingface/pytorch-image-models@abc123",
            output.getvalue(),
        )

    def test_dashboard_performance_uses_warm_peak_memory(self) -> None:
        args = parse_args(
            [
                "-dcuda",
                "--inductor",
                "--inference",
                "--performance",
                "--dashboard",
            ]
        )
        self.assertTrue(args.use_warm_peak_memory)

        args = parse_args(
            [
                "-dcuda",
                "--inductor",
                "--inference",
                "--performance",
            ]
        )
        self.assertFalse(args.use_warm_peak_memory)

    def test_detectron2_maskrcnn_uses_iou_for_bool_masks(self) -> None:
        runner = TorchBenchmarkRunner()
        for name in (
            "detectron2_maskrcnn_r_101_fpn",
            "detectron2_maskrcnn_r_50_c4",
        ):
            self.assertTrue(runner.use_iou_for_bool_accuracy(name))
            self.assertEqual(runner.get_iou_threshold(name), 0.99)

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


if __name__ == "__main__":
    run_tests()
