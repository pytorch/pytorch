# Owner(s): ["module: dynamo"]

import csv
import io
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import torch
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from benchmarks.dynamo import common


sys.path.remove(str(REPO_ROOT))


requires_distributed = unittest.skipIf(
    not torch.distributed.is_available(), "requires distributed"
)


class AccuracyBenchmarkRunner(common.BenchmarkRunner):
    suite_name = "test"

    def forward_pass(self, model, inputs, collect_outputs=True):
        return model(*inputs)

    def pick_grad(self, name, is_training):
        return torch.no_grad()

    def get_tolerance_and_cosine_flag(self, is_training, current_device, name):
        return 1e-4, False


class BenchmarkRunnerTests(TestCase):
    @parametrize("preexisting_graphs", (0, 1, 3))
    @torch._dynamo.disable
    def test_accuracy_counters_exclude_model_setup(self, preexisting_graphs):
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)
        runner = AccuracyBenchmarkRunner()
        runner.args = common.parse_args(
            ["--accuracy", "--inference", "--device", "cpu", "--backend", "eager"]
        )
        runner.args.iterations = 1
        runner.model_iter_fn = runner.forward_pass
        output = io.StringIO()

        with tempfile.TemporaryDirectory() as directory:
            result_path = Path(directory, "accuracy.csv")
            with (
                mock.patch.multiple(
                    common,
                    current_device="cpu",
                    current_name="model",
                    current_batch_size=4,
                    output_filename=str(result_path),
                ),
                mock.patch.object(common, "output_signpost", return_value=0.0),
                mock.patch.dict(torch._dynamo.utils.counters, clear=True),
                redirect_stdout(output),
            ):
                torch._dynamo.utils.counters["stats"].update(
                    unique_graphs=preexisting_graphs,
                    calls_captured=100 * preexisting_graphs,
                )
                runner.run_one_model(
                    "model",
                    torch.nn.ReLU(),
                    (torch.randn(4, 4),),
                    torch._dynamo.optimize("eager"),
                    None,
                    explain=True,
                )
                captured = common.get_dynamo_stats()

            with result_path.open() as results:
                row = next(csv.DictReader(results))

        self.assertEqual(row["accuracy"], "pass")
        self.assertEqual(int(row["unique_graphs"]), 1)
        self.assertGreater(int(row["calls_captured"]), 0)
        for field in (
            "unique_graphs",
            "calls_captured",
            "graph_breaks",
            "fallbacks_to_eager",
        ):
            self.assertEqual(int(row[field]), captured[field])
        self.assertIn("Dynamo produced 1 graphs", output.getvalue())

    @requires_distributed
    def test_default_fsdp_policy_does_not_import_model_dependencies(self):
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

        model_deps = frozenset({"diffusers", "torchbenchmark", "transformers"})
        with mock.patch.dict(sys.modules):
            for name in list(sys.modules):
                if name.partition(".")[0] in model_deps:
                    del sys.modules[name]
            policy = common.BenchmarkRunner().get_fsdp_auto_wrap_policy("resnet50")
            loaded = {name.partition(".")[0] for name in sys.modules} & model_deps

        self.assertEqual(loaded, set())
        self.assertIs(policy.func, size_based_auto_wrap_policy)
        self.assertEqual(policy.keywords["min_num_params"], int(1e5))

    @requires_distributed
    def test_diffusion_fsdp_policy_imports_current_model_class(self):
        from torch.distributed.fsdp.wrap import ModuleWrapPolicy

        class Transformer2DModel(torch.nn.Module):
            pass

        module = types.SimpleNamespace(Transformer2DModel=Transformer2DModel)
        with mock.patch("importlib.import_module", return_value=module) as imported:
            policy = common.BenchmarkRunner().get_fsdp_auto_wrap_policy(
                "stable_diffusion_unet"
            )

        imported.assert_called_once_with("diffusers.models.transformers.transformer_2d")
        self.assertIsInstance(policy, ModuleWrapPolicy)
        self.assertTrue(policy(Transformer2DModel(), recurse=False))


instantiate_parametrized_tests(BenchmarkRunnerTests)


if __name__ == "__main__":
    run_tests()
