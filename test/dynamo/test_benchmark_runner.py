# Owner(s): ["module: dynamo"]

import sys
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from benchmarks.dynamo import common


sys.path.remove(str(REPO_ROOT))


requires_distributed = unittest.skipIf(
    not torch.distributed.is_available(), "requires distributed"
)


class BenchmarkRunnerTests(TestCase):
    def test_fp64_oom_falls_back_to_cosine(self):
        runner = mock.Mock(spec=common.BenchmarkRunner())
        runner.args = types.SimpleNamespace(
            backend="inductor",
            training=False,
            cosine=False,
            export=False,
            export_aot_inductor=False,
            export_nativert=False,
            torchscript_jit_trace=False,
            aot_precompile=False,
            freezing=False,
            save_model_outputs_to=None,
            compare_model_outputs_with=None,
        )
        runner.non_deterministic_models = set()
        runner.skip_accuracy_checks_large_models_dashboard = set()
        runner.skip_accuracy_check_as_eager_non_deterministic = set()
        runner.pick_grad = mock.Mock(return_value=nullcontext())
        runner.get_tolerance_and_cosine_flag.side_effect = lambda *args: (
            1e-4,
            runner.args.cosine,
        )
        runner.get_accuracy_check_runs.return_value = 1
        model = mock.Mock()
        runner.maybe_cast.return_value = (model, [])
        result = object()
        runner.run_n_iterations.side_effect = [
            torch.cuda.OutOfMemoryError("fp64 reference OOM"),
            result,
            result,
            result,
        ]

        with (
            mock.patch.object(common, "cast_to_fp64", return_value=(model, [])),
            mock.patch.object(common, "empty_gpu_cache"),
            mock.patch.object(common, "reset_rng_state"),
            mock.patch.object(common, "same", return_value=True) as same,
            self.assertLogs(common.log, level="WARNING") as logs,
        ):
            status = common.BenchmarkRunner.check_accuracy(
                runner, "test_model", model, [], lambda fn: fn, None, None
            )

        self.assertEqual(status, "pass")
        self.assertTrue(runner.args.cosine)
        self.assertEqual(runner.run_n_iterations.call_count, 4)
        self.assertIsNone(same.call_args.args[2])
        self.assertTrue(same.call_args.kwargs["cos_similarity"])
        self.assertIn("fp64 reference OOM", "\n".join(logs.output))

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


if __name__ == "__main__":
    run_tests()
