# Owner(s): ["module: dynamo"]

import collections
import contextlib
import copy
import functools
import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch._dynamo.testing import AotEagerAndRecordGraphs
from torch.testing._internal.common_device_type import instantiate_device_type_tests
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


class BenchmarkRunnerTests(TestCase):
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


@instantiate_parametrized_tests
class TestHuggingFaceLLMPerformance(TestCase):
    @parametrize("training", [False, True])
    def test_compilation_latency_uses_matched_work(self, training) -> None:
        calls = collections.Counter()
        clock = [0.0]
        optimized = []
        result = {}

        class Model(torch.nn.Module):
            def forward(self, inputs):
                calls["eager"] += 1
                clock[0] += 10.0

        class Runner(common.BenchmarkRunner):
            hf_llm = True
            suite_name = "test"

            def deepcopy_and_maybe_parallelize(self, model):
                return model

            def init_optimizer(self, name, device, params):
                pass

            def pick_grad(self, name, is_training):
                return contextlib.nullcontext()

            def generate(self, model, example_inputs):
                return model(example_inputs)

            def forward_and_backward_pass(self, model, example_inputs):
                return model(example_inputs)

        def optimize_ctx(fn):
            optimized.append(fn)

            def compiled(*args, **kwargs):
                calls["compiled"] += 1
                clock[0] += 30.0 if calls["compiled"] == 1 else 10.0

            return compiled

        def experiment(args, model_iter_fn, model, example_inputs, **kwargs):
            result["model_iter_fn"] = model_iter_fn
            result.update(kwargs)
            return "done"

        runner = Runner()
        runner.args = common.parse_args(
            [
                "-dcpu",
                "--backend=eager",
                "--performance",
                "--training" if training else "--inference",
                "--only=model",
            ]
        )
        runner.model_iter_fn = runner.forward_and_backward_pass
        model = Model()

        def get_peak_memory():
            return calls["compiled"] or calls["eager"]

        def get_dynamo_stats():
            return collections.Counter(calls_captured=calls["compiled"])

        with (
            mock.patch.object(common, "current_device", "cuda"),
            mock.patch.object(common, "empty_gpu_cache"),
            mock.patch.object(common, "get_dynamo_stats", side_effect=get_dynamo_stats),
            mock.patch.object(common, "get_peak_memory", side_effect=get_peak_memory),
            mock.patch.object(
                common.time, "perf_counter", side_effect=lambda: clock[0]
            ),
            mock.patch.object(torch.cuda, "reset_peak_memory_stats"),
            mock.patch.object(common, "speedup_experiment", experiment),
        ):
            runner.run_performance_test(
                "model",
                model,
                (),
                optimize_ctx,
                functools.partial(experiment, object()),
            )

        eager_calls = 5 if training else 1
        self.assertEqual(calls, {"eager": eager_calls, "compiled": 5})
        self.assertEqual(result["compilation_latency"], 20.0)
        self.assertEqual(result["eager_peak_mem"], eager_calls)
        self.assertEqual(result["dynamo_peak_mem"], 5)
        self.assertEqual(result["dynamo_stats"], {"calls_captured": 5})
        if training:
            self.assertEqual(optimized, [runner.forward_and_backward_pass])
            self.assertEqual(result["model_iter_fn"], runner.forward_and_backward_pass)
        else:
            self.assertEqual(len(optimized), 1)
            self.assertIs(optimized[0].__self__, model)
            self.assertIs(optimized[0].__func__, Model.forward)
            self.assertEqual(result["model_iter_fn"], runner.generate)
        self.assertEqual(result["hf_llm"], not training)
        self.assertTrue(runner.hf_llm)


class TestHuggingFaceLLMTraining(TestCase):
    def setUp(self):
        super().setUp()
        module_name = "benchmarks.dynamo._test_huggingface"
        spec = importlib.util.spec_from_file_location(
            module_name, REPO_ROOT / "benchmarks/dynamo/huggingface.py"
        )
        if spec is None or spec.loader is None:
            raise AssertionError("could not load huggingface.py")
        self.assertNotIn(module_name, sys.modules)
        self.huggingface = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = self.huggingface
        self.addCleanup(sys.modules.pop, module_name, None)
        self.addCleanup(torch._dynamo.reset)

        llm_models = types.ModuleType("benchmarks.dynamo.huggingface_llm_models")
        llm_models.HF_LLM_MODELS = {}
        with (
            mock.patch.dict(
                sys.modules,
                {
                    "transformers": mock.Mock(),
                    "benchmarks.dynamo.huggingface_llm_models": llm_models,
                },
            ),
            mock.patch.dict(os.environ, {"TORCHBENCH_ONLY_MODELS": ""}),
            torch._inductor.config.patch(fx_graph_cache=True),
            torch._functorch.config.patch(enable_autograd_cache=True),
        ):
            spec.loader.exec_module(self.huggingface)

    def _load_model(self, device, inputs, *, training=True, nested_config=False):
        output_type = collections.namedtuple("Output", ["logits"])

        class Config:
            def __init__(self, text_config=None):
                self.use_cache = True
                self.text_config = text_config

            def get_text_config(self):
                return self.text_config if self.text_config is not None else self

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.5, device=device))
                self.config = Config(Config() if nested_config else None)
                self.generation_config = types.SimpleNamespace()

            def forward(self, input_ids, labels=None):
                logits = input_ids.float() * self.weight
                if labels is None:
                    return output_type(logits)
                loss = (logits - labels.float()).square().mean()
                return (loss, logits)

        class Adapter:
            @staticmethod
            def get_model_and_inputs(model_name, device):
                return Model(), {key: value.clone() for key, value in inputs.items()}

        runner = self.huggingface.HuggingfaceRunner()
        runner.args = common.parse_args(
            [
                "--device",
                device,
                "--backend=eager",
                "--accuracy",
                "--training" if training else "--inference",
                "--only=tiny",
                "--iterations=1",
            ]
        )
        runner.model_iter_fn = (
            runner.forward_and_backward_pass if training else runner.forward_pass
        )
        validate_ctx = (
            contextlib.nullcontext()
            if "input_ids" in inputs
            else mock.patch.object(runner, "validate_model")
        )
        llm_models = self.huggingface.HF_LLM_MODELS
        with (
            mock.patch.dict(llm_models, {"tiny": Adapter}, clear=True),
            validate_ctx,
        ):
            return runner, runner.load_model(device, "tiny", batch_size=1)

    @parametrize("dynamic", [False, True])
    @parametrize("nested_config", [False, True])
    @torch._dynamo.config.patch(suppress_errors=False)
    def test_text_training_loss_and_gradients(self, device, dynamic, nested_config):
        input_ids = torch.tensor([[1, 2, 3]], device=device)
        runner, loaded = self._load_model(
            device, {"input_ids": input_ids}, nested_config=nested_config
        )
        _, _, model, example_inputs, _ = loaded

        self.assertEqual(set(example_inputs), {"input_ids", "labels"})
        self.assertEqual(example_inputs["labels"], input_ids)
        self.assertNotEqual(
            example_inputs["labels"].data_ptr(), example_inputs["input_ids"].data_ptr()
        )
        self.assertFalse(model.config.get_text_config().use_cache)
        self.assertTrue(model.training)

        compiled_model = copy.deepcopy(model)
        eager_result = runner.forward_and_backward_pass(model, example_inputs)
        self.assertEqual(eager_result[1].dim(), 0)
        self.assertIsNotNone(model.weight.grad)
        self.assertNotEqual(model.weight.grad.item(), 0.0)

        backend = AotEagerAndRecordGraphs()
        compiled = torch.compile(
            runner.forward_and_backward_pass, backend=backend, dynamic=dynamic
        )
        result = compiled(compiled_model, example_inputs)

        self.assertTrue(backend.bw_graphs)
        self.assertEqual(result[1], eager_result[1])
        self.assertEqual(compiled_model.weight.grad, model.weight.grad)

    @parametrize(
        "training,input_name",
        [(False, "input_ids"), (False, "input_features"), (True, "input_features")],
    )
    def test_other_inputs_are_unchanged(self, device, training, input_name):
        inputs = {input_name: torch.ones(1, 2, dtype=torch.long, device=device)}
        if input_name == "input_features":
            inputs["decoder_input_ids"] = torch.ones(
                1, 1, dtype=torch.long, device=device
            )
        _, loaded = self._load_model(device, inputs, training=training)
        self.assertTrue(loaded[2].config.get_text_config().use_cache)
        self.assertEqual(loaded[3], inputs)


instantiate_device_type_tests(TestHuggingFaceLLMTraining, globals())


if __name__ == "__main__":
    run_tests()
