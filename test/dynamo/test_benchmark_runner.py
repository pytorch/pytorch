# Owner(s): ["module: dynamo"]

import collections
import contextlib
import csv
import functools
import importlib.metadata
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from packaging.version import Version

import torch
from torch._dynamo.testing import CompileCounter
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCUDA,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    TestCase,
)
from torch.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON


REPO_ROOT = Path(__file__).resolve().parents[2]
# Bind the namespace package explicitly: some CI images install pysbd, whose wheel
# ships a regular top-level `benchmarks` package that would shadow the repo's.
sys.modules["benchmarks"] = types.ModuleType("benchmarks")
sys.modules["benchmarks"].__path__ = [str(REPO_ROOT / "benchmarks")]
from benchmarks.dynamo import common


requires_distributed = unittest.skipIf(
    not torch.distributed.is_available(), "requires distributed"
)


class BenchmarkRunnerTests(TestCase):
    def test_timed_excludes_external_setup(self):
        clock = [0.0]
        queued = [0.0]
        calls = collections.Counter()

        def setup(model, inputs):
            calls["setup"] += 1
            clock[0] += 100.0
            queued[0] += 50.0

        def forward(model, inputs, collect_outputs=False):
            calls["forward"] += 1
            queued[0] += 10.0

        def synchronize():
            clock[0] += queued[0]
            queued[0] = 0.0

        with (
            mock.patch.object(common, "synchronize", side_effect=synchronize),
            mock.patch.object(common, "reset_rng_state"),
            mock.patch.object(common.time, "perf_counter", new=lambda: clock[0]),
        ):
            elapsed = common.timed(None, forward, (), times=2, setup_fn=setup)
        self.assertEqual(elapsed, 20.0)
        self.assertEqual(calls, {"setup": 2, "forward": 2})

    @skipIfTorchDynamo("speedup_experiment calls set_stance")
    @parametrize("mode", ("generate", "prefill", "decode"))
    def test_prefill_timing_units_and_setup(self, mode):
        args = common.parse_args(
            [
                "--performance",
                "--inference",
                "--backend=eager",
                f"--hf-inference-mode={mode}",
                "--iterations-per-run=4",
                "--repeat=1",
                "--export-profiler-trace",
            ]
        )
        args.profile_details = {}
        args.profiler_trace_name = "profile"
        profile = mock.Mock()
        setup = mock.Mock() if mode != "generate" else None
        model = torch.nn.Identity()
        model.name = "Qwen/Qwen3-0.6B"
        inputs = {
            "input_ids": torch.zeros(2, 3, dtype=torch.long),
            "decode_ids": torch.zeros(2, 5, dtype=torch.long),
        }
        with (
            mock.patch.object(common, "current_batch_size", 2),
            mock.patch.object(common, "output_filename", "results.csv"),
            mock.patch.object(
                common, "timed", side_effect=[(4.0, None), (2.0, None), None, None]
            ) as timed,
            mock.patch.object(torch._dynamo, "run", side_effect=lambda fn: fn),
            mock.patch.object(common, "write_outputs") as write,
            mock.patch("torch._dynamo.utils.compile_times", return_value=([], [])),
            mock.patch.object(common, "output_signpost"),
            mock.patch(
                "torch._inductor.utils.maybe_profile",
                return_value=contextlib.nullcontext(profile),
            ),
        ):
            common.speedup_experiment(
                args, mock.Mock(), model, inputs, hf_llm=True, setup_fn=setup
            )
        _, headers, values = write.call_args_list[0].args
        row = dict(zip(headers, values))
        self.assertEqual(row["speedup"], 2.0)
        self.assertEqual(row["abs_latency"], 2000.0)
        if mode == "generate":
            self.assertNotIn("eager_latency", row)
        else:
            self.assertEqual(row["eager_latency"], 4000.0)
        if mode == "prefill":
            self.assertEqual(row["input_tokens_per_second"], 12.0)
        else:
            self.assertNotIn("input_tokens_per_second", row)
        if mode == "decode":
            self.assertEqual(row["output_tokens_per_second"], 20.0)
        else:
            self.assertNotIn("output_tokens_per_second", row)
        self.assertEqual(timed.call_count, 4)
        for call in timed.call_args_list:
            self.assertIs(call.kwargs["setup_fn"], setup)
        trace = Path(profile.export_chrome_trace.call_args.args[0])
        self.assertEqual(trace.name, "profile_Qwen_Qwen3-0.6B.json")

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

    @skipIfTorchDynamo("parse_args is slow to trace")
    @parametrize(
        "flag",
        (
            "--hf-inference-mode=prefill",
            "--hf-inference-mode=decode",
            "--prompt-length=8",
            "--decode-length=8",
        ),
    )
    def test_prefill_options_require_huggingface(self, flag):
        args = common.parse_args(["--performance", "--inference", flag])
        with self.assertRaisesRegex(ValueError, "require the huggingface suite"):
            common.BenchmarkRunner().validate_args(args)

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


class TestHuggingFaceLLMPerformance(TestCase):
    @skipIfTorchDynamo("run_performance_test calls set_stance")
    @parametrize("prefill", (False, True))
    def test_compilation_latency_uses_matched_work(self, prefill):
        calls = collections.Counter()
        clock = [0.0]
        result = {}

        class Model(torch.nn.Module):
            def forward(self, inputs):
                calls["eager"] += 1
                clock[0] += 10.0

        class Runner(common.BenchmarkRunner):
            hf_llm = True
            suite_name = "test"

            def pick_grad(self, name, is_training):
                return contextlib.nullcontext()

            def generate(self, model, inputs, collect_outputs=False):
                return model(inputs)

            def get_performance_workload(self):
                return self.generate, self.setup if prefill else None

            def setup(self, model, inputs):
                calls["setup"] += 1
                clock[0] += 100.0

        def optimize_ctx(fn):
            def compiled(*args, **kwargs):
                calls["compiled"] += 1
                clock[0] += 30.0 if calls["compiled"] == 1 else 10.0

            return compiled

        def experiment(args, model_iter_fn, model, example_inputs, **kwargs):
            result.update(kwargs)
            return "done"

        runner = Runner()
        runner.args = common.parse_args(
            ["-dcpu", "--backend=eager", "--performance", "--inference", "--only=model"]
        )
        with (
            mock.patch.object(common, "current_device", "cuda"),
            mock.patch.object(common, "synchronize"),
            mock.patch.object(common, "reset_rng_state"),
            mock.patch.object(common, "empty_gpu_cache"),
            mock.patch.object(
                common,
                "get_dynamo_stats",
                side_effect=lambda: collections.Counter(
                    calls_captured=calls["compiled"]
                ),
            ),
            mock.patch.object(
                common,
                "get_peak_memory",
                side_effect=lambda: calls["compiled"] or calls["eager"],
            ),
            mock.patch.object(common.time, "perf_counter", new=lambda: clock[0]),
            mock.patch.object(torch.cuda, "reset_peak_memory_stats"),
            mock.patch.object(common, "speedup_experiment", experiment),
        ):
            runner.run_performance_test(
                "model", Model(), (), optimize_ctx, functools.partial(experiment, None)
            )
        self.assertEqual(calls["setup"], 6 if prefill else 0)
        self.assertEqual(calls["eager"], 1)
        self.assertEqual(calls["compiled"], 5)
        self.assertEqual(result["compilation_latency"], 20.0)
        self.assertEqual(result["eager_peak_mem"], 1)
        self.assertEqual(result["dynamo_peak_mem"], 5)
        self.assertEqual(result["dynamo_stats"], {"calls_captured": 5})


@skipIfTorchDynamo("drives the benchmark harness, which calls set_stance")
class HuggingFacePrefillTestCase(TestCase):
    def setUp(self):
        super().setUp()
        try:
            version = importlib.metadata.version("transformers")
        except importlib.metadata.PackageNotFoundError:
            raise unittest.SkipTest("requires transformers") from None
        if Version(version) < Version("5.13.0"):
            raise unittest.SkipTest("requires transformers >= 5.13.0")
        from transformers import AutoConfig

        with (
            mock.patch.object(AutoConfig, "from_pretrained", return_value=None),
            mock.patch(
                "subprocess.check_call",
                side_effect=AssertionError("unexpected install"),
            ),
        ):
            from benchmarks.dynamo import huggingface

        self.huggingface = huggingface

    def _runner_args(self, *extra, mode="prefill"):
        return common.parse_args(
            [
                "-dcpu",
                "--backend=eager",
                "--performance",
                "--inference",
                f"--hf-inference-mode={mode}",
                *extra,
            ]
        )

    def _make_model(self, model_type, device, dtype=torch.float32):
        import transformers

        from benchmarks.dynamo.huggingface_llm_models import TextGenerationPrefillModel

        model_cls = getattr(transformers, model_type + "ForCausalLM")
        config_name = "Qwen3_5Text" if model_type == "Qwen3_5" else model_type
        config_cls = getattr(transformers, config_name + "Config")
        config_kwargs = {
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "max_position_embeddings": 4096,
        }
        if model_type != "Llama":
            config_kwargs["head_dim"] = 8
        if model_type == "Gemma2":
            config_kwargs["sliding_window"] = 8
        if model_type == "Qwen3_5":
            config_kwargs.update(
                layer_types=["linear_attention", "full_attention"],
                linear_num_key_heads=1,
                linear_num_value_heads=2,
                linear_key_head_dim=8,
                linear_value_head_dim=8,
            )
        config = config_cls(**config_kwargs)
        model = model_cls(config).eval().to(device=device, dtype=dtype)
        return TextGenerationPrefillModel(model, 2, 4, 11).eval()

    def _decode_inputs(self, device):
        return {
            "input_ids": torch.randint(0, 32, (2, 4), device=device),
            "decode_ids": torch.randint(0, 32, (2, 5), device=device),
        }

    @contextlib.contextmanager
    def _record_cache_lengths(self, runner, forward_name):
        original_forward = getattr(runner, forward_name)
        cache_lengths = []

        def forward(model, example_inputs, collect_outputs=True):
            before = int(model.cache.get_seq_length())
            result = original_forward(model, example_inputs, collect_outputs)
            cache_lengths.append((before, int(model.cache.get_seq_length())))
            return result

        with mock.patch.object(runner, forward_name, forward):
            yield cache_lengths

    @contextlib.contextmanager
    def _mock_pretrained(self):
        from benchmarks.dynamo import huggingface_llm_models as llm

        model = self._make_model("Llama", "cpu").model
        tokenizer = types.SimpleNamespace(vocab_size=32, eos_token_id=2)
        with (
            mock.patch.object(
                llm.AutoTokenizer, "from_pretrained", return_value=tokenizer
            ),
            mock.patch.object(
                llm.AutoModelForCausalLM, "from_pretrained", return_value=model
            ),
        ):
            yield


class TestHuggingFacePrefill(HuggingFacePrefillTestCase):
    @parametrize("mode", ("prefill", "decode"))
    @parametrize(
        "extra, message",
        [
            ("--amp", "does not support --amp"),
            ("--dynamic-shapes", "does not support --dynamic-shapes"),
            ("--trace-on-xla", "does not support --trace-on-xla"),
            ("--batch-size=0", "--batch-size must be positive"),
            ("--prompt-length=0", "--prompt-length must be positive"),
            ("--iterations-per-run=0", "--iterations-per-run must be positive"),
        ],
    )
    def test_invalid_prefill_options(self, mode, extra, message):
        runner = self.huggingface.HuggingfaceRunner()
        with self.assertRaisesRegex(ValueError, message):
            runner.validate_args(self._runner_args(extra, mode=mode))

    @parametrize("decode_length", (0, 2000))
    def test_decode_length_range(self, decode_length):
        runner = self.huggingface.HuggingfaceRunner()
        args = self._runner_args(f"--decode-length={decode_length}", mode="decode")
        with self.assertRaisesRegex(ValueError, "must be between 1 and 1999"):
            runner.validate_args(args)
        args.decode_length = 1999
        runner.validate_args(args)

    @parametrize("mode", ("prefill", "decode"))
    def test_prefill_requires_inference_backend(self, mode):
        runner = self.huggingface.HuggingfaceRunner()
        args = self._runner_args(mode=mode)
        args.inference = False
        with self.assertRaisesRegex(ValueError, "requires --inference"):
            runner.validate_args(args)
        args.inference = True
        args.backend = None
        with self.assertRaisesRegex(ValueError, "requires --backend or --inductor"):
            runner.validate_args(args)

    @parametrize("mode", ("prefill", "decode"))
    @parametrize("name", ("Qwen/Qwen3-0.6B", "Qwen/Qwen3.5-0.8B"))
    def test_prefill_honors_model_filter(self, mode, name):
        runner = self.huggingface.HuggingfaceRunner()
        args = self._runner_args(mode=mode)
        args.filter, args.exclude, args.exclude_exact = ["."], ["^$"], []
        runner.args = args
        with mock.patch.dict(
            self.huggingface.BATCH_SIZE_KNOWN_MODELS, {name: 8}, clear=True
        ):
            self.assertEqual(list(runner.iter_model_names(args)), [name])
        args.only = name
        runner.validate_args(args)
        args.only = "openai/whisper-tiny"
        with self.assertRaisesRegex(ValueError, "does not support openai/whisper-tiny"):
            runner.validate_args(args)

    def test_decode_excludes_gemma2(self):
        runner = self.huggingface.HuggingfaceRunner()
        args = self._runner_args(mode="decode")
        args.filter, args.exclude, args.exclude_exact = ["."], ["^$"], []
        runner.args = args
        names = {"google/gemma-2-2b": 8, "Qwen/Qwen3-0.6B": 8}
        with mock.patch.dict(
            self.huggingface.BATCH_SIZE_KNOWN_MODELS, names, clear=True
        ):
            self.assertEqual(list(runner.iter_model_names(args)), ["Qwen/Qwen3-0.6B"])
        args.only = "google/gemma-2-2b"
        with self.assertRaisesRegex(ValueError, "does not support google/gemma-2-2b"):
            runner.validate_args(args)

    def test_prompt_length_requires_prefill(self):
        runner = self.huggingface.HuggingfaceRunner()
        args = common.parse_args(["--performance", "--inference", "--prompt-length=8"])
        with self.assertRaisesRegex(ValueError, "requires --hf-inference-mode=prefill"):
            runner.validate_args(args)

    @parametrize("mode", ("generate", "prefill"))
    def test_decode_length_requires_decode(self, mode):
        runner = self.huggingface.HuggingfaceRunner()
        args = self._runner_args("--decode-length=8", mode=mode)
        with self.assertRaisesRegex(ValueError, "requires --hf-inference-mode=decode"):
            runner.validate_args(args)

    @parametrize("mode", ("prefill", "decode"))
    @parametrize("prompt_length", (None, 7))
    def test_requested_input_dimensions(self, mode, prompt_length):
        runner = self.huggingface.HuggingfaceRunner()
        extra = () if prompt_length is None else (f"--prompt-length={prompt_length}",)
        runner.args = self._runner_args("--batch-size=3", *extra, mode=mode)
        with self._mock_pretrained(), mock.patch.object(runner, "validate_model"):
            _, _, model, inputs, batch_size = runner.load_model(
                "cpu", "Qwen/Qwen3-0.6B", batch_size=3
            )
        expected_length = prompt_length or 1000
        self.assertEqual(inputs["input_ids"].shape, (3, expected_length))
        self.assertEqual(batch_size, 3)
        self.assertEqual(model.cache_capacity, expected_length + 1999)
        if mode == "decode":
            self.assertEqual(inputs["decode_ids"].shape, (3, 128))
            self.assertIs(runner.model_iter_fn.__func__, type(runner).decode)
        else:
            self.assertNotIn("decode_ids", inputs)

    def test_requested_decode_length(self):
        runner = self.huggingface.HuggingfaceRunner()
        runner.args = self._runner_args("--decode-length=5", mode="decode")
        with self._mock_pretrained(), mock.patch.object(runner, "validate_model"):
            _, _, _, inputs, _ = runner.load_model(
                "cpu", "Qwen/Qwen3-0.6B", batch_size=2
            )
        self.assertEqual(inputs["decode_ids"].shape, (2, 5))

    def test_prompt_must_fit_generation_context(self):
        runner = self.huggingface.HuggingfaceRunner()
        runner.args = self._runner_args("--prompt-length=2097")
        with (
            self._mock_pretrained(),
            self.assertRaisesRegex(ValueError, "exceeds .* context length 4096"),
        ):
            runner.load_model("cpu", "Qwen/Qwen3-0.6B", batch_size=1)

    @parametrize("mode", ("prefill", "decode"))
    def test_prefill_performance_run(self, mode):
        runner = self.huggingface.HuggingfaceRunner()
        with (
            tempfile.TemporaryDirectory() as tmp,
            torch._dynamo.config.patch(base_dir=tmp),
            # run() rebinds module globals such as output_filename and current_name.
            mock.patch.dict(vars(common)),
            self._record_cache_lengths(runner, f"{mode}_forward") as cache_lengths,
            self._mock_pretrained(),
        ):
            extra = ("--decode-length=3",) if mode == "decode" else ()
            args = self._runner_args(
                "--only=Qwen/Qwen3-0.6B",
                "--batch-size=2",
                "--prompt-length=7",
                "--repeat=1",
                "--iterations-per-run=2",
                "--export-profiler-trace",
                *extra,
                mode=mode,
            )
            common.run(runner, args)
            trace = f"eager_{mode}_Qwen_Qwen3-0.6B.json"
            self.assertTrue(os.path.exists(os.path.join(tmp, trace)))
            with open(os.path.join(tmp, f"speedup_eager_{mode}.csv")) as f:
                (row,) = csv.DictReader(f)
        # validate_model, eager and compiled warmup, timed runs, and profiled runs.
        expected_lengths = (0, 7) if mode == "prefill" else (7, 10)
        self.assertEqual(cache_lengths, [expected_lengths] * 15)
        self.assertEqual(row["name"], "Qwen/Qwen3-0.6B")
        self.assertEqual(int(row["batch_size"]), 2)
        self.assertGreater(float(row["eager_latency"]), 0)
        tokens_column = "input" if mode == "prefill" else "output"
        tokens = 7 if mode == "prefill" else 3
        tokens_per_second = 2 * 2 * tokens / (float(row["abs_latency"]) / 1000)
        self.assertEqual(
            float(row[f"{tokens_column}_tokens_per_second"]),
            tokens_per_second,
            rtol=1e-3,
            atol=0,
        )


class TestHuggingFacePrefillDevice(HuggingFacePrefillTestCase):
    def _cache_tensors(self, model):
        return [
            (layer.conv_states, layer.recurrent_states)
            if is_linear
            else (layer.keys, layer.values)
            for layer, is_linear in zip(model.cache.layers, model.cache.is_linear)
        ]

    @parametrize("model_type", ("Llama", "Gemma2", "Qwen3", "Qwen3_5"))
    def test_prefill_resets_cache(self, device, model_type):
        runner = self.huggingface.HuggingfaceRunner()
        runner.args = self._runner_args("--iterations=2")
        model = self._make_model(model_type, device)
        inputs = {"input_ids": torch.randint(0, 32, (2, 4), device=device)}
        original_forward = model.model.forward
        starting_lengths = []

        def forward(*args, **kwargs):
            cache = kwargs["past_key_values"]
            starting_lengths.append(int(cache.get_seq_length()))
            for layer, is_linear in zip(cache.layers, cache.is_linear):
                if is_linear:
                    self.assertFalse(layer.has_previous_state)
                    for states in (layer.conv_states, layer.recurrent_states):
                        self.assertEqual(states, torch.zeros_like(states))
            return original_forward(*args, **kwargs)

        with (
            torch.no_grad(),
            mock.patch.object(model.model, "forward", side_effect=forward),
        ):
            output = runner.run_n_iterations(model, inputs, runner.prefill)
        self.assertEqual(starting_lengths, [0, 0])
        self.assertEqual(output.shape, (2, 32))
        with torch.no_grad():
            expected = model.model(**inputs, use_cache=False, logits_to_keep=1)
        self.assertEqual(output, expected.logits[:, -1, :])

    @parametrize("model_type", ("Llama", "Qwen3", "Qwen3_5"))
    def test_decode_matches_full_forward(self, device, model_type):
        runner = self.huggingface.HuggingfaceRunner()
        runner.args = self._runner_args("--iterations=2", mode="decode")
        model = self._make_model(model_type, device)
        inputs = self._decode_inputs(device)
        with self._record_cache_lengths(runner, "decode_forward") as cache_lengths:
            output = runner.run_n_iterations(model, inputs, runner.decode)
        self.assertEqual(cache_lengths, [(4, 9)] * 2)
        full_ids = torch.cat([inputs["input_ids"], inputs["decode_ids"]], dim=1)
        with torch.no_grad():
            expected = model.model(full_ids, use_cache=False, logits_to_keep=1)
        self.assertEqual(output, expected.logits[:, -1, :])

    @parametrize("model_type", ("Llama", "Qwen3", "Qwen3_5"))
    def test_decode_compiles_once(self, device, model_type):
        runner = self.huggingface.HuggingfaceRunner()
        model = self._make_model(model_type, device)
        inputs = self._decode_inputs(device)
        torch._dynamo.reset()
        counter = CompileCounter()
        model.forward = torch.compile(model.forward, backend=counter, fullgraph=True)
        runner.decode(model, {**inputs, "decode_ids": inputs["decode_ids"][:, :1]})
        with torch._dynamo.config.patch(error_on_recompile=True):
            for _ in range(2):
                runner.decode(model, inputs)
        self.assertEqual(counter.frame_count, 1)

    @parametrize("mode", ("prefill", "decode"))
    def test_accuracy_compilation_boundary(self, device, mode):
        runner = self.huggingface.HuggingfaceRunner()
        runner.args = self._runner_args(mode=mode)
        runner.args.performance, runner.args.accuracy = False, True
        runner.hf_llm = True
        runner.model_iter_fn = getattr(runner, mode)
        model = self._make_model("Llama", device)
        inputs = {"input_ids": torch.randint(0, 32, (2, 4), device=device)}
        if mode == "decode":
            inputs["decode_ids"] = torch.randint(0, 32, (2, 3), device=device)
        compiled_functions = []

        def optimize(fn):
            compiled_functions.append(fn.__name__)
            return torch._dynamo.optimize("eager", nopython=True)(fn)

        with (
            mock.patch.object(common, "current_name", "tiny"),
            mock.patch.object(common, "current_device", device),
            mock.patch.object(common, "write_outputs"),
            mock.patch.object(common, "output_signpost", return_value=0),
        ):
            status = runner.check_accuracy("tiny", model, inputs, optimize, None, None)
        self.assertEqual(status, "pass")
        self.assertEqual(compiled_functions, ["forward"])

    @onlyCUDA
    @unittest.skipIf(not HAS_CUDA_AND_TRITON, "CUDA and Triton are required")
    @parametrize(
        "model_type, cudagraphs",
        [
            ("Llama", False),
            ("Llama", True),
            ("Gemma2", True),
            ("Qwen3", True),
            ("Qwen3_5", False),
            ("Qwen3_5", True),
        ],
    )
    def test_inductor_prefill_cache(self, device, model_type, cudagraphs):
        from torch._dynamo.utils import counters
        from torch._inductor.cudagraph_trees import ExecutionState, get_manager

        model = self._make_model(model_type, device)
        inputs = torch.randint(0, 32, (2, 4), device=device)
        skips_before = counters["inductor"]["cudagraph_skips"]
        with (
            torch.no_grad(),
            torch._inductor.config.patch(
                {"triton.cudagraphs": cudagraphs, "triton.cudagraph_trees": True}
            ),
        ):
            model.prepare_for_prefill(inputs)
            cache = self._cache_tensors(model)
            pointers = [
                (first.data_ptr(), second.data_ptr()) for first, second in cache
            ]
            expected = model(inputs)
            expected_cache = [
                (first.clone(), second.clone()) for first, second in cache
            ]
            compiled = torch.compile(model.forward, backend="inductor", fullgraph=True)
            for _ in range(3):
                torch.compiler.cudagraph_mark_step_begin()
                model.prepare_for_prefill(inputs)
                self.assertEqual(int(model.cache.get_seq_length()), 0)
                self.assertEqual(compiled(inputs), expected)
                self.assertEqual(int(model.cache.get_seq_length()), 4)
                cache = self._cache_tensors(model)
                self.assertEqual(cache, expected_cache)
                actual_pointers = [(k.data_ptr(), v.data_ptr()) for k, v in cache]
                self.assertEqual(actual_pointers, pointers)
            if cudagraphs:
                manager = get_manager(inputs.device.index, create_if_none_exists=False)
                self.assertIsNotNone(manager)
                self.assertEqual(manager.path_state, ExecutionState.EXECUTION)
        self.assertEqual(counters["inductor"]["cudagraph_skips"], skips_before)

    @onlyCUDA
    @unittest.skipIf(not HAS_CUDA_AND_TRITON, "CUDA and Triton are required")
    @parametrize("model_type", ("Llama", "Qwen3", "Qwen3_5"))
    def test_inductor_decode_cudagraphs(self, device, model_type):
        from torch._dynamo.utils import counters
        from torch._inductor.cudagraph_trees import get_manager

        runner = self.huggingface.HuggingfaceRunner()
        model = self._make_model(model_type, device)
        inputs = self._decode_inputs(device)
        expected = runner.decode(model, inputs)
        pointers = [(k.data_ptr(), v.data_ptr()) for k, v in self._cache_tensors(model)]
        # Also resets the cudagraph trees, so the manager only holds this test's graphs.
        torch._dynamo.reset()
        skips_before = counters["inductor"]["cudagraph_skips"]
        model.forward = torch.compile(model.forward, backend="inductor", fullgraph=True)
        with torch._inductor.config.patch(
            {"triton.cudagraphs": True, "triton.cudagraph_trees": True}
        ):
            for _ in range(3):
                self.assertEqual(runner.decode(model, inputs), expected)
                cache = self._cache_tensors(model)
                actual_pointers = [(k.data_ptr(), v.data_ptr()) for k, v in cache]
                self.assertEqual(actual_pointers, pointers)
        manager = get_manager(torch.device(device).index, create_if_none_exists=False)
        roots = [node for nodes in manager.roots.values() for node in nodes]
        self.assertEqual(len(roots), 1)
        self.assertEqual(roots[0].num_descendants(), 0)
        self.assertEqual(counters["inductor"]["cudagraph_skips"], skips_before)


instantiate_parametrized_tests(BenchmarkRunnerTests)
instantiate_parametrized_tests(TestHuggingFaceLLMPerformance)
instantiate_parametrized_tests(TestHuggingFacePrefill)
instantiate_device_type_tests(
    TestHuggingFacePrefillDevice, globals(), only_for=("cpu", "cuda")
)


if __name__ == "__main__":
    run_tests()
