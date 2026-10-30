# Owner(s): ["module: dynamo"]

import collections
import contextlib
import csv
import functools
import importlib.metadata
import itertools
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from packaging.version import Version

import torch
from torch._dynamo.testing import CompileCounter, CompileCounterWithBackend
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCUDA,
)
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    TEST_WITH_SLOW,
    TestCase,
)
from torch.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON


REPO_ROOT = Path(__file__).resolve().parents[2]
# Skips before setUp spawns workers. The inductor_huggingface_tp job runs these.
launches_processes = unittest.skipIf(
    not TEST_WITH_SLOW, "launches processes; set PYTORCH_TEST_WITH_SLOW=1"
)
# Bind the namespace package explicitly: some CI images install pysbd, whose wheel
# ships a regular top-level `benchmarks` package that would shadow the repo's.
sys.modules["benchmarks"] = types.ModuleType("benchmarks")
sys.modules["benchmarks"].__path__ = [str(REPO_ROOT / "benchmarks")]
from benchmarks.dynamo import common


requires_distributed = unittest.skipIf(
    not torch.distributed.is_available(), "requires distributed"
)


def _require_transformers():
    try:
        version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        raise unittest.SkipTest("requires transformers") from None
    try:
        with mock.patch(
            "subprocess.check_call",
            side_effect=unittest.SkipTest("requires compatible transformers"),
        ):
            from benchmarks.dynamo.huggingface_llm_models import MIN_TRANSFORMERS_VERSION
    except ImportError:
        raise unittest.SkipTest("requires compatible transformers") from None
    if Version(version) < Version(MIN_TRANSFORMERS_VERSION):
        raise unittest.SkipTest(f"requires transformers >= {MIN_TRANSFORMERS_VERSION}")


def _import_huggingface():
    from transformers import AutoConfig

    with (
        mock.patch.object(AutoConfig, "from_pretrained", return_value=None),
        mock.patch(
            "subprocess.check_call",
            side_effect=AssertionError("unexpected install"),
        ),
    ):
        from benchmarks.dynamo import huggingface

    return huggingface


def _tiny_causal_lm(model_type, **config_overrides):
    import transformers

    model_cls = getattr(transformers, model_type + "ForCausalLM")
    config_name = "Qwen3_5Text" if model_type == "Qwen3_5" else model_type
    config_cls = getattr(transformers, config_name + "Config")
    config_kwargs = {
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 3 if model_type == "Qwen3_5" else 2,
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
            layer_types=["linear_attention", "linear_attention", "full_attention"],
            linear_num_key_heads=1,
            linear_num_value_heads=2,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
        )
    config_kwargs.update(config_overrides)
    return model_cls(config_cls(**config_kwargs))


class BenchmarkRunnerTests(TestCase):
    def test_non_tp_json_preserves_metadata(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.object(common, "current_name", "model"),
            mock.patch.object(common, "current_device", "cuda"),
            mock.patch.object(common, "current_batch_size", 32),
            mock.patch.object(common, "current_dtype", "float32"),
            mock.patch.object(common, "current_quantization", ""),
            mock.patch.object(
                common, "current_settings", {"batch_size": None, "tp_size": 1}
            ),
        ):
            filename = os.path.join(tmp, "torchbench.csv")
            common.output_json(
                filename,
                ["dev", "name", "batch_size", "speedup"],
                ["cuda", "model", 32, 1.5],
            )
            with open(os.path.join(tmp, "torchbench.json")) as f:
                (record,) = [json.loads(line) for line in f]
        self.assertEqual(
            record["benchmark"]["extra_info"],
            {"device": "cuda", "quantization": "", "batch_size": None},
        )
        self.assertEqual(record["benchmark"]["dtype"], "float32")
        self.assertEqual(record["model"]["name"], "model")
        self.assertEqual(
            record["metric"], {"name": "speedup", "benchmark_values": [1.5]}
        )

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
        request_tokens = {
            "prefill": {"input_tokens_per_second": 6},
            "decode": {"output_tokens_per_second": 10},
        }.get(mode)
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
                args,
                mock.Mock(),
                model,
                inputs,
                hf_llm=True,
                setup_fn=setup,
                request_tokens=request_tokens,
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
            "--tp-size=2",
        ),
    )
    def test_prefill_options_require_huggingface(self, flag):
        args = common.parse_args(["--performance", "--inference", flag])
        with self.assertRaisesRegex(ValueError, "require the huggingface suite"):
            common.BenchmarkRunner().validate_args(args)

    @skipIfTorchDynamo("speedup_experiment calls set_stance")
    def test_tp_speedup_row_accounting(self):
        args = common.parse_args(
            [
                "--performance",
                "--inference",
                "--backend=eager",
                "--hf-inference-mode=prefill",
                "--tp-size=2",
                "--prompt-length=3",
                "--iterations-per-run=4",
                "--repeat=1",
            ]
        )
        args.rank = 0
        group = object()
        setup = mock.Mock()
        model = torch.nn.Identity()
        inputs = {"input_ids": torch.zeros(2, 3, dtype=torch.long)}
        with (
            mock.patch.object(common, "current_batch_size", 2),
            mock.patch.object(common, "output_filename", "results.csv"),
            mock.patch.object(
                common, "timed", side_effect=[(4.0, None), (2.0, None)]
            ) as timed,
            mock.patch.object(torch._dynamo, "run", side_effect=lambda fn: fn),
            mock.patch.object(common, "write_outputs") as write,
            mock.patch("torch._dynamo.utils.compile_times", return_value=([], [])),
            mock.patch.object(common, "output_signpost"),
        ):
            common.speedup_experiment(
                args,
                mock.Mock(),
                model,
                inputs,
                hf_llm=True,
                setup_fn=setup,
                request_tokens={"input_tokens_per_second": 6},
                tp_group=group,
                compilation_latency=1.0,
                compression_ratio=1.0,
                eager_peak_mem=2.0,
                dynamo_peak_mem=1.5,
                eager_peak_mem_per_rank=[1.0, 2.0],
                dynamo_peak_mem_per_rank=[1.5, 1.25],
            )
        _, headers, values = write.call_args_list[0].args
        row = dict(zip(headers, values))
        identity = headers[3:7]
        self.assertEqual(identity, list(common.TP_IDENTITY_HEADERS))
        self.assertEqual([row[header] for header in identity], ["prefill", 2, 3, 0])
        self.assertEqual(row["abs_latency"], 2000.0)
        self.assertEqual(row["eager_latency"], 4000.0)
        # Logical tokens only: 4 requests of 2 x 3 tokens, not multiplied by TP size.
        self.assertEqual(row["input_tokens_per_second"], 12.0)
        self.assertEqual(row["eager_peak_mem"], 2.0)
        self.assertEqual(row["eager_peak_mem_per_rank"], "1.000;2.000")
        self.assertEqual(row["dynamo_peak_mem_per_rank"], "1.500;1.250")
        for call in timed.call_args_list:
            self.assertIs(call.kwargs["group"], group)
            self.assertIs(call.kwargs["setup_fn"], setup)

    @requires_distributed
    @skipIfTorchDynamo("parse_args is slow to trace")
    @parametrize(
        "env, extra, message",
        [
            ({"WORLD_SIZE": "4"}, (), "must equal WORLD_SIZE=4"),
            ({"LOCAL_WORLD_SIZE": "1"}, (), "single node"),
            ({}, ("--only=",), "require --only"),
        ],
    )
    def test_tensor_parallel_entry_validation(self, env, extra, message):
        args = common.parse_args(
            ["--performance", "--inference", "--tp-size=2", "--only=model", *extra]
        )
        env = {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "2", **env}
        with (
            mock.patch.dict(os.environ, env),
            mock.patch.object(torch.distributed, "init_process_group") as init,
            self.assertRaisesRegex(ValueError, message),
        ):
            common.tensor_parallel_entry(None, None, args)
        init.assert_not_called()

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
        _require_transformers()
        self.huggingface = _import_huggingface()

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
        from benchmarks.dynamo.huggingface_llm_models import TextGenerationPrefillModel

        model = _tiny_causal_lm(model_type).eval().to(device=device, dtype=dtype)
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
        from transformers import AutoConfig

        model = self._make_model("Llama", "cpu").model
        tokenizer = types.SimpleNamespace(vocab_size=32, eos_token_id=2)
        with (
            mock.patch.object(
                llm.AutoTokenizer, "from_pretrained", return_value=tokenizer
            ),
            mock.patch.object(
                llm.AutoModelForCausalLM, "from_pretrained", return_value=model
            ),
            mock.patch.object(AutoConfig, "from_pretrained", return_value=model.config),
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

    def test_prefill_requires_transformers_version(self):
        runner = self.huggingface.HuggingfaceRunner()
        with (
            mock.patch("importlib.metadata.version", return_value="5.0.0"),
            self.assertRaisesRegex(ValueError, "transformers>=.*, found 5.0.0"),
        ):
            runner.validate_args(self._runner_args())

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

    @parametrize(
        "failing_run, mismatch, status",
        [
            (2, None, "eager_1st_run_fail"),
            (3, None, "eager_2nd_run_fail"),
            (None, 0, "eager_two_runs_differ"),
            (4, None, "fail_to_run"),
            (None, 1, "fail_accuracy"),
        ],
    )
    def test_non_tp_accuracy_failure_status(self, failing_run, mismatch, status):
        runner = self.huggingface.HuggingfaceRunner()
        runner.args = self._runner_args()
        runner.args.performance, runner.args.accuracy = False, True
        runner.hf_llm = True
        runner.model_iter_fn = runner.prefill
        model = self._make_model("Llama", "cpu")
        inputs = {"input_ids": torch.randint(0, 32, (2, 4))}
        # Runs: fp64 reference, two eager runs, compiled run. Comparisons: the two
        # eager runs, then compiled against eager.
        run_n_iterations, same = runner.run_n_iterations, common.same
        runs, comparisons = itertools.count(1), itertools.count()

        def run(*args, **kwargs):
            if next(runs) == failing_run:
                raise RuntimeError("injected failure")
            return run_n_iterations(*args, **kwargs)

        def compare(*args, **kwargs):
            return next(comparisons) != mismatch and same(*args, **kwargs)

        with (
            mock.patch.object(runner, "run_n_iterations", run),
            mock.patch.object(common, "same", compare),
            mock.patch.object(common, "current_name", "tiny"),
            mock.patch.object(common, "current_device", "cpu"),
            mock.patch.object(common, "write_outputs"),
            mock.patch.object(common, "output_signpost", return_value=0),
        ):
            optimize = functools.partial(torch.compile, backend="eager", fullgraph=True)
            result = runner.check_accuracy("tiny", model, inputs, optimize, None, None)
        self.assertEqual(result, status)

    def test_prompt_must_fit_generation_context(self):
        runner = self.huggingface.HuggingfaceRunner()
        args = self._runner_args("--only=Qwen/Qwen3-0.6B", "--prompt-length=2096")
        with self._mock_pretrained():
            runner.validate_args(args)
            args.prompt_length = 2097
            with self.assertRaisesRegex(ValueError, "exceeds .* context length 4096"):
                runner.validate_args(args)

    def test_prompt_validation_does_not_download_sweep_configs(self):
        runner = self.huggingface.HuggingfaceRunner()
        args = self._runner_args("--prompt-length=7")
        with mock.patch.object(
            self.huggingface.AutoConfig,
            "from_pretrained",
            side_effect=OSError("config unavailable"),
        ) as download:
            runner.validate_args(args)
        download.assert_not_called()

    def test_prompt_config_download_retries(self):
        runner = self.huggingface.HuggingfaceRunner()
        args = self._runner_args("--only=Qwen/Qwen3-0.6B", "--prompt-length=7")
        config = mock.Mock()
        config.get_text_config.return_value.max_position_embeddings = 4096
        with (
            mock.patch.object(
                self.huggingface.AutoConfig,
                "from_pretrained",
                side_effect=[OSError("temporary download failure"), config],
            ) as download,
            mock.patch.object(common.time, "sleep") as sleep,
        ):
            runner.validate_args(args)
        self.assertEqual(download.call_args_list, [mock.call(args.only)] * 2)
        sleep.assert_called_once_with(30)

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


class TestHuggingFaceTensorParallelOptions(HuggingFacePrefillTestCase):
    def setUp(self):
        super().setUp()
        # Validate as on a machine without an accelerator, where -dcpu TP is allowed.
        patcher = mock.patch.object(
            torch.accelerator, "current_accelerator", return_value=None
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _tp_args(self, *extra, mode="decode"):
        return self._runner_args(
            "--tp-size=2", "--float32", "--disable-cudagraphs", *extra, mode=mode
        )

    @parametrize(
        "extra, message",
        [
            ("--tp-size=0", "--tp-size must be positive"),
            ("--multiprocess", "does not support --multiprocess"),
            ("--device-index=0", "does not support --device-index"),
            ("--warm-start-latency", "does not support --warm-start-latency"),
            ("--minify", "does not support --minify"),
            ("--ddp", "does not support --ddp"),
        ],
    )
    def test_invalid_tp_options(self, extra, message):
        runner = self.huggingface.HuggingfaceRunner()
        with self.assertRaisesRegex(ValueError, message):
            runner.validate_args(self._tp_args(extra))

    def test_tp_requires_prefill_or_decode(self):
        runner = self.huggingface.HuggingfaceRunner()
        with self.assertRaisesRegex(ValueError, "requires --hf-inference-mode"):
            runner.validate_args(self._tp_args(mode="generate"))

    def test_tp_requires_dtype_and_disabled_cudagraphs(self):
        runner = self.huggingface.HuggingfaceRunner()
        args = self._runner_args("--tp-size=2", "--float32")
        with self.assertRaisesRegex(ValueError, "requires --disable-cudagraphs"):
            runner.validate_args(args)
        args = self._runner_args("--tp-size=2", "--disable-cudagraphs")
        with self.assertRaisesRegex(ValueError, "--bfloat16, --float16, or --float32"):
            runner.validate_args(args)
        args = self._tp_args("--backend=cudagraphs")
        with self.assertRaisesRegex(ValueError, "a backend other than cudagraphs"):
            runner.validate_args(args)

    @parametrize("mode", ("reduce-overhead", "max-autotune"))
    def test_tp_rejects_cudagraph_compile_modes(self, mode):
        runner = self.huggingface.HuggingfaceRunner()
        args = self._tp_args("--inductor")
        args.backend = None
        args.devices = ["cuda"]
        args.inductor_compile_mode = mode
        args.inductor_config = ["triton.cudagraphs=False"]
        with self.assertRaisesRegex(ValueError, "re-enable CUDA graphs"):
            runner.validate_args(args)
        args.inductor_compile_mode = "max-autotune-no-cudagraphs"
        runner.validate_args(args)

    @parametrize("use_inductor_flag", (False, True))
    @parametrize("enabled", ("True", "1"))
    def test_tp_rejects_cudagraph_config(self, use_inductor_flag, enabled):
        runner = self.huggingface.HuggingfaceRunner()
        args = self._tp_args()
        args.devices = ["cuda"]
        args.backend = None if use_inductor_flag else "inductor"
        args.inductor = use_inductor_flag
        override = f"triton.cudagraphs={enabled}"
        args.inductor_config = ["triton.cudagraphs=False", override]
        with self.assertRaisesRegex(ValueError, "re-enable CUDA graphs"):
            runner.validate_args(args)
        args.inductor_config.append("triton.cudagraphs=False")
        runner.validate_args(args)

    def test_cpu_tp_requires_no_accelerator(self):
        runner = self.huggingface.HuggingfaceRunner()
        cuda = torch.device("cuda")
        with (
            mock.patch.object(
                torch.accelerator, "current_accelerator", return_value=cuda
            ),
            self.assertRaisesRegex(ValueError, "requires a machine without an accel"),
        ):
            runner.validate_args(self._tp_args())

    @parametrize("use_inductor_flag", (False, True))
    def test_inductor_tp_requires_cuda(self, use_inductor_flag):
        runner = self.huggingface.HuggingfaceRunner()
        args = self._tp_args()
        args.backend = None if use_inductor_flag else "inductor"
        args.inductor = use_inductor_flag
        with self.assertRaisesRegex(ValueError, "Inductor TP requires CUDA/NCCL"):
            runner.validate_args(args)
        args.devices = []
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=False),
            self.assertRaisesRegex(ValueError, "Inductor TP requires CUDA/NCCL"),
        ):
            runner.validate_args(args)
        args.devices = ["cuda"]
        runner.validate_args(args)

    @parametrize("mode", ("prefill", "decode"))
    def test_tp_resolves_result_identity(self, mode):
        runner = self.huggingface.HuggingfaceRunner()
        args = self._tp_args(mode=mode)
        runner.validate_args(args)
        expected = [mode, 2, 1000, 128 if mode == "decode" else 0]
        self.assertEqual(common.tensor_parallel_identity(args)[1], expected)

    def test_tp_requires_divisible_heads(self):
        from benchmarks.dynamo import huggingface_llm_models as llm

        config = types.SimpleNamespace(num_attention_heads=8, num_key_value_heads=2)
        config.get_text_config = lambda decoder: config
        with (
            mock.patch.object(llm.AutoConfig, "from_pretrained", return_value=config),
            mock.patch.object(llm.AutoModelForCausalLM, "from_pretrained") as load,
            self.assertRaisesRegex(
                ValueError, "must divide .* 8 attention heads and 2 KV"
            ),
        ):
            llm.load_tensor_parallel_model("Qwen/Qwen3.5-0.8B", torch.bfloat16, 4)
        load.assert_not_called()


class TestHuggingFacePrefillDevice(HuggingFacePrefillTestCase):
    def _cache_tensors(self, model):
        return [
            (layer.conv_states[0], layer.recurrent_states[0])
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
                    self.assertFalse(layer.has_previous_state[0])
                    for states in (layer.conv_states[0], layer.recurrent_states[0]):
                        self.assertEqual(states, torch.zeros_like(states))
            return original_forward(*args, **kwargs)

        with (
            torch.no_grad(),
            mock.patch.object(model.model, "forward", side_effect=forward),
        ):
            output = runner.run_n_iterations(model, inputs, runner.prefill)
        self.assertEqual(starting_lengths, [0, 0])
        linear_states = [
            states.data_ptr()
            for layer, is_linear in zip(model.cache.layers, model.cache.is_linear)
            if is_linear
            for states in (layer.conv_states[0], layer.recurrent_states[0])
        ]
        self.assertEqual(len(set(linear_states)), len(linear_states))
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
            return torch.compile(fn, backend="eager", fullgraph=True)

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

        torch._dynamo.reset()
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


def _tp_device():
    """Device for two-rank TP tests: Hugging Face shards onto the accelerator, if any."""
    accelerator = torch.accelerator.current_accelerator()
    if accelerator is None:
        return "cpu"
    if accelerator.type == "cuda" and torch.cuda.device_count() >= 2:
        return "cuda"
    return None


@skipIfTorchDynamo("drives the benchmark harness, which calls set_stance")
@requires_distributed
@launches_processes
class TestTensorParallelHarness(MultiProcessTestCase):
    @property
    def world_size(self):
        return 2

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    def _init_gloo(self):
        store = torch.distributed.FileStore(self.file_name, self.world_size)
        torch.distributed.init_process_group(
            "gloo", rank=self.rank, world_size=self.world_size, store=store
        )

    def test_timed_takes_slowest_rank_per_request(self):
        self._init_gloo()
        durations = iter([[1.0, 5.0], [4.0, 2.0]][self.rank])
        clock = [0.0]

        def forward(model, inputs, collect_outputs=False):
            clock[0] += next(durations)

        barrier = torch.distributed.barrier
        with (
            mock.patch.object(common, "synchronize"),
            mock.patch.object(common, "reset_rng_state"),
            mock.patch.object(common.time, "perf_counter", new=lambda: clock[0]),
            mock.patch.object(
                torch.distributed, "barrier", side_effect=barrier
            ) as barriers,
        ):
            elapsed = common.timed(
                None,
                forward,
                (),
                times=2,
                setup_fn=lambda model, inputs: None,
                group=torch.distributed.group.WORLD,
            )
        # max(1, 4) + max(5, 2), rather than either rank's own total.
        self.assertEqual(elapsed, 9.0)
        self.assertEqual(barriers.call_count, 2)

    def test_agree_on_status_takes_first_failure_in_rank_order(self):
        self._init_gloo()
        runner = common.BenchmarkRunner()
        runner.args = common.parse_args(["--performance", "--inference", "--tp-size=2"])
        statuses = ["pass", "fail_accuracy"]
        self.assertEqual(runner.agree_on_status(statuses[self.rank]), "fail_accuracy")
        self.assertEqual(runner.agree_on_status("pass"), "pass")


# Divisible by a TP size of 2, with tied embeddings like the benchmarked checkpoints.
TP_CONFIG = {
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "tie_word_embeddings": True,
}
TP_QWEN3_5_CONFIG = {"linear_num_key_heads": 2, "linear_num_value_heads": 4}


@skipIfTorchDynamo("drives the benchmark harness, which calls set_stance")
@requires_distributed
@launches_processes
class TestHuggingFaceTensorParallel(MultiProcessTestCase):
    @property
    def world_size(self):
        return 2

    def setUp(self):
        super().setUp()
        _require_transformers()
        if _tp_device() != self.device_type:
            raise unittest.SkipTest("requires no accelerator or two CUDA devices")
        if "backend_inductor" in self._testMethodName and self.device_type != "cuda":
            raise unittest.SkipTest("Inductor TP requires CUDA/NCCL")
        self._spawn_processes()

    def _init_process_group(self, device):
        device = torch.device(device).type
        # Hugging Face picks each rank's shard device from LOCAL_RANK.
        os.environ["LOCAL_RANK"] = str(self.rank)
        if device == "cuda":
            torch.cuda.set_device(self.rank)
        store = torch.distributed.FileStore(self.file_name, self.world_size)
        torch.distributed.init_process_group(
            "nccl" if device == "cuda" else "gloo",
            rank=self.rank,
            world_size=self.world_size,
            store=store,
        )
        return device

    @contextlib.contextmanager
    def _checkpoint(self, model_type):
        path = f"{self.file_name}_{model_type}"
        if self.rank == 0:
            torch.manual_seed(0)
            overrides = dict(TP_CONFIG)
            if model_type == "Qwen3_5":
                overrides.update(TP_QWEN3_5_CONFIG)
            _tiny_causal_lm(model_type, **overrides).save_pretrained(path)
        torch.distributed.barrier()
        try:
            yield path
        finally:
            torch.distributed.barrier()
            if self.rank == 0:
                shutil.rmtree(path)

    def _load(self, path, device, mode):
        from benchmarks.dynamo import huggingface_llm_models as llm

        tokenizer = types.SimpleNamespace(vocab_size=32, eos_token_id=2)
        with mock.patch.object(
            llm.AutoTokenizer, "from_pretrained", return_value=tokenizer
        ):
            return llm.TextGenerationBenchmark.get_model_and_inputs(
                path,
                device,
                batch_size=2,
                prompt_length=4,
                inference_mode=mode,
                decode_length=3,
                dtype=torch.float32,
                tp_size=2,
            )

    @contextlib.contextmanager
    def _count_eager_collectives(self):
        counts = collections.Counter()

        def counted(name):
            collective = getattr(torch.distributed, name)

            def wrapper(*args, **kwargs):
                counts[name] += 1
                return collective(*args, **kwargs)

            return mock.patch.object(torch.distributed, name, wrapper)

        with counted("all_reduce"), counted("all_gather"):
            yield counts

    def _check_sharding(self, model, reference):
        layer_types = getattr(model.model.config, "layer_types", None) or [
            "full_attention"
        ]
        full_attention = layer_types.index("full_attention")
        sharded = [
            (f"layers.{full_attention}.self_attn.q_proj.weight", 0),
            (f"layers.{full_attention}.self_attn.k_proj.weight", 0),
            (f"layers.{full_attention}.self_attn.o_proj.weight", 1),
            ("layers.0.mlp.down_proj.weight", 1),
        ]
        if "linear_attention" in layer_types:
            linear = layer_types.index("linear_attention")
            sharded.append((f"layers.{linear}.linear_attn.in_proj_qkv.weight", 0))
        local = model.model.model.get_parameter
        full = reference.model.model.get_parameter
        for name, dim in sharded:
            expected = list(full(name).shape)
            expected[dim] //= 2
            self.assertEqual(list(local(name).shape), expected, msg=name)
        # Each rank holds a different shard, so this is not replicated execution.
        weight = local(sharded[0][0])
        shards = [torch.empty_like(weight) for _ in range(2)]
        torch.distributed.all_gather(shards, weight.contiguous())
        self.assertNotEqual(shards[0], shards[1])

    def _check_cache_shapes(self, model):
        config = model.model.config
        for layer, is_linear in zip(model.cache.layers, model.cache.is_linear):
            if is_linear:
                # The linear-attention projections gather their outputs, so the
                # recurrent state keeps every head.
                key_dim = config.linear_num_key_heads * config.linear_key_head_dim
                value_dim = config.linear_num_value_heads * config.linear_value_head_dim
                conv_shape = (2, 2 * key_dim + value_dim, config.linear_conv_kernel_dim)
                self.assertEqual(layer.conv_states[0].shape, conv_shape)
                self.assertEqual(layer.recurrent_states[0].shape, (2, 4, 8, 8))
            else:
                # One of two KV heads per rank, not divided by the TP size twice.
                self.assertEqual(layer.keys.shape, (2, 1, model.cache_capacity, 8))

    @parametrize("backend", ("aot_eager", "inductor"))
    @parametrize("model_type", ("Llama", "Qwen3", "Qwen3_5"))
    def test_tp_matches_unsharded_reference(self, device, model_type, backend):
        from benchmarks.dynamo.huggingface_llm_models import TextGenerationPrefillModel
        from transformers import AutoModelForCausalLM

        device = self._init_process_group(device)
        runner = _import_huggingface().HuggingfaceRunner()
        tolerance = {"atol": 1e-5, "rtol": 1e-4}
        with self._checkpoint(model_type) as path:
            # Rank-dependent seeds: the loader must broadcast rank 0's request.
            torch.manual_seed(self.rank)
            model, inputs = self._load(path, device, "decode")
            for tensor in inputs.values():
                gathered = [torch.empty_like(tensor) for _ in range(2)]
                torch.distributed.all_gather(gathered, tensor)
                self.assertEqual(gathered[0], gathered[1])
            reference_model = AutoModelForCausalLM.from_pretrained(
                path, dtype=torch.float32
            )
            reference = TextGenerationPrefillModel(
                reference_model.to(device).eval(), 2, 4, model.cache_capacity
            )
        self._check_sharding(model, reference)

        prompt = {"input_ids": inputs["input_ids"]}
        with self._count_eager_collectives() as counts:
            prefill = runner.prefill(model, prompt)
        self.assertGreater(counts["all_reduce"], 0)
        self.assertEqual(prefill, runner.prefill(reference, prompt), **tolerance)
        self.assertEqual(int(model.cache.get_seq_length()), 4)
        self._check_cache_shapes(model)
        self.assertEqual(runner.prefill(model, prompt), prefill)
        changed = {"input_ids": (prompt["input_ids"] + 1) % 32}
        changed_prefill = runner.prefill(model, changed)
        self.assertNotEqual(changed_prefill, prefill)
        self.assertEqual(
            changed_prefill, runner.prefill(reference, changed), **tolerance
        )

        decode = runner.decode(model, inputs)
        self.assertEqual(int(model.cache.get_seq_length()), 4 + 3)
        full_ids = torch.cat([inputs["input_ids"], inputs["decode_ids"]], dim=1)
        with torch.no_grad():
            expected = reference.model(full_ids, use_cache=False, logits_to_keep=1)
        self.assertEqual(decode, expected.logits[:, -1, :], **tolerance)
        self.assertEqual(runner.decode(model, inputs), decode)

        for mode, mode_inputs, eager_result in (
            ("prefill", prompt, prefill),
            ("decode", inputs, decode),
        ):
            torch._dynamo.reset()
            counter = CompileCounterWithBackend(backend)
            compiled = model.copy_sharing_weights()
            compiled.forward = torch.compile(
                compiled.forward, backend=counter, fullgraph=True
            )
            request = getattr(runner, mode)
            self.assertEqual(request(compiled, mode_inputs), eager_result, **tolerance)
            with torch._dynamo.config.patch(error_on_recompile=True):
                self.assertEqual(
                    request(compiled, mode_inputs), eager_result, **tolerance
                )
                changed_inputs = {
                    key: (value + 1) % 32 for key, value in mode_inputs.items()
                }
                self.assertEqual(
                    request(compiled, changed_inputs),
                    request(model, changed_inputs),
                    **tolerance,
                )
            self.assertEqual(counter.frame_count, 1)
            (graph,) = counter.graphs
            self.assertIn("_c10d_functional.all_reduce", graph.code)
            # The compiled copy has its own cache and leaves the eager model's alone.
            self.assertIsNot(compiled.cache, model.cache)

    def test_tp_accuracy_outcomes_agree(self, device):
        device = self._init_process_group(device)
        runner = _import_huggingface().HuggingfaceRunner()
        runner.args = common.parse_args(
            [
                f"-d{device}",
                "--accuracy",
                "--inference",
                "--float32",
                "--backend=eager",
                "--hf-inference-mode=prefill",
                "--tp-size=2",
                "--disable-cudagraphs",
            ]
        )
        runner.hf_llm = True
        runner.model_iter_fn = runner.prefill
        with self._checkpoint("Llama") as path:
            model, inputs = self._load(path, device, "prefill")
        weight = model.model.model.layers[0].self_attn.q_proj.weight
        original_weight = weight.detach().clone()

        def check_accuracy():
            with (
                mock.patch.object(common, "current_name", "tiny"),
                mock.patch.object(common, "current_device", device),
                mock.patch.object(common, "write_outputs"),
                mock.patch.object(common, "output_signpost", return_value=0),
            ):
                return runner.check_accuracy(
                    "tiny",
                    model,
                    inputs,
                    functools.partial(torch.compile, backend="eager", fullgraph=True),
                    None,
                    None,
                )

        self.assertEqual(check_accuracy(), "pass")
        # The fp64 reference cast the shared shards; check_accuracy casts them back.
        self.assertEqual(weight.dtype, torch.float32)
        self.assertEqual(weight, original_weight, atol=0, rtol=0)

        # Only rank 1 sees an accuracy mismatch, yet both ranks report it.
        same = common.same
        comparisons = iter(range(100))

        def mismatch_on_rank_1(*args, **kwargs):
            is_compiled_comparison = next(comparisons) == 1
            if self.rank == 1 and is_compiled_comparison:
                return False
            return same(*args, **kwargs)

        with mock.patch.object(common, "same", mismatch_on_rank_1):
            self.assertEqual(check_accuracy(), "fail_accuracy")


@skipIfTorchDynamo("checks benchmark result files")
class TestTensorParallelResults(TestCase):
    @parametrize("mode", ("prefill", "decode"))
    def test_tp_json_result_identity_and_metrics(self, mode):
        args = common.parse_args(
            [
                "--performance",
                "--inference",
                "--float32",
                "--tp-size=2",
                f"--hf-inference-mode={mode}",
            ]
        )
        metrics = {
            "speedup": 1.5,
            "abs_latency": 2.0,
            "eager_peak_mem_per_rank": "1.000;2.000",
            "dynamo_peak_mem_per_rank": "0.500;1.500",
        }
        decode_length = 128 if mode == "decode" else 0
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.dict(vars(common)),
        ):
            common.set_current_metadata(args)
            common.current_name = "Qwen/Qwen3-0.6B"
            common.current_device = "cuda"
            common.current_batch_size = 4
            filename = os.path.join(tmp, "results.csv")
            identity = ["cuda", common.current_name, 4, mode, 2, 1000, decode_length]
            common.output_json(
                filename,
                ["dev", "name", "batch_size", *common.TP_IDENTITY_HEADERS, *metrics],
                [*identity, *metrics.values()],
            )
            with open(os.path.join(tmp, "results.json")) as f:
                records = [json.loads(line) for line in f]
        names = [record["metric"]["name"] for record in records]
        self.assertEqual(names, list(metrics))
        for record in records:
            self.assertEqual(record["model"]["name"], "Qwen/Qwen3-0.6B")
            self.assertEqual(record["benchmark"]["dtype"], "float32")
            info = record["benchmark"]["extra_info"]
            self.assertEqual(info["device"], "cuda")
            self.assertEqual(info["batch_size"], 4)
            self.assertEqual(info["phase"], mode)
            self.assertEqual(info["tp_size"], 2)
            self.assertEqual(info["prompt_length"], 1000)
            self.assertEqual(info["decode_length"], decode_length)
            metric = record["metric"]
            value = metrics[metric["name"]]
            expected = {"benchmark_values": [value]}
            if isinstance(value, str):
                expected = {"extra_info": expected}
            self.assertEqual(metric, {"name": metric["name"], **expected})


instantiate_parametrized_tests(BenchmarkRunnerTests)
instantiate_parametrized_tests(TestHuggingFaceLLMPerformance)
instantiate_parametrized_tests(TestHuggingFacePrefill)
instantiate_parametrized_tests(TestHuggingFaceTensorParallelOptions)
instantiate_device_type_tests(
    TestHuggingFaceTensorParallel, globals(), only_for=("cpu", "cuda")
)
instantiate_parametrized_tests(TestTensorParallelResults)
instantiate_device_type_tests(
    TestHuggingFacePrefillDevice, globals(), only_for=("cpu", "cuda")
)


if __name__ == "__main__":
    run_tests()
