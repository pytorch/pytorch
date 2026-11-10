# Owner(s): ["module: dynamo"]

import collections
import contextlib
import csv
import functools
import importlib.metadata
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from packaging.version import Version

import torch
from torch.testing._internal.common_device_type import (
    dtypes,
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
    @parametrize("flag", ("--hf-inference-mode=prefill", "--prompt-length=8"))
    def test_prefill_options_require_huggingface(self, flag):
        args = common.parse_args(["--performance", "--inference", flag])
        with self.assertRaisesRegex(ValueError, "requires the huggingface suite"):
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
        test_case = self
        calls = collections.Counter()
        clock = [0.0]
        result = {}
        warmup_times = []
        timed = common.timed

        def measure_warmup(*args, **kwargs):
            elapsed = timed(*args, **kwargs)
            warmup_times.append(elapsed)
            return elapsed

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

            def get_workload(self, example_inputs, *, is_performance=False):
                test_case.assertTrue(is_performance)
                return common.BenchmarkWorkload(
                    self.generate,
                    setup_fn=self.setup if prefill else None,
                    compile_forward=True,
                )

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
            mock.patch.object(common, "timed", side_effect=measure_warmup),
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
        self.assertEqual(warmup_times, [10.0, 30.0] if prefill else [])
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
        try:
            with mock.patch(
                "subprocess.check_call",
                side_effect=unittest.SkipTest("requires compatible transformers"),
            ):
                from benchmarks.dynamo.huggingface_llm_models import (
                    PREFILL_TRANSFORMERS_VERSION,
                )
        except ImportError:
            raise unittest.SkipTest("requires compatible transformers") from None
        if Version(version) != Version(PREFILL_TRANSFORMERS_VERSION):
            raise unittest.SkipTest(
                f"requires transformers == {PREFILL_TRANSFORMERS_VERSION}"
            )
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
        # common.run() adds the CPU skips to the cached skip list.
        common.load_yaml_file.cache_clear()
        self.addCleanup(common.load_yaml_file.cache_clear)

    def _runner_args(self, *extra):
        return common.parse_args(
            [
                "-dcpu",
                "--backend=eager",
                "--performance",
                "--inference",
                "--hf-inference-mode=prefill",
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
        config = config_cls(**config_kwargs)
        model = model_cls(config).eval().to(device=device, dtype=dtype)
        return TextGenerationPrefillModel(model, 2, 4, 11).eval()

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
    @torch._dynamo.config.patch(repro_tolerance=1e-4)
    @parametrize("prefill_first", (False, True))
    def test_timing_units_setup_and_csv_columns(self, prefill_first):
        clock = [0.0]
        calls = collections.Counter()

        class Model(torch.nn.Module):
            name = "Qwen/Qwen3-0.6B"

            def forward(self, input_ids):
                clock[0] += 1.0 if calls["forward"] % 8 < 4 else 0.5
                calls["forward"] += 1

            def generate(self, input_ids):
                calls["generate"] += 1
                return self(input_ids)

            def prepare_for_prefill(self, input_ids):
                calls["setup"] += 1
                clock[0] += 100.0

        inputs = {"input_ids": torch.zeros(2, 3, dtype=torch.long)}
        modes = ("prefill", "generate") if prefill_first else ("generate", "prefill")
        with tempfile.TemporaryDirectory() as tmp:
            output = os.path.join(tmp, "results.csv")
            for mode in modes:
                calls.clear()
                runner = self.huggingface.HuggingfaceRunner()
                runner.args = self._runner_args(
                    f"--hf-inference-mode={mode}",
                    "--iterations-per-run=4",
                    "--repeat=1",
                    "--export-profiler-trace",
                )
                runner.args.profile_details = {}
                runner.args.profiler_trace_name = "profile"
                runner.hf_llm = True
                runner.model_iter_fn = runner.forward_pass
                workload = runner.get_workload(inputs, is_performance=True)
                self.assertTrue(workload.compile_forward)
                if mode == "generate":
                    self.assertEqual(workload.iter_fn, runner.generate)
                    self.assertIsNone(workload.setup_fn)
                    self.assertFalse(workload.report_eager_latency)
                    self.assertIsNone(workload.request_tokens)
                    accuracy = runner.get_workload(inputs)
                    self.assertEqual(accuracy.iter_fn, runner.forward_pass)
                    self.assertFalse(accuracy.compile_forward)
                profile = mock.Mock()
                with (
                    mock.patch.object(common, "current_name", mode),
                    mock.patch.object(common, "current_batch_size", 2),
                    mock.patch.object(common, "output_filename", output),
                    mock.patch.object(common, "synchronize"),
                    mock.patch.object(common, "reset_rng_state"),
                    mock.patch("time.perf_counter", new=lambda: clock[0]),
                    mock.patch.object(torch._dynamo, "run", side_effect=lambda fn: fn),
                    mock.patch.object(common, "write_outputs", common.output_csv),
                    mock.patch(
                        "torch._dynamo.utils.compile_times", return_value=([], [])
                    ),
                    mock.patch.object(common, "output_signpost"),
                    mock.patch(
                        "torch._inductor.utils.maybe_profile",
                        return_value=contextlib.nullcontext(profile),
                    ),
                ):
                    common.speedup_experiment(
                        runner.args,
                        workload.iter_fn,
                        Model(),
                        inputs,
                        compile_forward=workload.compile_forward,
                        setup_fn=workload.setup_fn,
                        request_tokens=workload.request_tokens,
                        report_eager_latency=workload.report_eager_latency,
                        compilation_latency=3.0,
                        compression_ratio=4.0,
                        eager_peak_mem=5.0,
                        dynamo_peak_mem=6.0,
                        dynamo_stats={"calls_captured": 7},
                    )
                self.assertEqual(calls["forward"], 16)
                self.assertEqual(calls["generate"], 16 if mode == "generate" else 0)
                self.assertEqual(calls["setup"], 16 if mode == "prefill" else 0)
                trace = Path(profile.export_chrome_trace.call_args.args[0])
                self.assertEqual(trace.name, "profile_Qwen_Qwen3-0.6B.json")
            with open(output) as f:
                rows = {row["name"]: row for row in csv.DictReader(f)}
        self.assertEqual(set(rows), {"generate", "prefill"})
        for mode, row in rows.items():
            self.assertEqual(float(row["speedup"]), 2.0)
            self.assertEqual(float(row["abs_latency"]), 2000.0)
            self.assertEqual(float(row["compilation_latency"]), 3.0)
            self.assertEqual(float(row["compression_ratio"]), 4.0)
            self.assertEqual(float(row["eager_peak_mem"]), 5.0)
            self.assertEqual(float(row["dynamo_peak_mem"]), 6.0)
            self.assertEqual(int(row["calls_captured"]), 7)
            self.assertEqual(
                float(row["eager_latency"]), 4000.0 if mode == "prefill" else 0.0
            )
            tokens_per_second = 12.0 if mode == "prefill" else 0.0
            self.assertEqual(float(row["input_tokens_per_second"]), tokens_per_second)

    @parametrize(
        "extra, message",
        [
            ("--amp", "does not support --amp"),
            ("--dynamic-shapes", "does not support --dynamic-shapes"),
            ("--trace-on-xla", "does not support --trace-on-xla"),
            ("--multiprocess", "does not support --multiprocess"),
            ("--batch-size=0", "--batch-size must be positive"),
            ("--prompt-length=0", "--prompt-length must be positive"),
            ("--iterations-per-run=0", "--iterations-per-run must be positive"),
        ],
    )
    def test_invalid_prefill_options(self, extra, message):
        runner = self.huggingface.HuggingfaceRunner()
        with self.assertRaisesRegex(ValueError, message):
            runner.validate_args(self._runner_args(extra))

    @parametrize("version", ("5.0.0", "5.14.2"))
    def test_prefill_requires_transformers_version(self, version):
        runner = self.huggingface.HuggingfaceRunner()
        with (
            mock.patch("importlib.metadata.version", return_value=version),
            self.assertRaisesRegex(ValueError, f"transformers==.*, found {version}"),
        ):
            runner.validate_args(self._runner_args())

    def test_generate_import_does_not_require_prefill_cache_classes(self):
        from benchmarks.dynamo import huggingface_llm_models as llm

        transformers = types.ModuleType("transformers")
        for name in (
            "AutoModelForCausalLM",
            "AutoTokenizer",
            "WhisperForConditionalGeneration",
            "WhisperProcessor",
        ):
            setattr(transformers, name, getattr(llm, name))
        spec = importlib.util.spec_from_file_location("llm_without_cache", llm.__file__)
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {"transformers": transformers}):
            spec.loader.exec_module(module)
        self.assertIn("Qwen/Qwen3-0.6B", module.HF_LLM_MODELS)

    def test_invalid_inference_mode_does_not_download(self):
        from benchmarks.dynamo import huggingface_llm_models as llm

        with (
            mock.patch.object(llm.AutoTokenizer, "from_pretrained") as tokenizer,
            mock.patch.object(llm.AutoModelForCausalLM, "from_pretrained") as model,
            self.assertRaisesRegex(ValueError, "unsupported inference mode: prefil"),
        ):
            llm.TextGenerationBenchmark.get_model_and_inputs(
                "Qwen/Qwen3-0.6B", "cpu", inference_mode="prefil"
            )
        tokenizer.assert_not_called()
        model.assert_not_called()

    @parametrize(
        "extra, message",
        [
            (
                ["--hf-inference-mode=prefill", "--multiprocess", "--backend=eager"],
                "does not support --multiprocess",
            ),
            (
                ["--prompt-length=7", "--multiprocess"],
                "--prompt-length requires --hf-inference-mode=prefill",
            ),
            (
                ["--hf-inference-mode=prefill", "--quantization=int8dynamic"],
                "does not support .*torchao",
            ),
            (
                ["--hf-inference-mode=prefill", "--optimus=all"],
                "does not support .*optimus",
            ),
        ],
    )
    def test_invalid_options_rejected_before_launch(self, extra, message):
        runner = self.huggingface.HuggingfaceRunner()
        with (
            mock.patch.object(common.mp, "spawn") as spawn,
            mock.patch.object(subprocess, "check_call") as child,
            self.assertRaisesRegex(ValueError, message),
        ):
            common.main(
                runner,
                args=[
                    "--performance",
                    "--inference",
                    "--only=Qwen/Qwen3-0.6B",
                    *extra,
                ],
            )
        spawn.assert_not_called()
        child.assert_not_called()

    def test_prefill_requires_inference_backend(self):
        runner = self.huggingface.HuggingfaceRunner()
        args = self._runner_args()
        args.inference = False
        with self.assertRaisesRegex(ValueError, "requires --inference"):
            runner.validate_args(args)
        args.inference = True
        args.backend = None
        with self.assertRaisesRegex(ValueError, "requires --backend or --inductor"):
            runner.validate_args(args)

    @parametrize("name", ("Qwen/Qwen3-0.6B", "Qwen/Qwen3.5-0.8B"))
    def test_prefill_honors_model_filter(self, name):
        runner = self.huggingface.HuggingfaceRunner()
        args = self._runner_args()
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

    def test_prompt_length_requires_prefill(self):
        runner = self.huggingface.HuggingfaceRunner()
        args = common.parse_args(["--performance", "--inference", "--prompt-length=8"])
        with self.assertRaisesRegex(ValueError, "requires --hf-inference-mode=prefill"):
            runner.validate_args(args)

    @parametrize("prompt_length", (None, 7))
    def test_requested_input_dimensions(self, prompt_length):
        runner = self.huggingface.HuggingfaceRunner()
        extra = () if prompt_length is None else (f"--prompt-length={prompt_length}",)
        runner.args = self._runner_args("--batch-size=3", *extra)
        with self._mock_pretrained(), mock.patch.object(runner, "validate_model"):
            _, _, _, inputs, batch_size = runner.load_model(
                "cpu", "Qwen/Qwen3-0.6B", batch_size=3
            )
        expected_length = prompt_length or 1000
        self.assertEqual(inputs["input_ids"].shape, (3, expected_length))
        self.assertEqual(batch_size, 3)

    def test_prompt_must_fit_generation_context(self):
        runner = self.huggingface.HuggingfaceRunner()
        name = "Qwen/Qwen3-0.6B"
        runner.args = self._runner_args(f"--only={name}", "--prompt-length=2096")
        benchmark = self.huggingface.HF_LLM_MODELS["Qwen/Qwen3-0.6B"]
        with (
            self._mock_pretrained(),
            mock.patch.object(runner, "validate_model"),
            mock.patch.object(
                benchmark, "get_model_and_inputs", wraps=benchmark.get_model_and_inputs
            ) as load,
        ):
            runner.load_model("cpu", name, batch_size=1)
            runner.args.prompt_length = 2097
            with self.assertRaisesRegex(ValueError, "exceeds .* context length 4096"):
                runner.load_model("cpu", name, batch_size=1)
            load.assert_called_once()

    def test_prompt_config_download_retries(self):
        runner = self.huggingface.HuggingfaceRunner()
        runner.args = self._runner_args("--only=Qwen/Qwen3-0.6B", "--prompt-length=7")
        config = mock.Mock()
        config.get_text_config.return_value.max_position_embeddings = 4096
        with (
            self._mock_pretrained(),
            mock.patch.object(runner, "validate_model"),
            mock.patch.object(
                self.huggingface.AutoConfig,
                "from_pretrained",
                side_effect=[OSError("temporary download failure"), config],
            ) as download,
            mock.patch.object(common.time, "sleep") as sleep,
        ):
            runner.load_model("cpu", runner.args.only, batch_size=1)
        download.assert_has_calls([mock.call(runner.args.only)] * 2)
        self.assertEqual(download.call_count, 2)
        sleep.assert_called_once_with(30)

    @parametrize("sweep", (False, True))
    def test_config_download_failure_records_result(self, sweep):
        name = "Qwen/Qwen3-0.6B"

        def child(cmd, **kwargs):
            with mock.patch.dict(vars(common)):
                try:
                    common.run(
                        self.huggingface.HuggingfaceRunner(), common.parse_args(cmd[2:])
                    )
                except OSError as error:
                    raise subprocess.CalledProcessError(1, cmd) from error

        with (
            tempfile.TemporaryDirectory() as tmp,
            torch._dynamo.config.patch(base_dir=tmp, repro_tolerance=1e-4),
            torch._inductor.config.patch(fallback_random=True),
            mock.patch.dict(os.environ),
            mock.patch.dict(vars(common)),
            mock.patch.object(common, "setup_determinism"),
            mock.patch.object(
                self.huggingface.HuggingfaceRunner,
                "_download_config",
                side_effect=OSError("config unavailable"),
            ),
            mock.patch.object(
                self.huggingface.HF_LLM_MODELS["Qwen/Qwen3-0.6B"],
                "get_model_and_inputs",
            ) as text_load,
            mock.patch.object(subprocess, "check_call", side_effect=child),
        ):
            output = os.path.join(tmp, "results.csv")
            argv = [
                "benchmarks/dynamo/huggingface.py",
                "--accuracy",
                "--inference",
                "--backend=eager",
                "-dcpu",
                "--no-skip",
                "--hf-inference-mode=prefill",
                "--prompt-length=7",
                f"--output={output}",
                f"--filter=^{name}$" if sweep else f"--only={name}",
            ]
            with mock.patch.object(sys, "argv", argv):
                args = common.parse_args(argv[1:])
                common.run(self.huggingface.HuggingfaceRunner(), args)
            with open(output) as f:
                (row,) = csv.DictReader(f)
            text_load.assert_not_called()
        self.assertEqual(row["name"], name)
        self.assertEqual(row["accuracy"], "eager_fail_to_run")

    @torch._dynamo.config.patch(repro_tolerance=1e-4)
    def test_prefill_performance_run(self):
        runner = self.huggingface.HuggingfaceRunner()
        original_forward = runner.prefill_forward
        cache_lengths = []

        def prefill_forward(model, example_inputs, collect_outputs=True):
            before = int(model.cache.get_seq_length())
            result = original_forward(model, example_inputs, collect_outputs)
            cache_lengths.append((before, int(model.cache.get_seq_length())))
            return result

        with (
            tempfile.TemporaryDirectory() as tmp,
            torch._dynamo.config.patch(base_dir=tmp),
            # run() rebinds module globals such as output_filename and current_name.
            mock.patch.dict(vars(common)),
            mock.patch.object(runner, "prefill_forward", prefill_forward),
            self._mock_pretrained(),
        ):
            args = self._runner_args(
                "--only=Qwen/Qwen3-0.6B",
                "--batch-size=2",
                "--prompt-length=7",
                "--repeat=1",
                "--iterations-per-run=2",
                "--export-profiler-trace",
            )
            common.run(runner, args)
            self.assertTrue(
                os.path.exists(os.path.join(tmp, "eager_prefill_Qwen_Qwen3-0.6B.json"))
            )
            with open(os.path.join(tmp, "speedup_eager_prefill.csv")) as f:
                (row,) = csv.DictReader(f)
        # validate_model, eager and compiled warmup, timed runs, and profiled runs.
        self.assertEqual(cache_lengths, [(0, 7)] * 15)
        self.assertEqual(row["name"], "Qwen/Qwen3-0.6B")
        self.assertEqual(int(row["batch_size"]), 2)
        self.assertGreater(float(row["eager_latency"]), 0)
        tokens_per_second = 2 * 2 * 7 / (float(row["abs_latency"]) / 1000)
        self.assertEqual(
            float(row["input_tokens_per_second"]), tokens_per_second, rtol=1e-3, atol=0
        )

    def test_prefill_operator_logging_prepares_cache(self):
        runner = self.huggingface.HuggingfaceRunner()
        original_forward = runner.prefill_forward

        def forward(model, example_inputs, collect_outputs=True):
            self.assertIsNotNone(model.cache)
            return original_forward(model, example_inputs, collect_outputs)

        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.dict(vars(common)),
            mock.patch.dict(os.environ),
            mock.patch.object(
                runner, "setup_prefill", wraps=runner.setup_prefill
            ) as setup,
            mock.patch.object(
                runner, "prefill_forward", side_effect=forward
            ) as run_forward,
            self._mock_pretrained(),
        ):
            args = self._runner_args(
                "--only=Qwen/Qwen3-0.6B",
                "--batch-size=2",
                "--prompt-length=7",
                "--log-operator-inputs",
                f"--output={tmp}/results.csv",
            )
            common.run(runner, args)
            self.assertGreaterEqual(run_forward.call_count, 2)
            self.assertEqual(setup.call_count, run_forward.call_count)
            log = Path(tmp) / "Qwen_Qwen3-0.6B_eval.txt"
            self.assertIn("aten.", log.read_text())


class TestHuggingFacePrefillDevice(HuggingFacePrefillTestCase):
    def _cache_tensors(self, model):
        return [
            (layer.conv_states[0], layer.recurrent_states[0])
            if is_linear
            else (layer.keys, layer.values)
            for layer, is_linear in zip(model.cache.layers, model.cache.is_linear)
        ]

    @parametrize("model_type", ("Llama", "Gemma2", "Qwen3", "Qwen3_5"))
    @parametrize("length", (4, 7))
    def test_prefill_cache_capacity_matches_generate(self, device, model_type, length):
        from benchmarks.dynamo import huggingface_llm_models as llm
        from transformers import StaticCache, StoppingCriteriaList

        native = self._make_model(model_type, device).model
        tokenizer = types.SimpleNamespace(vocab_size=32, eos_token_id=2)
        benchmark = llm.TextGenerationBenchmark
        with (
            mock.patch.object(
                llm.AutoTokenizer, "from_pretrained", return_value=tokenizer
            ),
            mock.patch.object(
                llm.AutoModelForCausalLM, "from_pretrained", return_value=native
            ),
        ):
            model, inputs = benchmark.get_model_and_inputs(
                "tiny",
                device,
                batch_size=2,
                prompt_length=length,
                inference_mode="prefill",
            )
        native.generation_config.disable_compile = True
        with torch.no_grad():
            # Stop after prefill while retaining generate's full cache capacity.
            generated = native.generate(
                **inputs,
                max_new_tokens=benchmark.OUTPUT_LENGTH,
                return_dict_in_generate=True,
                stopping_criteria=StoppingCriteriaList([lambda *args, **kwargs: True]),
            )
            model.prepare_for_prefill(inputs["input_ids"])
        self.assertEqual(generated.sequences.shape, (2, length + 1))
        native_cache = generated.past_key_values
        self.assertIsInstance(native_cache, StaticCache)
        self.assertEqual(model.cache.max_cache_len, native_cache.max_cache_len)

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    def test_qwen_recurrent_state_matches_native_cache(self, device, dtype):
        from transformers import StaticCache

        model = self._make_model("Qwen3_5", device, dtype)
        prompt = torch.randint(0, 32, (2, 4), device=device)
        tokens = torch.randint(0, 32, (2, 6), device=device)
        with torch.no_grad():
            for _ in range(2):
                model.prepare_for_prefill(prompt)
                native = StaticCache(
                    config=model.model.config, max_cache_len=model.cache_capacity
                )
                for layer, linear in zip(model.cache.layers, model.cache.is_linear):
                    if linear:
                        self.assertEqual(layer.conv_states[0].dtype, dtype)
                        self.assertEqual(layer.recurrent_states[0].dtype, torch.float32)
                        self.assertEqual(
                            layer.recurrent_states[0],
                            torch.zeros_like(layer.recurrent_states[0]),
                        )
                for ids in (prompt, *tokens.split(1, dim=1)):
                    expected = model.model(
                        ids, past_key_values=native, use_cache=True, logits_to_keep=1
                    ).logits[:, -1, :]
                    self.assertEqual(model(ids), expected, atol=0, rtol=0)
                    for actual, reference, linear in zip(
                        model.cache.layers, native.layers, model.cache.is_linear
                    ):
                        if linear:
                            self.assertEqual(
                                actual.recurrent_states[0],
                                reference.recurrent_states[0],
                                atol=0,
                                rtol=0,
                            )

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
            if model_type == "Qwen3_5":
                linear_layers = [
                    layer
                    for layer, linear in zip(cache.layers, cache.is_linear)
                    if linear
                ]
                self.assertTrue(linear_layers)
                for layer in linear_layers:
                    self.assertFalse(layer.has_previous_state[0])
                    for states in (layer.conv_states[0], layer.recurrent_states[0]):
                        self.assertEqual(states, torch.zeros_like(states))
            return original_forward(*args, **kwargs)

        with (
            torch.no_grad(),
            mock.patch.object(model.model, "forward", side_effect=forward),
        ):
            workload = runner.get_workload(inputs)
            output = runner.run_n_iterations(
                model, inputs, workload.iter_fn, setup_fn=workload.setup_fn
            )
        self.assertEqual(starting_lengths, [0, 0])
        if model_type == "Qwen3_5":
            linear_states = [
                states.data_ptr()
                for layer, is_linear in zip(model.cache.layers, model.cache.is_linear)
                if is_linear
                for states in (layer.conv_states[0], layer.recurrent_states[0])
            ]
            self.assertTrue(linear_states)
            self.assertEqual(len(set(linear_states)), len(linear_states))
        self.assertEqual(output.shape, (2, 32))
        with torch.no_grad():
            expected = model.model(**inputs, use_cache=False, logits_to_keep=1)
        self.assertEqual(output, expected.logits[:, -1, :])

    def test_accuracy_compilation_boundary(self, device):
        runner = self.huggingface.HuggingfaceRunner()
        runner.args = self._runner_args()
        runner.args.performance, runner.args.accuracy = False, True
        runner.hf_llm = True
        runner.model_iter_fn = runner.forward_pass
        model = self._make_model("Llama", device)
        inputs = {"input_ids": torch.randint(0, 32, (2, 4), device=device)}
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


instantiate_parametrized_tests(BenchmarkRunnerTests)
instantiate_parametrized_tests(TestHuggingFaceLLMPerformance)
instantiate_parametrized_tests(TestHuggingFacePrefill)
instantiate_device_type_tests(
    TestHuggingFacePrefillDevice, globals(), only_for=("cpu", "cuda")
)


if __name__ == "__main__":
    run_tests()
