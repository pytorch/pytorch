# Owner(s): ["module: dynamo"]

import csv
import dataclasses
import json
import os
import sys
import tempfile
import weakref
from fnmatch import fnmatchcase
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


REPO_ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO_ROOT)
try:
    from benchmarks.diffusion import compile_benchmark as benchmark
finally:
    sys.path.remove(REPO_ROOT)


class TestDiffusionCompileBenchmark(TestCase):
    def _toy_scenarios(self, modes):
        recipe = benchmark.BENCHMARKS["toy"]
        execution = benchmark.ExecutionConfig(
            modes=tuple(modes),
            backend="eager",
            cudagraphs=False,
            warmups=0,
            repetitions=1,
            timeout_s=60,
            device="cpu",
            dtype="float32",
            num_threads=1,
        )
        return [
            benchmark.Scenario(
                recipe.model, recipe.workload, execution, mode, recipe.loader
            )
            for mode in modes
        ]

    def _success_result(self, scenario):
        return {
            "scenario_id": scenario.scenario_id,
            "status": "success",
            "error": None,
            "model": dataclasses.asdict(scenario.model),
            "workload": dataclasses.asdict(scenario.workload),
            "execution": dataclasses.asdict(scenario.execution),
            "mode": scenario.mode,
            "dtype": "float32",
            "compiled_targets": [] if scenario.mode == "eager" else ["ToyDenoiser"],
            "model_setup_s": 0.1,
            "compile_wrapper_setup_s": 0.0,
            "first_request_s": 0.2,
            "steady_state_samples_s": [0.3] * scenario.execution.repetitions,
            "steady_state_median_s": 0.3,
            "steady_state_mad_s": 0.0,
            "setup_device_peak": {"allocated_bytes": None, "reserved_bytes": None},
            "request_device_peak": {"allocated_bytes": None, "reserved_bytes": None},
            "compiler_diagnostics": {"compiler_times_s": {}},
        }

    def test_invalid_mode_and_missing_region(self):
        model = torch.nn.Linear(2, 2)
        with self.assertRaisesRegex(ValueError, "invalid compilation mode"):
            with benchmark.configure_compilation(
                model, "invalid", ("Linear",), "eager", False
            ):
                pass
        with self.assertRaisesRegex(ValueError, "no repeated compilation targets"):
            with benchmark.configure_compilation(
                model, "regional", ("MissingBlock",), "eager", False
            ):
                pass
        with self.assertRaisesRegex(ValueError, "CUDA graphs require the inductor"):
            benchmark._compile_kwargs("eager", True)

        scenario = self._toy_scenarios(("eager",))[0]
        invalid_workload = dataclasses.replace(
            scenario.workload, output_boundary="unspecified"
        )
        with self.assertRaisesRegex(ValueError, "invalid output boundary"):
            benchmark._validate_worker_scenario(
                dataclasses.replace(scenario, workload=invalid_workload), "cpu"
            )

    def test_regional_prefers_model_entry_point(self):
        class Block(torch.nn.Module):
            def forward(self, value):
                return value + 1

        class Model(torch.nn.Module):
            _repeated_blocks = ("Block",)

            def __init__(self):
                super().__init__()
                self.block = Block()
                self.called = False

            def compile_repeated_blocks(self, **kwargs):
                self.called = True

        model = Model()
        with benchmark.configure_compilation(
            model, "regional", ("Block",), "eager", False
        ) as details:
            self.assertTrue(model.called)
            self.assertEqual(
                details["regional_entry_point"], "model.compile_repeated_blocks"
            )

    def test_hierarchical_marking_is_idempotent_and_restored(self):
        class Block(torch.nn.Module):
            def forward(self, value):
                return value + 1

        model = torch.nn.Sequential(Block(), Block())
        original = Block.forward
        with benchmark.configure_compilation(
            model, "hierarchical", ("Block",), "eager", False
        ):
            marked = Block.forward
            self.assertTrue(hasattr(marked, "__marked_compile_region_fn__"))
            with benchmark.configure_compilation(
                model, "hierarchical", ("Block",), "eager", False
            ):
                self.assertIs(Block.forward, marked)
            self.assertIs(Block.forward, marked)
        self.assertIs(Block.forward, original)

    def test_request_state_is_reconstructed(self):
        class Scheduler:
            config = {"value": 1}

            def __init__(self, value):
                self.value = value

            @classmethod
            def from_config(cls, config):
                return cls(**config)

        class Pipeline:
            scheduler = Scheduler(1)

        pipeline = Pipeline()
        make_request = benchmark._fresh_request(pipeline, {"input": 1}, 7, "cpu")
        _, first = make_request()
        first_value = torch.rand(2, generator=first["generator"])
        first_scheduler = pipeline.scheduler
        pipeline.scheduler.value = 9
        _, second = make_request()
        second_value = torch.rand(2, generator=second["generator"])
        self.assertIsNot(pipeline.scheduler, first_scheduler)
        self.assertEqual(pipeline.scheduler.value, 1)
        self.assertEqual(first_value, second_value)

    def test_explicit_cuda_device_initializes_before_memory_reset(self):
        scenario = self._toy_scenarios(("eager",))[0]
        scenario = dataclasses.replace(
            scenario,
            execution=dataclasses.replace(
                scenario.execution, device="cuda:0", num_threads=None
            ),
        )
        initialized = False

        def synchronize(device):
            nonlocal initialized
            self.assertEqual(device, "cuda:0")
            initialized = True

        def reset_peaks(device):
            self.assertTrue(initialized, "CUDA allocator must be initialized")
            self.assertEqual(device, "cuda:0")

        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "synchronize", side_effect=synchronize),
            mock.patch.object(
                torch.cuda, "reset_peak_memory_stats", side_effect=reset_peaks
            ) as reset_memory,
            mock.patch.object(
                benchmark, "_load_toy", side_effect=RuntimeError("reached model setup")
            ),
            self.assertRaisesRegex(RuntimeError, "reached model setup"),
        ):
            benchmark.execute_scenario(scenario)
        reset_memory.assert_called_once_with("cuda:0")

    @parametrize("modes", (("all",), ("full", "eager")))
    def test_cudagraphs_with_eager_reference(self, modes):
        def run_worker(scenario_id, command, result_path, log_path, timeout_s, env):
            config = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
            scenario = benchmark._scenario_from_dict(config["scenario"])
            benchmark._validate_worker_scenario(scenario, "cuda:0")
            torch.save(torch.ones(1), config["sample_path"])
            return self._success_result(scenario)

        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(benchmark, "run_worker_process", side_effect=run_worker),
        ):
            output = Path(directory) / "result.csv"
            args = ["--model", "toy", "--device", "cuda:0", "--cudagraphs"]
            args.extend(("--check-outputs", "--output", str(output)))
            for mode in modes:
                args.extend(("--mode", mode))
            self.assertEqual(benchmark.main(args), 0)
            with output.open(newline="") as file:
                rows = list(csv.DictReader(file))

        expected_modes = benchmark.MODES if modes == ("all",) else modes
        self.assertEqual([row["mode"] for row in rows], list(expected_modes))
        for row in rows:
            info = json.loads(row["provenance"])
            eager = row["mode"] == "eager"
            self.assertEqual(info["execution"]["cudagraphs"], not eager)
            check = "passed with torch.testing.assert_close defaults"
            self.assertEqual(info["output_check"], "reference" if eager else check)

    def test_cudagraphs_requires_compiled_mode(self):
        with (
            mock.patch.object(benchmark, "_run_scenarios") as run_scenarios,
            mock.patch("sys.stderr"),
            self.assertRaises(SystemExit) as error,
        ):
            benchmark.main(["--model", "toy", "--mode", "eager", "--cudagraphs"])
        self.assertEqual(error.exception.code, 2)
        run_scenarios.assert_not_called()

    @parametrize("cudagraphs", (True, False))
    def test_timed_request_cudagraph_boundary(self, cudagraphs):
        output = object()

        def make_request():
            mark.assert_not_called()
            return (), {}

        def pipeline():
            self.assertEqual(mark.call_count, int(cudagraphs))
            return output

        loaded = benchmark.LoadedPipeline(pipeline, None, make_request, ())
        with mock.patch.object(torch.compiler, "cudagraph_mark_step_begin") as mark:
            for _ in range(3):
                mark.reset_mock()
                _, result = benchmark._timed_request(
                    loaded, "cpu", cudagraphs=cudagraphs
                )
                self.assertIs(result, output)
                self.assertEqual(mark.call_count, int(cudagraphs))

    @parametrize("cli", (True, False))
    @parametrize("filename", ("results.json", "results.JSON", "results.JsOn"))
    def test_json_output_rejected(self, cli, filename):
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(
                benchmark, "_run_scenarios", return_value=([], [])
            ) as run_scenarios,
            mock.patch("sys.stderr"),
        ):
            output = Path(directory) / filename
            output.write_text("existing results", encoding="utf-8")
            if cli:
                with self.assertRaises(SystemExit) as error:
                    benchmark.main(
                        ["--model", "toy", "--mode", "eager", "--output", str(output)]
                    )
                self.assertEqual(error.exception.code, 2)
            else:
                with self.assertRaisesRegex(ValueError, "must not end in .json"):
                    benchmark.write_outputs(output, [])
            run_scenarios.assert_not_called()
            self.assertEqual(output.read_text(encoding="utf-8"), "existing results")

    def test_fresh_worker_timing_and_output_boundary(self):
        scenarios = self._toy_scenarios(("eager", "full"))
        run_worker = benchmark.run_worker_process
        cache_dirs = []
        probe = """
import runpy
import sys
from pathlib import Path
from torch._functorch import config as aot_config
from torch._inductor import config
from torch._inductor.runtime.cache_dir_utils import cache_dir, triton_cache_dir

if any(value is not False for value in (
    config.fx_graph_cache,
    config.fx_graph_remote_cache,
    config.autotune_remote_cache,
    config.bundled_autotune_remote_cache,
    aot_config.enable_autograd_cache,
    aot_config.enable_remote_autograd_cache,
)):
    raise RuntimeError("worker inherited compiler cache settings")
if not Path(triton_cache_dir(0)).is_relative_to(Path(cache_dir())):
    raise RuntimeError("worker inherited a shared Triton cache")
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""

        def run_isolated(scenario_id, command, result_path, log_path, timeout_s, env):
            cache_dirs.append(env["TORCHINDUCTOR_CACHE_DIR"])
            return run_worker(
                scenario_id,
                [sys.executable, "-c", probe, *command[1:]],
                result_path,
                log_path,
                timeout_s,
                env,
            )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.csv"
            inherited = {
                "TRITON_CACHE_DIR": str(Path(directory) / "shared"),
                "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
                "TORCHINDUCTOR_AUTOGRAD_CACHE": "1",
                "TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE": "1",
                "TORCHINDUCTOR_AUTOGRAD_REMOTE_CACHE": "1",
                "TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE": "1",
                "TORCHINDUCTOR_BUNDLED_AUTOTUNE_REMOTE_CACHE": "1",
            }
            with (
                mock.patch.dict(os.environ, inherited),
                mock.patch.object(
                    benchmark, "run_worker_process", side_effect=run_isolated
                ),
            ):
                results, errors = benchmark._run_scenarios(scenarios, output, True)
        self.assertEqual(errors, [])
        self.assertEqual(len({result["pid"] for result in results}), 2)
        self.assertEqual(len(set(cache_dirs)), 2)
        self.assertEqual(
            results[0]["output_summary"]["sha256"],
            results[1]["output_summary"]["sha256"],
        )
        for result in results:
            self.assertEqual(result["workload"]["output_boundary"], "latent")
            self.assertEqual(len(result["steady_state_samples_s"]), 1)
            self.assertGreaterEqual(result["model_setup_s"], 0)
            self.assertGreaterEqual(result["first_request_s"], 0)

    @parametrize("filename", ("results", "results.csv", "results.txt"))
    def test_result_schema_and_scenario_accounting(self, filename):
        scenarios = self._toy_scenarios(("eager",))
        scenario_id = scenarios[0].scenario_id
        result = self._success_result(scenarios[0])
        self.assertEqual(
            benchmark.validate_scenario_results([scenario_id], [result]), []
        )
        accounting_errors = benchmark.validate_scenario_results(
            [scenario_id, "missing"], [result, result, {"status": "success"}]
        )
        self.assertTrue(any("duplicate" in error for error in accounting_errors))
        self.assertTrue(any("missing result" in error for error in accounting_errors))
        self.assertTrue(
            any("without a string scenario_id" in error for error in accounting_errors)
        )
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / filename
            json_path = benchmark.write_outputs(csv_path, [result])
            self.assertNotEqual(csv_path, json_path)
            with csv_path.open(newline="") as file:
                rows = list(csv.DictReader(file))
            records = [
                json.loads(line)
                for line in json_path.read_text(encoding="utf-8").splitlines()
            ]
        self.assertEqual(rows[0]["scenario_id"], scenario_id)
        self.assertEqual(records[0]["benchmark"]["name"], benchmark.BENCHMARK_NAME)
        self.assertIn("benchmark_values", records[0]["metric"])
        failure_record = benchmark.dashboard_records(
            {"scenario_id": "failed", "status": "failed", "error": "boom"}
        )[0]
        self.assertNotIn("benchmark_values", failure_record["metric"])
        self.assertEqual(
            failure_record["metric"]["extra_info"]["benchmark_values"], ["failed"]
        )

    @parametrize("failure", ("failed", "timed_out", "killed", "missing"))
    def test_worker_failure_stays_visible_alongside_success(self, failure):
        if failure == "killed" and sys.platform == "win32":
            self.skipTest("SIGKILL is unavailable on Windows")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            success_path = root / "success.json"
            success_value = self._success_result(self._toy_scenarios(("eager",))[0])
            success_value["scenario_id"] = "success"
            success_code = (
                "from pathlib import Path; "
                f"Path({str(success_path)!r}).write_text({json.dumps(success_value)!r})"
            )
            success = benchmark.run_worker_process(
                "success",
                [sys.executable, "-c", success_code],
                success_path,
                root / "success.log",
                10,
            )

            failure_path = root / "failure.json"
            if failure == "failed":
                failure_value = {
                    "scenario_id": failure,
                    "status": "failed",
                    "error": "intentional failure",
                }
                code = (
                    "from pathlib import Path; "
                    f"Path({str(failure_path)!r}).write_text({json.dumps(failure_value)!r}); "
                    "raise SystemExit(1)"
                )
                timeout = 10
            elif failure == "timed_out":
                code = "import time; time.sleep(10)"
                timeout = 0.05
            elif failure == "missing":
                code = "pass"
                timeout = 10
            else:
                code = "import os, signal; os.kill(os.getpid(), signal.SIGKILL)"
                timeout = 10
            failed = benchmark.run_worker_process(
                failure,
                [sys.executable, "-c", code],
                failure_path,
                root / f"{failure}.log",
                timeout,
            )
        self.assertEqual(success["status"], "success")
        self.assertEqual(failed["status"], failure)
        errors = benchmark.validate_scenario_results(
            ["success", failure], [success, failed]
        )
        self.assertTrue(any(failure in error for error in errors))
        self.assertEqual(len(errors), 1)

    @parametrize(
        "field",
        (
            "status",
            "compile_wrapper_setup_s",
            "execution",
            "setup_device_peak",
            "compiler_diagnostics",
            "steady_state_samples_s",
            "compiler_times_s",
            "allocated_bytes",
            "nonfinite_timing",
            "failed_metadata",
        ),
    )
    def test_malformed_child_preserves_other_results(self, field):
        scenarios = self._toy_scenarios(("eager", "full"))
        payloads = {
            scenario.scenario_id: self._success_result(scenario)
            for scenario in scenarios
        }
        malformed = payloads[scenarios[1].scenario_id]
        if field in ("status", "compile_wrapper_setup_s"):
            del malformed[field]
        elif field == "compiler_times_s":
            malformed["compiler_diagnostics"]["compiler_times_s"] = []
        elif field == "allocated_bytes":
            del malformed["request_device_peak"]["allocated_bytes"]
        elif field == "nonfinite_timing":
            malformed["first_request_s"] = float("nan")
        elif field == "failed_metadata":
            malformed.update(status="failed", model=None)
        else:
            malformed[field] = None
        run_worker = benchmark.run_worker_process

        def run_child(scenario_id, command, result_path, log_path, timeout_s, env):
            code = (
                "import sys; from pathlib import Path; "
                "Path(sys.argv[1]).write_text(sys.argv[2])"
            )
            command = [
                sys.executable,
                "-c",
                code,
                str(result_path),
                json.dumps(payloads[scenario_id]),
            ]
            return run_worker(
                scenario_id, command, result_path, log_path, timeout_s, env
            )

        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(benchmark, "run_worker_process", side_effect=run_child),
        ):
            output = Path(directory) / "results.csv"
            exit_code = benchmark.main(
                [
                    "--model",
                    "toy",
                    "--mode",
                    "eager",
                    "--mode",
                    "full",
                    "--backend",
                    "eager",
                    "--repetitions",
                    "1",
                    "--output",
                    str(output),
                ]
            )
            with output.open(newline="") as file:
                rows = list(csv.DictReader(file))
            records = [
                json.loads(line)
                for line in output.with_suffix(".json").read_text().splitlines()
            ]
        self.assertEqual(exit_code, 1)
        self.assertEqual([row["status"] for row in rows], ["success", "malformed"])
        failures = [
            record
            for record in records
            if record["metric"]["name"] == "scenario_status"
        ]
        self.assertEqual(len(failures), 1)
        self.assertEqual(
            failures[0]["metric"]["extra_info"]["benchmark_values"], ["malformed"]
        )

    def test_interruption_reaps_worker(self):
        wait = benchmark.subprocess.Popen.wait
        processes = []

        def interrupt_wait(process, timeout=None):
            if timeout is not None:
                processes.append(process)
                raise KeyboardInterrupt("cancelled worker")
            return wait(process, timeout)

        try:
            with (
                tempfile.TemporaryDirectory() as directory,
                mock.patch.object(benchmark.subprocess.Popen, "wait", interrupt_wait),
            ):
                root = Path(directory)
                with self.assertRaisesRegex(KeyboardInterrupt, "cancelled worker"):
                    benchmark.run_worker_process(
                        "interrupted",
                        [sys.executable, "-c", "import time; time.sleep(60)"],
                        root / "result.json",
                        root / "worker.log",
                        30,
                    )
                self.assertEqual(len(processes), 1)
                self.assertIsNotNone(processes[0].returncode)
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                    wait(process)

    def test_measured_outputs_are_released_between_requests(self):
        scenario = self._toy_scenarios(("eager",))[0]
        scenario = dataclasses.replace(
            scenario,
            execution=dataclasses.replace(
                scenario.execution, warmups=1, repetitions=3, num_threads=None
            ),
        )
        outputs = []

        def pipeline():
            if outputs:
                self.assertIsNone(outputs[-1]())
            value = torch.ones(1)
            outputs.append(weakref.ref(value))
            return {"images": value}

        loaded = benchmark.LoadedPipeline(
            pipeline, torch.nn.Identity(), lambda: ((), {}), ("images",)
        )
        with (
            mock.patch.object(benchmark, "_load_toy", return_value=loaded),
            torch.random.fork_rng(devices=[]),
        ):
            benchmark.execute_scenario(scenario)
        self.assertEqual(len(outputs), 5)

    @parametrize("captured", (True, False))
    def test_cudagraph_count_excludes_empty_nodes(self, captured):
        from torch._dynamo.utils import counters

        leaf = SimpleNamespace(graph=object() if captured else None, children={})
        child = SimpleNamespace(
            graph=object() if captured else None, children={1: [leaf]}
        )
        root = SimpleNamespace(graph=None, children={0: [child]})
        manager = SimpleNamespace(get_roots=lambda: iter([root]))
        module = SimpleNamespace(get_manager=lambda *args, **kwargs: manager)
        with (
            mock.patch.dict(counters, {"stats": {"unique_graphs": 1}}, clear=True),
            mock.patch.dict(sys.modules, {"torch._inductor.cudagraph_trees": module}),
        ):
            if captured:
                diagnostics = benchmark._compiler_diagnostics("full", "cuda:0", True)
                self.assertEqual(diagnostics["cudagraph_capture_count"], 2)
            else:
                with self.assertRaisesRegex(RuntimeError, "no graph capture"):
                    benchmark._compiler_diagnostics("full", "cuda:0", True)

    def test_output_check_rejects_matching_nans(self):
        scenarios = self._toy_scenarios(("eager", "full"))
        results = [self._success_result(scenario) for scenario in scenarios]
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for index, scenario in enumerate(scenarios):
                path = Path(directory) / f"{index}.pt"
                torch.save(torch.tensor([float("nan")]), path)
                paths[scenario.scenario_id] = path
            benchmark._compare_outputs(scenarios, results, paths)
        self.assertEqual(results[1]["status"], "failed")
        self.assertEqual(results[1]["output_check"], "failed")

    @parametrize("name", ("auraflow", "flux"))
    def test_prefetch_excludes_unused_weights(self, name):
        artifacts = benchmark.BENCHMARKS[name].model.artifacts
        snapshot = mock.Mock(return_value="snapshot")
        hub = SimpleNamespace(
            snapshot_download=snapshot, hf_hub_download=mock.Mock(return_value="gguf")
        )
        with mock.patch.dict(sys.modules, {"huggingface_hub": hub}):
            benchmark._prefetch(artifacts)
        patterns = snapshot.call_args.kwargs["ignore_patterns"]
        needed = (
            "model_index.json",
            "transformer/config.json",
            "text_encoder/model.safetensors",
        )
        unused = (
            (
                "aura_flow_0.3.safetensors",
                "transformer/model.safetensors",
                "vae/model.fp16.safetensors",
            )
            if name == "auraflow"
            else ("flux1-dev.safetensors", "ae.safetensors")
        )
        for filename in needed:
            self.assertFalse(
                any(fnmatchcase(filename, pattern) for pattern in patterns)
            )
        for filename in unused:
            self.assertTrue(any(fnmatchcase(filename, pattern) for pattern in patterns))

    def test_auraflow_uses_prefetched_config(self):
        recipe = benchmark.BENCHMARKS["auraflow"]
        scenario = dataclasses.replace(
            self._toy_scenarios(("eager",))[0],
            model=recipe.model,
            workload=recipe.workload,
            loader=recipe.loader,
        )
        serialized = json.loads(json.dumps(benchmark._scenario_to_dict(scenario)))
        self.assertEqual(benchmark._scenario_from_dict(serialized), scenario)
        base, gguf = recipe.model.artifacts
        paths = {base.key: "pinned-base", gguf.key: "pinned.gguf"}
        diffusers = SimpleNamespace(
            AuraFlowPipeline=mock.Mock(),
            AuraFlowTransformer2DModel=mock.Mock(),
            GGUFQuantizationConfig=mock.Mock(),
        )
        with (
            mock.patch.dict(sys.modules, {"diffusers": diffusers}),
            mock.patch.object(benchmark, "_finish_loading"),
        ):
            benchmark._load_auraflow(scenario, paths, "cpu", torch.float32)
        kwargs = diffusers.AuraFlowTransformer2DModel.from_single_file.call_args.kwargs
        self.assertEqual(kwargs["config"], paths[base.key])
        self.assertEqual(kwargs["subfolder"], "transformer")
        self.assertTrue(kwargs["local_files_only"])


class TestDiffusionCompileBenchmarkDeviceType(TestCase):
    def test_cudagraph_request_retains_predictions(self, device):
        from torch.utils._triton import has_triton

        if not has_triton():
            self.skipTest("CUDA graph compilation requires Triton")
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)

        def denoise(value):
            return value + 1

        compiled = torch.compile(
            denoise, fullgraph=True, options={"triton.cudagraphs": True}
        )
        value = torch.ones(8, device=device)

        def pipeline():
            first = compiled(value)
            second = compiled(value)
            return first + second

        loaded = benchmark.LoadedPipeline(pipeline, compiled, lambda: ((), {}), ())
        with torch.inference_mode():
            for _ in range(3):
                _, output = benchmark._timed_request(loaded, device, cudagraphs=True)
                self.assertEqual(output, torch.full_like(value, 4))
                del output
        benchmark._compiler_diagnostics("full", device, True)


instantiate_parametrized_tests(TestDiffusionCompileBenchmark)
instantiate_device_type_tests(
    TestDiffusionCompileBenchmarkDeviceType, globals(), only_for="cuda"
)


if __name__ == "__main__":
    run_tests()
