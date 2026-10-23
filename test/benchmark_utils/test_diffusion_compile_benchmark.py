# Owner(s): ["module: dynamo"]

import csv
import dataclasses
import importlib.util
import json
import os
import sys
import tempfile
import unittest
import weakref
from fnmatch import fnmatchcase
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    subtest,
    TestCase,
)


REPO_ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO_ROOT)
try:
    from benchmarks.diffusion import compile_benchmark as benchmark
finally:
    sys.path.remove(REPO_ROOT)

HAS_DIFFUSION_DEPS = all(
    importlib.util.find_spec(name) is not None for name in ("diffusers", "transformers")
)
skipIfNoDiffusionDeps = unittest.skipIf(
    not HAS_DIFFUSION_DEPS, "requires diffusers and transformers"
)
# The runner compiles and counts graphs itself; wrapping tests in Dynamo makes
# its steady-state checks see the test's own compilation.
skipIfWrappedByDynamo = skipIfTorchDynamo("the benchmark runner manages compilation")


def _denoise_scenario(
    mode="eager", *, device="cpu", backend="eager", cudagraphs=False, **parameters
):
    recipe = benchmark.RECIPES["denoise_step"]["tiny_sd3"]
    workload = dataclasses.replace(
        recipe.workload, parameters={**recipe.workload.parameters, **parameters}
    )
    execution = benchmark.ExecutionConfig(
        modes=(mode,),
        backend=backend,
        cudagraphs=cudagraphs,
        warmups=1,
        repetitions=2,
        timeout_s=60,
        device=device,
        dtype="float32",
        num_threads=None,
    )
    return benchmark.Scenario(recipe.model, workload, execution, mode, recipe.loader)


@skipIfWrappedByDynamo
class TestDiffusionCompileBenchmark(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def _toy_scenarios(self, modes):
        recipe = benchmark.RECIPES["pipeline"]["toy"]
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
            "workload_details": {},
            "execution": dataclasses.asdict(scenario.execution),
            "mode": scenario.mode,
            "dtype": scenario.model.default_dtype
            if scenario.execution.dtype == "auto"
            else scenario.execution.dtype,
            "device": scenario.model.default_device
            if scenario.execution.device == "auto"
            else scenario.execution.device,
            "compiled_targets": [] if scenario.mode == "eager" else ["ToyDenoiser"],
            "model_setup_s": 0.1,
            "compile_wrapper_setup_s": 0.0,
            "first_request_s": 0.2,
            "steady_state_samples_s": [0.3] * scenario.execution.repetitions,
            "steady_state_median_s": 0.3,
            "steady_state_mad_s": 0.0,
            "setup_device_peak": {"allocated_bytes": None, "reserved_bytes": None},
            "request_device_peak": {"allocated_bytes": None, "reserved_bytes": None},
            "compiler_diagnostics": {
                "compiler_times_s": {},
                "unique_graphs": int(scenario.mode != "eager"),
            },
            "output_summary": {
                "shape": [1],
                "dtype": "torch.float32",
                "sha256": "0" * 64,
            },
            "output_check": "not requested",
        }

    @parametrize("timeout", ("nan", "inf", "-inf", "0", "-1"))
    def test_timeout_must_be_finite_and_positive(self, timeout):
        args = benchmark._parser().parse_args([f"--timeout={timeout}"])
        with self.assertRaisesRegex(ValueError, "timeout must be finite and positive"):
            benchmark._execution_from_args(args, ("eager",))
        scenario = self._toy_scenarios(("eager",))[0]
        scenario = dataclasses.replace(
            scenario,
            execution=dataclasses.replace(scenario.execution, timeout_s=float(timeout)),
        )
        with self.assertRaisesRegex(ValueError, "timeout must be finite and positive"):
            benchmark._validate_worker_scenario(scenario, "cpu")

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
        make_request = benchmark._fresh_request(
            pipeline, {}, 7, "cpu", mutable_parameters={"input": lambda: [1]}
        )
        _, first = make_request()
        first["input"].append(2)
        first_value = torch.rand(2, generator=first["generator"])
        first_scheduler = pipeline.scheduler
        pipeline.scheduler.value = 9
        _, second = make_request()
        second_value = torch.rand(2, generator=second["generator"])
        self.assertIsNot(pipeline.scheduler, first_scheduler)
        self.assertEqual(pipeline.scheduler.value, 1)
        self.assertEqual(second["input"], [1])
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
        scenarios = self._toy_scenarios(benchmark.MODES)
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
        self.assertEqual(len({result["pid"] for result in results}), len(scenarios))
        self.assertEqual(len(set(cache_dirs)), len(scenarios))
        for result in results:
            self.assertEqual(result["output_summary"], results[0]["output_summary"])
            graphs = result["compiler_diagnostics"]["unique_graphs"]
            self.assertEqual(graphs > 0, result["mode"] != "eager")
            self.assertEqual(result["workload"]["output_boundary"], "latent")
            self.assertEqual(len(result["steady_state_samples_s"]), 1)
            self.assertGreaterEqual(result["model_setup_s"], 0)
            self.assertGreaterEqual(result["first_request_s"], 0)

    @parametrize("backend", ("eager", "inductor"))
    def test_dashboard_distinguishes_compilation_modes(self, backend):
        scenarios = self._toy_scenarios(benchmark.MODES)
        scenarios += [
            dataclasses.replace(
                scenario,
                execution=dataclasses.replace(scenario.execution, cudagraphs=True),
            )
            for scenario in scenarios
            if scenario.mode != "eager"
        ]
        backends = []
        for scenario in scenarios:
            scenario = dataclasses.replace(
                scenario,
                execution=dataclasses.replace(scenario.execution, backend=backend),
            )
            records = benchmark.dashboard_records(self._success_result(scenario))
            backends.append(records[0]["model"]["backend"])
        self.assertEqual(len(set(backends)), len(scenarios))

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
            "device",
            "dtype",
            "output_summary",
            "unique_graphs",
            "requested_mode",
            "requested_model",
            "requested_workload",
            "requested_repetitions",
            "resolved_device",
            "resolved_dtype",
        ),
    )
    def test_malformed_child_preserves_other_results(self, field):
        args = ["--model", "toy", "--backend", "eager"]
        args += ["--mode", "eager", "--mode", "full"]
        execution = benchmark._execution_from_args(
            benchmark._parser().parse_args(args), ("eager", "full")
        )
        scenarios = [
            dataclasses.replace(scenario, execution=execution)
            for scenario in self._toy_scenarios(("eager", "full"))
        ]
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
        elif field == "unique_graphs":
            del malformed["compiler_diagnostics"]["unique_graphs"]
        elif field == "requested_mode":
            malformed.update(mode="eager", compiled_targets=[])
            malformed["compiler_diagnostics"]["unique_graphs"] = 0
        elif field == "requested_model":
            malformed["model"]["revision"] = "wrong-revision"
        elif field == "requested_workload":
            malformed["workload"]["parameters"]["num_inference_steps"] = 1
        elif field == "requested_repetitions":
            malformed["execution"]["repetitions"] = 1
            malformed["steady_state_samples_s"] = [0.3]
        elif field == "resolved_device":
            malformed["device"] = "cuda:0"
        elif field == "resolved_dtype":
            malformed["dtype"] = "float16"
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
            exit_code = benchmark.main([*args, "--output", str(output)])
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

    def test_reference_failure_keeps_eager_timings(self):
        scenarios = [_denoise_scenario(mode) for mode in ("eager", "full")]

        def high_precision_reference():
            raise torch.OutOfMemoryError("out of memory")

        loaded = benchmark.LoadedPipeline(
            lambda: torch.ones(2),
            torch.nn.Identity(),
            lambda: ((), {}),
            (),
            high_precision_reference=high_precision_reference,
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = {
                scenario.scenario_id: Path(directory) / f"{scenario.mode}.pt"
                for scenario in scenarios
            }
            with (
                mock.patch.object(
                    benchmark, "_load_tiny_sd3_denoise_step", return_value=loaded
                ),
                mock.patch("traceback.print_exc"),
                torch.random.fork_rng(devices=[]),
            ):
                eager = benchmark.execute_scenario(
                    scenarios[0], str(paths[scenarios[0].scenario_id])
                )
            self.assertFalse(
                benchmark._reference_path(paths[scenarios[0].scenario_id]).exists()
            )
            torch.save(torch.ones(2), paths[scenarios[1].scenario_id])
            results = [eager, self._success_result(scenarios[1])]
            benchmark._compare_outputs(scenarios, results, paths)
        self.assertEqual(eager["status"], "success")
        self.assertEqual(eager["output_check"], "reference failed")
        self.assertIn("OutOfMemoryError", eager["reference_error"])
        records = benchmark.dashboard_records(eager)
        metrics = [record["metric"]["name"] for record in records]
        self.assertIn("steady_state_median_s", metrics)
        compiled = results[1]
        self.assertEqual(compiled["status"], "failed")
        self.assertEqual(compiled["output_check"], "reference unavailable")
        self.assertIn("high-precision reference failed", compiled["error"])

    def test_high_precision_reference_bounds_compiled_error(self):
        scenarios = [_denoise_scenario(mode) for mode in ("eager", "full", "regional")]
        generator = torch.Generator().manual_seed(0)
        high = torch.randn(4096, generator=generator, dtype=torch.float64)

        def perturbed(scale):
            noise = torch.randn(4096, generator=generator, dtype=torch.float64)
            return (high + scale * noise).float()

        eager = perturbed(1e-2)
        loaded = benchmark.LoadedPipeline(
            lambda: eager,
            torch.nn.Identity(),
            lambda: ((), {}),
            (),
            high_precision_reference=lambda: high,
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = {
                scenario.scenario_id: Path(directory) / f"{scenario.mode}.pt"
                for scenario in scenarios
            }
            eager_path = str(paths[scenarios[0].scenario_id])
            with mock.patch.object(
                benchmark, "_load_tiny_sd3_denoise_step", return_value=loaded
            ):
                results = [benchmark.execute_scenario(scenarios[0], eager_path)]
            torch.save(perturbed(1e-2), paths[scenarios[1].scenario_id])
            torch.save(perturbed(1e-1), paths[scenarios[2].scenario_id])
            results += [self._success_result(scenario) for scenario in scenarios[1:]]
            benchmark._compare_outputs(scenarios, results, paths)
        eager_result, close, far = results
        self.assertEqual(eager_result["output_check"], "reference")
        self.assertIsNone(eager_result["reference_error"])
        self.assertEqual(close["status"], "success", close.get("error"))
        self.assertIn("relative RMS error vs torch.float64", close["output_check"])
        self.assertEqual(far["status"], "failed")
        self.assertIn("bound 2 x eager", far["error"])

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

    @parametrize("failure", ("worker_failed", "missing_sample"))
    def test_output_check_requires_eager_reference(self, failure):
        scenarios = self._toy_scenarios(("eager", "full"))
        results = [self._success_result(scenario) for scenario in scenarios]
        if failure == "worker_failed":
            results[0].update(status="failed", error="eager worker failed")
        with tempfile.TemporaryDirectory() as directory:
            paths = {
                scenario.scenario_id: Path(directory) / f"{scenario.mode}.pt"
                for scenario in scenarios
            }
            benchmark._compare_outputs(scenarios, results, paths)
            output = Path(directory) / "results.csv"
            benchmark.write_outputs(output, results)
            with output.open(newline="") as file:
                rows = list(csv.DictReader(file))
        compiled = results[1]
        self.assertEqual(compiled["status"], "failed")
        self.assertEqual(compiled["output_check"], "reference unavailable")
        self.assertEqual(json.loads(rows[1]["steady_state_samples_s"]), [0.3])
        records = benchmark.dashboard_records(compiled)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["metric"]["name"], "scenario_status")
        self.assertNotIn("benchmark_values", records[0]["metric"])

    @parametrize("name", ("auraflow", "flux"))
    def test_prefetch_excludes_unused_weights(self, name):
        artifacts = benchmark.RECIPES["pipeline"][name].model.artifacts
        snapshot = mock.Mock(return_value="snapshot")
        hub = SimpleNamespace(
            snapshot_download=snapshot, hf_hub_download=mock.Mock(return_value="gguf")
        )
        errors = SimpleNamespace(
            GatedRepoError=type("GatedRepoError", (Exception,), {})
        )
        with mock.patch.dict(
            sys.modules, {"huggingface_hub": hub, "huggingface_hub.errors": errors}
        ):
            benchmark._prefetch(artifacts)
        self.assertEqual(snapshot.call_args.kwargs["repo_id"], artifacts[0].repo_id)
        self.assertEqual(snapshot.call_args.kwargs["revision"], artifacts[0].revision)
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
        recipe = benchmark.RECIPES["pipeline"]["auraflow"]
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
        self.assertEqual(kwargs["torch_dtype"], torch.float32)
        diffusers.GGUFQuantizationConfig.assert_called_once_with(
            compute_dtype=torch.float32
        )
        diffusers.AuraFlowPipeline.from_pretrained.assert_called_once_with(
            paths[base.key],
            torch_dtype=torch.float32,
            transformer=diffusers.AuraFlowTransformer2DModel.from_single_file.return_value,
        )

    def test_wan_request_contract(self):
        from PIL import Image

        recipe = benchmark.RECIPES["pipeline"]["wan"]
        scenario = dataclasses.replace(
            self._toy_scenarios(("eager",))[0],
            model=recipe.model,
            workload=recipe.workload,
            loader=recipe.loader,
        )
        model, image = recipe.model.artifacts
        paths = {model.key: "pinned-model", image.key: "pinned-image.png"}
        diffusers = SimpleNamespace(
            AutoencoderKLWan=mock.Mock(), WanImageToVideoPipeline=mock.Mock()
        )
        encoder = mock.Mock()
        transformers = SimpleNamespace(CLIPVisionModel=encoder)
        pipeline = diffusers.WanImageToVideoPipeline.from_pretrained.return_value
        pipeline.vae_scale_factor_spatial = 8
        pipeline.transformer.config.patch_size = (1, 2, 2)
        source = Image.new("RGB", (640, 480), "red")
        with (
            mock.patch.dict(
                sys.modules, {"diffusers": diffusers, "transformers": transformers}
            ),
            mock.patch.object(Image, "open", return_value=source) as open_image,
            mock.patch.object(benchmark, "_finish_loading") as finish,
        ):
            benchmark._load_wan(scenario, paths, "cpu", torch.bfloat16)
        open_image.assert_called_once_with(paths[image.key])
        encoder.from_pretrained.assert_called_once_with(
            paths[model.key], subfolder="image_encoder", torch_dtype=torch.float32
        )
        diffusers.AutoencoderKLWan.from_pretrained.assert_called_once_with(
            paths[model.key], subfolder="vae", torch_dtype=torch.float32
        )
        diffusers.WanImageToVideoPipeline.from_pretrained.assert_called_once_with(
            paths[model.key],
            vae=diffusers.AutoencoderKLWan.from_pretrained.return_value,
            image_encoder=encoder.from_pretrained.return_value,
            torch_dtype=torch.bfloat16,
        )
        parameters = dict(recipe.workload.parameters, height=544, width=720)
        self.assertEqual(finish.call_args.args, (pipeline, scenario, "cpu", parameters))
        self.assertEqual(finish.call_args.kwargs["output_fields"], ("frames",))
        image_factory = finish.call_args.kwargs["mutable_parameters"]["image"]
        first = image_factory()
        first.putpixel((0, 0), (0, 0, 0))
        second = image_factory()
        self.assertEqual(second.size, (720, 544))
        self.assertEqual(second.getpixel((0, 0)), (255, 0, 0))

    @parametrize(
        "model,workload,loader",
        (
            ("toy", None, "_load_toy"),
            ("sd35_medium", None, "_load_sd35_medium"),
            ("sd35_medium", "denoise_step", "_load_sd35_medium_denoise_step"),
            ("tiny_sd3", "denoise_step", "_load_tiny_sd3_denoise_step"),
            ("toy", "denoise_step", None),
            ("tiny_sd3", "pipeline", None),
        ),
    )
    def test_workload_dispatch(self, model, workload, loader):
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(
                benchmark, "_run_scenarios", return_value=([], [])
            ) as run_scenarios,
            mock.patch("sys.stderr"),
        ):
            args = ["--model", model, "--mode", "eager", "--mode", "regional"]
            args += ["--output", str(Path(directory) / "results.csv")]
            if workload is not None:
                args += ["--workload", workload]
            if loader is None:
                with self.assertRaises(SystemExit) as error:
                    benchmark.main(args)
                self.assertEqual(error.exception.code, 2)
                run_scenarios.assert_not_called()
                return
            self.assertEqual(benchmark.main(args), 0)
        scenarios = run_scenarios.call_args.args[0]
        self.assertEqual(
            [scenario.mode for scenario in scenarios], ["eager", "regional"]
        )
        for scenario in scenarios:
            self.assertEqual(scenario.workload.name, workload or "pipeline")
            self.assertEqual(scenario.loader, loader)
            self.assertIn(f":{scenario.workload.name}:", scenario.scenario_id)

    @parametrize(
        "flags,expected,label",
        (
            subtest(
                (
                    ("--workload", "denoise_step", "--batch-size", "3"),
                    {"num_images_per_prompt": 3},
                    "denoise_step[num_images_per_prompt=3]",
                ),
                name="denoise_batch",
            ),
            subtest(
                (
                    ("--guidance-scale", "1", "--height", "512", "--width", "768"),
                    {"guidance_scale": 1.0, "height": 512, "width": 768},
                    "pipeline[guidance_scale=1.0,height=512,width=768]",
                ),
                name="pipeline_shape_and_cfg",
            ),
            subtest(
                (("--height", "1024", "--guidance-scale", "4.5"), {}, "pipeline"),
                name="recipe_defaults",
            ),
            subtest((("--batch-size", "0"), None, None), name="zero_batch"),
            subtest((("--guidance-scale", "nan"), None, None), name="nan_guidance"),
            subtest((("--guidance-scale", "-1"), None, None), name="negative_guidance"),
        ),
    )
    def test_workload_overrides(self, flags, expected, label):
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(
                benchmark, "_run_scenarios", return_value=([], [])
            ) as run_scenarios,
            mock.patch("sys.stderr"),
        ):
            args = ["--model", "sd35_medium", "--mode", "eager", *flags]
            args += ["--output", str(Path(directory) / "results.csv")]
            if expected is None:
                with self.assertRaises(SystemExit):
                    benchmark.main(args)
                run_scenarios.assert_not_called()
                return
            benchmark.main(args)
        scenario = run_scenarios.call_args.args[0][0]
        workload = scenario.workload
        recipe = benchmark.RECIPES[workload.name]["sd35_medium"]
        self.assertEqual(
            workload.parameters, {**recipe.workload.parameters, **expected}
        )
        self.assertEqual(workload.overrides, expected)
        self.assertTrue(scenario.scenario_id.startswith(f"sd35_medium:{label}:"))
        record = benchmark.dashboard_records(self._success_result(scenario))[0]
        self.assertEqual(record["model"]["name"], f"sd35_medium:{label}")

    def test_overrides_rejected_for_fixed_recipes(self):
        with (
            mock.patch.object(benchmark, "_run_scenarios") as run_scenarios,
            mock.patch("sys.stderr"),
            self.assertRaises(SystemExit) as error,
        ):
            benchmark.main(["--model", "toy", "--mode", "eager", "--batch-size", "2"])
        self.assertEqual(error.exception.code, 2)
        run_scenarios.assert_not_called()

    def test_workload_is_part_of_result_identity(self):
        results = []
        for workload in benchmark.WORKLOADS:
            recipe = benchmark.RECIPES[workload]["sd35_medium"]
            scenario = dataclasses.replace(
                self._toy_scenarios(("full",))[0],
                model=recipe.model,
                workload=recipe.workload,
                loader=recipe.loader,
            )
            result = self._success_result(scenario)
            result["workload_details"] = {
                "height": 1024,
                "width": 1024,
                "requested_batch_size": 1,
                "effective_batch_size": 2,
                "conditioning_length": 333,
                "guidance_scale": 4.5,
            }
            results.append(result)
        pipeline, denoise_step = results
        self.assertNotEqual(pipeline["scenario_id"], denoise_step["scenario_id"])
        records = [benchmark.dashboard_records(result)[0] for result in results]
        self.assertEqual(records[0]["model"]["backend"], records[1]["model"]["backend"])
        self.assertEqual(
            [record["model"]["name"] for record in records],
            ["sd35_medium:pipeline", "sd35_medium:denoise_step"],
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.csv"
            benchmark.write_outputs(output, results)
            with output.open(newline="") as file:
                rows = list(csv.DictReader(file))
        self.assertEqual([row["workload"] for row in rows], list(benchmark.WORKLOADS))
        self.assertEqual(
            [row["output_boundary"] for row in rows], ["latent", "denoiser_prediction"]
        )
        for row in rows:
            self.assertEqual(row["checkpoint_revision"], pipeline["model"]["revision"])
            self.assertEqual(row["cudagraphs"], "False")
            self.assertEqual(
                (row["requested_batch_size"], row["effective_batch_size"]), ("1", "2")
            )
            self.assertEqual(row["conditioning_length"], "333")

    def test_csv_falls_back_to_workload_parameters(self):
        recipe = benchmark.RECIPES["pipeline"]["flux"]
        scenario = dataclasses.replace(
            self._toy_scenarios(("full",))[0],
            model=recipe.model,
            workload=recipe.workload,
            loader=recipe.loader,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.csv"
            benchmark.write_outputs(output, [self._success_result(scenario)])
            with output.open(newline="") as file:
                (row,) = csv.DictReader(file)
        dimensions = (row["height"], row["width"], row["guidance_scale"])
        self.assertEqual(dimensions, ("1024", "1024", "3.5"))
        self.assertEqual(row["effective_batch_size"], "")

    def test_sd35_artifacts_and_gated_access(self):
        def selected(artifact, filename):
            ignored = any(fnmatchcase(filename, p) for p in artifact.ignore_patterns)
            allowed = not artifact.allow_patterns or any(
                fnmatchcase(filename, p) for p in artifact.allow_patterns
            )
            return allowed and not ignored

        (pipeline,) = benchmark.RECIPES["pipeline"]["sd35_medium"].model.artifacts
        (denoise,) = benchmark.RECIPES["denoise_step"]["sd35_medium"].model.artifacts
        self.assertEqual(
            (pipeline.repo_id, pipeline.revision), (denoise.repo_id, denoise.revision)
        )
        files = {
            "transformer/diffusion_pytorch_model.safetensors": (True, True),
            "transformer/config.json": (True, True),
            "scheduler/scheduler_config.json": (True, True),
            "tokenizer/vocab.json": (True, True),
            "vae/config.json": (True, True),
            "vae/diffusion_pytorch_model.safetensors": (True, False),
            "text_encoder_3/model-00001-of-00002.safetensors": (True, False),
            "text_encoder_3/model.fp16-00001-of-00002.safetensors": (False, False),
            "text_encoder/model.fp16.safetensors": (False, False),
            "text_encoders/t5xxl_fp16.safetensors": (False, False),
            "sd3.5_medium.safetensors": (False, False),
        }
        for filename, expected in files.items():
            actual = (selected(pipeline, filename), selected(denoise, filename))
            self.assertEqual(actual, expected, filename)

        gated = type("GatedRepoError", (Exception,), {})
        snapshot = mock.Mock(side_effect=gated("401"))
        hub = SimpleNamespace(snapshot_download=snapshot, hf_hub_download=mock.Mock())
        errors = SimpleNamespace(GatedRepoError=gated)
        with (
            mock.patch.dict(
                sys.modules, {"huggingface_hub": hub, "huggingface_hub.errors": errors}
            ),
            self.assertRaisesRegex(RuntimeError, "is gated: accept its license"),
        ):
            benchmark._prefetch((denoise,))
        self.assertEqual(
            snapshot.call_args.kwargs["allow_patterns"], list(denoise.allow_patterns)
        )

    def test_sd35_pipeline_contract(self):
        recipe = benchmark.RECIPES["pipeline"]["sd35_medium"]
        scenario = dataclasses.replace(
            self._toy_scenarios(("eager",))[0],
            model=recipe.model,
            workload=recipe.workload,
            loader=recipe.loader,
        )
        self.assertEqual(
            recipe.model.repeated_block_classes, ("JointTransformerBlock",)
        )
        self.assertEqual(recipe.workload.output_boundary, "latent")
        self.assertEqual(recipe.workload.parameters["output_type"], "latent")
        diffusers = SimpleNamespace(StableDiffusion3Pipeline=mock.Mock())
        pipeline = diffusers.StableDiffusion3Pipeline.from_pretrained.return_value
        pipeline.tokenizer_max_length = 77
        (artifact,) = recipe.model.artifacts
        with (
            mock.patch.dict(sys.modules, {"diffusers": diffusers}),
            mock.patch.object(benchmark, "_finish_loading") as finish,
        ):
            benchmark._load_sd35_medium(
                scenario, {artifact.key: "snapshot"}, "cuda", torch.bfloat16
            )
        diffusers.StableDiffusion3Pipeline.from_pretrained.assert_called_once_with(
            "snapshot", torch_dtype=torch.bfloat16
        )
        kwargs = finish.call_args.kwargs
        self.assertEqual(kwargs["output_fields"], ("images",))
        details = kwargs["workload_details"]
        self.assertEqual(details["requested_batch_size"], 1)
        self.assertEqual(details["effective_batch_size"], 2)
        self.assertEqual(details["conditioning_length"], 77 + 256)

    @parametrize("dtype", (torch.bfloat16, torch.float16, torch.float32))
    @parametrize("numel", (512, 4096))
    def test_prediction_check_is_anchored_to_high_precision(self, dtype, numel):
        generator = torch.Generator().manual_seed(0)
        high = torch.randn(numel, generator=generator, dtype=torch.float64)

        def perturbed(scale):
            noise = torch.randn(numel, generator=generator, dtype=torch.float64)
            return (high + scale * noise).to(dtype)

        eager = perturbed(1e-2)
        self.assertIn(
            "passed", benchmark._check_prediction(perturbed(1e-2), eager, high)
        )
        # As in torch._dynamo.utils.same, 2.5x eager's error passes only with the
        # 3x multiplier for half precision or small tensors.
        loose = perturbed(2.5e-2)
        if dtype == torch.float32 and numel >= 1000:
            with self.assertRaisesRegex(AssertionError, "bound 2 x eager"):
                benchmark._check_prediction(loose, eager, high)
        else:
            check = benchmark._check_prediction(loose, eager, high)
            self.assertIn("bound 3 x eager", check)
        for actual in (
            perturbed(5e-2),
            perturbed(1e-2).reshape(2, -1),
            perturbed(1e-2).to(torch.float64),
            torch.full_like(eager, float("nan")),
        ):
            with self.assertRaises(AssertionError):
                benchmark._check_prediction(actual, eager, high)

    def test_prediction_check_requires_high_precision_reference(self):
        scenarios = [_denoise_scenario(mode) for mode in ("eager", "full")]
        results = [self._success_result(scenario) for scenario in scenarios]
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for scenario in scenarios:
                paths[scenario.scenario_id] = Path(directory) / f"{scenario.mode}.pt"
                torch.save(torch.ones(2), paths[scenario.scenario_id])
            benchmark._compare_outputs(scenarios, results, paths)
        self.assertEqual(results[0]["status"], "failed")
        self.assertEqual(results[1]["output_check"], "reference unavailable")

    @skipIfNoDiffusionDeps
    @parametrize("guidance_scale", (1.0, 4.5))
    @parametrize("batch_size", (1, 2))
    def test_denoise_inputs_match_pipeline_contract(self, guidance_scale, batch_size):
        scenario = _denoise_scenario(
            guidance_scale=guidance_scale, num_images_per_prompt=batch_size
        )
        loads = [
            benchmark._load_tiny_sd3_denoise_step(scenario, {}, "cpu", torch.float32)
            for _ in range(2)
        ]
        requests = [loaded.make_request()[1] for loaded in loads]
        self.assertEqual(requests[0], requests[1])
        inputs = requests[0]
        effective = batch_size * (2 if guidance_scale > 1 else 1)
        details = loads[0].workload_details
        self.assertEqual(details["requested_batch_size"], batch_size)
        self.assertEqual(details["effective_batch_size"], effective)
        self.assertEqual(details["classifier_free_guidance"], guidance_scale > 1)
        self.assertEqual(details["conditioning_length"], 8)
        self.assertEqual(inputs["hidden_states"].shape, (effective, 4, 8, 8))
        self.assertEqual(inputs["encoder_hidden_states"].shape, (effective, 8, 16))
        self.assertEqual(inputs["pooled_projections"].shape, (effective, 8))
        self.assertEqual(inputs["timestep"].shape, (effective,))
        self.assertEqual(inputs["timestep"].stride(), (0,))
        self.assertEqual(inputs["timestep"].dtype, torch.float32)
        if guidance_scale > 1:
            latents = inputs["hidden_states"].chunk(2)
            self.assertEqual(latents[0], latents[1])
        with torch.inference_mode():
            prediction = loads[0].pipeline(**inputs)
        self.assertEqual(prediction.shape, inputs["hidden_states"].shape)

    @skipIfNoDiffusionDeps
    def test_denoise_step_times_one_forward_per_changing_timestep(self):
        from diffusers import FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel

        scenario = _denoise_scenario("full")
        loaded = benchmark._load_tiny_sd3_denoise_step(
            scenario, {}, "cpu", torch.float32
        )
        schedule = loaded.workload_details["timesteps"]
        forward = SD3Transformer2DModel.forward
        with (
            mock.patch.object(
                SD3Transformer2DModel, "forward", autospec=True, side_effect=forward
            ) as transformer_forward,
            mock.patch.object(
                FlowMatchEulerDiscreteScheduler,
                "step",
                side_effect=AssertionError("scheduler step inside the forward"),
            ),
            torch.inference_mode(),
        ):
            for count in range(len(schedule) + 1):
                _, prediction = benchmark._timed_request(
                    loaded, "cpu", cudagraphs=False
                )
                self.assertEqual(transformer_forward.call_count, count + 1)
                self.assertEqual(prediction.shape, (2, 4, 8, 8))
        requests = [call.kwargs for call in transformer_forward.call_args_list]
        timesteps = [request["timestep"][0].item() for request in requests]
        self.assertEqual(timesteps, schedule + schedule[:1])
        for request in requests:
            self.assertIs(request["hidden_states"], requests[0]["hidden_states"])
            self.assertEqual(request["timestep"].stride(), (0,))

        result = benchmark.execute_scenario(scenario)
        self.assertEqual(result["compiler_diagnostics"]["unique_graphs"], 1)
        self.assertEqual(result["compiled_component"], "transformer")
        self.assertEqual(result["output_summary"]["shape"], [2, 4, 8, 8])

    @skipIfNoDiffusionDeps
    def test_sd35_denoise_step_derives_inputs_from_checkpoint(self):
        import transformers
        from diffusers import (
            FlowMatchEulerDiscreteScheduler,
            SD3Transformer2DModel,
            StableDiffusion3Pipeline,
        )

        recipe = benchmark.RECIPES["denoise_step"]["sd35_medium"]
        parameters = {"height": 64, "width": 96, "max_sequence_length": 4}
        parameters["num_inference_steps"] = 3
        workload = dataclasses.replace(
            recipe.workload, parameters={**recipe.workload.parameters, **parameters}
        )
        scenario = dataclasses.replace(
            _denoise_scenario(),
            model=recipe.model,
            workload=workload,
            loader=recipe.loader,
        )
        tiny = benchmark._load_tiny_sd3_denoise_step(
            _denoise_scenario(), {}, "cpu", torch.float32
        ).component
        # SD3.5's dual-attention, plain, and final context_pre_only variants.
        dual = [block.attn2 is not None for block in tiny.transformer_blocks]
        pre_only = [block.context_pre_only for block in tiny.transformer_blocks]
        self.assertEqual(dual, [True, False, False])
        self.assertEqual(pre_only, [False, False, True])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tiny.save_pretrained(root / "transformer")
            FlowMatchEulerDiscreteScheduler(shift=3.0).save_pretrained(
                root / "scheduler"
            )
            (root / "vae").mkdir()
            vae_config = {"block_out_channels": [8, 16, 32, 64]}
            (root / "vae" / "config.json").write_text(json.dumps(vae_config))
            tokenizer = SimpleNamespace(model_max_length=77)
            (artifact,) = recipe.model.artifacts
            with (
                mock.patch.object(
                    transformers.CLIPTokenizer,
                    "from_pretrained",
                    return_value=tokenizer,
                ),
                mock.patch.object(
                    StableDiffusion3Pipeline, "from_pretrained"
                ) as load_pipeline,
            ):
                loaded = benchmark._load_sd35_medium_denoise_step(
                    scenario, {artifact.key: str(root)}, "cpu", torch.float32
                )
                with self.assertRaisesRegex(ValueError, "divisible by 16"):
                    benchmark._load_sd35_medium_denoise_step(
                        dataclasses.replace(
                            scenario,
                            workload=dataclasses.replace(
                                workload,
                                parameters={**workload.parameters, "width": 72},
                            ),
                        ),
                        {artifact.key: str(root)},
                        "cpu",
                        torch.float32,
                    )
        load_pipeline.assert_not_called()
        self.assertIsInstance(loaded.component, SD3Transformer2DModel)
        config = loaded.component.config
        inputs = loaded.make_request()[1]
        conditioning_length = 77 + 4
        self.assertEqual(
            loaded.workload_details["conditioning_length"], conditioning_length
        )
        self.assertEqual(
            inputs["hidden_states"].shape, (2, config.in_channels, 64 // 8, 96 // 8)
        )
        self.assertEqual(
            inputs["encoder_hidden_states"].shape,
            (2, conditioning_length, config.joint_attention_dim),
        )
        self.assertEqual(
            inputs["pooled_projections"].shape, (2, config.pooled_projection_dim)
        )
        scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)
        scheduler.set_timesteps(3)
        self.assertEqual(
            loaded.workload_details["timesteps"], scheduler.timesteps.tolist()
        )
        classes, count = benchmark._matched_region_classes(
            loaded.component, recipe.model.repeated_block_classes
        )
        self.assertEqual([cls.__name__ for cls in classes], ["JointTransformerBlock"])
        self.assertEqual(count, config.num_layers)

    @skipIfNoDiffusionDeps
    def test_denoise_step_fresh_workers_check_outputs(self):
        scenarios = [_denoise_scenario(mode) for mode in benchmark.MODES]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.csv"
            results, errors = benchmark._run_scenarios(scenarios, output, True)
        self.assertEqual(errors, [])
        self.assertEqual(len({result["pid"] for result in results}), len(scenarios))
        for result in results:
            graphs = result["compiler_diagnostics"]["unique_graphs"]
            self.assertEqual(graphs == 0, result["mode"] == "eager")
            self.assertEqual(result["workload"]["name"], "denoise_step")
            self.assertEqual(result["workload_details"]["effective_batch_size"], 2)
            check = result["output_check"]
            self.assertTrue(
                check == "reference"
                if result["mode"] == "eager"
                else check.startswith("passed"),
                check,
            )


@skipIfWrappedByDynamo
class TestDiffusionCompileBenchmarkDeviceType(TestCase):
    hw_classification = HardwareClassification.CUDA

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
        other = torch.full_like(value, 2)

        def pipeline():
            first = compiled(value)
            second = compiled(other)
            return first, second

        loaded = benchmark.LoadedPipeline(pipeline, compiled, lambda: ((), {}), ())
        with torch.inference_mode():
            for _ in range(3):
                _, output = benchmark._timed_request(loaded, device, cudagraphs=True)
                self.assertEqual(output[0], torch.full_like(value, 2))
                self.assertEqual(output[1], torch.full_like(value, 3))
                del output
        benchmark._compiler_diagnostics("full", device, True)


@skipIfNoDiffusionDeps
@skipIfWrappedByDynamo
class TestDenoiseStepNumericsDeviceType(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    @parametrize("mode", ("full", "regional", "hierarchical"))
    def test_compiled_prediction_matches_eager(self, device, mode):
        from torch.utils._triton import has_triton

        cudagraphs = torch.device(device).type == "cuda"
        if cudagraphs and not has_triton():
            self.skipTest("CUDA graph compilation requires Triton")
        self.addCleanup(torch._dynamo.reset)
        scenarios = [
            _denoise_scenario(
                scenario_mode,
                device=device,
                backend="inductor",
                cudagraphs=cudagraphs and scenario_mode != "eager",
            )
            for scenario_mode in ("eager", mode)
        ]
        with tempfile.TemporaryDirectory() as directory:
            paths = {
                scenario.scenario_id: Path(directory) / f"{scenario.mode}.pt"
                for scenario in scenarios
            }
            results = [
                benchmark.execute_scenario(scenario, str(paths[scenario.scenario_id]))
                for scenario in scenarios
            ]
            benchmark._compare_outputs(scenarios, results, paths)
        compiled = results[1]
        self.assertEqual(compiled["status"], "success", compiled.get("error"))
        self.assertTrue(compiled["output_check"].startswith("passed"))
        if cudagraphs:
            self.assertGreater(
                compiled["compiler_diagnostics"]["cudagraph_capture_count"], 0
            )


instantiate_parametrized_tests(TestDiffusionCompileBenchmark)
instantiate_device_type_tests(
    TestDiffusionCompileBenchmarkDeviceType, globals(), only_for="cuda"
)
instantiate_device_type_tests(TestDenoiseStepNumericsDeviceType, globals())


if __name__ == "__main__":
    run_tests()
