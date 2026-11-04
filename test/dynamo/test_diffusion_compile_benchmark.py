# Owner(s): ["module: dynamo"]

import csv
import dataclasses
import io
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
from torch._inductor.test_case import TestCase as InductorTestCase
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    skipCPUIf,
    skipGPUIf,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    IS_WINDOWS,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    subtest,
    TestCase,
)
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_GPU
from torch.testing._internal.triton_utils import requires_cuda_and_triton


REPO_ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO_ROOT)
try:
    from benchmarks.diffusion import compile_benchmark as benchmark
finally:
    sys.path.remove(REPO_ROOT)

# The runner compiles and counts graphs itself; wrapping tests in Dynamo makes
# its steady-state checks see the test's own compilation.
skipIfWrappedByDynamo = skipIfTorchDynamo("the benchmark runner manages compilation")


@skipIfWrappedByDynamo
class TestDiffusionCompileBenchmark(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def _toy_scenarios(self, modes):
        recipe = benchmark.BENCHMARKS["toy"]
        execution = benchmark.ExecutionConfig(
            backend="eager",
            cudagraphs=False,
            warmups=0,
            repetitions=1,
            timeout_s=600,
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
            "dtype": scenario.model.default_dtype
            if scenario.execution.dtype == "auto"
            else scenario.execution.dtype,
            "device": scenario.model.default_device
            if scenario.execution.device == "auto"
            else scenario.execution.device,
            "compiled_targets": [] if scenario.mode == "eager" else ["ToyDenoiser"],
            "model_setup_s": 0.1,
            "compile_wrapper_setup_s": 0.0,
            "compile_time_s": 0.0 if scenario.mode == "eager" else 0.5,
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

    def _main_error(self, args):
        stderr = io.StringIO()
        with mock.patch("sys.stderr", stderr), self.assertRaises(SystemExit) as error:
            benchmark.main(args)
        self.assertEqual(error.exception.code, 2)
        return stderr.getvalue()

    @parametrize("timeout", ("nan", "inf", "-inf", "0", "-1"))
    def test_timeout_must_be_finite_and_positive(self, timeout):
        args = benchmark._parser().parse_args([f"--timeout={timeout}"])
        with self.assertRaisesRegex(ValueError, "timeout must be finite and positive"):
            benchmark._execution_from_args(args)
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

    def test_regional_falls_back_to_explicit_classes(self):
        class Block(torch.nn.Module):
            def forward(self, value):
                return value + 1

        class Mismatched(torch.nn.Sequential):
            _repeated_blocks = ("Other",)

            def compile_repeated_blocks(self, **kwargs):
                raise AssertionError("mismatched entry point must not be used")

        model = torch.nn.Sequential(Block(), Block())
        with mock.patch.object(torch.nn.Module, "compile") as compile_module:
            with benchmark.configure_compilation(
                model, "regional", ("Block",), "eager", False
            ) as details:
                pass
        self.assertEqual(details["regional_entry_point"], "explicit class selection")
        self.assertEqual(compile_module.call_count, 2)
        with self.assertRaisesRegex(ValueError, "do not match recipe"):
            with benchmark.configure_compilation(
                Mismatched(Block()), "regional", ("Block",), "eager", False
            ):
                pass

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

    def test_device_initializes_before_setup_timing(self):
        scenario = self._toy_scenarios(("eager",))[0]
        scenario = dataclasses.replace(
            scenario,
            execution=dataclasses.replace(
                scenario.execution, device="cuda:0", num_threads=None
            ),
        )
        calls = []
        resolved = ("cuda:0", "float32", torch.float32)

        def record(name):
            return lambda device: calls.append((name, device))

        with (
            mock.patch.object(
                benchmark, "_resolve_device_and_dtype", return_value=resolved
            ),
            mock.patch.object(
                torch, "empty", side_effect=lambda size, device: record("init")(device)
            ),
            mock.patch.object(benchmark, "_synchronize", side_effect=record("sync")),
            mock.patch.object(
                benchmark, "_reset_device_memory", side_effect=record("reset")
            ),
            mock.patch.object(
                benchmark, "_load_toy", side_effect=RuntimeError("reached model setup")
            ),
            self.assertRaisesRegex(RuntimeError, "reached model setup"),
        ):
            benchmark.execute_scenario(scenario)
        # Synchronizing and resetting peaks do not start the runtime of an indexed
        # device, so without the explicit initialization setup would include it.
        expected = [("init", "cuda:0"), ("sync", "cuda:0"), ("reset", "cuda:0")]
        self.assertEqual(calls, expected)

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
            check = "unbounded: relative RMS difference vs eager 0"
            self.assertEqual(info["output_check"], "reference" if eager else check)

    @parametrize(
        "flags,message",
        (
            subtest((("--mode", "eager"), "at least one compiled mode"), name="eager"),
            subtest(
                (("--mode", "full", "--backend", "eager"), "--backend inductor"),
                name="backend",
            ),
            subtest(
                (("--mode", "full", "--device", "cpu"), "requires a CUDA device"),
                name="device",
            ),
            subtest(
                (("--mode", "full", "--warmups", "0"), "at least one warmup"),
                name="warmups",
            ),
        ),
    )
    def test_cudagraphs_rejected_before_workers(self, flags, message):
        args = ["--model", "toy", "--device", "cuda:0", "--cudagraphs", *flags]
        with mock.patch.object(benchmark, "_run_scenarios") as run_scenarios:
            self.assertIn(message, self._main_error(args))
        run_scenarios.assert_not_called()

    @parametrize(
        "flags,message",
        (
            subtest((("--warmups", "-1"), "--warmups must"), name="warmups"),
            subtest((("--repetitions", "0"), "--repetitions must"), name="repetitions"),
            subtest((("--num-threads", "0"), "--num-threads must"), name="num_threads"),
            subtest((("--mode", "all"), "cannot be combined"), name="all"),
            subtest((("--mode", "full"), "only once"), name="duplicate"),
            subtest((("--check-outputs",), "--check-outputs requires"), name="check"),
        ),
    )
    def test_invalid_arguments_rejected_before_workers(self, flags, message):
        args = ["--model", "toy", "--mode", "full", *flags]
        with mock.patch.object(benchmark, "_run_scenarios") as run_scenarios:
            self.assertIn(message, self._main_error(args))
        run_scenarios.assert_not_called()

    def test_unwritable_output_rejected_before_workers(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(benchmark, "_run_scenarios") as run_scenarios,
        ):
            args = ["--model", "toy", "--mode", "eager", "--output", directory]
            self.assertIn("cannot write", self._main_error(args))
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
        ):
            output = Path(directory) / filename
            output.write_text("existing results", encoding="utf-8")
            if cli:
                args = ["--model", "toy", "--mode", "eager", "--output", str(output)]
                self.assertIn("must not end in .json", self._main_error(args))
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
from torch._dynamo import config as dynamo_config
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
    dynamo_config.automatic_dynamic_remote_pgo,
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
            stale_log = Path(directory) / "result_logs" / "9-hierarchical.log"
            stale_log.parent.mkdir()
            stale_log.touch()
            inherited = {
                "TRITON_CACHE_DIR": str(Path(directory) / "shared"),
                "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
                "TORCHINDUCTOR_AUTOGRAD_CACHE": "1",
                "TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE": "1",
                "TORCHINDUCTOR_AUTOGRAD_REMOTE_CACHE": "1",
                "TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE": "1",
                "TORCHINDUCTOR_BUNDLED_AUTOTUNE_REMOTE_CACHE": "1",
                "TORCH_DYNAMO_AUTOMATIC_DYNAMIC_REMOTE_PGO": "1",
            }
            with (
                mock.patch.dict(os.environ, inherited),
                mock.patch.object(
                    benchmark, "run_worker_process", side_effect=run_isolated
                ),
            ):
                results, errors = benchmark._run_scenarios(scenarios, output, True)
            self.assertFalse(stale_log.exists())
        self.assertEqual(errors, [])
        self.assertNotIn(os.getpid(), {result["pid"] for result in results})
        self.assertEqual(len(set(cache_dirs)), len(scenarios))
        for result in results:
            self.assertEqual(result["output_summary"], results[0]["output_summary"])
            graphs = result["compiler_diagnostics"]["unique_graphs"]
            self.assertEqual(graphs > 0, result["mode"] != "eager")
            self.assertEqual(result["compile_time_s"] > 0, result["mode"] != "eager")
            self.assertEqual(set(result["compiler_caches"].values()), {False})
            environment = result["environment"]
            self.assertEqual(environment["TORCHINDUCTOR_FX_GRAPH_CACHE"], "0")
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
        self.assertEqual(rows[0]["backend"], records[0]["model"]["backend"])
        self.assertEqual(records[0]["benchmark"]["name"], benchmark.BENCHMARK_NAME)
        self.assertIn("benchmark_values", records[0]["metric"])
        metrics = {record["metric"]["name"] for record in records}
        self.assertIn("compile_time_s", metrics)
        self.assertFalse(any(name.startswith("compiler_") for name in metrics))
        extra_info = records[0]["benchmark"]["extra_info"]
        self.assertEqual(extra_info["device"], "cpu")
        self.assertTrue(all(not isinstance(v, dict) for v in extra_info.values()))
        failure_record = benchmark.dashboard_records(
            {"scenario_id": "failed", "status": "failed", "error": "boom /home/user"}
        )[0]
        self.assertNotIn("benchmark_values", failure_record["metric"])
        self.assertEqual(
            failure_record["metric"]["extra_info"], {"benchmark_values": ["failed"]}
        )

    def test_worker_exception_is_reported(self):
        scenario = self._toy_scenarios(("full",))[0]
        execution = dataclasses.replace(scenario.execution, backend="no_such_backend")
        scenario = dataclasses.replace(scenario, execution=execution)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.csv"
            (result,), errors = benchmark._run_scenarios([scenario], output, False)
        self.assertEqual(result["status"], "failed")
        self.assertIn("no_such_backend", result["error"])
        self.assertEqual(len(errors), 1)

    @parametrize(
        "failure",
        ("failed", "timed_out", "killed", "crashed", "reported_then_killed", "missing"),
    )
    def test_worker_failure_stays_visible_alongside_success(self, failure):
        signals = ("killed", "crashed", "reported_then_killed")
        if failure in signals and sys.platform == "win32":
            self.skipTest("POSIX signals are unavailable on Windows")
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
            failure_value = {
                "scenario_id": failure,
                "status": "failed",
                "error": "intentional failure",
            }
            report = (
                "from pathlib import Path; "
                f"Path({str(failure_path)!r}).write_text({json.dumps(failure_value)!r}); "
            )
            kill = "import os, signal; os.kill(os.getpid(), signal.{})"
            code = {
                "failed": report + "raise SystemExit(1)",
                "timed_out": "import time; time.sleep(10)",
                "killed": kill.format("SIGKILL"),
                "crashed": kill.format("SIGABRT"),
                "reported_then_killed": report + kill.format("SIGKILL"),
                "missing": "pass",
            }[failure]
            timeout = 0.05 if failure == "timed_out" else 10
            failed = benchmark.run_worker_process(
                failure,
                [sys.executable, "-c", code],
                failure_path,
                root / f"{failure}.log",
                timeout,
            )
        self.assertEqual(success["status"], "success")
        # Only an external kill is "killed"; a crash or a worker's own report is not.
        crashed = failure in ("crashed", "reported_then_killed")
        self.assertEqual(failed["status"], "failed" if crashed else failure)
        if failure == "reported_then_killed":
            self.assertEqual(failed["error"], "intentional failure")
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
            "compile_time_s",
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
        args = ["--model", "toy", "--backend", "eager", "--timeout", "60"]
        args += ["--mode", "eager", "--mode", "full"]
        execution = benchmark._execution_from_args(benchmark._parser().parse_args(args))
        scenarios = [
            dataclasses.replace(scenario, execution=execution)
            for scenario in self._toy_scenarios(("eager", "full"))
        ]
        payloads = {
            scenario.scenario_id: self._success_result(scenario)
            for scenario in scenarios
        }
        malformed = payloads[scenarios[1].scenario_id]
        if field in ("status", "compile_wrapper_setup_s", "compile_time_s"):
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
        self.assertIn("; log: ", rows[1]["error"])
        # Failures resolve "auto" so they share a dashboard series with successes.
        self.assertEqual([row["dtype"] for row in rows], ["float32", "float32"])
        self.assertEqual([row["device"] for row in rows], ["cpu", "cpu"])
        failures = [
            record
            for record in records
            if record["metric"]["name"] == "scenario_status"
        ]
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["benchmark"]["dtype"], "float32")
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
                # The worker sleeps for 60s, so only a kill ends it this early.
                if IS_WINDOWS:
                    self.assertIsNotNone(processes[0].returncode)
                else:
                    self.assertEqual(processes[0].returncode, -benchmark.signal.SIGKILL)
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

    def test_steady_state_recompilation_is_rejected(self):
        scenario = self._toy_scenarios(("full",))[0]
        scenario = dataclasses.replace(
            scenario,
            execution=dataclasses.replace(
                scenario.execution, repetitions=2, num_threads=None
            ),
        )
        model = torch.nn.Linear(4, 4)
        rows = iter((1, 2, 3))

        def make_request():
            return (torch.randn(next(rows), 4),), {}

        loaded = benchmark.LoadedPipeline(model, model, make_request, ())
        with (
            mock.patch.object(benchmark, "_load_toy", return_value=loaded),
            self.assertRaisesRegex(RuntimeError, "during steady-state measurement"),
        ):
            benchmark.execute_scenario(scenario)

    @parametrize("case", ("captured", "empty", "skipped"))
    def test_cudagraph_diagnostics(self, case):
        from torch._dynamo.utils import counters

        captured = case != "empty"
        leaf = SimpleNamespace(graph=object() if captured else None, children={})
        child = SimpleNamespace(
            graph=object() if captured else None, children={1: [leaf]}
        )
        root = SimpleNamespace(graph=None, children={0: [child]})
        manager = SimpleNamespace(get_roots=lambda: iter([root]))
        module = SimpleNamespace(get_manager=lambda *args, **kwargs: manager)
        values = {
            "stats": {"unique_graphs": 1},
            "inductor": {"cudagraph_skips": int(case == "skipped")},
        }
        with (
            mock.patch.dict(counters, values, clear=True),
            mock.patch.dict(sys.modules, {"torch._inductor.cudagraph_trees": module}),
        ):
            if case == "captured":
                diagnostics = benchmark._compiler_diagnostics("full", "cuda:0", True)
                self.assertEqual(diagnostics["cudagraph_capture_count"], 2)
            else:
                message = "skipped them 1 time" if captured else "no graph capture"
                with self.assertRaisesRegex(RuntimeError, message):
                    benchmark._compiler_diagnostics("full", "cuda:0", True)

    def test_hierarchical_requires_nested_region(self):
        from torch._dynamo.utils import compilation_time_metrics, counters

        with (
            mock.patch.dict(counters, {"stats": {"unique_graphs": 1}}, clear=True),
            mock.patch.dict(compilation_time_metrics, {}, clear=True),
        ):
            with self.assertRaisesRegex(RuntimeError, "no nested region"):
                benchmark._compiler_diagnostics("hierarchical", "cpu", False)
            compilation_time_metrics["invoke_subgraph_trace"] = [0.1]
            benchmark._compiler_diagnostics("hierarchical", "cpu", False)

    def test_extract_tensor_handles_pipeline_outputs(self):
        import numpy as np
        from PIL import Image

        pixels = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        expected = torch.from_numpy(pixels).expand(2, 2, 2, 3)
        fields = ("images", "frames")
        for output in (
            {"images": [Image.fromarray(pixels)] * 2},
            SimpleNamespace(frames=np.stack([pixels, pixels])),
        ):
            self.assertEqual(benchmark._extract_tensor(output, fields), expected)
        stacked = benchmark._extract_tensor((torch.ones(3), torch.ones(3)), fields)
        self.assertEqual(stacked, torch.ones(2, 3))
        with self.assertRaisesRegex(TypeError, "cannot extract a tensor"):
            benchmark._extract_tensor(object(), fields)

    def test_pipeline_output_check(self):
        eager = torch.tensor([3.0, 4.0])
        check = benchmark._check_pipeline_output(torch.tensor([3.0, 4.5]), eager)
        self.assertEqual(check, "unbounded: relative RMS difference vs eager 0.1")
        for actual, message in (
            (eager.reshape(2, 1), "does not match eager"),
            (eager.double(), "does not match eager"),
            (torch.tensor([3.0, float("inf")]), "non-finite"),
        ):
            with self.assertRaisesRegex(AssertionError, message):
                benchmark._check_pipeline_output(actual, eager)

    @parametrize("case", ("eager_nan", "eager_zeros", "compiled_nan"))
    def test_output_check_rejects_unusable_outputs(self, case):
        scenarios = self._toy_scenarios(("eager", "full"))
        results = [self._success_result(scenario) for scenario in scenarios]
        values = {
            "eager_nan": (float("nan"), 1.0),
            "eager_zeros": (0.0, 1.0),
            "compiled_nan": (1.0, float("nan")),
        }[case]
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for scenario, value in zip(scenarios, values):
                path = Path(directory) / f"{scenario.mode}.pt"
                torch.save(torch.tensor([value]), path)
                paths[scenario.scenario_id] = path
            benchmark._compare_outputs(scenarios, results, paths)
        eager, compiled = results
        self.assertEqual(compiled["status"], "failed")
        if case == "compiled_nan":
            self.assertEqual(eager["output_check"], "reference")
            self.assertEqual(compiled["output_check"], "failed")
            self.assertIn("non-finite", compiled["error"])
        else:
            self.assertEqual(eager["status"], "failed")
            problem = "non-finite" if case == "eager_nan" else "all zeros"
            self.assertIn(problem, eager["error"])
            self.assertEqual(compiled["output_check"], "reference unavailable")

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

    # File names from the pinned revisions; "unused" files must not be downloaded.
    @parametrize(
        "name,needed,unused",
        (
            subtest(
                (
                    "auraflow",
                    ("model_index.json", "text_encoder/model.safetensors"),
                    (
                        "aura_flow_0.3.safetensors",
                        "text_encoder/model.fp16.safetensors",
                        "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
                        "vae/diffusion_pytorch_model.fp16.safetensors",
                    ),
                ),
                name="auraflow",
            ),
            subtest(
                (
                    "flux",
                    ("model_index.json", "transformer/config.json"),
                    ("flux1-dev.safetensors", "ae.safetensors"),
                ),
                name="flux",
            ),
            subtest(
                (
                    "wan",
                    ("image_encoder/model.safetensors", "vae/config.json"),
                    ("assets/logo.png", "examples/i2v_input.JPG"),
                ),
                name="wan",
            ),
            subtest(
                (
                    "ltx",
                    ("model_index.json", "vae/diffusion_pytorch_model.safetensors"),
                    ("media/trailer.gif",),
                ),
                name="ltx",
            ),
        ),
    )
    def test_prefetch_excludes_unused_weights(self, name, needed, unused):
        artifacts = benchmark.BENCHMARKS[name].model.artifacts
        snapshot = mock.Mock(return_value="snapshot")
        download = mock.Mock(return_value="file")
        hub = SimpleNamespace(snapshot_download=snapshot, hf_hub_download=download)
        with mock.patch.dict(sys.modules, {"huggingface_hub": hub}):
            benchmark._prefetch(artifacts)
        self.assertEqual(snapshot.call_args.kwargs["repo_id"], artifacts[0].repo_id)
        self.assertEqual(snapshot.call_args.kwargs["revision"], artifacts[0].revision)
        patterns = snapshot.call_args.kwargs["ignore_patterns"]
        for filename in needed:
            self.assertFalse(
                any(fnmatchcase(filename, pattern) for pattern in patterns)
            )
        for filename in unused:
            self.assertTrue(any(fnmatchcase(filename, pattern) for pattern in patterns))
        files = [
            {
                "repo_id": artifact.repo_id,
                "revision": artifact.revision,
                "filename": artifact.filename,
                "repo_type": artifact.repo_type,
            }
            for artifact in artifacts
            if artifact.filename is not None
        ]
        self.assertEqual([call.kwargs for call in download.call_args_list], files)

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

        recipe = benchmark.BENCHMARKS["wan"]
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
        self.assertEqual(finish.call_args.args, (pipeline, scenario, "cpu"))
        self.assertEqual(finish.call_args.kwargs["output_fields"], ("frames",))
        image_factory = finish.call_args.kwargs["mutable_parameters"]["image"]
        first = image_factory()
        first.putpixel((0, 0), (0, 0, 0))
        second = image_factory()
        self.assertEqual(second.size, (720, 544))
        self.assertEqual(second.getpixel((0, 0)), (255, 0, 0))


def _toy_inductor_scenario(mode, device, cudagraphs):
    recipe = benchmark.BENCHMARKS["toy"]
    execution = benchmark.ExecutionConfig(
        backend="inductor",
        cudagraphs=cudagraphs,
        warmups=1,
        repetitions=2,
        timeout_s=600,
        device=device,
        dtype="float32",
        num_threads=None,
    )
    loader = recipe.loader
    return benchmark.Scenario(recipe.model, recipe.workload, execution, mode, loader)


@requires_cuda_and_triton
@skipIfWrappedByDynamo
class TestDiffusionCompileBenchmarkDeviceType(InductorTestCase):
    hw_classification = HardwareClassification.CUDA

    def test_cudagraph_request_retains_predictions(self, device):
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

    @parametrize("mode", ("full", "regional", "hierarchical"))
    def test_cudagraph_scenario(self, device, mode):
        # execute_scenario raises unless every graph was captured without a skip.
        scenario = _toy_inductor_scenario(mode, device, cudagraphs=True)
        result = benchmark.execute_scenario(scenario)
        options = result["effective_compile_options"]["options"]
        self.assertTrue(options["triton.cudagraphs"])


@skipIfWrappedByDynamo
class TestInductorScenarioDeviceType(InductorTestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    @skipCPUIf(not HAS_CPU, "requires a C++ compiler")
    @skipGPUIf(not HAS_GPU, "requires a GPU with Triton")
    @parametrize("mode", ("full", "regional", "hierarchical"))
    def test_compiled_scenario(self, device, mode):
        scenario = _toy_inductor_scenario(mode, device, cudagraphs=False)
        run_worker = benchmark.run_worker_process
        cache_entries = []

        def run_tracked(scenario_id, command, result_path, log_path, timeout_s, env):
            result = run_worker(
                scenario_id, command, result_path, log_path, timeout_s, env
            )
            cache_entries.extend(Path(env["TORCHINDUCTOR_CACHE_DIR"]).glob("*"))
            return result

        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(benchmark, "run_worker_process", side_effect=run_tracked),
        ):
            output = Path(directory) / "result.csv"
            (result,), errors = benchmark._run_scenarios([scenario], output, False)
        self.assertEqual(errors, [], result.get("error"))
        self.assertEqual(result["device"], device)
        self.assertGreater(result["compile_time_s"], 0)
        self.assertTrue(cache_entries, "the worker did not use its private cache")


instantiate_parametrized_tests(TestDiffusionCompileBenchmark)
instantiate_device_type_tests(
    TestDiffusionCompileBenchmarkDeviceType, globals(), only_for="cuda"
)
instantiate_device_type_tests(TestInductorScenarioDeviceType, globals())


if __name__ == "__main__":
    run_tests()
