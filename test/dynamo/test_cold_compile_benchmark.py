# Owner(s): ["module: dynamo"]

import json
import os
import subprocess
import sys
import tempfile

import torch
from torch._dynamo.test_case import run_tests, TestCase
from torch._inductor.utils import fresh_cache


class TestColdCompileBenchmark(TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        augmented_pp = ":".join(sys.path)
        if os.environ.get("PYTHONPATH"):
            augmented_pp = f"{os.environ.get('PYTHONPATH')}:{augmented_pp}"
        cls.python_path = augmented_pp

        cls.benchmark_script = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "benchmarks",
            "dynamo",
            "microbenchmarks",
            "cold_compile_benchmark.py",
        )

    def test_benchmark_module_imports(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "cold_compile_benchmark", self.benchmark_script
        )
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self.assertTrue(hasattr(module, "BENCHMARKS"))
        self.assertTrue(hasattr(module, "run_cold_compile_benchmark"))
        self.assertTrue(hasattr(module, "BenchmarkResult"))
        self.assertTrue(len(module.BENCHMARKS) > 0)

    def test_simple_function_cold_compile(self):
        with fresh_cache():
            torch._dynamo.reset()

            @torch.compile
            def f(x):
                return x.sin() + x.cos()

            x = torch.randn(100)

            import time

            start = time.perf_counter()
            result = f(x)
            cold_time = time.perf_counter() - start

            expected = x.sin() + x.cos()
            self.assertEqual(result, expected)
            self.assertGreater(cold_time, 0)

    def test_timing_breakdown_available(self):
        with fresh_cache():
            import torch._dynamo.utils as dynamo_utils

            torch._dynamo.reset()
            dynamo_utils.reset_frame_count()

            @torch.compile
            def f(x):
                return x.sin() + x.cos()

            x = torch.randn(100)
            f(x)

            timing = dynamo_utils.calculate_time_spent()

            self.assertIn("total_wall_time", timing)
            self.assertGreater(timing["total_wall_time"], 0)

    def test_benchmark_script_help(self):
        result = subprocess.run(
            [sys.executable, self.benchmark_script, "--help"],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": self.python_path},
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("cold compile", result.stdout.lower())

    def test_benchmark_script_runs_simple(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_output = os.path.join(tmpdir, "results.json")

            result = subprocess.run(
                [
                    sys.executable,
                    self.benchmark_script,
                    "--device",
                    "cpu",
                    "--benchmarks",
                    "simple_function",
                    "--json",
                    json_output,
                ],
                capture_output=True,
                text=True,
                timeout=120,  # 2 minute timeout
                env={**os.environ, "PYTHONPATH": self.python_path},
            )

            if result.returncode != 0:
                print(f"stdout: {result.stdout}")
                print(f"stderr: {result.stderr}")

            self.assertEqual(result.returncode, 0)

            self.assertTrue(os.path.exists(json_output))

            with open(json_output) as f:
                data = json.load(f)

            self.assertIsInstance(data, list)
            self.assertGreater(len(data), 0)

            cold_compile_metric = None
            for record in data:
                if (
                    record.get("metric", {}).get("name") == "cold_compile_time_s"
                    and record.get("model", {}).get("name") == "simple_function"
                ):
                    cold_compile_metric = record
                    break

            self.assertIsNotNone(cold_compile_metric)
            benchmark_values = cold_compile_metric["metric"]["benchmark_values"]
            self.assertEqual(len(benchmark_values), 1)
            self.assertGreater(benchmark_values[0], 0)

    def test_fresh_cache_provides_cold_compile(self):
        times = []

        for _ in range(2):
            with fresh_cache():
                torch._dynamo.reset()

                @torch.compile
                def f(x):
                    return x.sin() + x.cos()

                x = torch.randn(100)

                import time

                start = time.perf_counter()
                f(x)
                times.append(time.perf_counter() - start)

        self.assertGreater(times[0], 0)
        self.assertGreater(times[1], 0)

    def test_benchmark_result_dataclass(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "cold_compile_benchmark", self.benchmark_script
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        result = module.BenchmarkResult(
            name="test",
            device="cpu",
            cold_compile_time_s=1.5,
            dynamo_tracing_s=0.5,
        )

        self.assertEqual(result.name, "test")
        self.assertEqual(result.device, "cpu")
        self.assertEqual(result.cold_compile_time_s, 1.5)
        self.assertEqual(result.dynamo_tracing_s, 0.5)
        self.assertIsNone(result.aot_autograd_s)

        d = result.to_dict()
        self.assertIn("name", d)
        self.assertIn("dynamo_tracing_s", d)
        self.assertNotIn("aot_autograd_s", d)


class TestImportTime(TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        augmented_pp = ":".join(sys.path)
        if os.environ.get("PYTHONPATH"):
            augmented_pp = f"{os.environ.get('PYTHONPATH')}:{augmented_pp}"
        cls.python_path = augmented_pp

        cls.import_time_script = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "benchmarks",
            "dynamo",
            "pr_time_benchmarks",
            "benchmarks",
            "import_time.py",
        )

    def _measure_import_time(self, module_name: str) -> int:
        result = subprocess.run(
            [
                sys.executable,
                "-X",
                "importtime",
                "-c",
                f"import {module_name}",
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": self.python_path},
        )

        for line in result.stderr.splitlines():
            if "import time:" in line:
                parts = line.split("|")
                if len(parts) >= 3:
                    module = parts[2].strip()
                    if module == module_name:
                        return int(parts[1].strip())

        return 0

    def test_dynamo_import_time_measurable(self):
        import_time_us = self._measure_import_time("torch._dynamo")
        self.assertGreater(import_time_us, 0, "torch._dynamo import time should be measurable")

    def test_inductor_import_time_measurable(self):
        import_time_us = self._measure_import_time("torch._inductor")
        self.assertGreater(import_time_us, 0, "torch._inductor import time should be measurable")

    def test_triton_import_time_measurable(self):
        import_time_us = self._measure_import_time("triton")
        self.assertGreater(import_time_us, 0, "triton import time should be measurable")

    def test_import_time_benchmark_module_imports(self):
        self.assertTrue(
            os.path.exists(self.import_time_script),
            f"import_time.py should exist at {self.import_time_script}",
        )

        result = subprocess.run(
            [sys.executable, "-m", "py_compile", self.import_time_script],
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode, 0, f"import_time.py should have valid syntax: {result.stderr}"
        )

    def test_import_time_benchmark_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_output = os.path.join(tmpdir, "results.csv")

            result = subprocess.run(
                [
                    sys.executable,
                    self.import_time_script,
                    csv_output,
                ],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout for all 3 modules
                env={**os.environ, "PYTHONPATH": self.python_path},
                cwd=os.path.dirname(self.import_time_script),
            )

            if result.returncode != 0:
                print(f"stdout: {result.stdout}")
                print(f"stderr: {result.stderr}")

            self.assertEqual(result.returncode, 0)
            self.assertTrue(os.path.exists(csv_output))

            with open(csv_output) as f:
                content = f.read()

            self.assertIn("import_time_torch__dynamo", content)
            self.assertIn("import_time_torch__inductor", content)
            self.assertIn("import_time_triton", content)


class TestColdCompileBaseline(TestCase):

    def test_subprocess_cold_compile_isolation(self):
        script = '''
import json
import os
import time
import torch
import torch._dynamo
from torch._dynamo.utils import calculate_time_spent

torch._dynamo.reset()

@torch.compile
def f(x):
    return x.sin() + x.cos()

start = time.perf_counter()
f(torch.randn(100))
cold_time = time.perf_counter() - start

timing = calculate_time_spent()
print(json.dumps({
    "cold_time": cold_time,
    "timing": timing,
}))
'''

        times = []
        with tempfile.TemporaryDirectory() as cache_dir:
            for _ in range(2):
                run_cache = os.path.join(cache_dir, f"run_{len(times)}")
                os.makedirs(run_cache)

                env = os.environ.copy()
                env["TORCHINDUCTOR_CACHE_DIR"] = run_cache
                env["TRITON_CACHE_DIR"] = os.path.join(run_cache, "triton")

                result = subprocess.run(
                    [sys.executable, "-c", script],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    env=env,
                )

                self.assertEqual(result.returncode, 0, f"stderr: {result.stderr}")
                data = json.loads(result.stdout)
                times.append(data["cold_time"])

        self.assertGreater(times[0], 0)
        self.assertGreater(times[1], 0)


if __name__ == "__main__":
    run_tests()
