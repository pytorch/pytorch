import subprocess
import sys
from benchmark_base import BenchmarkBase


class ImportTimeBenchmark(BenchmarkBase):

    def __init__(self, module_name: str):
        super().__init__(
            category="import_time",
            backend="",
            device="cpu",
        )
        self._module_name = module_name

    def name(self):
        return f"import_time_{self._module_name.replace('.', '_')}"

    def description(self):
        return f"Import time for {self._module_name}"

    def _prepare_once(self):
        pass

    def _prepare(self):
        pass

    def _work(self):
        pass

    def _measure_import_time(self):
        result = subprocess.run(
            [
                sys.executable,
                "-X",
                "importtime",
                "-c",
                f"import {self._module_name}",
            ],
            capture_output=True,
            text=True,
        )

        for line in result.stderr.splitlines():
            if "import time:" in line:
                parts = line.split("|")
                if len(parts) >= 3:
                    module = parts[2].strip()
                    if module == self._module_name:
                        return int(parts[1].strip())

        return 0

    def collect_all(self):
        self._prepare_once()
        self.results = []

        print(f"Measuring import time for {self._module_name}...")

        times = []
        for i in range(self._num_iterations):
            import_time_us = self._measure_import_time()
            print(f"  Iteration {i}: {import_time_us} us")
            times.append(import_time_us)

        self.results.append((self.name(), "import_time_us", min(times) if times else 0))

        return self


def main():
    result_path = sys.argv[1] if len(sys.argv) > 1 else None

    modules_to_benchmark = [
        "torch._dynamo",
        "torch._inductor",
        "triton",
    ]

    benchmarks = [
        ImportTimeBenchmark(module).with_iterations(5) for module in modules_to_benchmark
    ]

    for benchmark in benchmarks:
        benchmark.collect_all()
        if result_path:
            benchmark.append_results(result_path)
        else:
            benchmark.print()


if __name__ == "__main__":
    main()
