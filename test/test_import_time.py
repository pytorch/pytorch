import subprocess
import re
import sys
import unittest
import pathlib

from torch.testing._internal.common_utils import TestCase, run_tests, IS_LINUX, IS_IN_CI


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

try:
    # Just in case PyTorch was not built in 'develop' mode
    sys.path.append(str(REPO_ROOT))
    from tools.stats.scribe import rds_write, register_rds_schema
except ImportError:
    register_rds_schema = None
    rds_write = None


# these tests could eventually be changed to fail if the import/init
# time is greater than a certain threshold, but for now we just use them
# as a way to track the duration of `import torch` in our ossci-metrics
# S3 bucket (see tools/stats/print_test_stats.py)
class TestImportTime(TestCase):
    def test_time_import_torch(self):
        TestCase.runWithPytorchAPIUsageStderr("import torch")

    def test_time_cuda_device_count(self):
        TestCase.runWithPytorchAPIUsageStderr(
            "import torch; torch.cuda.device_count()",
        )

    @unittest.skipIf(not IS_LINUX, "Memory test is only implemented for Linux")
    # @unittest.skipIf(not IS_IN_CI, "Memory test only runs in CI")
    def test_peak_memory(self):
        def profile(command, name):
            result = subprocess.run(
                ["/usr/bin/time", "-v", sys.executable, "-c", command],
                stderr=subprocess.PIPE,
            )
            lines = result.stderr.decode().split("\n")

            def parse_time(pattern):
                search_re = re.compile(pattern)
                for line in lines:
                    match = search_re.search(line)
                    if match:
                        return float(match.groups()[0])

                raise RuntimeError(f"Unable to find '{pattern}' in /usr/bin/time -v output")

            return {
                "test_name": name,
                "peak_memory_bytes": int(
                    parse_time(r"Maximum resident set size \(kbytes\): (\d+)")
                ),
                "time_ms": int(parse_time(r"User time \(seconds\): ([0-9.]+)") * 1000),
            }

        data = profile("import torch", "pytorch")
        rds_write("import_stats", data, only_on_master=False)  # TODO: remove only_on_master arg
        baseline = profile("import sys", "baseline")
        rds_write("import_stats", baseline, only_on_master=False)  # TODO: remove only_on_master arg


if __name__ == "__main__":
    if register_rds_schema and IS_IN_CI:
        register_rds_schema(
            "import_stats",
            {
                "test_name": "string",
                "peak_memory_bytes": "int",
                "time_ms": "int",
            },
        )

    run_tests()
