# Owner(s): ["module: inductor"]
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest

from torch.testing._internal.common_utils import (
    HardwareClassification,
    IS_WINDOWS,
    run_tests,
    TestCase,
)


@unittest.skipUnless(IS_WINDOWS, "Windows OpenMP DLL lifetime")
class AOTIRuntimeLifetimeTests(TestCase):
    hw_classification = HardwareClassification.CPU

    def test_destroy_parallel_model(self):
        # Isolate native crashes and start without another model holding OpenMP.
        source = textwrap.dedent(
            """
            import concurrent.futures
            import gc
            import pathlib
            import sys
            import threading

            import torch
            from torch._inductor import aoti_compile_and_package, aoti_load_package, config

            class Model(torch.nn.Module):
                def forward(self, x):
                    return x + 1

            torch.manual_seed(0)
            torch.set_num_threads(4)
            x = torch.randn(65536)
            expected = x + 1
            with config.patch({"cpp.threads": 4, "cpp.min_chunk_size": 1}):
                package = aoti_compile_and_package(
                    torch.export.export(Model(), (x,)),
                    package_path=str(pathlib.Path(sys.argv[1]) / "model.pt2"),
                )

            def run_once(barrier=None):
                runner = aoti_load_package(package)
                if barrier is not None:
                    barrier.wait(timeout=60)
                output = runner(x)
                del runner
                gc.collect()
                if not torch.equal(output, expected):
                    raise AssertionError("Output changed after runner destruction")
                return output

            outputs = [run_once() for _ in range(2)]
            barrier = threading.Barrier(4)
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                outputs.extend(pool.map(run_once, [barrier] * 4))
            for output in outputs:
                if not torch.equal(output, expected):
                    raise AssertionError("A surviving output was invalidated")
            print("parallel runners destroyed OK", flush=True)
            """
        )
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ)
            env["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(directory, "cache")
            env["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
            with subprocess.Popen(
                [sys.executable, "-I", "-u", "-c", source, directory],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            ) as process:
                try:
                    stdout, stderr = process.communicate(timeout=300)
                except subprocess.TimeoutExpired:
                    # Python's Windows venv launcher can have interpreter children.
                    cleanup = subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    stdout, stderr = process.communicate(timeout=30)
                    self.fail(
                        f"Child timed out\n{stdout}\n{stderr}\n"
                        f"Cleanup: {cleanup.returncode}\n"
                        f"{cleanup.stdout}\n{cleanup.stderr}"
                    )
            self.assertEqual(
                process.returncode,
                0,
                msg=f"Child return code {process.returncode}\n{stdout}\n{stderr}",
            )
            self.assertIn("parallel runners destroyed OK", stdout)


if __name__ == "__main__":
    run_tests()
