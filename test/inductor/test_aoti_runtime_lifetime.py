# Owner(s): ["module: inductor"]
import os
import subprocess
import sys
import textwrap
import unittest
from unittest import mock

import torch
from torch._inductor.codecache import AotCodeCompiler
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    IS_WINDOWS,
    parametrize,
    run_tests,
    TestCase,
)


@unittest.skipUnless(IS_WINDOWS, "Windows OpenMP DLL lifetime")
class AOTIRuntimeLifetimeTests(TestCase):
    hw_classification = HardwareClassification.CPU

    def _run_isolated(self, args, env):
        # Do not use Popen as a context manager: its exit wait has no timeout.
        process = subprocess.Popen(
            args,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=180)
        except subprocess.TimeoutExpired as original:
            diagnostics = [
                f"Original timeout: {original}",
                f"Partial stdout: {original.output!r}",
                f"Partial stderr: {original.stderr!r}",
            ]
            if process.poll() is None:
                cleanup = None
                try:
                    # The Windows venv launcher can have interpreter children.
                    cleanup = subprocess.Popen(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    cleanup_out, cleanup_err = cleanup.communicate(timeout=15)
                    diagnostics.append(
                        f"Tree cleanup: {cleanup.returncode}\n"
                        f"{cleanup_out}\n{cleanup_err}"
                    )
                except (OSError, subprocess.TimeoutExpired) as error:
                    diagnostics.append(f"Tree cleanup failed: {error!r}")
                finally:
                    if cleanup is not None:
                        if cleanup.poll() is None:
                            try:
                                cleanup.kill()
                            except OSError as error:
                                diagnostics.append(
                                    f"Cleanup termination failed: {error!r}"
                                )
                        try:
                            cleanup.wait(timeout=5)
                        except (OSError, subprocess.TimeoutExpired) as error:
                            diagnostics.append(f"Cleanup reap unconfirmed: {error!r}")
            else:
                diagnostics.append("Child exited before tree cleanup")
            if process.poll() is None:
                try:
                    process.kill()
                except OSError as error:
                    diagnostics.append(f"Child termination failed: {error!r}")
            try:
                stdout, stderr = process.communicate(timeout=15)
                diagnostics.append(f"Final stdout: {stdout}\nFinal stderr: {stderr}")
            except (OSError, subprocess.TimeoutExpired) as error:
                diagnostics.append(f"Final pipe collection failed: {error!r}")
            try:
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired) as error:
                diagnostics.append(f"Child reap unconfirmed: {error!r}")
            self.fail("\n".join(diagnostics))
        return process.returncode, stdout, stderr

    @parametrize("size", [1024, 65536])
    @parametrize("concurrent", [False, True])
    @parametrize("packaged", [False, True])
    def test_model_dll_unloaded(self, size, concurrent, packaged):
        class Model(torch.nn.Module):
            def forward(self, x):
                return (x + 1,)

        inputs = (torch.randn(size),)
        exported = torch.export.export(Model(), inputs)
        options = {"cpp.threads": 2, "cpp.min_chunk_size": 4096}
        with mock.patch.object(
            AotCodeCompiler, "compile", wraps=AotCodeCompiler.compile
        ) as compile_model:
            if packaged:
                model_path = torch._inductor.aoti_compile_and_package(
                    exported, inductor_configs=options
                )
            else:
                model_path = torch._inductor.aot_compile(
                    exported.module(), inputs, options=options
                )
        kernel_code = compile_model.call_args.args[2]
        self.assertEqual("#pragma omp parallel" in kernel_code, size == 65536)

        # A fresh process must not inherit another model's runtime reference.
        source = textwrap.dedent(
            """
            import ctypes
            import gc
            import os
            import shutil
            import sys
            import tempfile
            import zipfile
            from concurrent.futures import ThreadPoolExecutor
            from pathlib import Path
            from threading import Barrier

            import torch
            from torch.testing._internal.common_utils import TestCase

            model_path = sys.argv[1]
            size, concurrent, packaged = map(int, sys.argv[2:])
            torch.set_num_threads(2)
            check = TestCase()
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
            kernel32.GetModuleHandleW.restype = ctypes.c_void_p
            barrier = Barrier(4) if concurrent else None

            def run_once(index):
                x = torch.full((size,), float(index))
                expected = x + 1
                # Each runner gets a different DLL, including concurrent runners.
                with tempfile.TemporaryDirectory() as directory:
                    if packaged:
                        with zipfile.ZipFile(model_path) as archive:
                            archive.extractall(directory)
                        paths = [
                            path for path in Path(directory).rglob("*")
                            if path.suffix in (".dll", ".pyd")
                        ]
                        check.assertEqual(len(paths), 1)
                        path = str(paths[0].resolve())
                    else:
                        path = os.path.join(directory, "model.dll")
                        shutil.copyfile(model_path, path)
                    check.assertIsNone(kernel32.GetModuleHandleW(path))
                    if packaged:
                        runner = torch._inductor.aoti_load_package(directory)
                    else:
                        runner = torch._C._aoti.AOTIModelContainerRunnerCpu(path, 1)
                    check.assertIsNotNone(kernel32.GetModuleHandleW(path))
                    if barrier is not None:
                        barrier.wait(timeout=60)
                    output = runner(x)[0] if packaged else runner.run([x])[0]
                    del runner
                    gc.collect()
                    check.assertIsNone(kernel32.GetModuleHandleW(path))
                    os.remove(path)
                    check.assertFalse(os.path.exists(path))
                    check.assertEqual(output, expected)
                    return output, expected

            if concurrent:
                with ThreadPoolExecutor(max_workers=4) as executor:
                    results = list(executor.map(run_once, range(16)))
            else:
                results = [run_once(index) for index in range(16)]
            for output, expected in results:
                check.assertEqual(output, expected)
            print("16 model DLLs unloaded and deleted; surviving outputs OK", flush=True)
            """
        )
        env = {**os.environ, "OMP_NUM_THREADS": "2", "OMP_DYNAMIC": "FALSE"}
        env.pop("OMP_WAIT_POLICY", None)
        env.pop("KMP_BLOCKTIME", None)
        returncode, stdout, stderr = self._run_isolated(
            [
                sys.executable,
                "-I",
                "-X",
                "faulthandler",
                "-u",
                "-c",
                source,
                model_path,
                str(size),
                str(int(concurrent)),
                str(int(packaged)),
            ],
            env,
        )
        self.assertEqual(
            returncode,
            0,
            msg=f"Child return code {returncode}\n{stdout}\n{stderr}",
        )
        self.assertIn(
            "16 model DLLs unloaded and deleted; surviving outputs OK", stdout
        )


instantiate_parametrized_tests(AOTIRuntimeLifetimeTests)

if __name__ == "__main__":
    run_tests()
