# Owner(s): ["module: inductor"]
import subprocess
import sys

from torch._inductor.test_case import run_tests, TestCase


class TestCompileFxImport(TestCase):
    def test_compile_fx_import_time(self):
        code = """
import time
start = time.perf_counter()
import torch._inductor.compile_fx
elapsed = time.perf_counter() - start
print(elapsed)
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, f"Import failed: {result.stderr}")
        elapsed = float(result.stdout.strip())
        self.assertLess(
            elapsed,
            2.0,
            f"torch._inductor.compile_fx import took {elapsed:.2f}s, should be <2.0s",
        )

    def test_lazy_imports_not_loaded_at_import_time(self):
        code = """
import sys
for mod in list(sys.modules.keys()):
    if 'torch' in mod:
        del sys.modules[mod]

import torch._inductor.compile_fx

lazy_modules = [
    'torch._inductor.triton_bundler',
    'torch._inductor.decomposition',
    'torch.fx.passes.fake_tensor_prop',
    'functorch.compile',
    'torch._inductor.async_compile',
    'torch.monitor',
    'torch._functorch._aot_autograd.subclass_parametrization',
    'torch._inductor.distributed_autotune',
]

loaded = []
for mod in lazy_modules:
    if mod in sys.modules:
        loaded.append(mod)

if loaded:
    print('LOADED:' + ','.join(loaded))
else:
    print('OK')
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, f"Check failed: {result.stderr}")
        output = result.stdout.strip()
        self.assertEqual(
            output,
            "OK",
            f"Some modules were loaded at import time that should be lazy: {output}",
        )


if __name__ == "__main__":
    run_tests()
