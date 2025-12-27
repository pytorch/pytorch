# Owner(s): ["module: dynamo"]

import subprocess
import sys

from torch.testing._internal.common_utils import run_tests, TestCase


class TestLazyImports(TestCase):
    def test_convert_frame_import_does_not_load_rarely_used_modules(self):
        code = """
import sys
import torch._dynamo.convert_frame

rarely_used_modules = ['cProfile', 'pstats', 'gc']
loaded = [mod for mod in rarely_used_modules if mod in sys.modules]
if loaded:
    print(f"FAIL: eagerly loaded: {loaded}")
    sys.exit(1)
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"stdout: {result.stdout}\nstderr: {result.stderr}",
        )


if __name__ == "__main__":
    run_tests()
