# Owner(s): ["module: inductor"]
import functools
import textwrap
import unittest

import torch
import torch._dynamo
import torch._inductor.utils
from torch._dynamo.test_minifier_common import MinifierTestBase
from torch.testing._internal.common_utils import IS_JETSON, IS_MACOS, TEST_WITH_ASAN

_HAS_TRITON = torch._inductor.utils.has_triton()
requires_cuda = functools.partial(unittest.skipIf, not _HAS_TRITON, "requires cuda")


# These minifier tests are slow, because they must be run in separate
# subprocesses
class MinifierIsolateTests(MinifierTestBase):
    def _test_after_aot_runtime_error(self, device, bug_type):
        run_code = textwrap.dedent(
            f"""\
            @torch.compile()
            def inner(x):
                x = torch.relu(x)
                x = torch.cos(x)
                return x

            inner(torch.randn(20, 20).to("{device}"))
        """
        )
        patch_code = self._gen_codegen_fn_patch_code(device, bug_type)
        self.assertIsNotNone(patch_code)

        # These must isolate because they crash the process
        test_proc, _, repro_proc = self._run_full_test_nocode(
            run_code, "aot", 3, patch_code, isolate=True
        )

        self.assertNotIn("CompilerError", test_proc.stderr.decode("utf-8"))

        self.assertEqual(test_proc.returncode, repro_proc.returncode)
        self.assertNotEqual(test_proc.returncode, 0)

    @unittest.skipIf(IS_JETSON, "Fails on Jetson")
    def test_after_aot_cpu_runtime_error(self):
        self._test_after_aot_runtime_error("cpu", "runtime_error")

    @requires_cuda()
    def test_after_aot_cuda_runtime_error(self):
        self._test_after_aot_runtime_error("cuda", "runtime_error")


if __name__ == "__main__":
    import sys

    from torch._dynamo.test_case import run_tests

    # Skip CI tests on mac since CPU inductor does not seem to work due to C++ compile errors,
    # also skip on ASAN due to https://github.com/pytorch/pytorch/issues/98262
    # also skip on Py 3.11+ since unhandled exceptions can cause segfaults
    if not IS_MACOS and not TEST_WITH_ASAN and sys.version_info < (3, 11):
        run_tests()
