# Owner(s): ["module: inductor"]
import functools
import unittest

import torch
import torch._dynamo
import torch._inductor.config as inductor_config
import torch._inductor.utils
from torch._dynamo.test_minifier_common import MinifierTestBase
from torch.testing._internal.common_utils import (
    IS_JETSON,
    IS_MACOS,
    skipIfRocm,
    TEST_WITH_ASAN,
)

_HAS_TRITON = torch._inductor.utils.has_triton()
requires_cuda = functools.partial(unittest.skipIf, not _HAS_TRITON, "requires cuda")


# These minifier tests are slow, because they must be run in separate
# subprocesses
class MinifierIsolateTests(MinifierTestBase):
    def _test_after_aot_runtime_error(self, device, expected_error):
        run_code = f"""\
@torch.compile()
def inner(x):
    x = torch.relu(x)
    x = torch.cos(x)
    return x

inner(torch.randn(2, 2).to("{device}"))
"""
        # These must isolate because they crash the process
        self._run_full_test(run_code, "aot", expected_error, isolate=True)

    @unittest.skipIf(IS_JETSON, "Fails on Jetson")
    @inductor_config.patch("cpp.inject_relu_bug_TESTING_ONLY", "runtime_error")
    def test_after_aot_cpu_runtime_error(self):
        self._test_after_aot_runtime_error("cpu", "")

    @skipIfRocm
    @requires_cuda()
    @inductor_config.patch("triton.inject_relu_bug_TESTING_ONLY", "runtime_error")
    def test_after_aot_cuda_runtime_error(self):
        self._test_after_aot_runtime_error("cuda", "device-side assert")


if __name__ == "__main__":
    import sys

    from torch._dynamo.test_case import run_tests

    # Skip CI tests on mac since CPU inductor does not seem to work due to C++ compile errors,
    # also skip on ASAN due to https://github.com/pytorch/pytorch/issues/98262
    # also skip on Py 3.11+ since unhandled exceptions can cause segfaults
    if not IS_MACOS and not TEST_WITH_ASAN and sys.version_info < (3, 11):
        run_tests()
