# Owner(s): ["module: inductor"]
import unittest

import torch
import torch._inductor.config as inductor_config
from torch._dynamo.test_minifier_common import MinifierTestBase
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    HardwareClassification,
    IS_JETSON,
    IS_MACOS,
    skipIfWindows,
    TEST_WITH_ASAN,
    TEST_WITH_ROCM,
)
from torch.utils._triton import get_triton_version, has_triton


skipIfRocmWithoutDebugAsserts = unittest.skipIf(
    TEST_WITH_ROCM and get_triton_version() < (3, 7),
    "requires Triton 3.7+ on ROCm for debug asserts",
)


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


class MinifierIsolateTestsOnlyCPU(MinifierIsolateTests):
    hw_classification = HardwareClassification.CPU

    @unittest.skipIf(IS_JETSON, "Fails on Jetson")
    @inductor_config.patch("cpp.inject_relu_bug_TESTING_ONLY", "runtime_error")
    @skipIfWindows(
        msg="Build Failed: fatal error C1083: Cannot open include file: 'Python.h': No such file or directory"
    )
    def test_after_aot_cpu_runtime_error(self, device):
        self._test_after_aot_runtime_error(device, "")


class MinifierIsolateTestsACCELERATOR(MinifierIsolateTests):
    hw_classification = HardwareClassification.ACCELERATOR

    @skipIfRocmWithoutDebugAsserts
    @unittest.skipUnless(has_triton(), "Triton not available")
    @inductor_config.patch("triton.inject_relu_bug_TESTING_ONLY", "runtime_error")
    def test_after_aot_gpu_runtime_error(self, device):
        # CUDA's __assertfail surfaces through PyTorch as "device-side assert";
        # ROCm's Triton AMD lowering prints the injected assertion text before trapping.
        device_type = torch.device(device).type
        expected_error = (
            "injected assert fail"
            if device_type == "xpu" or TEST_WITH_ROCM
            else "device-side assert"
        )
        self._test_after_aot_runtime_error(device, expected_error)


instantiate_device_type_tests(MinifierIsolateTestsOnlyCPU, globals(), only_for="cpu")
instantiate_device_type_tests(
    MinifierIsolateTestsACCELERATOR, globals(), except_for="cpu", allow_xpu=True
)


if __name__ == "__main__":
    import sys

    from torch._dynamo.test_case import run_tests

    # Skip CI tests on mac since CPU inductor does not seem to work due to C++ compile errors,
    # also skip on ASAN due to https://github.com/pytorch/pytorch/issues/98262
    # also skip on Py 3.11+ since unhandled exceptions can cause segfaults
    if not IS_MACOS and not TEST_WITH_ASAN and sys.version_info < (3, 11):
        run_tests()
