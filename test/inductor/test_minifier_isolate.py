# Owner(s): ["module: inductor"]
import unittest

from torch._dynamo.test_minifier_common import MinifierTestBase
from torch._inductor.utils import with_device_backend
from torch.testing._internal.common_utils import (
    IS_JETSON,
    IS_MACOS,
    skipIfWindows,
    TEST_WITH_ASAN,
    TEST_WITH_ROCM,
)
from torch.testing._internal.inductor_utils import (
    requires_triton,
    TRITON_TYPE,
    try_patch_inductor_backend_config,
)
from torch.utils._triton import get_triton_version


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

    @unittest.skipIf(IS_JETSON, "Fails on Jetson")
    @skipIfWindows(
        msg="Build Failed: fatal error C1083: Cannot open include file: 'Python.h': No such file or directory"
    )
    def test_after_aot_cpp_runtime_error(self):
        with (
            with_device_backend("cpp", "cpu"),
            try_patch_inductor_backend_config(
                "cpu", "inject_relu_bug_TESTING_ONLY", "runtime_error"
            ),
        ):
            self._test_after_aot_runtime_error("cpu", "")

    @skipIfRocmWithoutDebugAsserts
    @requires_triton()
    def test_after_aot_triton_runtime_error(self):
        # CUDA's __assertfail surfaces through PyTorch as "device-side assert";
        # ROCm's Triton AMD lowering prints the injected assertion text before trapping.
        expected_errors = {
            "cuda": "injected assert fail"
            if TEST_WITH_ROCM
            else "device-side assert",
            "xpu": "injected assert fail",
        }
        if TRITON_TYPE not in expected_errors:
            raise unittest.SkipTest(
                f"Unknown Triton runtime assert message for {TRITON_TYPE}"
            )
        with (
            with_device_backend("triton", TRITON_TYPE),
            try_patch_inductor_backend_config(
                TRITON_TYPE, "inject_relu_bug_TESTING_ONLY", "runtime_error"
            ),
        ):
            self._test_after_aot_runtime_error(
                TRITON_TYPE, expected_errors[TRITON_TYPE]
            )


if __name__ == "__main__":
    import sys

    from torch._dynamo.test_case import run_tests

    # Skip CI tests on mac since CPU inductor does not seem to work due to C++ compile errors,
    # also skip on ASAN due to https://github.com/pytorch/pytorch/issues/98262
    # also skip on Py 3.11+ since unhandled exceptions can cause segfaults
    if not IS_MACOS and not TEST_WITH_ASAN and sys.version_info < (3, 11):
        run_tests()
