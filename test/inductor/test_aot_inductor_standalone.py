# Owner(s): ["module: inductor"]
import copy
import functools
import sys
import unittest

from torch._inductor import config
from torch._inductor.test_case import TestCase
from torch.testing._internal.common_utils import (
    IS_CI,
    IS_WINDOWS,
    skipIfRocm,
    skipIfXpu,
)
from torch.testing._internal.inductor_utils import GPU_TYPE
from torch.testing._internal.triton_utils import HAS_GPU


if IS_WINDOWS and IS_CI:
    sys.stderr.write(
        "Windows CI does not have necessary dependencies for test_torchinductor yet\n"
    )
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires sympy/functorch/filelock")

try:
    try:
        from .test_aot_inductor import (
            AOTInductorTestsTemplate,
            check_model,
            check_model_with_multiple_inputs,
            code_check_count,
        )
    except ImportError:
        from test_aot_inductor import (  # @manual
            AOTInductorTestsTemplate,
            check_model,
            check_model_with_multiple_inputs,
            code_check_count,
        )
except (unittest.SkipTest, ImportError):
    if __name__ == "__main__":
        sys.exit(0)
    raise


# Similar to copy_tests in test_torchinductor.py, but only takes a whitelist of tests
def copy_tests(my_cls, other_cls, suffix, whitelist):  # noqa: B902
    for name, value in my_cls.__dict__.items():
        if name.startswith("test_") and name in whitelist:
            # You cannot copy functions in Python, so we use closures here to
            # create objects with different ids. Otherwise, unittest.skip
            # would modify all methods sharing the same object id. Also, by
            # using a default argument, we create a copy instead of a
            # reference. Otherwise, we would lose access to the value.

            @functools.wraps(value)
            @config.patch(
                {
                    "aot_inductor.codegen_standalone": True,
                    "max_autotune_gemm_backends": "TRITON",
                    "max_autotune_conv_backends": "TRITON",
                }
            )
            @skipIfXpu
            @skipIfRocm
            def new_test(self, value=value):
                return value(self)

            # Copy __dict__ which may contain test metadata
            new_test.__dict__ = copy.deepcopy(value.__dict__)
            setattr(other_cls, f"{name}_{suffix}", new_test)

    # Special case convenience routine
    if hasattr(my_cls, "is_dtype_supported"):
        other_cls.is_dtype_supported = my_cls.is_dtype_supported


test_list_cpu = {
    # Need to sort out third-party library build issues, e.g. blas, sleef
}


class AOTInductorTestStandaloneCpu(TestCase):
    device = "cpu"
    device_type = "cpu"
    check_model = check_model
    check_model_with_multiple_inputs = check_model_with_multiple_inputs
    code_check_count = code_check_count
    allow_stack_allocation = False
    use_minimal_arrayref_interface = False


copy_tests(
    AOTInductorTestsTemplate,
    AOTInductorTestStandaloneCpu,
    "cpu_standalone",
    test_list_cpu,
)

test_list_gpu = {
    "test_cos",
}


@unittest.skipIf(sys.platform == "darwin", "No CUDA on MacOS")
class AOTInductorTestStandaloneGpu(TestCase):
    device = GPU_TYPE
    device_type = GPU_TYPE
    check_model = check_model
    check_model_with_multiple_inputs = check_model_with_multiple_inputs
    code_check_count = code_check_count
    allow_stack_allocation = False
    use_minimal_arrayref_interface = False


copy_tests(
    AOTInductorTestsTemplate,
    AOTInductorTestStandaloneGpu,
    f"{GPU_TYPE}_standalone",
    test_list_gpu,
)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_GPU:
        run_tests(needs="filelock")
