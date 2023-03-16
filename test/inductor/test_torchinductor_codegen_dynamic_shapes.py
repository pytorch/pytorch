# Owner(s): ["module: inductor"]
import importlib
import os
import sys
import unittest

from torch.testing._internal.common_utils import (
    IS_CI,
    IS_WINDOWS,
    TEST_WITH_ASAN,
    TEST_WITH_ROCM,
    TestCase,
)
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA

if IS_WINDOWS and IS_CI:
    sys.stderr.write(
        "Windows CI does not have necessary dependencies for test_torchinductor_codegen_dynamic_shapes yet\n"
    )
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires sympy/functorch/filelock")

importlib.import_module("filelock")

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
# Access tests classes via a module import or they'll run as part of this file
import inductor.test_torchinductor_dynamic_shapes as inductor_tests
from inductor.test_torchinductor_codegen import check_codegen


if HAS_CPU:

    class DynamicShapesCodegenCpuTests(inductor_tests.DynamicShapesCpuTests):
        maxDiff = None

        def common(self: TestCase, model, example_inputs, kwargs=None):
            return check_codegen(
                self=self,
                model=model,
                example_inputs=example_inputs,
                kwargs=kwargs,
                is_cpp_code=True,
            )


if HAS_CUDA and not TEST_WITH_ASAN:

    class DynamicShapesCodegenCudaTests(inductor_tests.DynamicShapesCudaTests):
        maxDiff = None

        def common(self: TestCase, model, example_inputs, kwargs=None):
            return check_codegen(
                self=self,
                model=model,
                example_inputs=example_inputs,
                kwargs=kwargs,
                is_cpp_code=False,
            )


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if (HAS_CPU or HAS_CUDA) and not TEST_WITH_ROCM:
        run_tests(needs="filelock")
