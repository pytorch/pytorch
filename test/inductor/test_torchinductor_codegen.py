# Owner(s): ["module: inductor"]
import importlib
import os
import sys
import unittest

import torch
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
        "Windows CI does not have necessary dependencies for test_torchinductor_codegen yet\n"
    )
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires sympy/functorch/filelock")

importlib.import_module("filelock")

from torch._inductor.compile_fx import compile_fx

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
# Access tests classes via a module import or they'll run as part of this file
import inductor.test_torchinductor as inductor_tests
from inductor.test_torchinductor import run_and_get_cpp_code, run_and_get_triton_code


def check_codegen(
    self: TestCase,
    model,
    example_inputs,
    kwargs=None,
    *,
    is_cpp_code: bool,
):
    kwargs = kwargs or {}

    if is_cpp_code is False:
        if hasattr(model, "to"):
            model = model.to("cuda")

        def copy_fn(x):
            # preserve strides of the input on the device
            if not isinstance(x, torch.Tensor):
                return x
            return torch.empty_strided(
                x.size(), x.stride(), device="cuda", dtype=x.dtype
            ).copy_(x)

        example_inputs = tuple(copy_fn(x) for x in example_inputs)

    torch._dynamo.reset()

    called = False

    def compile_fx_wrapper(model_, example_inputs_):
        nonlocal called
        called = True
        return compile_fx(model_, example_inputs_)

    def run(*ex, **kwargs):
        return model(*ex, **kwargs)

    run = torch._dynamo.optimize(compile_fx_wrapper, nopython=True)(run)

    if is_cpp_code:
        code = run_and_get_cpp_code(run, *example_inputs, **kwargs)
    else:
        code = run_and_get_triton_code(run, *example_inputs, **kwargs)

    assert called, "Ran graph without calling compile_fx"

    # Check that the output code matches
    # Run test with EXPECTTEST_ACCEPT=1 to update the expect file
    self.assertExpected(code)

    torch._dynamo.reset()


if HAS_CPU:

    class CodegenCpuTests(inductor_tests.CpuTests):
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

    class CodegenCudaTests(inductor_tests.CudaTests):
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
