# Owner(s): ["module: inductor"]
import importlib
import os
import sys
import unittest

import torch
from torch._inductor.compile_fx import compile_fx
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
from inductor.test_torchinductor import (
    CommonTemplate,
    copy_tests,
    run_and_get_cpp_code,
    run_and_get_triton_code,
)
from inductor.test_torchinductor_dynamic_shapes import make_dynamic_cls


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
        for_loop_found = False
        lines = code.split("\n")
        for line in lines:
            if "for(" in line:
                for_loop_found = True
                self.assertTrue(
                    "ks" in line,
                    msg=f"Failed to find dynamic for loop variable\n{code}",
                )
        self.assertTrue(for_loop_found, f"Failed to find for loop\n{code}")
    else:
        code = run_and_get_triton_code(run, *example_inputs, **kwargs)
        # XXX: remove
        print(code)
        raise NotImplementedError("TBD: triton checks")

    assert called, "Ran graph without calling compile_fx"

    torch._dynamo.reset()


test_skips = {
    # The following tests do not support dynamic shapes yet:
    "test_cpp_wrapper_dynamic_shapes": ("cpu",),
    "test_cudnn_rnn_dynamic_shapes": ("cuda",),
    "test_kwargs_dynamic_shapes": ("cpu",),
    # test_roi_align uses torchvision, which doesn't work with dynamic shapes
    "test_roi_align_dynamic_shapes": ("cpu", "cuda"),
}


DynamicShapesCodegenCommonTemplate = make_dynamic_cls(CommonTemplate)


if HAS_CPU:

    class DynamicShapesCodegenCpuTests(TestCase):
        maxDiff = None
        device = "cpu"

        def common(self: TestCase, model, example_inputs, kwargs=None, **_rest):
            return check_codegen(
                self=self,
                model=model,
                example_inputs=example_inputs,
                kwargs=kwargs,
                is_cpp_code=True,
            )

    copy_tests(
        DynamicShapesCodegenCommonTemplate,
        DynamicShapesCodegenCpuTests,
        "cpu",
        test_skips,
    )


if HAS_CUDA and not TEST_WITH_ASAN:

    class DynamicShapesCodegenCudaTests(TestCase):
        maxDiff = None
        device = "cuda"

        def common(self: TestCase, model, example_inputs, kwargs=None, **_rest):
            return check_codegen(
                self=self,
                model=model,
                example_inputs=example_inputs,
                kwargs=kwargs,
                is_cpp_code=False,
            )

    copy_tests(
        DynamicShapesCodegenCommonTemplate,
        DynamicShapesCodegenCudaTests,
        "cuda",
        test_skips,
    )


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if (HAS_CPU or HAS_CUDA) and not TEST_WITH_ROCM:
        run_tests(needs="filelock")
