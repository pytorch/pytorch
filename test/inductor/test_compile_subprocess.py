# Owner(s): ["module: fx"]

#
# Tests compiling the inductor tests in a subprocess.
#

import contextlib
import importlib
import os
import sys
from unittest.mock import patch

import torch
import torch.library
from torch._dynamo.testing import make_test_cls_with_patches
from torch._inductor.test_case import TestCase
from torch.testing._internal.common_utils import TEST_WITH_ASAN
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CPU, HAS_GPU


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from inductor.test_torchinductor import (  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
    check_model,
    check_model_gpu,
    CommonTemplate,
    copy_tests,
    TestFailure,
)


importlib.import_module("filelock")

# xfail by default, set is_skip=True to skip
test_failures = {
    # TypeError: cannot pickle 'generator' object
    "test_layer_norm": TestFailure(("cpu", "cuda"), is_skip=True),
}


def make_test_cls(cls):
    return make_test_cls_with_patches(
        cls,
        "Subprocess",
        "",
        (
            torch._inductor.compile_fx,
            "fx_compile_mode",
            torch._inductor.compile_fx.FxCompileMode.SUBPROCESS,
        ),
    )


CommonTemplate = make_test_cls(CommonTemplate)


if HAS_CPU:

    class CpuTests(TestCase):
        common = check_model
        device = "cpu"

    copy_tests(CommonTemplate, CpuTests, "cpu", test_failures)

if HAS_GPU and not TEST_WITH_ASAN:

    class GpuTests(TestCase):
        common = check_model_gpu
        device = GPU_TYPE

    copy_tests(CommonTemplate, GpuTests, GPU_TYPE, test_failures)


class TestSubprocess(TestCase):
    def setUp(self):
        torch._dynamo.reset()
        TestCase.setUp(self)

        self._stack = contextlib.ExitStack()
        self._stack.enter_context(
            patch(
                "torch._inductor.compile_fx.fx_compile_mode",
                torch._inductor.compile_fx.FxCompileMode.SERIALIZE,
            )
        )

    def tearDown(self):
        self._stack.close()
        TestCase.tearDown(self)
        torch._dynamo.reset()


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_CPU or HAS_GPU:
        run_tests(needs="filelock")
