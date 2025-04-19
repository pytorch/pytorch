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
from torch._inductor.compile_fx import _InProcessFxCompile, FxCompile, FxCompileMode
from torch._inductor.test_case import TestCase
from torch.testing._internal.common_utils import TEST_WITH_ASAN
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CPU, HAS_GPU


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
import inductor.test_torchinductor  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
from inductor.test_torchinductor import (  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
    check_model,
    check_model_gpu,
    copy_tests,
    TestFailure,
)


importlib.import_module("filelock")

# xfail by default, set is_skip=True to skip
test_failures = {
    # TypeError: cannot pickle 'generator' object
    "test_layer_norm": TestFailure(("cpu", "cuda"), is_skip=True),
}


class TestSubprocess(TestCase):
    def setUp(self):
        torch._dynamo.reset()
        FxCompile._reset_stats()

        TestCase.setUp(self)

        self._stack = contextlib.ExitStack()
        self._stack.enter_context(
            patch(
                "torch._inductor.compile_fx.fx_compile_mode",
                FxCompileMode.SUBPROCESS,
            )
        )

    def tearDown(self):
        # Check that the test didn't instigate an in-process compile - which
        # would mean that something about the fx graph failed to serialize. If
        # some tests are expected to fail then we should probably add a list of
        # expected failures here.
        self.assertEqual(
            FxCompile._compile_stats[type(_InProcessFxCompile)].codegen_and_compile, 0
        )
        self._stack.close()
        TestCase.tearDown(self)
        torch._dynamo.reset()


if HAS_CPU:

    class CpuTests(TestSubprocess):
        common = check_model
        device = "cpu"

    copy_tests(
        inductor.test_torchinductor.CommonTemplate, CpuTests, "cpu", test_failures
    )

if HAS_GPU and not TEST_WITH_ASAN:

    class GPUTests(TestSubprocess):
        common = check_model_gpu
        device = GPU_TYPE

    copy_tests(
        inductor.test_torchinductor.CommonTemplate, GPUTests, GPU_TYPE, test_failures
    )


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_CPU or HAS_GPU:
        run_tests(needs="filelock")
