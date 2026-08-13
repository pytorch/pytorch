# Owner(s): ["module: inductor"]

import functools
import inspect
import os
import re
import subprocess
import sys

import torch
import torch._inductor.async_compile
from torch._inductor.codecache import PyCodeCache
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_device_type import (
    Capability,
    instantiate_device_type_tests,
    onlyAccelerator,
    requires_capabilities,
)
from torch.testing._internal.common_utils import HardwareClassification
from torch.utils._triton import has_triton


class TestTritonWrapper(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    def setUp(self):
        super().setUp()
        PyCodeCache.cache_clear()

    def get_compiled_module(self):
        compiled_module = None
        for v in PyCodeCache.modules:
            if hasattr(v, "benchmark_compiled_module"):
                self.assertTrue(
                    compiled_module is None, "Found multiple compiled modules"
                )
                compiled_module = v

        self.assertTrue(compiled_module is not None)
        return compiled_module

    @requires_capabilities(Capability.lib.triton)
    @onlyAccelerator
    def test_wrapper_using_gpu_seed(self, device):
        """
        Make sure the subprocess.check_output does not throw.
        """

        @torch.compile
        def f(x, y):
            # dropout will result in usage of cuda_seed
            z = torch.nn.functional.dropout(x, 0.5)
            return z + y

        N = 10
        x = torch.rand(N).to(device=device)
        y = torch.rand(N).to(device=device)
        out = f(x, y)  # noqa: F841
        compiled_module = self.get_compiled_module()
        # to make sure the subprocess runs on the exact same path as the parent process
        # we augment the PYTHONPATH env var
        augmented_pp = ":".join(sys.path)
        if os.environ.get("PYTHONPATH"):
            augmented_pp = f"{os.environ.get('PYTHONPATH')}:{augmented_pp}"
        # now run the compiled module in subprocess and check its output
        bench_out = subprocess.check_output(
            f"{sys.executable} {compiled_module.__file__}".split(),
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONPATH": augmented_pp},
        ).decode()

        self.assertTrue(len(bench_out) > 0)

    @requires_capabilities(Capability.lib.triton)
    @onlyAccelerator
    def test_get_args_and_benchmark_compiled_module(self, device):
        @torch.compile
        def f(x, y):
            return x * y + x

        N = 10
        x = torch.rand(N).to(device=device)
        y = torch.rand(N).to(device=device)
        f(x, y)

        compiled_module = self.get_compiled_module()

        # Verify get_args function exists and is callable
        self.assertTrue(hasattr(compiled_module, "get_args"))
        self.assertTrue(callable(compiled_module.get_args))

        self.assertTrue(hasattr(compiled_module, "benchmark_compiled_module"))
        sig = inspect.signature(compiled_module.benchmark_compiled_module)
        self.assertIn("args", sig.parameters)

        args = compiled_module.get_args()
        self.assertIsInstance(args, list)
        self.assertTrue(len(args) > 0)

        # Verify that running the compiled module as a subprocess works
        augmented_pp = ":".join(sys.path)
        if os.environ.get("PYTHONPATH"):
            augmented_pp = f"{os.environ.get('PYTHONPATH')}:{augmented_pp}"
        bench_out = subprocess.check_output(
            f"{sys.executable} {compiled_module.__file__}".split(),
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONPATH": augmented_pp},
        ).decode()
        self.assertTrue(len(bench_out) > 0)

    @requires_capabilities(Capability.lib.triton)
    @onlyAccelerator
    def test_get_args_preserves_aliased_inputs(self, device):
        @torch.compile
        def f(x, y, empty_bool, empty_long):
            return x + y, empty_bool.logical_not(), empty_long + 1

        base = torch.arange(64, device=device, dtype=torch.long)
        x = torch.as_strided(base, (3, 4), (4, 1), 2)
        y = torch.as_strided(base, (3, 4), (4, 1), 10)
        empty_bool = torch.empty((0, 3), device=device, dtype=torch.bool)
        empty_long = torch.empty((0, 2), device=device, dtype=torch.long)
        f(x, y, empty_bool, empty_long)

        compiled_module = self.get_compiled_module()
        get_args_src = inspect.getsource(compiled_module.get_args)
        shared_storage_names = re.findall(
            r"(_shared_storage_\d+) = rand_strided\(",
            get_args_src,
        )
        self.assertEqual(len(shared_storage_names), 1, get_args_src)

        shared_storage = shared_storage_names[0]
        aliased_view_lines = re.findall(
            rf"^\s+\w+ = torch\.as_strided\({shared_storage}, .*$",
            get_args_src,
            re.MULTILINE,
        )
        self.assertEqual(len(aliased_view_lines), 2, get_args_src)

        args = compiled_module.get_args()
        recreated_x, recreated_y, _, _ = args

        self.assertEqual(
            recreated_x.untyped_storage().data_ptr(),
            recreated_y.untyped_storage().data_ptr(),
        )
        self.assertEqual(len(args), 4)


_TRITON_WRAPPER_MTIA_TESTS = {
    name: test
    for name, test in vars(TestTritonWrapper).items()
    if name.startswith("test") and callable(test)
}
_TRITON_WRAPPER_CLASS = TestTritonWrapper
instantiate_device_type_tests(
    TestTritonWrapper,
    globals(),
    except_for=("hpu",),
    allow_xpu=True,
)


if torch.mtia.is_available() and has_triton():

    class TestTritonWrapperMTIA(_TRITON_WRAPPER_CLASS):
        pass

    for name, test in _TRITON_WRAPPER_MTIA_TESTS.items():
        test = test.__wrapped__.__wrapped__

        @functools.wraps(test)
        def mtia_test(self, test=test):
            return test(self, "mtia")

        setattr(TestTritonWrapperMTIA, name, mtia_test)


if __name__ == "__main__":
    run_tests()
