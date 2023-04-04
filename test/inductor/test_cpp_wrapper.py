# Owner(s): ["module: inductor"]
import sys
import unittest
from typing import NamedTuple

import torch._dynamo
from torch._inductor import config
from torch.testing._internal.common_utils import IS_MACOS, TestCase as TorchTestCase
from torch.testing._internal.inductor_utils import HAS_CPU

try:
    try:
        from . import test_torchinductor
    except ImportError:
        import test_torchinductor
except unittest.SkipTest:
    if __name__ == "__main__":
        sys.exit(0)
    raise


class CppWrapperTemplate:
    pass


class TestCppWrapper(TorchTestCase):
    device = "cpu"


def make_test_case(name, device, tests):
    test_name = f"{name}_{device}" if device else name

    @config.patch(cpp_wrapper=True, search_autotune_cache=False)
    def fn(self):
        tests.setUpClass()
        tests.setUp()
        try:
            func = getattr(tests, test_name)
            assert callable(func), "not a callable"
            code = test_torchinductor.run_and_get_cpp_code(func)
            self.assertEqual("load_inline" in code, True)
        finally:
            tests.tearDown()
            tests.tearDownClass()

    fn.__name__ = test_name
    setattr(CppWrapperTemplate, test_name, fn)


class BaseTest(NamedTuple):
    name: str
    device: str = "cpu"
    tests: TorchTestCase = test_torchinductor.CpuTests()


for item in [
    BaseTest("test_as_strided"),  # buffer reuse
    BaseTest("test_bitwise"),  # int32
    BaseTest("test_bmm1"),
    BaseTest("test_bmm2"),
    BaseTest("test_cat"),  # alias
    BaseTest("test_int_div", "", test_torchinductor.CPUReproTests()),
    BaseTest("test_linear1"),
    BaseTest("test_linear2"),
    BaseTest("test_lowmem_dropout1"),  # None as output
    BaseTest("test_mm_views"),
    BaseTest("test_profiler_mark_wrapper_call"),
    BaseTest("test_reduction1"),  # Reduction
    BaseTest("test_relu"),  # multiple inputs
    BaseTest("test_scalar_input"),
    BaseTest("test_silu"),  # single input, single output
    BaseTest("test_sum_dtype"),  # float64
    BaseTest("test_sum_int"),  # bool, int64, int8, uint8
    BaseTest("test_transpose"),  # multiple outputs, buffer clear
]:
    make_test_case(item.name, item.device, item.tests)


test_torchinductor.copy_tests(CppWrapperTemplate, TestCppWrapper, "cpp_wrapper")

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if HAS_CPU and not torch.backends.mps.is_available() and not IS_MACOS:
        run_tests(needs="filelock")
