# Owner(s): ["module: inductor"]
import sys
import unittest

import torch._dynamo
from torch._inductor import config
from torch.testing._internal.common_utils import TestCase as TorchTestCase
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


class TestCppWrapper(TorchTestCase):
    pass


def make_test_case(name, device="cpu"):
    test_name = f"{name}_{device}"

    @config.patch(cpp_wrapper=True, search_autotune_cache=False)
    def fn(self):
        tests = test_torchinductor.CpuTests()
        tests.setUpClass()
        tests.setUp()
        try:
            func = getattr(tests, test_name)
            assert callable(func), "not a callable"
            code = test_torchinductor.run_and_get_cpp_code(func, [])
            self.assertEqual("load_inline" in code, True)
        finally:
            tests.tearDown()
            tests.tearDownClass()

    fn.__name__ = test_name
    setattr(TestCppWrapper, test_name, fn)


for name in [
    "test_as_strided",  # buffer reuse
    "test_bitwise",  # int32
    "test_bmm1",
    "test_bmm2",
    "test_cat",  # alias
    "test_linear1",
    "test_linear2",
    "test_lowmem_dropout1",  # None as output
    "test_mm_views",
    "test_profiler_mark_wrapper_call",
    "test_reduction1",  # Reduction
    "test_relu",  # multiple inputs
    "test_silu",  # single input, single output
    "test_sum_dtype",  # float64
    "test_sum_int",  # bool, int64, int8, uint8
    "test_transpose",  # multiple outputs, buffer clear
]:
    make_test_case(name)

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if HAS_CPU and not torch.backends.mps.is_available():
        run_tests(needs="filelock")
