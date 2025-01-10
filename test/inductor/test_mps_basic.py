# Owner(s): ["module: mps"]
import importlib
import os
import sys

import torch
from torch.testing import make_tensor
from torch.testing._internal.common_dtype import get_all_dtypes
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    MACOS_VERSION,
    parametrize,
)


MPS_UNSUPPORTED_TYPES = [torch.double, torch.cdouble] + (
    [torch.bfloat16] if MACOS_VERSION < 14.0 else []
)
MPS_DTYPES = [t for t in get_all_dtypes() if t not in MPS_UNSUPPORTED_TYPES]

importlib.import_module("filelock")

pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

from inductor.test_torchinductor import (  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
    check_model_gpu,
    CommonTemplate,
    TestCase,
)


# TODO: Remove this file.
# This tests basic MPS compile functionality


class MPSBasicTests(TestCase):
    common = check_model_gpu
    device = "mps"

    test_add_const_int = CommonTemplate.test_add_const_int
    test_add_inplace_permuted_mps = CommonTemplate.test_add_inplace_permuted
    test_addmm = CommonTemplate.test_addmm
    test_cat_empty = CommonTemplate.test_cat_empty
    test_floordiv = CommonTemplate.test_floordiv
    test_fmod = CommonTemplate.test_fmod
    test_fmod_zero_dim = CommonTemplate.test_fmod_zero_dim
    test_inf = CommonTemplate.test_inf
    test_isinf = CommonTemplate.test_isinf
    test_isinf2 = CommonTemplate.test_isinf2
    test_low_memory_max_pool = CommonTemplate.test_low_memory_max_pool
    test_max_min = CommonTemplate.test_max_min
    test_max_pool2d2 = CommonTemplate.test_max_pool2d2
    test_nan_to_num = CommonTemplate.test_nan_to_num
    test_remainder = CommonTemplate.test_remainder
    test_rsqrt = CommonTemplate.test_rsqrt
    test_signbit = CommonTemplate.test_signbit
    test_tanh = CommonTemplate.test_tanh
    test_view_as_complex = CommonTemplate.test_view_as_complex
    test_views6 = CommonTemplate.test_views6
    test_zero_dim_reductions = CommonTemplate.test_zero_dim_reductions

    @parametrize("dtype", MPS_DTYPES)
    def test_add(self, dtype):
        self.common(
            lambda a, b: a + b,
            (
                make_tensor(1024, dtype=dtype, device=self.device),
                make_tensor(1024, dtype=dtype, device=self.device),
            ),
            check_lowp=False,
        )

    def test_log(self):
        self.common(lambda x: x.log(), (torch.rand(1024),))

    def test_acos(self):
        self.common(lambda x: x.acos(), (torch.rand(1024),))

    def test_atanh(self):
        self.common(lambda x: x.atanh(), (torch.rand(1024),))

    def test_floor(self):
        self.common(lambda x: x.floor(), (torch.rand(1024),))

    def test_sign(self):
        self.common(lambda x: x.sign(), (torch.rand(1024),))

    def test_sliced_input(self):
        self.common(
            lambda x: x[:, ::2].sin() + x[:, 1::2].cos(), (torch.rand(32, 1024),)
        )

    def test_where(self):
        def foo(x):
            rc = x.abs().sqrt()
            rc[x < 0] = -5
            return rc

        self.common(foo, (torch.rand(1024),))

    @parametrize("dtype", MPS_DTYPES)
    def test_cast(self, dtype):
        self.common(lambda a: a.to(dtype), (torch.rand(1024),))

    def test_broadcast(self):
        self.common(torch.add, (torch.rand(32, 1024), torch.rand(1024)))

    def test_inplace(self):
        def inc_(x):
            x += 1
            return x

        self.common(inc_, (torch.rand(1024),))


instantiate_parametrized_tests(MPSBasicTests)

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if torch.backends.mps.is_available():
        run_tests(needs="filelock")
