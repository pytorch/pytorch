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
    is_dtype_supported = CommonTemplate.is_dtype_supported
    common = check_model_gpu
    device = "mps"

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

    def test_pointwise_i0(self):
        self.common(torch.special.i0, (torch.rand(128, 128),), check_lowp=False)

    def test_pointwise_i1(self):
        self.common(torch.special.i1, (torch.rand(128, 128),), check_lowp=False)

    def test_pointwise_erf(self):
        self.common(torch.special.erf, (torch.rand(128, 128),), check_lowp=False)

    def test_pointwise_polygamma(self):
        self.common(
            torch.special.polygamma,
            (
                1,
                torch.rand(128, 128),
            ),
            check_lowp=False,
        )

    def test_pointwise_digamma(self):
        self.common(torch.special.digamma, (torch.rand(128, 128),), check_lowp=False)

    def test_pointwise_sinc(self):
        self.common(torch.special.sinc, (torch.rand(128, 128),), check_lowp=False)

    def test_pointwise_zeta(self):
        self.common(
            torch.special.zeta,
            (torch.rand(128, 128), torch.rand(128, 128)),
            check_lowp=False,
        )

    def test_broadcast(self):
        self.common(torch.add, (torch.rand(32, 1024), torch.rand(1024)))

    def test_inplace(self):
        def inc_(x):
            x += 1
            return x

        self.common(inc_, (torch.rand(1024),))

    # TODO(NS): Replace me with full test_prod when multi-stage reductions are implemented
    def test_prod(self):
        def fn(a):
            return a.prod(0), a.prod(1), a.prod()

        self.common(fn, (torch.rand((10, 10)),))


# Copy tests
for test_name in [
    "test_min_max_reduction",
    "test_add_const_int",
    "test_add_inplace_permuted",
    "test_addmm",
    "test_any",
    "test_arange5",
    "test_argmax_min_int32",
    "test_argmax_argmin2",
    "test_avg_pool2d5",
    "test_avg_pool2d8",
    "test_builtins_round",
    "test_builtins_round_float_ndigits_neg",
    "test_cat_empty",
    "test_cat_unbacked_empty_1d",
    "test_consecutive_split_cumprod",
    "test_consecutive_split_cumsum",
    "test_constant_pad_float64",
    "test_cumsum_inf",
    "test_custom_op_2",
    "test_div1",
    "test_div3",
    "test_erfinv",
    "test_floordiv",
    "test_full_truncation",
    "test_fmod",
    "test_fmod_zero_dim",
    "test_index_dynamic_shapes",
    "test_inf",
    "test_isinf",
    "test_isinf2",
    "test_layer_norm",
    "test_lgamma",
    "test_linear_float64",
    "test_log_fp64",
    "test_low_memory_max_pool",
    "test_max_min",
    "test_max_pool2d2",
    "test_min_max_reduction_nan",
    "test_nan_to_num",
    "test_pow2",
    "test_randint_int64_mod",
    "test_randn_generator",
    "test_remainder",
    "test_remove_no_ops",
    "test_reflection_pad2d",
    "test_rsqrt",
    "test_scalar_cpu_tensor_arg",
    "test_scalar_output",
    "test_setitem_with_int_parameter",
    "test_signbit",
    "test_silu",
    "test_slice_scatter4",
    "test_softmax",
    "test_sort",
    "test_split_cumsum",
    "test_sum_int",
    "test_sum_keepdims",
    "test_tanh",
    "test_view_as_complex",
    "test_view_on_aliased",
    "test_views3",
    "test_views6",
    "test_views7",
    "test_zero_dim_reductions",
]:
    setattr(MPSBasicTests, test_name, getattr(CommonTemplate, test_name))

instantiate_parametrized_tests(MPSBasicTests)

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if torch.backends.mps.is_available():
        run_tests(needs="filelock")
