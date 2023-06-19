# Owner(s): ["module: inductor"]
import atexit
import os
import sys
import unittest
from collections import defaultdict
from enum import Enum
from functools import partial
from unittest.mock import patch

import torch

import torch._dynamo
from torch._dynamo.test_case import run_tests
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyNativeDeviceTypes,
    OpDTypes,
    ops,
    skipCPUIf,
    skipCUDAIf,
)
from torch.testing._internal.common_methods_invocations import op_db, skipOps
from torch.testing._internal.common_utils import (
    dtype_abbrs,
    IS_MACOS,
    IS_X86,
    skipCUDAMemoryLeakCheckIf,
    skipIfCrossRef,
    skipIfTorchDynamo,
    suppress_warnings,
    TEST_WITH_ASAN,
    TEST_WITH_ROCM,
    TestCase,
)
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA

try:
    try:
        from .test_torchinductor import check_model, check_model_cuda
    except ImportError:
        from test_torchinductor import check_model, check_model_cuda
except (unittest.SkipTest, ImportError) as e:
    sys.stderr.write(f"{type(e)}: {e}\n")
    if __name__ == "__main__":
        sys.exit(0)
    raise

bf16 = torch.bfloat16  # not tested
f64 = torch.float64
f32 = torch.float32
f16 = torch.float16
i8 = torch.int8  # not tested
i16 = torch.int16  # not tested
i32 = torch.int32
i64 = torch.int64
b8 = torch.bool
u8 = torch.uint8  # not tested

_ops = partial(
    ops, dtypes=OpDTypes.supported, allowed_dtypes=[f16, f32, f64, i32, i64, b8]
)

# Success forces pass; failure forces fail; skip unconditionally skips testing
ExpectedTestResult = Enum("ExpectedTestResult", ("SUCCESS", "XFAILURE", "SKIP"))

COLLECT_EXPECT = os.getenv("PYTORCH_COLLECT_EXPECT", "0") == "1"
FAIL_ON_SUCCESS = os.getenv("PYTORCH_FAIL_ON_SUCCESS", "1") == "1"
ALL_SAMPLES = os.getenv("PYTORCH_ALL_SAMPLES", "0") == "1"
START = os.getenv("PYTORCH_TEST_RANGE_START", None)
END = os.getenv("PYTORCH_TEST_RANGE_END", None)

if START is not None or END is not None:
    assert END is not None
    assert START is not None
    START = int(START)
    END = int(END)
    assert START < END
else:
    START = 0
    END = len(op_db)

seen_succeeded = defaultdict(dict)
seen_failed = defaultdict(dict)
failed_reasons = defaultdict(set)


def print_seen():
    expected_failures = defaultdict(list)

    def fmt_dtypes(dtypes):
        r = ", ".join(sorted(dtype_abbrs[d] for d in dtypes))
        return "{" + r + "}"

    def process(device_type):
        for op, failed_dtypes in seen_failed[device_type].items():
            succeeded_dtypes = seen_succeeded.get(op, set())
            expected_failures_dtypes = failed_dtypes - succeeded_dtypes

            reasons = ""
            if failed_reasons[op]:
                reasons = "  # " + ", ".join(sorted(failed_reasons[op]))
            if expected_failures_dtypes:
                expected_failures[device_type].append(
                    f'   "{op}": {fmt_dtypes(expected_failures_dtypes)},{reasons}'
                )

        expected_failures[device_type].sort()
        nl = "\n"
        print(
            f"""
inductor_expected_failures_single_sample[\"{device_type}\"] = {{
{nl.join(expected_failures[device_type])}
}}
"""
        )

    process("cpu")
    process("cuda")


if COLLECT_EXPECT:
    atexit.register(print_seen)

# Note, in these skip/xfail dictionaries use a string as the key
# for the default test, and a tuple of two strings for variants

inductor_skips = defaultdict(dict)


inductor_skips["cpu"] = {
    "linalg.ldl_solve": {b8, f16, f32, f64, i32, i64},  # segfault
    "linalg.ldl_factor": {f32, f64},  # flaky
    "__rdiv__": {b8, f16, f32, f64, i32, i64},  # flaky
    "nn.functional.cosine_embedding_loss": {b8},  # flaky
}

if IS_MACOS and IS_X86:
    inductor_skips["cpu"]["rsqrt"] = {b8, i32}

inductor_skips["cuda"] = {
    # Jiterator kernel is not expected to work with inductor
    "jiterator_2inputs_2outputs": {b8, f16, f32, f64, i32, i64},
    "jiterator_4inputs_with_extra_args": {b8, f16, f32, f64, i32, i64},
    "jiterator_binary": {b8, f16, f32, f64, i32, i64},
    "jiterator_binary_return_by_ref": {b8, f16, f32, f64, i32, i64},
    "jiterator_unary": {b8, f16, f32, f64, i32, i64},
    # flaky
    "nn.functional.cosine_embedding_loss": {b8},
    "native_batch_norm": {f16, f32, f64},
    "_native_batch_norm_legit": {f16, f32, f64},
}

if TEST_WITH_ROCM:
    # Tensors are not alike
    inductor_skips["cuda"]["logcumsumexp"] = {f32}

inductor_expected_failures_single_sample = defaultdict(dict)

inductor_expected_failures_single_sample["cpu"] = {
    "__getitem__": {b8, f16, f32, f64, i32, i64},
    "allclose": {f16, f32, f64},
    "amax": {f16},
    "amin": {f16},
    "angle": {f16, f32, f64},
    "argwhere": {b8, f16, f32, f64, i32, i64},
    "bernoulli": {f32, f64},
    "bincount": {i32, i64},
    "bucketize": {b8, f16, f32, f64, i32, i64},
    "cholesky": {f32, f64},
    "combinations": {b8, f16, f32, f64, i32, i64},
    "corrcoef": {f32, f64, i32, i64},
    "cov": {f32, f64, i32, i64},
    "equal": {b8, f16, f32, f64, i32, i64},
    "index_add": {f16},
    "index_reduce": {f16, f32, f64},
    "istft": {f32, f64},
    # Unsupported: data dependent operator: aten._local_scalar_dense.default
    "item": {b8, f16, f32, f64, i32, i64},
    "linalg.eig": {f32, f64},
    "linalg.eigh": {f32, f64},
    "linalg.eigvals": {f32, f64},
    "linalg.eigvalsh": {f32, f64},
    "linalg.lstsq": {f32, f64},
    # This pair of strings denotes a test variant
    ("linalg.lstsq", "grad_oriented"): {f32, f64},
    "masked.var": {f16},
    "masked_scatter": {f16, f32, f64},
    "masked_select": {b8, f16, f32, f64, i32, i64},
    ("max", "reduction_with_dim"): {b8},
    ("min", "reduction_with_dim"): {b8},
    "multinomial": {f32, f64},
    "nanquantile": {f32, f64},
    "nn.functional.avg_pool1d": {i64},
    "nn.functional.avg_pool2d": {i64},
    "nn.functional.adaptive_avg_pool2d": {f16},
    "nn.functional.ctc_loss": {f32, f64},
    "nn.functional.gaussian_nll_loss": {f16, f32, f64},
    "nn.functional.local_response_norm": {i64},
    "nn.functional.one_hot": {i64},
    "nn.functional.rrelu": {f32, f64},
    "nn.functional.triplet_margin_with_distance_loss": {f16, f32, f64, i32, i64},
    "nonzero": {b8, f16, f32, f64, i32, i64},
    "normal": {f16, f32, f64},
    ("normal", "number_mean"): {f16, f32, f64},
    "polar": {f32, f64},
    "quantile": {f32, f64},
    "rand_like": {f16, f32, f64},
    "randint_like": {f16, f32, f64, i32, i64},
    "randint": {f16, f32, f64, i32, i64},
    "randn_like": {f16, f32, f64},
    "repeat_interleave": {b8, f16, f32, f64, i32, i64},
    "scatter_add": {f16},
    ("scatter_reduce", "sum"): {f16},
    ("scatter_reduce", "prod"): {f16, f32, f64},
    ("_segment_reduce", "lengths"): {f16, f32, f64},
    "sparse.sampled_addmm": {f32, f64},
    ("sparse.mm", "reduce"): {bf16, f32, f64},
    "stft": {f32, f64},
    "svd": {f32, f64},
    "svd_lowrank": {f32, f64},
    "linalg.cond": {f32, f64},
    "linalg.svd": {f32, f64},
    "linalg.svdvals": {f32, f64},
    "linalg.matrix_rank": {f32, f64},
    "pca_lowrank": {f32, f64},
    "tensor_split": {b8, f16, f32, f64, i32, i64},
    "to_sparse": {f32, f64},
    # AssertionError: Tensor-likes are not close!
    "cauchy": {f16},
    "exponential": {f16},
    "geometric": {f16},
    "log_normal": {f16},
    ("normal", "in_place"): {f16, f32, f64},
    "uniform": {f16},
    "unique": {b8, f16, f32, f64, i32, i64},
    "unique_consecutive": {b8, f16, f32, f64, i32, i64},
    "var": {f16},
    "var_mean": {f16},
    "view_as_complex": {f16},
    "fft.fft": {b8, f16, f32, f64, i32, i64},
    "fft.fft2": {b8, f16, f32, f64, i32, i64},
    "fft.fftn": {b8, f16, f32, f64, i32, i64},
    "fft.hfft": {b8, f16, f32, f64, i32, i64},
    "fft.hfft2": {b8, f16, f32, f64, i32, i64},
    "fft.hfftn": {b8, f16, f32, f64, i32, i64},
    "fft.ifft": {f16, f32, f64, b8, i32, i64},
    "fft.ifft2": {b8, f16, f32, f64, i32, i64},
    "fft.ifftn": {b8, f16, f32, f64, i32, i64},
    "fft.ihfft": {f16, f32, f64, b8, i32, i64},
    "fft.ihfft2": {f16, f32, f64, b8, i32, i64},
    "fft.ihfftn": {f16, f32, f64, b8, i32, i64},
    "fft.irfft": {b8, f16, f32, f64, i32, i64},
    "fft.irfft2": {b8, f16, f32, f64, i32, i64},
    "fft.irfftn": {b8, f16, f32, f64, i32, i64},
    "fft.rfft": {f16, f32, f64, b8, i32, i64},
    "fft.rfft2": {b8, f16, f32, f64, i32, i64},
    "fft.rfftn": {b8, f16, f32, f64, i32, i64},
    # These return complex tensors
    "cdouble": {b8, i32, i64, f16, f32, f64},
    "cfloat": {b8, i32, i64, f16, f32, f64},
    "chalf": {b8, i32, i64, f16, f32, f64},
    "complex": {f16, f32, f64},
}


inductor_expected_failures_single_sample["cuda"] = {
    "__getitem__": {b8, f16, f32, f64, i32, i64},
    "__rdiv__": {b8, f16, f32, f64, i32, i64},
    "addr": {f16},
    "allclose": {f16, f32, f64},
    "angle": {f32, f64},
    "argwhere": {b8, f16, f32, f64, i32, i64},
    ("as_strided", "partial_views"): {b8, f16, f32, f64, i32, i64},
    "baddbmm": {f16},
    "bernoulli": {f16, f32, f64},
    "bincount": {i32, i64},
    "bucketize": {b8, f16, f32, f64, i32, i64},
    "cholesky": {f32, f64},
    "combinations": {b8, f16, f32, f64, i32, i64},
    "corrcoef": {f16, f32, f64, i32, i64},
    "cov": {f16, f32, f64, i32, i64},
    "equal": {b8, f16, f32, f64, i32, i64},
    "index_reduce": {f16, f32, f64},
    "istft": {f32, f64},
    # Unsupported: data dependent operator: aten._local_scalar_dense.default
    "item": {b8, f16, f32, f64, i32, i64},
    "linalg.eig": {f32, f64},
    "linalg.eigh": {f32, f64},
    "linalg.eigvals": {f32, f64},
    "linalg.eigvalsh": {f32, f64},
    "linalg.householder_product": {f32, f64},
    "linalg.lstsq": {f32, f64},
    ("linalg.lstsq", "grad_oriented"): {f32, f64},
    "masked_scatter": {f16, f32, f64},
    "masked_select": {b8, f16, f32, f64, i32, i64},
    ("max", "reduction_with_dim"): {b8},
    ("min", "reduction_with_dim"): {b8},
    "multinomial": {f16, f32, f64},
    "nn.functional.adaptive_avg_pool2d": {f16},
    "nn.functional.ctc_loss": {f32, f64},
    "nn.functional.grid_sample": {f16},
    "grid_sampler_2d": {f16},
    "nn.functional.gaussian_nll_loss": {f16, f32, f64},
    "nn.functional.one_hot": {i64},
    "nn.functional.rrelu": {f16, f32, f64},
    "nn.functional.triplet_margin_with_distance_loss": {f16, f32, f64, i32, i64},
    "nonzero": {b8, f16, f32, f64, i32, i64},
    "normal": {f16, f32, f64},
    ("normal", "number_mean"): {f16, f32, f64},
    "polar": {f32, f64},
    "rand_like": {f16, f32, f64},
    "randint_like": {f16, f32, f64, i32, i64},
    "randint": {f16, f32, f64, i32, i64},
    "randn_like": {f16, f32, f64},
    "repeat_interleave": {b8, f16, f32, f64, i32, i64},
    ("round", "decimals_3"): {f16},
    ("scatter_reduce", "prod"): {f16, f32, f64},
    ("_segment_reduce", "lengths"): {f16, f32, f64},
    "sparse.sampled_addmm": {f32, f64},
    ("std_mean", "unbiased"): {f16},
    "stft": {f32, f64},
    "tensor_split": {b8, f16, f32, f64, i32, i64},
    "to_sparse": {f16, f32, f64},
    # AssertionError: Tensor-likes are not close!
    "cauchy": {f16, f32, f64},
    "exponential": {f16, f32, f64},
    "geometric": {f16, f32, f64, i32, i64},
    ("normal", "in_place"): {f16, f32, f64},
    "log_normal": {f16, f32, f64},
    "uniform": {f16, f32, f64},
    "unique": {b8, f16, f32, f64, i32, i64},
    "unique_consecutive": {b8, f16, f32, f64, i32, i64},
    # AssertionError: Tensor-likes are not close!
    "nn.functional.triplet_margin_loss": {f16},
    # The following 3 tests fail on CUDA with AssertionError: expected size 5==5, stride 5==1 at dim=0
    # linalg._svd's return value has different strides on CUDA vs CPU which causes this
    # In test_meta.py there is a mechanism to skipping strides checks for some ops
    # (including _linalg_svd), possibly we should have something similar here
    "linalg.cond": {f32, f64},
    "linalg.svdvals": {f32, f64},
    "linalg.matrix_rank": {f32, f64},
    "linalg.svd": {f32, f64},
    "pca_lowrank": {f32, f64},
    "svd_lowrank": {f32, f64},
    "svd": {f32, f64},
    # AssertionError: Scalars are not close!
    "nn.functional.soft_margin_loss": {f16},
    "fft.fft": {b8, f16, f32, f64, i32, i64},
    "fft.fft2": {b8, f16, f32, f64, i32, i64},
    "fft.fftn": {b8, f16, f32, f64, i32, i64},
    "fft.hfft": {b8, f16, f32, f64, i32, i64},
    "fft.hfft2": {b8, f16, f32, f64, i32, i64},
    "fft.hfftn": {b8, f16, f32, f64, i32, i64},
    "fft.ifft": {f16, f32, f64, b8, i32, i64},
    "fft.ifft2": {b8, f16, f32, f64, i32, i64},
    "fft.ifftn": {b8, f16, f32, f64, i32, i64},
    "fft.ihfft": {f16, f32, f64, b8, i32, i64},
    "fft.ihfft2": {f16, f32, f64, b8, i32, i64},
    "fft.ihfftn": {f16, f32, f64, b8, i32, i64},
    "fft.irfft": {b8, f16, f32, f64, i32, i64},
    "fft.irfft2": {b8, f16, f32, f64, i32, i64},
    "fft.irfftn": {b8, f16, f32, f64, i32, i64},
    "fft.rfft": {f16, f32, f64, b8, i32, i64},
    "fft.rfft2": {b8, f16, f32, f64, i32, i64},
    "fft.rfftn": {b8, f16, f32, f64, i32, i64},
    # These return complex tensors
    "cdouble": {b8, i32, i64, f16, f32, f64},
    "cfloat": {b8, i32, i64, f16, f32, f64},
    "chalf": {b8, i32, i64, f16, f32, f64},
    "complex": {f16, f32, f64},
}


inductor_gradient_expected_failures_single_sample = defaultdict(dict)

inductor_gradient_expected_failures_single_sample["cuda"] = {
    "asin": {f16},
    "atanh": {f16, f32},
    "cumprod": {f16},
    "linalg.vector_norm": {f64, f64},
    "kron": {f16},
    "nanquantile": {f32, f64},
    "nn.functional.avg_pool2d": {f16, f32, f64},
    ("nn.functional.batch_norm", "without_cudnn"): {f16},
    "nn.functional.batch_norm": {f16},
    "nn.functional.cosine_similarity": {f16},
    "nn.functional.instance_norm": {f16},
    "nn.functional.normalize": {f16},
    "nn.functional.softsign": {f16},
    "nn.functional.local_response_norm": {f16},
    "outer": {f16},
    "quantile": {f32, f64},
}

if not TEST_WITH_ROCM:
    inductor_gradient_expected_failures_single_sample["cuda"]["tanh"] = {f16}
else:
    # aten.miopen_batch_norm is unsupported for lowering
    inductor_expected_failures_single_sample["cuda"]["nn.functional.batch_norm"] = {
        f16,
        f32,
    }
    inductor_expected_failures_single_sample["cuda"]["nn.functional.instance_norm"] = {
        f16,
        f32,
    }

inductor_should_fail_with_exception = defaultdict(dict)

inductor_should_fail_with_exception["cpu"] = {}


inductor_should_fail_with_exception["cuda"] = {
    "__rpow__": {
        i32: "Pow input must be floating point.",
        i64: "Pow input must be floating point.",
    }
}


def get_skips_and_xfails(from_dict, xfails=True):
    retval = set()
    for device, d in from_dict.items():
        for op, dtypes in d.items():
            if type(op) is tuple:
                op, variant_name = op
            else:
                variant_name = ""
            retval.add((op, variant_name, device, tuple(dtypes), xfails))
    return retval


# Note: if you get a "AssertionError: Couldn't find OpInfo for ..." error for an OpInfo you are sure
# exists, you might be trying to use a test variant and you need to replace, for example,
# "max.reduction_no_dim" with ("max", "reduction_no_dim") as the key of one of these dictionaries
test_skips_or_fails = (
    get_skips_and_xfails(inductor_skips, xfails=False)
    | get_skips_and_xfails(inductor_expected_failures_single_sample, xfails=True)
    | get_skips_and_xfails(
        inductor_gradient_expected_failures_single_sample, xfails=True
    )
)


def wrapper_set_seed(op, *args, **kwargs):
    """Wrapper to set seed manually for some functions like dropout
    See: https://github.com/pytorch/pytorch/pull/62315#issuecomment-896143189 for more details.
    """
    torch.manual_seed(42)
    return op(*args, **kwargs)


torch.testing._internal.common_methods_invocations.wrapper_set_seed = wrapper_set_seed

# This file does a global patch to `disable_global_flags()` - which we should not invoke in non testing cases.
torch._dynamo.variables.torch.tensor_dunder_fns.append(
    torch.testing._internal.common_utils.disable_functorch
)

# key can be either op_name, or (op_name, deivce_type), or (op_name, device_type, dtype)
inductor_override_kwargs = {
    # the return value of empty is undefined
    "empty": {"assert_equal": False},
    "empty_permuted": {"assert_equal": False},
    "empty_like": {"assert_equal": False},
    "new_empty": {"assert_equal": False},
    "empty_strided": {"assert_equal": False},
    "new_empty_strided": {"assert_equal": False},
    "randn": {"assert_equal": False},
    ("masked.softmin", "cuda", f16): {"atol": 1e-4, "rtol": 0.01},
    ("nn.functional.tanhshrink", "cuda", f16): {"atol": 3e-4, "rtol": 0.001},
    ("nn.functional.softmin", "cuda", f16): {"atol": 1e-4, "rtol": 0.01},
    ("special.log_ndtr", "cuda", f64): {"atol": 1e-6, "rtol": 1e-5},
    ("cummax", "cuda", f16): {"atol": 5e-4, "rtol": 0.002},
    ("softmax", "cuda", f16): {"atol": 1e-4, "rtol": 0.02},
    ("softmax", "cpu", f16): {"atol": 1e-4, "rtol": 0.02},
    ("_softmax_backward_data", "cuda", f16): {"atol": 0.008, "rtol": 0.002},
    "gradient": {"check_gradient": False},  # segfault on check_gradient
    # Following tests failed, and causing subsequent tests failing with unrecoverable CUDA error
    "linalg.solve_triangular": {"check_gradient": False},
    "linalg.lu_factor": {"check_gradient": False},
    "linalg.lu_factor_ex": {"check_gradient": False},
}

# Always test with all sample for following ops
inductor_all_samples = {
    "arange",
    "softmax.with_dtype",
    "index_add",
    "index_copy",
    "scatter_reduce.sum",
    "select_scatter",
    "squeeze",
    "unsqueeze",
    "sum",
    "amax",
    "amin",
    "all",
    "T",
    "H",
    "isinf",
    "isposinf",
    "isneginf",
    "nan_to_num",
    "mT",
    "mH",
    "rsub",
    "triu",
}


class TestInductorOpInfo(TestCase):
    check_model = check_model
    check_model_cuda = check_model_cuda

    @onlyNativeDeviceTypes
    @suppress_warnings
    @skipCUDAMemoryLeakCheckIf(
        True
    )  # inductor kernels failing this test intermittently
    @skipCUDAIf(not HAS_CUDA, "Skipped! Triton not found")
    @skipCPUIf(not HAS_CPU, "Skipped! Supported CPU compiler not found")
    @unittest.skipIf(TEST_WITH_ASAN, "Skipped under ASAN")
    @skipIfTorchDynamo("Test uses dynamo already")
    @skipIfCrossRef
    @_ops(op_db[START:END])
    @skipOps("TestInductorOpInfo", "test_comprehensive", test_skips_or_fails)
    @patch("torch._dynamo.config.raise_on_unsafe_aot_autograd", True)
    @torch._inductor.config.patch(
        {"implicit_fallbacks": False, "triton.autotune_pointwise": False}
    )
    def test_comprehensive(self, device, dtype, op):
        torch._dynamo.reset()
        with torch.no_grad():
            torch.cuda.empty_cache()
        op_name = op.name
        if op.variant_test_name:
            op_name += f".{op.variant_test_name}"

        device_type = torch.device(device).type

        assert device_type in ("cuda", "cpu")

        # with open("test_output.txt", "a") as f:
        #     print(f"CONSIDERING OP {op_name} on {device_type} with {dtype} |
        # {inductor_skips[device_type].get(op_name, set())}", flush=True, file=f)
        #     print(f"CONSIDERING OP {op_name} on {device_type} with {dtype} |
        # {inductor_skips[device_type].get(op_name, set())}", flush=True)
        if dtype in inductor_skips[device_type].get(op_name, set()):
            test_expect = ExpectedTestResult.SKIP
            # with open("test_output.txt", "a") as f:
            #     print(f"SKIPPING OP {op_name} on {device_type}", flush=True, file=f)
            #     print(f"SKIPPING OP {op_name} on {device_type}", flush=True)
        elif dtype in inductor_expected_failures_single_sample[device_type].get(
            op_name, set()
        ) or dtype in inductor_gradient_expected_failures_single_sample[
            device_type
        ].get(
            op_name, set()
        ):
            test_expect = ExpectedTestResult.XFAILURE
        else:
            test_expect = ExpectedTestResult.SUCCESS

        overridden_kwargs = {}
        if op_name in inductor_override_kwargs:
            overridden_kwargs = inductor_override_kwargs[op_name]
        elif (op_name, device_type) in inductor_override_kwargs:
            overridden_kwargs = inductor_override_kwargs[(op_name, device_type)]
        elif (op_name, device_type, dtype) in inductor_override_kwargs:
            overridden_kwargs = inductor_override_kwargs[(op_name, device_type, dtype)]

        func = op.get_op()

        def fn(*args, **kwargs):
            return func(*args, **kwargs)

        requires_grad = (
            op.supports_autograd
            and dtype in op.supported_backward_dtypes(device_type)
            # TODO: OpInfo really ought to error out for this case, but it's
            # not exercised in test_ops_gradients atm.  The problem is not
            # complex32 per-se (which is supported by data movement only ops)
            # but that when we do backwards we expect other ops like add to work
            and not dtype == torch.complex32
        )
        samples = op.sample_inputs(device, dtype, requires_grad=requires_grad)

        if op_name not in inductor_all_samples and not ALL_SAMPLES:
            if isinstance(samples, (list, tuple)):
                samples = [samples[0]]
            else:
                samples = [next(samples)]

        try:
            for sample_input in samples:
                args = [sample_input.input] + list(sample_input.args)
                kwargs = sample_input.kwargs
                # UNCOMMENT TO DEBUG SEGFAULTS
                # with open("test_output.txt", "a") as f:
                #     print(f"RUNNING OP {op_name} on {device_type} with {dtype}", flush=True, file=f)
                #     print(f"RUNNING OP {op_name} on {device_type} with {dtype}", flush=True)
                if device_type == "cuda":
                    # opinfo test case have already place the input on the correct device
                    # so we don't need do additional copy by setting copy_to_cuda=False
                    adjusted_kwargs = {
                        "check_lowp": False,
                        "nopython": True,
                        "copy_to_cuda": False,
                        "reference_in_float": False,
                        "check_gradient": requires_grad,
                    }
                    adjusted_kwargs.update(overridden_kwargs)
                    self.check_model_cuda(
                        fn,
                        args,
                        kwargs,
                        **adjusted_kwargs,
                    )
                elif device_type == "cpu":
                    adjusted_kwargs = {
                        "check_lowp": False,
                        "nopython": True,
                        # skip checking gradient on CPU for now
                        "check_gradient": False,
                    }
                    adjusted_kwargs.update(overridden_kwargs)

                    self.check_model(
                        fn,
                        args,
                        kwargs,
                        **adjusted_kwargs,
                    )

        except Exception as e:
            if test_expect is ExpectedTestResult.XFAILURE:
                raise e

            seen_failed[device_type].setdefault(op_name, set()).add(dtype)

            if COLLECT_EXPECT:
                return

            known_failure = False
            if dtype in inductor_should_fail_with_exception[device_type].get(
                op_name, set()
            ):
                failure = inductor_should_fail_with_exception[device_type][op_name][
                    dtype
                ]
                if failure in str(e):
                    known_failure = True
            if not known_failure:
                raise e

        # with open("test_output.txt", "a") as f:
        #     print(f"SUCCEEDED OP {op_name} on {device_type} with {dtype}", flush=True, file=f)
        seen_succeeded[device_type].setdefault(op_name, set()).add(dtype)

        if test_expect is ExpectedTestResult.XFAILURE and not COLLECT_EXPECT:
            if FAIL_ON_SUCCESS:
                raise RuntimeError(
                    f"unexpected success {op_name}, {dtype}, {device_type}"
                )


instantiate_device_type_tests(TestInductorOpInfo, globals())

if __name__ == "__main__":
    run_tests()
