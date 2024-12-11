# Owner(s): ["module: inductor"]
import atexit
import contextlib
import functools
import os
import sys
import unittest
from collections import defaultdict
from enum import Enum
from functools import partial
from unittest.mock import patch

import torch
from torch._dispatch.python import enable_python_dispatcher
from torch._inductor.test_case import run_tests, TestCase
from torch._subclasses.fake_tensor import (
    DataDependentOutputException,
    DynamicOutputShapeException,
    FakeTensorMode,
)
from torch.testing._internal.common_cuda import SM80OrLater
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyNativeDeviceTypes,
    OpDTypes,
    ops,
    skipCPUIf,
    skipCUDAIf,
    skipXPUIf,
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
    TEST_MKL,
    TEST_WITH_ASAN,
    TEST_WITH_ROCM,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CPU, HAS_CUDA, HAS_XPU
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map


try:
    try:
        from .test_torchinductor import check_model, check_model_gpu
    except ImportError:
        from test_torchinductor import (  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
            check_model,
            check_model_gpu,
        )
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
u8 = torch.uint8  # not tested except upsampling and interpolate ops
u16 = torch.uint16  # not tested
u32 = torch.uint32  # not tested
u64 = torch.uint64  # not tested

_ops = partial(
    ops,
    dtypes=OpDTypes.supported,
    allowed_dtypes=[f16, f32, f64, i32, i64, b8, u8, u16, u32, u64],
)

# Success forces pass; failure forces fail; skip unconditionally skips testing
ExpectedTestResult = Enum("ExpectedTestResult", ("SUCCESS", "XFAILURE", "SKIP"))

COLLECT_EXPECT = os.getenv("PYTORCH_COLLECT_EXPECT", "0") == "1"
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

seen_failed = defaultdict(set)
failed_reasons = defaultdict(set)


def print_seen():
    expected_failures = defaultdict(list)

    def fmt_dtypes(dtypes):
        r = ", ".join(sorted(dtype_abbrs[d] for d in dtypes))
        return "{" + r + "}"

    def sort_key(kv):
        k, v = kv
        device_type, op = k
        if isinstance(op, tuple):
            return op
        else:
            return op, ""

    for (device_type, op), failed_dtypes in sorted(seen_failed.items(), key=sort_key):
        key = device_type, op
        reasons = ""
        if failed_reasons[key]:

            def maybe_truncate(x, length=80):
                x = str(x).replace("\n", " ")

                idx = x.find("\\n")
                if idx >= 0:
                    x = f"{x[:idx]}..."
                if len(x) > length:
                    return f"{x[:length - 3]}..."
                return x

            reasons = sorted(set(map(maybe_truncate, failed_reasons[key])))
            reasons = "  # " + ", ".join(reasons)

        if failed_dtypes:

            def format_op(op):
                if isinstance(op, tuple):
                    return f'("{op[0]}", "{op[1]}")'
                else:
                    return f'"{op}"'

            expected_failures[device_type].append(
                f"    {format_op(op)}: {fmt_dtypes(failed_dtypes)},{reasons}"
            )

    for device_type in ("cpu", GPU_TYPE):
        expected_failures[device_type]
        nl = "\n"
        print(
            f"""
inductor_expected_failures_single_sample[\"{device_type}\"] = {{
{nl.join(expected_failures[device_type])}
}}
"""
        )


if COLLECT_EXPECT:
    atexit.register(print_seen)

# Note, in these skip/xfail dictionaries use a string as the key
# for the default test, and a tuple of two strings for variants

inductor_skips = defaultdict(dict)


inductor_skips["cpu"] = {
    "linalg.ldl_factor": {f32, f64},  # flaky
    "nn.functional.cosine_embedding_loss": {b8},  # flaky
    ("index_reduce", "prod"): {f16},  # flaky
    ("index_reduce", "mean"): {f16},  # flaky
}

if IS_MACOS and IS_X86:
    inductor_skips["cpu"]["rsqrt"] = {b8, i32}
    inductor_skips["cpu"]["nn.functional.multi_margin_loss"] = {
        b8,
        f16,
        f32,
        f64,
        i32,
        i64,
    }

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
    "_batch_norm_with_update": {f16, f32, f64},
}

if not SM80OrLater:
    inductor_skips["cuda"]["bfloat16"] = {b8, f16, f32, f64, i32, i64}

if TEST_WITH_ROCM:
    # Tensors are not alike
    inductor_skips["cuda"]["logcumsumexp"] = {f32}
    inductor_skips["cuda"]["special.modified_bessel_i1"] = {f64}

inductor_skips["xpu"] = {}

inductor_expected_failures_single_sample = defaultdict(dict)

inductor_expected_failures_single_sample["cpu"] = {
    "_softmax_backward_data": {
        f16
    },  # half_to_float is only valid for the CUDA implementation
    "_upsample_bilinear2d_aa": {f32, f64},
    "cholesky": {f32, f64},
    "complex": {f16},
    "resize_": {b8, f16, f32, f64, i32, i64},
    "resize_as_": {b8, f16, f32, f64, i32, i64},
    "histc": {f16},
    "multinomial": {f16, f32, f64},
    "nn.functional.avg_pool1d": {i64},
    "nn.functional.avg_pool2d": {i64},
    "nn.functional.avg_pool3d": {i64},
    "nn.functional.local_response_norm": {i64},
    "nonzero_static": {b8, f16, f32, f64, i32, i64},
    ("normal", "in_place"): {f16, f32, f64},
    ("normal", "number_mean"): {f16, f32, f64},
    "normal": {f16, f32, f64},
    ("sparse.mm", "reduce"): {f32, f64, f16},
    "sparse.sampled_addmm": {f32, f64},
    "to_sparse": {
        f32,
        f64,
    },  # NYI: could not find kernel for aten.view.default at dispatch key DispatchKey.SparseCPU
    "view_as_complex": {f16},
}


inductor_expected_failures_single_sample["cuda"] = {
    "_upsample_bilinear2d_aa": {f16, f32, f64},
    "cholesky": {f32, f64},
    "multinomial": {f16, f32, f64},
    ("normal", "in_place"): {f16, f32, f64},
    ("normal", "number_mean"): {f16, f32, f64},
    "normal": {f16, f32, f64},
    "sparse.sampled_addmm": {f32, f64},
    "torch.ops.aten._flash_attention_forward": {f16},
    "torch.ops.aten._efficient_attention_forward": {f16, f32},
    "to_sparse": {
        f16,
        f32,
        f64,
    },  # NYI: could not find kernel for aten.view.default at dispatch key DispatchKey.SparseCUDA
}

inductor_expected_failures_single_sample["xpu"] = {
    "_upsample_bilinear2d_aa": {f16, f32, f64},
    "cholesky": {f32, f64},
    "multinomial": {f16, f32, f64},
    ("normal", "in_place"): {f16, f32, f64},
    ("normal", "number_mean"): {f16, f32, f64},
    "normal": {f16, f32, f64},
    "sparse.sampled_addmm": {f32, f64},
    "tan": {f16},
    "torch.ops.aten._flash_attention_forward": {f16},
    "torch.ops.aten._efficient_attention_forward": {f16, f32},
    "to_sparse": {f32, f64},
    "linalg.eig": {f32, f64},
    "linalg.eigvals": {f32, f64},
    # Double and complex datatype matmul is not supported in oneDNN
    "__rmatmul__": {f64},
    ("addmm", "decomposed"): {f64},
    "addr": {f64},
    "baddbmm": {f64},
    "bmm": {f64},
    "byte": {f16, f32},
    "cdist": {f64},
    "corrcoef": {f64},
    "cov": {f64},
    "einsum": {f64},
    "inner": {f64},
    "linalg.cholesky_ex": {f64},
    "linalg.ldl_factor_ex": {f64},
    "linalg.ldl_factor": {f64},
    "linalg.ldl_solve": {f64},
    "linalg.matrix_power": {f64},
    "linalg.multi_dot": {f64},
    "matmul": {f64},
    "mm": {f64},
    "mv": {f64},
    "nn.functional.bilinear": {f64},
    "nn.functional.linear": {f64},
    "pca_lowrank": {f64},
    "svd_lowrank": {f64},
    "tensordot": {f64},
    "triangular_solve": {f64},
    "svd": {f64},
    "qr": {f64},
    "pinverse": {f64},
    "ormqr": {f64},
    ("norm", "nuc"): {f64},
    "lu": {f64},
    "lu_solve": {f64},
    "logdet": {f64},
    "linalg.tensorsolve": {f64},
    "linalg.tensorinv": {f64},
    "linalg.svdvals": {f64},
    "linalg.svd": {f64},
    "linalg.solve": {f64},
    "linalg.solve_triangular": {f64},
    "linalg.solve_ex": {f64},
    "linalg.slogdet": {f64},
    "linalg.qr": {f64},
    "linalg.pinv": {f64},
    ("linalg.pinv", "hermitian"): {f64},
    ("linalg.pinv", "singular"): {f64},
    "linalg.norm": {f64},
    ("linalg.norm", "subgradients_at_zero"): {f64},
    "linalg.matrix_rank": {f64},
    ("linalg.matrix_rank", "hermitian"): {f64},
    "linalg.matrix_norm": {f64},
    "linalg.lu": {f64},
    "linalg.lu_solve": {f64},
    "linalg.lu_factor": {f64},
    "linalg.lu_factor_ex": {f64},
    "linalg.lstsq": {f64},
    ("linalg.lstsq", "grad_oriented"): {f64},
    "linalg.inv": {f64},
    "linalg.inv_ex": {f64},
    "linalg.householder_product": {f64},
    "linalg.eigvalsh": {f64},
    "linalg.eigh": {f64},
    "linalg.det": {f64},
    "cholesky_solve": {f64},
    "cholesky_inverse": {f64},
    # could not create a primitive
    "addbmm": {f64},
    "addmm": {f64},
    "addmv": {f64},
    # could not create a primitive descriptor for
    # a deconvolution forward propagation primitive
    "nn.functional.conv_transpose2d": {f32, f64},
    "nn.functional.conv_transpose3d": {f32, f64},
    # not implemented for 'Half'
    "sort": {b8},
    "argsort": {b8},
}


# intentionally not handled
intentionally_not_handled = {
    "resize_": {b8, f16, f32, f64, i32, i64},
    "resize_as_": {b8, f16, f32, f64, i32, i64},
}
# This is only fixed when this config is set
# We should eventually always turn it on
import torch._functorch.config as functorch_config


if not functorch_config.view_replay_for_aliased_outputs:
    intentionally_not_handled['("as_strided", "partial_views")'] = {
        b8,
        f16,
        f32,
        f64,
        i32,
        i64,
    }

inductor_expected_failures_single_sample["cuda"].update(intentionally_not_handled)
inductor_expected_failures_single_sample["xpu"].update(intentionally_not_handled)


inductor_gradient_expected_failures_single_sample = defaultdict(dict)

inductor_gradient_expected_failures_single_sample["cuda"] = {}
inductor_gradient_expected_failures_single_sample["xpu"] = {}

if not TEST_MKL:
    inductor_expected_failures_single_sample["cpu"].update({})

inductor_should_fail_with_exception = defaultdict(dict)
inductor_should_fail_with_exception["cpu"] = {}
inductor_should_fail_with_exception["cuda"] = {}
inductor_should_fail_with_exception["xpu"] = {}


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


def wrapper_noop_set_seed(op, *args, **kwargs):
    return op(*args, **kwargs)


torch.testing._internal.common_methods_invocations.wrapper_set_seed = (
    wrapper_noop_set_seed
)

# key can be either op_name, or (op_name, dtype)
inductor_override_kwargs = defaultdict(dict)

inductor_override_kwargs["cpu"] = {
    # the return value of empty is undefined
    "empty": {"assert_equal": False},
    "empty_permuted": {"assert_equal": False},
    "empty_like": {"assert_equal": False},
    "new_empty": {"assert_equal": False},
    "empty_strided": {"assert_equal": False},
    "new_empty_strided": {"assert_equal": False},
    "randn": {"assert_equal": False},
    ("nn.functional.multilabel_soft_margin_loss", f16): {
        "atol": 3e-4,
        "rtol": 0.002,
    },
    ("nn.functional.triplet_margin_with_distance_loss", f16): {
        "atol": 3e-4,
        "rtol": 0.003,
    },
    ("softmax", f16): {"atol": 1e-4, "rtol": 0.02},
    ("polygamma.polygamma_n_0", f32): {"atol": 1e-3, "rtol": 1e-4},
    ("polygamma.polygamma_n_1", f32): {"atol": 1e-3, "rtol": 1e-4},
    ("polygamma.polygamma_n_2", f32): {"atol": 1e-3, "rtol": 1e-4},
    ("polygamma.polygamma_n_3", f32): {"atol": 1e-3, "rtol": 1e-4},
    ("polygamma.polygamma_n_4", f32): {"atol": 1e-3, "rtol": 1e-4},
    ("special.polygamma.special_polygamma_n_0", f32): {
        "atol": 1e-3,
        "rtol": 1e-4,
    },
    ("_unsafe_masked_index_put_accumulate", f16): {"atol": 1e-4, "rtol": 0.01},
    # Following tests are failing with strict comparision but atol=1 is acceptable due roundings errors
    ("nn.functional.interpolate.bilinear", u8): {"atol": 1, "rtol": 0},
    ("nn.functional.upsample_bilinear", u8): {"atol": 1, "rtol": 0},
    ("nn.functional.interpolate.bicubic", u8): {"atol": 1, "rtol": 0},
    # High atol due to precision loss
    ("nn.functional.interpolate.bicubic", f32): {"atol": 5e-3, "rtol": 0},
}

inductor_override_kwargs["cuda"] = {
    # the return value of empty is undefined
    "empty": {"assert_equal": False},
    "empty_permuted": {"assert_equal": False},
    "empty_like": {"assert_equal": False},
    "new_empty": {"assert_equal": False},
    "empty_strided": {"assert_equal": False},
    "new_empty_strided": {"assert_equal": False},
    "randn": {"assert_equal": False},
    ("cross", f16): {"reference_in_float": True},
    ("linalg.cross", f16): {"reference_in_float": True},
    ("addr", f16): {"reference_in_float": True},
    ("baddbmm", f16): {"atol": 2e-3, "rtol": 0.002},  # decomp affects accuracy
    ("angle", f64): {"reference_in_float": True},
    ("asin", f16): {"reference_in_float": True},
    ("atanh", f16): {"reference_in_float": True},
    "cauchy": {"reference_in_float": True},
    ("cummax", f16): {"atol": 5e-4, "rtol": 0.002},
    ("cumsum", f16): {"reference_in_float": True},
    "cumprod": {"reference_in_float": True, "atol": 7e-5, "rtol": 0.002},
    "logcumsumexp": {"grad_atol": 8e-4, "grad_rtol": 0.001},
    "exponential": {"reference_in_float": True},
    "geometric": {"reference_in_float": True},
    ("kron", f16): {"reference_in_float": True},
    "log_normal": {"reference_in_float": True},
    ("masked.softmin", f16): {"atol": 1e-4, "rtol": 0.01},
    ("nn.functional.batch_norm", f16): {"reference_in_float": True},
    ("nn.functional.batch_norm.without_cudnn", f16): {"reference_in_float": True},
    ("nn.functional.cosine_similarity", f16): {"reference_in_float": True},
    ("nn.functional.instance_norm", f16): {"reference_in_float": True},
    ("nn.functional.local_response_norm", f16): {"reference_in_float": True},
    ("nn.functional.normalize", f16): {"atol": 1e-3, "rtol": 0.05},
    ("nn.functional.rms_norm", f16): {"reference_in_float": True},
    ("nn.functional.soft_margin_loss", f16): {"reference_in_float": True},
    ("nn.functional.softmin", f16): {"atol": 1e-4, "rtol": 0.01},
    ("nn.functional.softsign", f16): {"reference_in_float": True},
    ("nn.functional.tanhshrink", f16): {"atol": 3e-4, "rtol": 0.001},
    ("outer", f16): {"reference_in_float": True},
    ("round.decimals_3", f16): {"reference_in_float": True},
    ("nn.functional.triplet_margin_loss", f16): {"atol": 1e-4, "rtol": 0.02},
    ("nn.functional.triplet_margin_with_distance_loss", f16): {
        "atol": 1e-4,
        "rtol": 0.02,
    },
    ("sinc", f16): {"atol": 0.008, "rtol": 0.002},
    ("torch.ops.aten._safe_softmax.default", f16): {"atol": 5e-4, "rtol": 0.02},
    ("softmax", f16): {"atol": 1e-4, "rtol": 0.02},
    ("_softmax_backward_data", f16): {"atol": 0.008, "rtol": 0.002},
    ("special.log_ndtr", f64): {"atol": 1e-6, "rtol": 1e-5},
    ("std_mean.unbiased", f16): {"reference_in_float": True},
    "uniform": {"reference_in_float": True},
    ("_unsafe_masked_index_put_accumulate", f16): {"atol": 1e-4, "rtol": 0.01},
    # High atol due to precision loss
    ("nn.functional.interpolate.bilinear", f64): {"atol": 5e-4, "rtol": 0},
    ("nn.functional.upsample_bilinear", f64): {"atol": 5e-4, "rtol": 0},
    ("nn.functional.interpolate.bicubic", f64): {"atol": 1e-3, "rtol": 0},
    # Unreasonably high atol requirement:
    ("index_reduce.mean", f16): {"check_gradient": False},
    ("index_reduce.mean", f32): {"check_gradient": False},
    ("index_reduce.mean", f64): {"check_gradient": False},
    # Gradient contains non-finite entries:
    ("index_reduce.amin", f64): {"check_gradient": False},
    ("index_reduce.amin", f32): {"check_gradient": False},
    ("index_reduce.amin", f16): {"check_gradient": False},
    ("index_reduce.amax", f64): {"check_gradient": False},
    ("index_reduce.amax", f32): {"check_gradient": False},
    ("index_reduce.amax", f16): {"check_gradient": False},
    ("tanh", f16): {"atol": 1e-4, "rtol": 1e-2},
}

inductor_override_kwargs["xpu"] = {
    # the return value of empty is undefined
    "empty": {"assert_equal": False},
    "empty_permuted": {"assert_equal": False},
    "empty_like": {"assert_equal": False},
    "new_empty": {"assert_equal": False},
    "empty_strided": {"assert_equal": False},
    "new_empty_strided": {"assert_equal": False},
    "randn": {"assert_equal": False},
    # XPU
    ("cross", f16): {"reference_in_float": True},
    ("addr", f16): {"reference_in_float": True},
    ("baddbmm", f16): {"atol": 2e-3, "rtol": 0.002},  # decomp affects accuracy
    ("angle", f64): {"reference_in_float": True},
    ("asin", f16): {"reference_in_float": True},
    ("atanh", f16): {"reference_in_float": True},
    "cauchy": {"reference_in_float": True},
    ("cummax", f16): {"atol": 5e-4, "rtol": 0.002},
    ("cumsum", f16): {"reference_in_float": True},
    "cumprod": {"reference_in_float": True, "atol": 7e-5, "rtol": 0.002},
    ("dot", f16): {"atol": 1e-5, "rtol": 0.002},
    "logcumsumexp": {
        "atol": 5e-5,
        "rtol": 0.005,
        "grad_atol": 8e-4,
        "grad_rtol": 0.001,
    },
    "exponential": {"reference_in_float": True},
    "geometric": {"reference_in_float": True},
    ("kron", f16): {"reference_in_float": True},
    ("linalg.cross", f16): {"reference_in_float": True},
    ("linalg.vecdot", f16): {"atol": 1e-5, "rtol": 2e-2},
    "log_normal": {"reference_in_float": True},
    ("logsumexp", f16): {"atol": 1e-5, "rtol": 1e-2},
    ("masked.cumprod", f16): {"atol": 1e-5, "rtol": 5e-2},
    ("masked.cumsum", f16): {"atol": 1e-5, "rtol": 5e-3},
    ("masked.softmin", f16): {"atol": 1e-4, "rtol": 0.01},
    ("masked.softmax", f16): {"atol": 2e-4, "rtol": 0.01},
    ("masked.var", f16): {"atol": 2e-5, "rtol": 5e-3},
    ("native_batch_norm", f64): {"atol": 1e-7, "rtol": 1e-5},
    ("_native_batch_norm_legit", f64): {"atol": 1e-7, "rtol": 5e-6},
    ("_batch_norm_with_update", f64): {"atol": 1e-7, "rtol": 1e-6},
    ("native_layer_norm", f16): {"atol": 5e-3, "rtol": 5e-3},
    ("native_layer_norm", f32): {"atol": 5e-3, "rtol": 5e-3},
    ("nn.functional.batch_norm", f16): {"reference_in_float": True},
    ("nn.functional.batch_norm", f64): {"atol": 1e-6, "rtol": 1e-6},
    ("nn.functional.batch_norm.without_cudnn", f16): {"reference_in_float": True},
    ("nn.functional.conv1d", f16): {"atol": 1e-5, "rtol": 6e-3},
    ("nn.functional.conv3d", f16): {"atol": 1e-5, "rtol": 2e-3},
    ("nn.functional.conv_transpose2d", f16): {"atol": 1e-5, "rtol": 2e-3},
    ("nn.functional.conv_transpose3d", f16): {"atol": 1e-5, "rtol": 5e-3},
    ("nn.functional.cosine_embedding_loss", f16): {"atol": 1e-5, "rtol": 2e-3},
    ("nn.functional.cosine_similarity", f16): {
        "reference_in_float": True,
        "atol": 1e-5,
        "rtol": 5e-3,
    },
    ("nn.functional.instance_norm", f16): {"reference_in_float": True},
    ("nn.functional.instance_norm", f64): {"atol": 1e-6, "rtol": 1e-6},
    ("nn.functional.layer_norm", f16): {"atol": 5e-3, "rtol": 2e-3},
    ("nn.functional.layer_norm", f32): {"atol": 5e-5, "rtol": 2e-3},
    ("nn.functional.local_response_norm", f16): {"reference_in_float": True},
    ("nn.functional.multilabel_soft_margin_loss", f16): {
        "atol": 3e-4,
        "rtol": 2e-3,
    },
    ("nn.functional.normalize", f16): {"atol": 1e-3, "rtol": 0.05},
    ("nn.functional.rms_norm", f16): {"reference_in_float": True},
    ("nn.functional.soft_margin_loss", f16): {"reference_in_float": True},
    ("nn.functional.softmin", f16): {"atol": 1e-4, "rtol": 0.01},
    ("nn.functional.softsign", f16): {
        "reference_in_float": True,
        "atol": 1e-5,
        "rtol": 0.005,
    },
    ("nn.functional.tanhshrink", f16): {"atol": 3e-4, "rtol": 0.001},
    ("outer", f16): {"reference_in_float": True},
    ("round.decimals_3", f16): {"reference_in_float": True},
    ("nn.functional.triplet_margin_loss", f16): {"atol": 1e-4, "rtol": 0.02},
    ("nn.functional.triplet_margin_with_distance_loss", f16): {
        "atol": 1e-4,
        "rtol": 0.02,
    },
    ("remainder", f16): {"atol": 1e-4, "rtol": 0.005},
    ("nn.functional.upsample_bilinear", f16): {"atol": 1e-5, "rtol": 0.002},
    ("sinc", f16): {"atol": 0.008, "rtol": 0.002},
    ("softmax", f16): {"atol": 1e-4, "rtol": 0.02},
    ("_softmax_backward_data", f16): {"atol": 0.008, "rtol": 0.002},
    ("special.log_ndtr", f64): {"atol": 1e-6, "rtol": 1e-5},
    ("std_mean.unbiased", f16): {
        "reference_in_float": True,
        "atol": 5e-5,
        "rtol": 5e-3,
    },
    ("trapezoid", f16): {"atol": 1e-5, "rtol": 5e-3},
    ("trapz", f16): {"atol": 1e-5, "rtol": 5e-3},
    "uniform": {"reference_in_float": True},
    ("var_mean", f16): {"atol": 1e-5, "rtol": 2e-3},
    ("var_mean.unbiased", f16): {"atol": 1e-5, "rtol": 2e-3},
    ("vdot", f16): {"atol": 1e-5, "rtol": 2e-3},
    # Following tests are failing with strict comparision but atol=1 is acceptable due roundings errors
    # High atol due to precision loss
    ("nn.functional.interpolate.bilinear", f64): {"atol": 5e-4, "rtol": 0},
    ("nn.functional.upsample_bilinear", f64): {"atol": 5e-4, "rtol": 0},
    ("nn.functional.interpolate.bicubic", f64): {"atol": 1e-3, "rtol": 0},
    # Unreasonably high atol requirement:
    ("index_reduce.mean", f16): {"check_gradient": False},
    ("index_reduce.mean", f32): {"check_gradient": False},
    ("index_reduce.mean", f64): {"check_gradient": False},
    # Gradient contains non-finite entries:
    ("index_reduce.amin", f64): {"check_gradient": False},
    ("index_reduce.amin", f32): {"check_gradient": False},
    ("index_reduce.amin", f16): {"check_gradient": False},
    ("index_reduce.amax", f64): {"check_gradient": False},
    ("index_reduce.amax", f32): {"check_gradient": False},
    ("index_reduce.amax", f16): {"check_gradient": False},
    ("tanh", f16): {"atol": 1e-4, "rtol": 1e-2},
    ("nn.functional.embedding_bag", f16): {"check_gradient": False},
    ("nn.functional.embedding_bag", f32): {"check_gradient": False},
    ("nn.functional.embedding_bag", f64): {"check_gradient": False},
    ("_unsafe_masked_index", f16): {"atol": 1e-5, "rtol": 2e-3},
    ("_unsafe_masked_index_put_accumulate", f16): {"atol": 1e-5, "rtol": 5e-3},
}

# Test with one sample only for following ops
inductor_one_sample = defaultdict(dict)

inductor_one_sample["cpu"] = {
    "_segment_reduce.lengths": {f16},
    "_segment_reduce.offsets": {f16},
    "addmv": {f16},
    "as_strided.partial_views": {f16},
    "corrcoef": {f16},
    "diff": {f16},
    "einsum": {f16, i32},
    "gradient": {f16},
    "histogram": {f32, f64},
    "histogramdd": {f32, f64},
    "index_put": {f16, f32, f64},
    "linalg.eig": {f32, f64},
    "linspace": {f16, i32, i64},
    "linspace.tensor_overload": {f16, f32, f64, i32, i64},
    "logspace": {f16},
    "logspace.tensor_overload": {f16, f32, f64, i32, i64},
    "masked_logsumexp": {i64},
    "max_pool2d_with_indices_backward": {f16, f32, f64},
    "new_empty_strided": {f16},
    "nn.functional.adaptive_avg_pool3d": {f16},
    "nn.functional.adaptive_max_pool1d": {f16, f32},
    "nn.functional.adaptive_max_pool2d": {f16, f32},
    "nn.functional.bilinear": {f16},
    "nn.functional.conv_transpose1d": {f16},
    "nn.functional.conv_transpose2d": {f16},
    "nn.functional.conv_transpose3d": {f16},
    "nn.functional.cosine_similarity": {f16},
    "nn.functional.cross_entropy": {f16, f32, f64},
    "nn.functional.gaussian_nll_loss": {f16},
    "nn.functional.grid_sample": {f32, f64, f16},
    "nn.functional.interpolate.area": {f16},
    "nn.functional.nll_loss": {f16, f32, f64},
    "normal": {f16, f32, f64},
    "put": {f16, f32, f64},
    "take": {b8, f16, f32, f64, i32, i64},
}

inductor_one_sample["cuda"] = {
    "_segment_reduce.lengths": {f16},
    "_segment_reduce.offsets": {f16},
    "addmv": {f16},
    "as_strided.partial_views": {f16},
    "corrcoef": {f16},
    "diff": {f16},
    "einsum": {f16, i32},
    "gradient": {f16},
    "histogram": {f32, f64},
    "histogramdd": {f32, f64},
    "index_put": {f16, f32, f64},
    "linalg.eig": {f32, f64},
    "linspace": {f16, i32, i64},
    "linspace.tensor_overload": {f16, f32, f64, i32, i64},
    "logspace": {f16, i32, i64},
    "logspace.tensor_overload": {f16, f32, f64, i32, i64},
    "masked_logsumexp": {i64},
    "max_pool2d_with_indices_backward": {f16, f32, f64},
    "new_empty_strided": {f16},
    "nn.functional.adaptive_avg_pool3d": {f16},
    "nn.functional.adaptive_max_pool1d": {f16, f32},
    "nn.functional.adaptive_max_pool2d": {f16, f32},
    "nn.functional.bilinear": {f16},
    "nn.functional.conv_transpose1d": {f16},
    "nn.functional.conv_transpose2d": {f16},
    "nn.functional.conv_transpose3d": {f16},
    "nn.functional.cosine_similarity": {f16},
    "nn.functional.cross_entropy": {f16, f32, f64},
    "nn.functional.gaussian_nll_loss": {f16},
    "nn.functional.grid_sample": {f16, f32, f64},
    "nn.functional.interpolate.area": {f16},
    "nn.functional.nll_loss": {f16, f32, f64},
    "normal": {f16, f32, f64},
    "put": {f16, f32, f64},
    "take": {b8, f16, f32, f64, i32, i64},
    "__rdiv__": {f16},
    "__rmod__": {f16, i64},
    "__rmul__": {f16},
    "__rpow__": {f16},
    "_unsafe_masked_index": {f16},
    "_unsafe_masked_index_put_accumulate": {f16},
    "addcdiv": {f16},
    "addcmul": {f16},
    "atan2": {f16},
    "cumsum": {f16},
    "cumulative_trapezoid": {f16},
    "dist": {f16},
    "div.no_rounding_mode": {f16},
    "fmod": {f16},
    "grid_sampler_2d": {f16},
    "index_fill": {f16, f32, f64},
    "ldexp": {f16},
    "lerp": {f16},
    "linalg.householder_product": {f32},
    "linalg.matrix_norm": {f16},
    "linalg.vector_norm": {f16},
    "masked.cumsum": {f16},
    "masked.logsumexp": {f16},
    "masked.mean": {b8},
    "masked.normalize": {f16},
    "masked.prod": {f16},
    "masked.std": {f16},
    "masked.var": {f16},
    "mul": {f16},
    "nn.functional.alpha_dropout": {f16, f32, f64},
    "nn.functional.avg_pool1d": {f16, f32, f64},
    "nn.functional.avg_pool2d": {f16, f32, f64},
    "nn.functional.avg_pool3d": {f16, f32, f64},
    "nn.functional.binary_cross_entropy": {f16},
    "nn.functional.binary_cross_entropy_with_logits": {f16},
    "nn.functional.conv2d": {f16},
    "nn.functional.cosine_embedding_loss": {f16},
    "nn.functional.dropout2d": {f16, f32, f64},
    "nn.functional.dropout3d": {f16, f32, f64},
    "nn.functional.dropout": {f16, f32, f64},
    "nn.functional.feature_alpha_dropout.with_train": {f16, f32, f64},
    "nn.functional.fractional_max_pool2d": {f16, f32, f64},
    "nn.functional.fractional_max_pool3d": {f16, f32, f64},
    "nn.functional.group_norm": {f16},
    "nn.functional.hinge_embedding_loss": {f16},
    # Enabling all tests for this test fails randomly
    # See https://github.com/pytorch/pytorch/issues/129238
    "nn.functional.huber_loss": {f16},
    "nn.functional.interpolate.bicubic": {f16},
    "nn.functional.interpolate.bilinear": {f16},
    "nn.functional.interpolate.trilinear": {f16},
    "nn.functional.kl_div": {f16},
    "nn.functional.margin_ranking_loss": {f16},
    "nn.functional.max_pool1d": {f16, f32, f64},
    "nn.functional.max_pool3d": {f16},
    "nn.functional.mse_loss": {f16},
    "nn.functional.multi_margin_loss": {f16},
    "nn.functional.multilabel_margin_loss": {f16},
    "nn.functional.multilabel_soft_margin_loss": {f16},
    "nn.functional.normalize": {f16},
    "nn.functional.pad.replicate": {f16, f32, f64},
    "nn.functional.pad.reflect": {f16},
    "nn.functional.pairwise_distance": {f16},
    "nn.functional.poisson_nll_loss": {f16},
    "nn.functional.rms_norm": {f16},
    "norm": {f16},
    "pow": {f16},
    "prod": {f16},
    "scatter_reduce.amax": {f16, f32, f64},
    "scatter_reduce.amin": {f16, f32, f64},
    "scatter_reduce.mean": {f16, f32, f64},
    "special.xlog1py": {f16},
    "std": {f16},
    "std_mean": {f16},
    "svd_lowrank": {f32, f64},
    "trapezoid": {f16},
    "trapz": {f16},
    "true_divide": {f16},
    "var": {f16},
    "var_mean": {f16},
    "xlogy": {f16},
}

inductor_one_sample["xpu"] = {
    "_segment_reduce.lengths": {f16},
    "_segment_reduce.offsets": {f16},
    "addmv": {f16},
    "as_strided.partial_views": {f16},
    "corrcoef": {f16},
    "diff": {f16},
    "einsum": {f16, i32},
    "gradient": {f16},
    "histogram": {f32, f64},
    "histogramdd": {f32, f64},
    "index_put": {f16, f32, f64},
    "linalg.eig": {f32, f64},
    "linspace": {f16, i32, i64},
    "linspace.tensor_overload": {f16, f32, f64, i32, i64},
    "logspace": {f16, i32, i64},
    "logspace.tensor_overload": {f16, f32, f64, i32, i64},
    "masked_logsumexp": {i64},
    "max_pool2d_with_indices_backward": {f16, f32, f64},
    "new_empty_strided": {f16},
    "nn.functional.adaptive_avg_pool3d": {f16},
    "nn.functional.adaptive_max_pool1d": {f16, f32},
    "nn.functional.adaptive_max_pool2d": {f16, f32},
    "nn.functional.bilinear": {f16},
    "nn.functional.conv_transpose1d": {f16},
    "nn.functional.conv_transpose2d": {f16},
    "nn.functional.conv_transpose3d": {f16},
    "nn.functional.cosine_similarity": {f16},
    "nn.functional.cross_entropy": {f16, f32, f64},
    "nn.functional.gaussian_nll_loss": {f16},
    "nn.functional.grid_sample": {f16, f32, f64},
    "nn.functional.interpolate.area": {f16},
    "nn.functional.nll_loss": {f16, f32, f64},
    "normal": {f16, f32, f64},
    "put": {f16, f32, f64},
    "take": {b8, f16, f32, f64, i32, i64},
    "__rdiv__": {f16},
    "__rmod__": {f16, i64},
    "__rmul__": {f16},
    "__rpow__": {f16},
    "_unsafe_masked_index": {f16},
    "_unsafe_masked_index_put_accumulate": {f16},
    "addcdiv": {f16},
    "addcmul": {f16},
    "atan2": {f16},
    "cumsum": {f16},
    "cumulative_trapezoid": {f16},
    "dist": {f16},
    "div.no_rounding_mode": {f16},
    "fmod": {f16},
    "grid_sampler_2d": {f16},
    "index_fill": {f16, f32, f64},
    "ldexp": {f16},
    "lerp": {f16},
    "linalg.householder_product": {f32},
    "linalg.matrix_norm": {f16},
    "linalg.vector_norm": {f16},
    "masked.cumsum": {f16},
    "masked.logsumexp": {f16},
    "masked.mean": {b8},
    "masked.normalize": {f16},
    "masked.prod": {f16},
    "masked.std": {f16},
    "masked.var": {f16},
    "mul": {f16},
    "nn.functional.alpha_dropout": {f16, f32, f64},
    "nn.functional.avg_pool1d": {f16, f32, f64},
    "nn.functional.avg_pool2d": {f16, f32, f64},
    "nn.functional.avg_pool3d": {f16, f32, f64},
    "nn.functional.binary_cross_entropy": {f16},
    "nn.functional.binary_cross_entropy_with_logits": {f16},
    "nn.functional.conv2d": {f16},
    "nn.functional.cosine_embedding_loss": {f16},
    "nn.functional.dropout2d": {f16, f32, f64},
    "nn.functional.dropout3d": {f16, f32, f64},
    "nn.functional.dropout": {f16, f32, f64},
    "nn.functional.feature_alpha_dropout.with_train": {f16, f32, f64},
    "nn.functional.fractional_max_pool2d": {f16, f32, f64},
    "nn.functional.fractional_max_pool3d": {f16, f32, f64},
    "nn.functional.group_norm": {f16},
    "nn.functional.hinge_embedding_loss": {f16},
    # Enabling all tests for this test fails randomly
    # See https://github.com/pytorch/pytorch/issues/129238
    "nn.functional.huber_loss": {f16},
    "nn.functional.interpolate.bicubic": {f16},
    "nn.functional.interpolate.bilinear": {f16},
    "nn.functional.interpolate.trilinear": {f16},
    "nn.functional.kl_div": {f16},
    "nn.functional.margin_ranking_loss": {f16},
    "nn.functional.max_pool1d": {f16, f32, f64},
    "nn.functional.max_pool3d": {f16},
    "nn.functional.mse_loss": {f16},
    "nn.functional.multi_margin_loss": {f16},
    "nn.functional.multilabel_margin_loss": {f16},
    "nn.functional.multilabel_soft_margin_loss": {f16},
    "nn.functional.normalize": {f16},
    "nn.functional.pad.replicate": {f16, f32, f64},
    "nn.functional.pad.reflect": {f16},
    "nn.functional.pairwise_distance": {f16},
    "nn.functional.poisson_nll_loss": {f16},
    "nn.functional.rms_norm": {f16},
    "norm": {f16},
    "pow": {f16},
    "prod": {f16},
    "scatter_reduce.amax": {f16, f32, f64},
    "scatter_reduce.amin": {f16, f32, f64},
    "scatter_reduce.mean": {f16, f32, f64},
    "special.xlog1py": {f16},
    "std": {f16},
    "std_mean": {f16},
    "svd_lowrank": {f32, f64},
    "trapezoid": {f16},
    "trapz": {f16},
    "true_divide": {f16},
    "var": {f16},
    "var_mean": {f16},
    "xlogy": {f16},
}


def collection_decorator(fn):
    @functools.wraps(fn)
    def inner(self, device, dtype, op):
        try:
            fn(self, device, dtype, op)
        except Exception as e:
            if COLLECT_EXPECT:
                variant = op.variant_test_name
                op_key = op.name if not variant else (op.name, variant)
                device_type = torch.device(device).type
                # failed_reasons[device_type, op_key].add(repr(e))
                seen_failed[device_type, op_key].add(dtype)
            raise e

    return inner


class TestInductorOpInfo(TestCase):
    def tearDown(self):
        torch._dynamo.reset()

    check_model = check_model
    check_model_gpu = check_model_gpu

    @onlyNativeDeviceTypes
    @suppress_warnings
    @skipCUDAMemoryLeakCheckIf(
        True
    )  # inductor kernels failing this test intermittently
    @skipCUDAIf(not HAS_CUDA, "Skipped! Triton not found")
    @skipXPUIf(not HAS_XPU, "Skipped! Supported XPU compiler not found")
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
    @torch._inductor.config.patch("test_configs.runtime_triton_dtype_assert", True)
    @collection_decorator
    def test_comprehensive(self, device, dtype, op):
        device_type = torch.device(device).type

        assert device_type in (GPU_TYPE, "cpu")

        torch._dynamo.reset()
        with torch.no_grad():
            # TODO: should we move empty_cache to the common device interface
            if device_type == "cuda":
                torch.cuda.empty_cache()
            elif device == "xpu":
                torch.xpu.empty_cache()
        op_name = op.name
        if op.variant_test_name:
            op_name += f".{op.variant_test_name}"

        # Skip dtype=torch.uint8 for all ops except upsample and interpolate:
        allowed_dtypes = [f16, f32, f64, i32, i64, b8]
        if op_name not in (
            "nn.functional.interpolate.bilinear",
            "nn.functional.interpolate.bicubic",
            "nn.functional.upsample_bilinear",
            "nn.functional.upsample_nearest",
        ):
            if dtype not in allowed_dtypes:
                raise unittest.SkipTest("Skipped!")

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
        overridden_kwargs.update(
            inductor_override_kwargs.get(device_type, {}).get(op_name, {})
        )
        overridden_kwargs.update(
            inductor_override_kwargs.get(device_type, {}).get((op_name, dtype), {})
        )
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

        if (
            dtype in inductor_one_sample.get(device_type, {}).get(op_name, {})
        ) and not ALL_SAMPLES:
            if isinstance(samples, (list, tuple)):
                samples = [samples[0]]
            else:
                samples = [next(samples)]

        class HasRngOp(TorchDispatchMode):
            def __init__(self) -> None:
                super().__init__()
                self.has_rng_op = False

            def __torch_dispatch__(self, func, types, args, kwargs=None):
                kwargs = kwargs if kwargs else {}
                if torch.Tag.nondeterministic_seeded in func.tags:
                    self.has_rng_op = True

                return func(*args, **kwargs)

        def do_nopython_and_has_rng(fn, args, kwargs):
            try:
                mode = FakeTensorMode()

                def map_to_fake(e):
                    if isinstance(e, torch.Tensor):
                        return mode.from_tensor(e)
                    else:
                        return e

                args, kwargs = tree_map(map_to_fake, (args, kwargs))
                with HasRngOp() as rng_mode, mode:
                    with enable_python_dispatcher():
                        fn(*args, **kwargs)

            except (DataDependentOutputException, DynamicOutputShapeException):
                return False, rng_mode.has_rng_op

            return True, rng_mode.has_rng_op

        def get_contexts(has_rng_op):
            if has_rng_op:
                # TODO - enable this, running into errors
                return (
                    # (
                    #     lambda: torch._inductor.config.patch(
                    #         {"fallback_random": True, "implicit_fallbacks": True}
                    #     ),
                    #     {"assert_equal": True},
                    # ),
                    (
                        contextlib.nullcontext,
                        {"assert_equal": False},
                    ),
                )
            return ((contextlib.nullcontext, {}),)

        try:

            def _get_tolerances(dtype):
                _custom_tolerances = {
                    torch.float32: (1.3e-5, 1.5e-5),
                }
                if dtype in _custom_tolerances:
                    return _custom_tolerances[dtype]
                else:
                    return None, None

            for sample_input in samples:
                args = [sample_input.input] + list(sample_input.args)
                kwargs = sample_input.kwargs
                # UNCOMMENT TO DEBUG SEGFAULTS

                # with open("test_output.txt", "a") as f:
                #     print(f"RUNNING OP {op_name} on {device_type} with {dtype}", flush=True, file=f)
                #     print(f"RUNNING OP {op_name} on {device_type} with {dtype}", flush=True)
                rtol, atol = _get_tolerances(dtype)
                if device_type == GPU_TYPE:
                    # opinfo test case have already place the input on the correct device
                    # so we don't need do additional copy by setting copy_to_gpu=False

                    no_python, has_rng_op = do_nopython_and_has_rng(fn, args, kwargs)
                    for context_fn, kwarg_overrides in get_contexts(has_rng_op):
                        with context_fn():
                            adjusted_kwargs = {
                                "check_lowp": False,
                                "nopython": no_python,
                                "copy_to_gpu": False,
                                "reference_in_float": False,
                                "check_gradient": requires_grad,
                                "check_has_compiled": no_python,
                                "output_process_fn_grad": sample_input.output_process_fn_grad,
                                "atol": atol,
                                "rtol": rtol,
                            }
                            adjusted_kwargs.update(overridden_kwargs)
                            adjusted_kwargs.update(kwarg_overrides)
                            self.check_model_gpu(
                                fn,
                                args,
                                kwargs,
                                **adjusted_kwargs,
                            )
                elif device_type == "cpu":
                    no_python, has_rng_op = do_nopython_and_has_rng(fn, args, kwargs)
                    for context_fn, kwarg_overrides in get_contexts(has_rng_op):
                        with context_fn():
                            adjusted_kwargs = {
                                "check_lowp": False,
                                "nopython": no_python,
                                "check_has_compiled": no_python,
                                # skip checking gradient on CPU for now
                                "check_gradient": False,
                                "atol": atol,
                                "rtol": rtol,
                            }
                            adjusted_kwargs.update(overridden_kwargs)
                            adjusted_kwargs.update(kwarg_overrides)

                            self.check_model(
                                fn,
                                args,
                                kwargs,
                                **adjusted_kwargs,
                            )

        except Exception as e:
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


instantiate_device_type_tests(TestInductorOpInfo, globals(), allow_xpu=True)

if __name__ == "__main__":
    run_tests()
