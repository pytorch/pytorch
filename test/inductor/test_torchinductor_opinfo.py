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
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CPU, HAS_CUDA
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map

try:
    try:
        from .test_torchinductor import check_model, check_model_gpu
    except ImportError:
        from test_torchinductor import check_model, check_model_gpu
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
    "nn.functional.rrelu": {f32, f64},
    "nonzero_static": {b8, f16, f32, f64, i32, i64},
    ("normal", "in_place"): {f16, f32, f64},
    ("normal", "number_mean"): {f16, f32, f64},
    ("sparse.mm", "reduce"): {f32, f64},
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
    "sparse.sampled_addmm": {f32, f64},
    "torch.ops.aten._flash_attention_forward": {f16},
    "torch.ops.aten._efficient_attention_forward": {f16, f32},
    "to_sparse": {
        f16,
        f32,
        f64,
    },  # NYI: could not find kernel for aten.view.default at dispatch key DispatchKey.SparseCUDA
}


# intentionally not handled
intentionally_not_handled = {
    "resize_": {b8, f16, f32, f64, i32, i64},
    "resize_as_": {b8, f16, f32, f64, i32, i64},
}

inductor_expected_failures_single_sample["cuda"].update(intentionally_not_handled)


inductor_gradient_expected_failures_single_sample = defaultdict(dict)

inductor_gradient_expected_failures_single_sample["cuda"] = {}

if not TEST_MKL:
    inductor_expected_failures_single_sample["cpu"].update({})

inductor_should_fail_with_exception = defaultdict(dict)
inductor_should_fail_with_exception["cpu"] = {}
inductor_should_fail_with_exception["cuda"] = {}


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
    ("cross", "cuda", f16): {"reference_in_float": True},
    ("linalg.cross", "cuda", f16): {"reference_in_float": True},
    ("addr", "cuda", f16): {"reference_in_float": True},
    ("baddbmm", "cuda", f16): {"atol": 2e-3, "rtol": 0.002},  # decomp affects accuracy
    ("angle", "cuda", f64): {"reference_in_float": True},
    ("asin", "cuda", f16): {"reference_in_float": True},
    ("atanh", "cuda", f16): {"reference_in_float": True},
    ("cauchy", "cuda"): {"reference_in_float": True},
    ("cummax", "cuda", f16): {"atol": 5e-4, "rtol": 0.002},
    ("cumsum", "cuda", f16): {"reference_in_float": True},
    ("cumprod", "cuda"): {"reference_in_float": True, "atol": 7e-5, "rtol": 0.002},
    ("logcumsumexp", "cuda"): {"grad_atol": 8e-4, "grad_rtol": 0.001},
    ("exponential", "cuda"): {"reference_in_float": True},
    ("geometric", "cuda"): {"reference_in_float": True},
    ("kron", "cuda", f16): {"reference_in_float": True},
    ("log_normal", "cuda"): {"reference_in_float": True},
    ("masked.softmin", "cuda", f16): {"atol": 1e-4, "rtol": 0.01},
    ("nn.functional.batch_norm", "cuda", f16): {"reference_in_float": True},
    ("nn.functional.batch_norm.without_cudnn", "cuda", f16): {
        "reference_in_float": True
    },
    ("nn.functional.cosine_similarity", "cuda", f16): {"reference_in_float": True},
    ("nn.functional.instance_norm", "cuda", f16): {"reference_in_float": True},
    ("nn.functional.local_response_norm", "cuda", f16): {"reference_in_float": True},
    ("nn.functional.normalize", "cuda", f16): {"atol": 1e-3, "rtol": 0.05},
    ("nn.functional.rms_norm", "cuda", f16): {"reference_in_float": True},
    ("nn.functional.soft_margin_loss", "cuda", f16): {"reference_in_float": True},
    ("nn.functional.softmin", "cuda", f16): {"atol": 1e-4, "rtol": 0.01},
    ("nn.functional.softsign", "cuda", f16): {"reference_in_float": True},
    ("nn.functional.tanhshrink", "cuda", f16): {"atol": 3e-4, "rtol": 0.001},
    ("outer", "cuda", f16): {"reference_in_float": True},
    ("round.decimals_3", "cuda", f16): {"reference_in_float": True},
    ("nn.functional.triplet_margin_loss", "cuda", f16): {"atol": 1e-4, "rtol": 0.02},
    ("nn.functional.triplet_margin_with_distance_loss", "cuda", f16): {
        "atol": 1e-4,
        "rtol": 0.02,
    },
    ("sinc", "cuda", f16): {"atol": 0.008, "rtol": 0.002},
    ("softmax", "cpu", f16): {"atol": 1e-4, "rtol": 0.02},
    ("softmax", "cuda", f16): {"atol": 1e-4, "rtol": 0.02},
    ("_softmax_backward_data", "cuda", f16): {"atol": 0.008, "rtol": 0.002},
    ("special.log_ndtr", "cuda", f64): {"atol": 1e-6, "rtol": 1e-5},
    ("polygamma.polygamma_n_0", "cpu", f32): {"atol": 1e-3, "rtol": 1e-4},
    ("polygamma.polygamma_n_1", "cpu", f32): {"atol": 1e-3, "rtol": 1e-4},
    ("polygamma.polygamma_n_2", "cpu", f32): {"atol": 1e-3, "rtol": 1e-4},
    ("polygamma.polygamma_n_3", "cpu", f32): {"atol": 1e-3, "rtol": 1e-4},
    ("polygamma.polygamma_n_4", "cpu", f32): {"atol": 1e-3, "rtol": 1e-4},
    ("special.polygamma.special_polygamma_n_0", "cpu", f32): {
        "atol": 1e-3,
        "rtol": 1e-4,
    },
    ("std_mean.unbiased", "cuda", f16): {"reference_in_float": True},
    ("uniform", "cuda"): {"reference_in_float": True},
    # Following tests are failing with strict comparision but atol=1 is acceptable due roundings errors
    ("nn.functional.interpolate.bilinear", "cpu", u8): {"atol": 1, "rtol": 0},
    ("nn.functional.upsample_bilinear", "cpu", u8): {"atol": 1, "rtol": 0},
    ("nn.functional.interpolate.bicubic", "cpu", u8): {"atol": 1, "rtol": 0},
    # High atol due to precision loss
    ("nn.functional.interpolate.bilinear", "cuda", f64): {"atol": 5e-4, "rtol": 0},
    ("nn.functional.upsample_bilinear", "cuda", f64): {"atol": 5e-4, "rtol": 0},
    ("nn.functional.interpolate.bicubic", "cpu", f32): {"atol": 5e-3, "rtol": 0},
    ("nn.functional.interpolate.bicubic", "cuda", f64): {"atol": 1e-3, "rtol": 0},
    # Unreasonably high atol requirement:
    ("index_reduce.mean", "cuda", f16): {"check_gradient": False},
    ("index_reduce.mean", "cuda", f32): {"check_gradient": False},
    ("index_reduce.mean", "cuda", f64): {"check_gradient": False},
    # Gradient contains non-finite entries:
    ("index_reduce.amin", "cuda", f64): {"check_gradient": False},
    ("index_reduce.amin", "cuda", f32): {"check_gradient": False},
    ("index_reduce.amin", "cuda", f16): {"check_gradient": False},
    ("index_reduce.amax", "cuda", f64): {"check_gradient": False},
    ("index_reduce.amax", "cuda", f32): {"check_gradient": False},
    ("index_reduce.amax", "cuda", f16): {"check_gradient": False},
    ("tanh", "cuda", f16): {"atol": 1e-4, "rtol": 1e-2},
}


# Always test with all sample for following ops
inductor_all_samples = {
    "arange",
    "diagonal",
    "diagonal_copy",
    "diagonal_scatter",
    "softmax.with_dtype",
    "index_add",
    "index_copy",
    "index_reduce.prod",
    "index_reduce.mean",
    "index_reduce.amax",
    "index_reduce.amin",
    "scatter_reduce.sum",
    "select_scatter",
    "squeeze",
    "unfold",
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
    @collection_decorator
    def test_comprehensive(self, device, dtype, op):
        torch._dynamo.reset()
        with torch.no_grad():
            # TODO: should we move empty_cache to the common device interface
            if device == "cuda":
                torch.cuda.empty_cache()
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

        device_type = torch.device(device).type

        assert device_type in (GPU_TYPE, "cpu")

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

        class HasRngOp(TorchDispatchMode):
            def __init__(self):
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


instantiate_device_type_tests(TestInductorOpInfo, globals())

if __name__ == "__main__":
    run_tests()
