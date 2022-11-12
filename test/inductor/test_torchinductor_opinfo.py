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
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyNativeDeviceTypes,
    OpDTypes,
    ops,
)
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_utils import (
    dtype_abbrs,
    run_tests,
    skipCUDAMemoryLeakCheckIf,
    suppress_warnings,
    TEST_WITH_ROCM,
    TestCase,
)

try:
    from torch._inductor.utils import has_triton

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
TestExpect = Enum("TestExpect", ("SUCCESS", "XFAILURE", "SKIP"))

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

inductor_skips = defaultdict(dict)

inductor_skips["cpu"] = {
    "linalg.ldl_solve": {b8, f16, f32, f64, i32, i64},  # segfault
    "__rdiv__": {b8, f16, f32, f64, i32, i64},  # flaky
}

inductor_skips["cuda"] = {
    # flaky
    "__rdiv__": {b8, f16, f32, f64, i32, i64},
    "masked.prod": {f16, f32, f64},
    "linalg.vander": {f32, f64},
    "sparse.sampled_addmm": {f32, f64},
    "broadcast_tensors": {f16, f32, f64},
    "dsplit": {f16, f32, f64},
    # Jiterator kernel is not expected to work with inductor
    "jiterator_2inputs_2outputs": {b8, f16, f32, f64, i32, i64},
    "jiterator_4inputs_with_extra_args": {b8, f16, f32, f64, i32, i64},
    "jiterator_binary": {b8, f16, f32, f64, i32, i64},
    "jiterator_binary_return_by_ref": {b8, f16, f32, f64, i32, i64},
    "jiterator_unary": {b8, f16, f32, f64, i32, i64},
    # Disabled on migration to core
    "linalg.pinv.singular": {f32, f64},
    "linalg.householder_product": {f32},
    # These might be passing now?
    "__getitem__": {b8, f16, f32, f64, i32, i64},
    "nn.functional.conv_transpose3d": {f16},
    "max.reduction_with_dim": {i32, i64},
    "min.reduction_with_dim": {i32, i64},
    "linalg.lu": {f32, f64},
    "lu_unpack": {f32, f64},
    "native_batch_norm": {f16, f32, f64},
    "native_layer_norm": {f16, f32, f64},
    # Issues on sm86 periodic job (complex numbers)
    "cdouble": {b8, f16, f32, f64, i32, i64},
    "cfloat": {b8, f16, f32, f64, i32, i64},
    "randint": {b8, f16, f32, f64, i32, i64},
}

inductor_expected_failures_single_sample = defaultdict(dict)

inductor_expected_failures_single_sample["cpu"] = {
    "T": {b8, f16, f32, f64, i32, i64},
    "H": {b8, f16, f32, f64, i32, i64},
    "mH": {b8, f16, f32, f64, i32, i64},
    "mT": {b8, f16, f32, f64, i32, i64},
    "__getitem__": {b8, f16, f32, f64, i32, i64},
    "addr": {f16},
    "allclose": {f16, f32, f64},
    "angle": {f16, f32, f64},
    "argwhere": {b8, f16, f32, f64, i32, i64},
    "bernoulli": {f32, f64},
    "bincount": {i32, i64},
    "chalf": {b8, f16, f32, f64, i32, i64},
    "cholesky": {f32, f64},
    "combinations": {b8, f16, f32, f64, i32, i64},
    "complex": {f16, f32, f64},
    "constant_pad_nd": {f16, f32, f64},
    "corrcoef": {f32, f64, i32, i64},
    "cov": {f32, f64, i32, i64},
    "equal": {b8, f16, f32, f64, i32, i64},
    "erf": {b8, f64},
    "fft.fft": {f32, f64},
    "fft.fft2": {b8, f32, f64, i32, i64},
    "fft.fftn": {b8, f32, f64, i32, i64},
    "fft.hfft": {b8, f32, f64, i32, i64},
    "fft.hfft2": {b8, f32, f64, i32, i64},
    "fft.hfftn": {b8, f32, f64, i32, i64},
    "fft.ifft": {f16, f32, f64},
    "fft.ifft2": {b8, f32, f64, i32, i64},
    "fft.ifftn": {b8, f32, f64, i32, i64},
    "fft.ihfft": {f16, f32, f64},
    "fft.ihfft2": {f32, f64},
    "fft.ihfftn": {f32, f64},
    "fft.irfft": {b8, f32, f64, i32, i64},
    "fft.irfft2": {b8, f32, f64, i32, i64},
    "fft.irfftn": {b8, f32, f64, i32, i64},
    "fft.rfft": {f32, f64},
    "fft.rfft2": {f32, f64},
    "fft.rfftn": {f32, f64},
    "index_add": {f16},
    "index_put": {f16, f32, f64},
    "index_reduce": {f16, f32, f64},
    "istft": {f32, f64},
    "linalg.eig": {f32, f64},
    "linalg.eigh": {f32, f64},
    "linalg.eigvals": {f32, f64},
    "linalg.eigvalsh": {f32, f64},
    "linalg.lstsq": {f32, f64},
    "linalg.lstsq.grad_oriented": {f32, f64},
    "linalg.matrix_rank": {f32, f64},
    "linalg.matrix_rank.hermitian": {f32, f64},
    "linalg.lu_solve": {f32, f64},
    "lu_solve": {f32, f64},
    "lu_unpack": {f32, f64},
    "logdet": {f32, f64},
    "masked.norm": {f16},
    "masked_fill": {f16},
    "masked_scatter": {f16, f32, f64},
    "masked_select": {b8, f16, f32, f64, i32, i64},
    "max.reduction_no_dim": {f16},
    "max.reduction_with_dim": {b8},
    "min.reduction_no_dim": {f16},
    "min.reduction_with_dim": {b8},
    "multinomial": {f32, f64},
    "nan_to_num": {f16},
    "nanquantile": {f32, f64},
    "nn.functional.avg_pool1d": {i64},
    "nn.functional.avg_pool2d": {i64},
    "nn.functional.adaptive_avg_pool2d": {f16},
    "nn.functional.ctc_loss": {f32, f64},
    "nn.functional.gaussian_nll_loss": {f32, f64},
    "nn.functional.gelu": {f64},
    "nn.functional.local_response_norm": {i64},
    "nn.functional.one_hot": {i64},
    "nn.functional.pairwise_distance": {f16},
    "nn.functional.rrelu": {f32, f64},
    "nn.functional.triplet_margin_with_distance_loss": {f32, f64, i32, i64},
    "nonzero": {b8, f16, f32, f64, i32, i64},
    "normal": {f16, f32, f64},
    "normal.number_mean": {f16, f32, f64},
    "pca_lowrank": {f32, f64},
    "polar": {f32, f64},
    "quantile": {f32, f64},
    "rand_like": {f16, f32, f64},
    "randint_like": {f16, f32, f64, i32, i64},
    "randn_like": {f16, f32, f64},
    "repeat_interleave": {b8, f16, f32, f64, i32, i64},
    "scatter_add": {f16},
    "scatter_reduce.sum": {f16},
    "scatter_reduce.prod": {f16, f32, f64},
    "segment_reduce.lengths": {f16, f32, f64},
    "sparse.sampled_addmm": {f32, f64},
    "stft": {f32, f64},
    "svd_lowrank": {f32, f64},
    "tensor_split": {b8, f16, f32, f64, i32, i64},
    "to": {b8, f16, f32, f64, i32, i64},
    "to_sparse": {f32, f64},
    "tril": {f16},
    "triu": {f16},
    "uniform": {f16, f32, f64},
    "unique": {b8, f32, f64, i32, i64},
    "unique_consecutive": {b8, f32, f64, i32, i64},
    "var": {f16},
    "var_mean": {f16},
    "view_as_complex": {f16, f32, f64},
}


inductor_expected_failures_single_sample["cuda"] = {
    "T": {b8, f16, f32, f64, i32, i64},
    "H": {b8, f16, f32, f64, i32, i64},
    "mH": {b8, f16, f32, f64, i32, i64},
    "mT": {b8, f16, f32, f64, i32, i64},
    "__getitem__": {b8, f16, f32, f64, i32, i64},
    "allclose": {f16, f32, f64},
    "angle": {f32, f64},
    "argwhere": {b8, f16, f32, f64, i32, i64},
    "baddbmm": {f16},
    "bernoulli": {f16, f32, f64},
    "bincount": {i32, i64},
    "chalf": {b8, f16, f32, f64, i32, i64},
    "cholesky": {f32, f64},
    "combinations": {b8, f16, f32, f64, i32, i64},
    "complex": {f16, f32, f64},
    "corrcoef": {f16, f32, f64, i32, i64},
    "cov": {f16, f32, f64, i32, i64},
    "equal": {b8, f16, f32, f64, i32, i64},
    "fft.fft": {f16, f32, f64},
    "fft.fft2": {b8, f16, f32, f64, i32, i64},
    "fft.fftn": {b8, f16, f32, f64, i32, i64},
    "fft.hfft": {b8, f16, f32, f64, i32, i64},
    "fft.hfft2": {b8, f16, f32, f64, i32, i64},
    "fft.hfftn": {b8, f16, f32, f64, i32, i64},
    "fft.ifft": {f16, f32, f64},
    "fft.ifft2": {b8, f16, f32, f64, i32, i64},
    "fft.ifftn": {b8, f16, f32, f64, i32, i64},
    "fft.ihfft": {f16, f32, f64},
    "fft.ihfft2": {f16, f32, f64},
    "fft.ihfftn": {f16, f32, f64},
    "fft.irfft": {b8, f16, f32, f64, i32, i64},
    "fft.irfft2": {b8, f16, f32, f64, i32, i64},
    "fft.irfftn": {b8, f16, f32, f64, i32, i64},
    "fft.rfft": {f16, f32, f64},
    "fft.rfft2": {f16, f32, f64},
    "fft.rfftn": {f16, f32, f64},
    "index_put": {f16, f32, f64},
    "index_reduce": {f16, f32, f64},
    "istft": {f32, f64},
    "linalg.eig": {f32, f64},
    "linalg.eigh": {f32, f64},
    "linalg.eigvals": {f32, f64},
    "linalg.eigvalsh": {f32, f64},
    "linalg.lstsq": {f32, f64},
    "linalg.lstsq.grad_oriented": {f32, f64},
    "linalg.matrix_rank": {f32, f64},
    "linalg.matrix_rank.hermitian": {f32, f64},
    "lu_unpack": {f32, f64},
    "masked.argmax": {f16, f32, f64, i32},
    "masked.argmin": {f16, f32, f64, i32},
    "masked_scatter": {f16, f32, f64},
    "masked_select": {b8, f16, f32, f64, i32, i64},
    "max.reduction_with_dim": {b8, i32, i64},
    "min.reduction_with_dim": {b8, i32, i64},
    "multinomial": {f16, f32, f64},
    "nn.functional.adaptive_avg_pool2d": {f16},
    "nn.functional.ctc_loss": {f32, f64},
    "nn.functional.grid_sample": {f16},
    "nn.functional.gaussian_nll_loss": {f16, f32, f64},
    "nn.functional.one_hot": {i64},
    "nn.functional.rrelu": {f16, f32, f64},
    "nn.functional.triplet_margin_with_distance_loss": {f16, f32, f64, i32, i64},
    "nonzero": {b8, f16, f32, f64, i32, i64},
    "normal": {f16, f32, f64},
    "normal.number_mean": {f16, f32, f64},
    "pca_lowrank": {f32, f64},
    "polar": {f32, f64},
    "pow": {i32, i64},
    "rand_like": {f16, f32, f64},
    "randint_like": {f16, f32, f64, i32, i64},
    "randn_like": {f16, f32, f64},
    "repeat_interleave": {b8, f16, f32, f64, i32, i64},
    "round.decimals_3": {f16},
    "scatter_reduce.prod": {f16, f32, f64},
    "segment_reduce.lengths": {f16, f32, f64},
    "stft": {f32, f64},
    "svd_lowrank": {f32, f64},
    "tensor_split": {b8, f16, f32, f64, i32, i64},
    "to": {b8, f16, f32, f64, i32, i64},
    "to_sparse": {f16, f32, f64},
    "uniform": {f16, f32, f64},
    "unique": {b8, f16, f32, f64, i32, i64},
    "unique_consecutive": {b8, f16, f32, f64, i32, i64},
    "view_as_complex": {f16, f32, f64},
    # AssertionError: Tensor-likes are not close!
    "erf": {b8, f64},
    "nn.functional.gelu": {f64},
    "nn.functional.triplet_margin_loss": {f16},
}

inductor_gradient_expected_failures_single_sample = defaultdict(dict)

inductor_gradient_expected_failures_single_sample["cuda"] = {
    "asin": {f16},
    "cumprod": {f16},
    "linalg.vector_norm": {f64, f64},
    "linalg.householder_product": {f32},
    "kron": {f16},
    "nanquantile": {f32, f64},
    "native_batch_norm": {f16, f32, f64},
    "native_layer_norm": {f16, f32, f64},
    "nn.functional._scaled_dot_product_attention": {f16},
    "nn.functional.avg_pool2d": {f16, f32, f64},
    "nn.functional.batch_norm.without_cudnn": {f16},
    "nn.functional.batch_norm": {f16},
    "nn.functional.cosine_similarity": {f16},
    "nn.functional.instance_norm": {f16},
    "nn.functional.normalize": {f16},
    "nn.functional.softsign": {f16},
    "nn.functional.local_response_norm": {f16},
    "outer": {f16},
    "quantile": {f32, f64},
    "scatter_reduce.amax": {f16, f32, f64},
    "scatter_reduce.amin": {f16, f32, f64},
    "tanh": {f16},
}

inductor_should_fail_with_exception = defaultdict(dict)

inductor_should_fail_with_exception["cpu"] = {}


inductor_should_fail_with_exception["cuda"] = {
    "__rpow__": {
        i32: "Pow input must be floating point.",
        i64: "Pow input must be floating point.",
    }
}


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
    "empty_like": {"assert_equal": False},
    "new_empty": {"assert_equal": False},
    "new_empty_strided": {"assert_equal": False},
    "randn": {"assert_equal": False},
    ("nn.functional.tanhshrink", "cuda", f16): {"atol": 3e-4, "rtol": 0.001},
    ("cummax", "cuda", f16): {"atol": 5e-4, "rtol": 0.002},
    "gradient": {"check_gradient": False},  # segfault on check_gradient
    # Following tests failed, and causing subsequent tests failing with unrecoverable CUDA error
    "linalg.solve_triangular": {"check_gradient": False},
    "linalg.lu_factor": {"check_gradient": False},
    "linalg.lu_factor_ex": {"check_gradient": False},
}

# Always test with all sample for following ops
inductor_all_samples = {
    "softmax.with_dtype",
    "index_add",
    "index_put",
    "index_copy",
    "scatter_reduce.sum",
    "select_scatter",
    "squeeze",
    "unsqueeze",
    "sum",
}


class TestInductorOpInfo(TestCase):
    check_model = check_model
    check_model_cuda = check_model_cuda

    @onlyNativeDeviceTypes
    @suppress_warnings
    @skipCUDAMemoryLeakCheckIf(
        True
    )  # inductor kernels failing this test intermittently
    @_ops(op_db[START:END])
    @patch("torch._dynamo.config.raise_on_unsafe_aot_autograd", True)
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
            test_expect = TestExpect.SKIP
            # with open("test_output.txt", "a") as f:
            #     print(f"SKIPPING OP {op_name} on {device_type}", flush=True, file=f)
            #     print(f"SKIPPING OP {op_name} on {device_type}", flush=True)
            self.skipTest(f"{op_name} in {dtype} not supported")
        elif dtype in inductor_expected_failures_single_sample[device_type].get(
            op_name, set()
        ) or dtype in inductor_gradient_expected_failures_single_sample[
            device_type
        ].get(
            op_name, set()
        ):
            test_expect = TestExpect.XFAILURE
        else:
            test_expect = TestExpect.SUCCESS

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

            if test_expect is TestExpect.XFAILURE:
                return

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

        if test_expect is TestExpect.XFAILURE and not COLLECT_EXPECT:
            if FAIL_ON_SUCCESS:
                raise RuntimeError(
                    f"unexpected success {op_name}, {dtype}, {device_type}"
                )


instantiate_device_type_tests(TestInductorOpInfo, globals())

if __name__ == "__main__":
    if has_triton() and not TEST_WITH_ROCM:
        run_tests()
