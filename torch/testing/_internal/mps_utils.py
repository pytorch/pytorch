import unittest
import warnings
from collections.abc import Iterable
from typing import Callable, Optional, Union

import torch
from torch.testing._internal.common_utils import MACOS_VERSION
from torch.testing._internal.opinfo.core import DecorateInfo, OpInfo


UNIMPLEMENTED = "unimplemented"
TEST_OUTPUT_MATCH = "test_output_match"
TEST_OUTPUT_GRAD_MATCH = "test_output_grad_match"
TEST_ERROR_INPUTS = "test_error_inputs"
COMMON = "TestCommon"
ERROR_INPUTS = "TestErrorInputs"


def xfailUnimplemented(test_func: Callable) -> Callable:
    def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
        try:
            test_func(*args, **kwargs)
        except NotImplementedError as e:
            raise unittest.SkipTest(
                "Requires op not currently implemented on MPS"
            ) from e
        except TypeError as e:
            raise unittest.SkipTest("Uses dtype not supported on MPS") from e
        except unittest.SkipTest as e:
            # Don't error out on tests that have been explicitly skipped for some other reason
            raise e
        except Exception as e:
            warnings.warn(
                "Test is marked as unimplemented on MPS,"
                "but instead of NotImplementedError or TypeError we received {type(e).__name__}:{e} "
            )
            raise unittest.SkipTest(
                "[WARNING]: Test is marked as unimplemented on MPS,"
                "but instead of NotImplementedError or TypeError we received {type(e).__name__}:{e} "
            ) from e
        else:
            raise RuntimeError(
                "Test is marked as unimplemented on MPS, but received unexpected success. Ensure CPU fallback is disabled"
            )

    return wrapper


class MPSSkipInfo:
    def __init__(
        self,
        *args: str,
        test_class: Optional[str] = None,
        variant: Optional[str] = None,
        dtypes: Optional[Union[torch.dtype, list[torch.dtype]]] = None,
        skip: Callable = unittest.expectedFailure,
        skip_msg: str = "Skipped!",
        upper: Optional[float] = None,
        lower: Optional[float] = None,
    ):
        """Basic struct for tracking MPS OpInfo xfails

        args: String names of test(s) to apply this xfail info to
        test_class: Test class, e.g. 'TestCommon' etc.
        variant: Variant name. Set to empty str ("") to explicitly specify the non-variant case
        If set to None, will instead apply to all variants of the test
        dtypes: If none specified, xfails all dtype variants
        skip: Type of decorator to add [expectedFailure, skipTest, xfailUnimplementedOpMPS, xfailUnimplementedDtypeMPS]
        upper: Upper bound MacOS version this xfail applies to (exclusive)
        lower: Lower bound MacOS version this xfail applies to (inclusive)
        """
        self.tests: list[str] = []
        for arg in args:
            self.tests.append(arg)
        self.test_class = test_class
        self.variant = variant
        self.dtypes = dtypes
        self.skip = skip
        self.skip_msg = skip_msg
        self.upper = upper
        self.lower = lower

        if UNIMPLEMENTED in self.tests:
            # Not all tests fail for unimplemented ops, so we specify here
            self.tests = [TEST_OUTPUT_GRAD_MATCH, TEST_OUTPUT_MATCH]
            self.skip = xfailUnimplemented


""" Failures due to lack of op implementation on MPS backend """
UNIMPLEMENTED_XFAILIST = {
    "login": MPSSkipInfo(UNIMPLEMENTED),
    "logspace": MPSSkipInfo(UNIMPLEMENTED),
    "logspacetensor_overload": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.eig": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.eigvals": MPSSkipInfo(UNIMPLEMENTED),
    "put": MPSSkipInfo(UNIMPLEMENTED),
    "nn.functional.conv_transpose3d": MPSSkipInfo(UNIMPLEMENTED),
    "__rsub__": MPSSkipInfo(UNIMPLEMENTED),
    "cauchy_": MPSSkipInfo(UNIMPLEMENTED),
    "cauchy": MPSSkipInfo(UNIMPLEMENTED),
    "cholesky_inverse": MPSSkipInfo(UNIMPLEMENTED),
    "cholesky_solve": MPSSkipInfo(UNIMPLEMENTED),
    "cummax": MPSSkipInfo(UNIMPLEMENTED),
    "cummin": MPSSkipInfo(UNIMPLEMENTED),
    "erfc": MPSSkipInfo(UNIMPLEMENTED),
    "frexp": MPSSkipInfo(UNIMPLEMENTED),
    "gcd": MPSSkipInfo(UNIMPLEMENTED),
    "geqrf": MPSSkipInfo(UNIMPLEMENTED),
    "nn.functional.grid_sample": MPSSkipInfo(
        UNIMPLEMENTED
    ),  # Unsupported Border padding mode
    "heaviside": MPSSkipInfo(UNIMPLEMENTED),
    "igamma": MPSSkipInfo(UNIMPLEMENTED),
    "igammac": MPSSkipInfo(UNIMPLEMENTED),
    "index_copy": MPSSkipInfo(UNIMPLEMENTED),
    "index_reduce": MPSSkipInfo(UNIMPLEMENTED),
    "kthvalue": MPSSkipInfo(UNIMPLEMENTED),
    "lcm": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.cond": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.eigh": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.eigvalsh": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.householder_product": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.ldl_factor": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.ldl_factor_ex": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.ldl_solve": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.lstsq": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.lstsqgrad_oriented": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.lu": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.lu_solve": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.matrix_norm": MPSSkipInfo(UNIMPLEMENTED, dtypes=[torch.float32]),
    "linalg.norm": MPSSkipInfo(UNIMPLEMENTED, dtypes=[torch.float32]),
    "linalg.normsubgradients_at_zero": MPSSkipInfo(
        UNIMPLEMENTED, dtypes=[torch.float32]
    ),
    "linalg.qr": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.slogdet": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.svdvals": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.vecdot": MPSSkipInfo(UNIMPLEMENTED),
    "logcumsumexp": MPSSkipInfo(UNIMPLEMENTED),
    "logdet": MPSSkipInfo(UNIMPLEMENTED),
    "lu_solve": MPSSkipInfo(UNIMPLEMENTED),
    "masked.median": MPSSkipInfo(UNIMPLEMENTED),
    "matrix_exp": MPSSkipInfo(UNIMPLEMENTED),
    "mode": MPSSkipInfo(UNIMPLEMENTED),
    "nanmedian": MPSSkipInfo(UNIMPLEMENTED),
    "native_dropout_backward": MPSSkipInfo(UNIMPLEMENTED),
    "norm": MPSSkipInfo(UNIMPLEMENTED, variant="nuc"),
    "nn.functional.fractional_max_pool2d": MPSSkipInfo(UNIMPLEMENTED),
    "nn.functional.fractional_max_pool3d": MPSSkipInfo(UNIMPLEMENTED),
    "nn.functional.adaptive_avg_pool3d": MPSSkipInfo(UNIMPLEMENTED),
    "nn.functional.adaptive_max_pool3d": MPSSkipInfo(UNIMPLEMENTED),
    "nn.functional.interpolate": [
        MPSSkipInfo(UNIMPLEMENTED, variant="area"),
        MPSSkipInfo(UNIMPLEMENTED, dtypes=[torch.uint8], variant="bicubic"),
        MPSSkipInfo(UNIMPLEMENTED, variant="trilinear"),
    ],
    "nn.functional.max_unpool1dgrad": MPSSkipInfo(UNIMPLEMENTED),
    "nn.functional.max_unpool2dgrad": MPSSkipInfo(UNIMPLEMENTED),
    "nn.functional.max_unpool3dgrad": MPSSkipInfo(UNIMPLEMENTED),
    "nn.functional.avg_pool3d": MPSSkipInfo(UNIMPLEMENTED),
    "nn.functional.ctc_loss": MPSSkipInfo(UNIMPLEMENTED),
    "nn.functional.embedding_bag": MPSSkipInfo(UNIMPLEMENTED),
    "nn.functional.hardshrink": MPSSkipInfo(UNIMPLEMENTED),
    "nn.functional.max_pool3d": MPSSkipInfo(UNIMPLEMENTED),
    "nn.functional.max_unpool1d": MPSSkipInfo(UNIMPLEMENTED),
    "nn.functional.max_unpool2d": MPSSkipInfo(UNIMPLEMENTED),
    "nn.functional.max_unpool3d": MPSSkipInfo(UNIMPLEMENTED),
    "nn.functional.multi_margin_loss": MPSSkipInfo(UNIMPLEMENTED),
    "nn.functional.multilabel_margin_loss": MPSSkipInfo(UNIMPLEMENTED),
    "nn.functional.pdist": MPSSkipInfo(UNIMPLEMENTED),
    "nn.functional.rrelu": MPSSkipInfo(UNIMPLEMENTED),
    "nn.functional.norm": MPSSkipInfo(UNIMPLEMENTED),
    "ormqr": MPSSkipInfo(UNIMPLEMENTED),
    "pca_lowrank": MPSSkipInfo(UNIMPLEMENTED),
    "qr": MPSSkipInfo(UNIMPLEMENTED),
    "rsub": MPSSkipInfo(UNIMPLEMENTED),
    "scatter_reduce": [
        MPSSkipInfo(
            UNIMPLEMENTED,
            dtypes=[torch.int64],
            lower=15.0,
            variant="amax",
        ),
        MPSSkipInfo(
            UNIMPLEMENTED,
            dtypes=[torch.int32, torch.int64],
            upper=15.0,
            variant="amax",
        ),
        MPSSkipInfo(
            UNIMPLEMENTED,
            dtypes=[torch.int64],
            lower=15.0,
            variant="amin",
        ),
        MPSSkipInfo(
            UNIMPLEMENTED,
            dtypes=[torch.int32, torch.int64],
            upper=15.0,
            variant="amin",
        ),
    ],
    "segment_reduce": MPSSkipInfo(UNIMPLEMENTED),
    "_segment.reduce": MPSSkipInfo(UNIMPLEMENTED),
    "segment.reduce": MPSSkipInfo(UNIMPLEMENTED),
    "segment_reduce_offsets": MPSSkipInfo(UNIMPLEMENTED),
    "_segment_reduce": MPSSkipInfo(UNIMPLEMENTED),
    "_segment_reducelengths": MPSSkipInfo(UNIMPLEMENTED),
    "_segment_reduceoffsets": MPSSkipInfo(UNIMPLEMENTED),
    "sparse.mm": MPSSkipInfo(UNIMPLEMENTED),
    "sparse.mmreduce": MPSSkipInfo(UNIMPLEMENTED),
    "special.airy_ai": MPSSkipInfo(UNIMPLEMENTED),
    "special.bessel_j0": MPSSkipInfo(UNIMPLEMENTED),
    "special.bessel_j1": MPSSkipInfo(UNIMPLEMENTED),
    "special.bessel_y0": MPSSkipInfo(UNIMPLEMENTED),
    "special.bessel_y1": MPSSkipInfo(UNIMPLEMENTED),
    "special.chebyshev_polynomial_t": MPSSkipInfo(UNIMPLEMENTED),
    "special.chebyshev_polynomial_u": MPSSkipInfo(UNIMPLEMENTED),
    "special.entr": MPSSkipInfo(UNIMPLEMENTED),
    "special.erfcx": MPSSkipInfo(UNIMPLEMENTED),
    "special.hermite_polynomial_h": MPSSkipInfo(UNIMPLEMENTED),
    "special.hermite_polynomial_he": MPSSkipInfo(UNIMPLEMENTED),
    "special.i0e": MPSSkipInfo(UNIMPLEMENTED),
    "special.i1e": MPSSkipInfo(UNIMPLEMENTED),
    "special.laguerre_polynomial_l": MPSSkipInfo(UNIMPLEMENTED),
    "special.log_ndtr": MPSSkipInfo(UNIMPLEMENTED),
    "special.modified_bessel_i0": MPSSkipInfo(UNIMPLEMENTED),
    "special.modified_bessel_i1": MPSSkipInfo(UNIMPLEMENTED),
    "special.modified_bessel_k0": MPSSkipInfo(UNIMPLEMENTED),
    "special.modified_bessel_k1": MPSSkipInfo(UNIMPLEMENTED),
    "special.ndtri": MPSSkipInfo(UNIMPLEMENTED),
    "special.scaled_modified_bessel_k0": MPSSkipInfo(UNIMPLEMENTED),
    "special.scaled_modified_bessel_k1": MPSSkipInfo(UNIMPLEMENTED),
    "special.xlog1py": MPSSkipInfo(
        UNIMPLEMENTED,
        dtypes=[
            torch.bool,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.int8,
            torch.uint8,
        ],
    ),
    "special.zeta": MPSSkipInfo(
        UNIMPLEMENTED,
        dtypes=[
            torch.bool,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.int8,
            torch.uint8,
        ],
    ),
    "svd_lowrank": MPSSkipInfo(UNIMPLEMENTED),
    "symeig": MPSSkipInfo(UNIMPLEMENTED),
    "take": MPSSkipInfo(UNIMPLEMENTED),
    "to": MPSSkipInfo(UNIMPLEMENTED),
    "to_sparse": MPSSkipInfo(UNIMPLEMENTED),
    "unique": MPSSkipInfo(UNIMPLEMENTED),
    "vdot": MPSSkipInfo(UNIMPLEMENTED),
    "segment_reduce_": MPSSkipInfo(UNIMPLEMENTED),
    "geometric": MPSSkipInfo(UNIMPLEMENTED),
    "geometric_": MPSSkipInfo(UNIMPLEMENTED),
    "log_normal_": MPSSkipInfo(UNIMPLEMENTED),
    "log_normal": MPSSkipInfo(UNIMPLEMENTED),
    "cdouble": MPSSkipInfo(UNIMPLEMENTED),
    "double": MPSSkipInfo(UNIMPLEMENTED),
    "nn.functional.softmin": MPSSkipInfo(UNIMPLEMENTED, variant="with_dtype"),
    "log_softmax": MPSSkipInfo(UNIMPLEMENTED, variant="with_dtype"),
    "softmax": MPSSkipInfo(UNIMPLEMENTED, variant="with_dtype"),
    "float_power": MPSSkipInfo(UNIMPLEMENTED),
    "full_like": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.pinv": MPSSkipInfo(UNIMPLEMENTED, variant="hermitian"),
    "nonzero_static": MPSSkipInfo(UNIMPLEMENTED),
    # MPS: input sizes must be divisible by output sizes
    "nn.functional.adaptive_avg_pool1d": MPSSkipInfo(UNIMPLEMENTED),
    "nn.functional.adaptive_avg_pool2d": MPSSkipInfo(UNIMPLEMENTED),
    # Unsupported dtypes
    "linalg.lu_factor": MPSSkipInfo(
        UNIMPLEMENTED,
        dtypes=[torch.float16],  # missing `aten::lu_unpack`.
    ),
    "ones_like": MPSSkipInfo(UNIMPLEMENTED),
    "zeros_like": MPSSkipInfo(UNIMPLEMENTED),
    # Convolution for integral types is not supported on MPS
    "nn.functional.conv1d": MPSSkipInfo(UNIMPLEMENTED, dtypes=[torch.int64]),
    "nn.functional.conv2d": MPSSkipInfo(UNIMPLEMENTED, dtypes=[torch.int64]),
    "nn.functional.conv3d": MPSSkipInfo(UNIMPLEMENTED, dtypes=[torch.int64]),
    "nn.functional.conv_transpose1d": MPSSkipInfo(UNIMPLEMENTED, dtypes=[torch.int64]),
    "nn.functional.conv_transpose2d": MPSSkipInfo(
        UNIMPLEMENTED, dtypes=[torch.int64, torch.bfloat16]
    ),
    # Unsupported dtypes
    "histc": MPSSkipInfo(UNIMPLEMENTED, dtypes=[torch.float16, torch.bfloat16]),
    "index_add": MPSSkipInfo(UNIMPLEMENTED, dtypes=[torch.int64]),
    "log1p": MPSSkipInfo(UNIMPLEMENTED, dtypes=[torch.int64]),
    "sigmoid": MPSSkipInfo(UNIMPLEMENTED, dtypes=[torch.int64]),
    "atan2": MPSSkipInfo(UNIMPLEMENTED, dtypes=[torch.int64]),
    "angle": MPSSkipInfo(UNIMPLEMENTED, dtypes=[torch.int64]),
    # GEMM on MPS is not supported for integral types
    "nn.functional.linear": MPSSkipInfo(
        UNIMPLEMENTED,
        dtypes=[torch.int16, torch.int32, torch.int64, torch.uint8, torch.int8],
    ),
    "addmmdecomposed": MPSSkipInfo(
        UNIMPLEMENTED,
        dtypes=[torch.int16, torch.int32, torch.int64, torch.uint8, torch.int8],
    ),
    "addbmm": MPSSkipInfo(
        UNIMPLEMENTED,
        dtypes=[torch.int16, torch.int32, torch.int64, torch.uint8, torch.int8],
    ),
    "addmm": MPSSkipInfo(
        UNIMPLEMENTED,
        dtypes=[torch.int16, torch.int32, torch.int64, torch.uint8, torch.int8],
    ),
    "baddbmm": MPSSkipInfo(
        UNIMPLEMENTED,
        dtypes=[torch.int16, torch.int32, torch.int64, torch.uint8, torch.int8],
    ),
    "mat": MPSSkipInfo(
        UNIMPLEMENTED,
        dtypes=[torch.int16, torch.int32, torch.int64, torch.uint8, torch.int8],
    ),
    "unravel_index": MPSSkipInfo(UNIMPLEMENTED, dtypes=[torch.int32, torch.int64]),
    # returned output on CPU is float64
    "bincount": MPSSkipInfo(
        UNIMPLEMENTED,
        dtypes=[torch.int16, torch.int32, torch.int64, torch.uint8, torch.int8],
    ),
    # trunc_tensor not working properly for float16 and bfloat16
    "div": [
        MPSSkipInfo(
            UNIMPLEMENTED,
            dtypes=[torch.float16, torch.bfloat16],
            variant="trunc_rounding",
        ),
        MPSSkipInfo(
            UNIMPLEMENTED,
            dtypes=[torch.bfloat16],  # bfloat16 has issues with rounding
            variant="floor_rounding",
        ),
    ],
    "fmod": MPSSkipInfo(UNIMPLEMENTED, dtypes=[torch.float16]),
    # bfloat16 have weird issues with rounding
    "floor_divide": MPSSkipInfo(UNIMPLEMENTED, dtypes=[torch.bfloat16]),
    "remainder": MPSSkipInfo(UNIMPLEMENTED, dtypes=[torch.bfloat16]),
    # atomic operations not supported
    "_unsafe_masked_index_put_accumulate": MPSSkipInfo(
        UNIMPLEMENTED,
        dtypes=[
            torch.bool,
            torch.int8,
            torch.uint8,
            torch.float16,
            torch.int16,
            torch.int64,
            torch.bfloat16,
        ],
    ),
    "_upsample_bilinear2d_aa": MPSSkipInfo(
        TEST_OUTPUT_MATCH,
        dtypes=[torch.uint8],
    ),
}

"""Expected failures due to backwards pass issues"""
XFAILLIST_GRAD = {
    # precision issues
    "special.polygamma": MPSSkipInfo(
        TEST_OUTPUT_GRAD_MATCH, dtypes=[torch.float16], variant="special_polygamma_n_0"
    ),
    "polygamma": MPSSkipInfo(
        TEST_OUTPUT_GRAD_MATCH, dtypes=[torch.float16], variant="polygamma_n_0"
    ),
    "nn.functional.binary_cross_entropy": MPSSkipInfo(
        TEST_OUTPUT_GRAD_MATCH, dtypes=[torch.float16]
    ),
    # Unimplemented ops
    "_upsample_bilinear2d_aa": MPSSkipInfo(TEST_OUTPUT_GRAD_MATCH),
    "linalg.det": MPSSkipInfo(TEST_OUTPUT_GRAD_MATCH),
    "linalg.solve": MPSSkipInfo(TEST_OUTPUT_GRAD_MATCH),
    "linalg.solve_ex": MPSSkipInfo(TEST_OUTPUT_GRAD_MATCH),
    "linalg.tensorsolve": MPSSkipInfo(TEST_OUTPUT_GRAD_MATCH),
    "special.zeta": MPSSkipInfo(TEST_OUTPUT_GRAD_MATCH),
    "__getitem__": MPSSkipInfo(TEST_OUTPUT_GRAD_MATCH, dtypes=[torch.float16]),
    "_segment_reduce": MPSSkipInfo(
        TEST_OUTPUT_GRAD_MATCH, dtypes=[torch.float16, torch.float32]
    ),
    "_chunk_cat": MPSSkipInfo(
        TEST_OUTPUT_GRAD_MATCH, dtypes=[torch.float16, torch.float32]
    ),
    "sparse.mmreduce": MPSSkipInfo(
        TEST_OUTPUT_GRAD_MATCH, dtypes=[torch.float32]
    ),  # csr not supported
    "unique_consecutive": MPSSkipInfo(
        TEST_OUTPUT_GRAD_MATCH, dtypes=[torch.float16, torch.float32]
    ),
    "special_modified_bessel_i0": MPSSkipInfo(
        TEST_OUTPUT_GRAD_MATCH, dtypes=[torch.float16, torch.float32]
    ),
    "scalar_tensor": MPSSkipInfo(
        TEST_OUTPUT_GRAD_MATCH, dtypes=[torch.float16, torch.float32]
    ),
    "cdist": MPSSkipInfo(TEST_OUTPUT_GRAD_MATCH, dtypes=[torch.float32]),
    "masked.scatter": MPSSkipInfo(
        TEST_OUTPUT_GRAD_MATCH, dtypes=[torch.float16, torch.float32]
    ),
    "index_fill": MPSSkipInfo(
        TEST_OUTPUT_GRAD_MATCH, dtypes=[torch.float16, torch.float32]
    ),  # missing `aten::_unique`.
    "aminmax": MPSSkipInfo(
        TEST_OUTPUT_GRAD_MATCH, dtypes=[torch.float32, torch.float16]
    ),
    "special.i1": MPSSkipInfo(
        TEST_OUTPUT_GRAD_MATCH, dtypes=[torch.float16]
    ),  # "i1_backward" not implemented for 'Half'
    # Correctness issues
    "atanh": MPSSkipInfo(TEST_OUTPUT_GRAD_MATCH, dtypes=[torch.float32]),
    # Random output
    "exponential": MPSSkipInfo(
        TEST_OUTPUT_GRAD_MATCH, dtypes=[torch.float16, torch.float32]
    ),
    # CPU errors
    # derivative for aten::nextafter is not implemented on CPU
    "nextafter": MPSSkipInfo(TEST_OUTPUT_GRAD_MATCH),
    # derivative for aten::floor_divide is not implemented on CPU
    "floor_divide": MPSSkipInfo(
        TEST_OUTPUT_GRAD_MATCH, dtypes=[torch.float16, torch.float32]
    ),
    # derivative for aten::narrow_copy is not implemented on CPU
    "narrow_copy": MPSSkipInfo(
        TEST_OUTPUT_GRAD_MATCH, dtypes=[torch.float16, torch.float32]
    ),
    # derivative for aten::_histogramdd_from_bin_cts is not implemented on CPU
    "histogramdd": MPSSkipInfo(
        TEST_OUTPUT_GRAD_MATCH, dtypes=[torch.float16, torch.float32]
    ),
    # derivative for aten::histogram is not implemented
    "histogram": MPSSkipInfo(
        TEST_OUTPUT_GRAD_MATCH, dtypes=[torch.float16, torch.float32]
    ),
    # 'bool' object is not iterable
    "allclose": MPSSkipInfo(
        TEST_OUTPUT_GRAD_MATCH, dtypes=[torch.float16, torch.float32]
    ),
    "equal": MPSSkipInfo(TEST_OUTPUT_GRAD_MATCH, dtypes=[torch.float16, torch.float32]),
    # 'float' object is not iterable
    "item": MPSSkipInfo(TEST_OUTPUT_GRAD_MATCH, dtypes=[torch.float16, torch.float32]),
    # "mse_backward_cpu_out" not implemented for 'Half'
    "nn.functional.mse_loss": MPSSkipInfo(
        TEST_OUTPUT_GRAD_MATCH, dtypes=[torch.float16]
    ),
    # "smooth_l1_backward_cpu_out" not implemented for 'Half'
    "nn.functional.smooth_l1_loss": MPSSkipInfo(
        TEST_OUTPUT_GRAD_MATCH, dtypes=[torch.float16]
    ),
    # cpu error: grad requires non-empty inputs
    "randn": MPSSkipInfo(TEST_OUTPUT_GRAD_MATCH, dtypes=[torch.float16, torch.float32]),
    "signal.windows.bartlett": MPSSkipInfo(
        TEST_OUTPUT_GRAD_MATCH, dtypes=[torch.float32]
    ),
    "signal.windows.blackman": MPSSkipInfo(
        TEST_OUTPUT_GRAD_MATCH, dtypes=[torch.float32]
    ),
    "signal.windows.cosine": MPSSkipInfo(
        TEST_OUTPUT_GRAD_MATCH, dtypes=[torch.float32]
    ),
    "signal.windows.exponential": MPSSkipInfo(
        TEST_OUTPUT_GRAD_MATCH, dtypes=[torch.float32]
    ),
    "signal.windows.gaussian": MPSSkipInfo(
        TEST_OUTPUT_GRAD_MATCH, dtypes=[torch.float32]
    ),
    "signal.windows.general_cosine": MPSSkipInfo(
        TEST_OUTPUT_GRAD_MATCH, dtypes=[torch.float32]
    ),
    "signal.windows.general_hamming": MPSSkipInfo(
        TEST_OUTPUT_GRAD_MATCH, dtypes=[torch.float32]
    ),
    "signal.windows.hamming": MPSSkipInfo(
        TEST_OUTPUT_GRAD_MATCH, dtypes=[torch.float32]
    ),
    "signal.windows.hann": MPSSkipInfo(TEST_OUTPUT_GRAD_MATCH, dtypes=[torch.float32]),
    "signal.windows.kaiser": MPSSkipInfo(
        TEST_OUTPUT_GRAD_MATCH, dtypes=[torch.float32]
    ),
    "signal.windows.nuttall": MPSSkipInfo(
        TEST_OUTPUT_GRAD_MATCH, dtypes=[torch.float32]
    ),
    "eye": MPSSkipInfo(TEST_OUTPUT_GRAD_MATCH, dtypes=[torch.float16, torch.float32]),
    # trunc_tensor not working properly for float16
    "divtrunc_rounding": MPSSkipInfo(TEST_OUTPUT_GRAD_MATCH, dtypes=[torch.float16]),
    "fmod": MPSSkipInfo(TEST_OUTPUT_GRAD_MATCH, dtypes=[torch.float16]),
    # atomic operation in backward pass
    "_unsafe_masked_index": MPSSkipInfo(TEST_OUTPUT_GRAD_MATCH, dtypes=[torch.float16]),
    "_unsafe_masked_index_put_accumulate": MPSSkipInfo(
        TEST_OUTPUT_GRAD_MATCH, dtypes=[torch.float16]
    ),
    # Same issue as `argsort` and `sort` with duplicate elements (undefined behaviour).
    # Forward pass is passing since `msort` doesn't return the indices, just the values, which match the CPU.
    # On the backward pass for `sort` both are used (values and indices), thus resulting in a issmatch between CPU and MPS.
    # Running `msort` with stable `sort` passes.
    "msort": MPSSkipInfo(TEST_OUTPUT_GRAD_MATCH, dtypes=[torch.float16]),
    "nn.functional.pairwise_distance": MPSSkipInfo(
        TEST_OUTPUT_GRAD_MATCH, dtypes=[torch.float16]
    ),
    "nn.functional.conv1d": MPSSkipInfo(
        TEST_OUTPUT_GRAD_MATCH, dtypes=[torch.float16], lower=14.0
    ),
    "nn.functional.conv2d": MPSSkipInfo(
        TEST_OUTPUT_GRAD_MATCH, dtypes=[torch.float16], lower=14.0
    ),
    "nn.functional.conv3d": MPSSkipInfo(
        TEST_OUTPUT_GRAD_MATCH, dtypes=[torch.float16, torch.float32], lower=14.0
    ),
    # Uncategorized grad failures
    "nanquantile": MPSSkipInfo(
        TEST_OUTPUT_GRAD_MATCH,
        dtypes=[torch.float32],
        upper=15.0,
    ),
    "quantile": MPSSkipInfo(
        TEST_OUTPUT_GRAD_MATCH,
        dtypes=[torch.float32],
        upper=15.0,
    ),
}

COMPLEX_DTYPES = [torch.complex32, torch.complex64]
# NOTE - torch.complex64 is always skipped when MacOS<14, see mps_op_db()
COMPLEX_XFAILLIST = {
    "__rpow__": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "_unsafe_masked_index_put_accumulate": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "addbmm": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "addmm": [
        MPSSkipInfo(
            dtypes=COMPLEX_DTYPES,
            variant=" ",
        ),
    ],
    "addr": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "asinh": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "baddbmm": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "block_diag": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "cholesky": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "cov": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "cross": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "cumprod": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "cumsum": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "cumulative_trapezoid": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "dist": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "div": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
        variant=" ",
    ),
    "gather": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "index_add": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "index_fill": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "index_put": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "istft": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "lerp": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "linalg.cholesky": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "linalg.cholesky_ex": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "linalg.cross": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "linalg.det": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "linalg.inv": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "linalg.inv_ex": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "linalg.lu_factor": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "linalg.lu_factor_ex": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "linalg.matrix_norm": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "linalg.matrix_power": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "linalg.norm": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "linalg.solve": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "linalg.solve_ex": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "linalg.solve_triangular": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "linalg.tensorinv": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "linalg.tensorsolve": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "linalg.vander": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "linalg.vector_norm": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "logaddexp": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "lu": MPSSkipInfo(dtypes=[torch.complex64]),
    "lu_unpack": MPSSkipInfo(dtypes=[torch.complex64]),
    "masked.cumprod": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "masked.cumsum": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "masked.normalize": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "nn.functional.conv3d": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "nn.functional.l1_loss": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "nn.functional.linear": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "nn.functional.normalize": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "nn.functional.padconstant": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "nn.functional.padreflect": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "nn.functional.padreplicate": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "nn.functional.pairwise_distance": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "nn.functional.silu": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "nn.functional.triplet_margin_loss": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "nn.functional.triplet_margin_with_distance_loss": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES
    ),
    "norm": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "normal": MPSSkipInfo(dtypes=COMPLEX_DTYPES, variant="in_place"),
    "pow": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "rand_like": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "renorm": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "repeat": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "resize_": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "resize_as_": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "scatter_add": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "scatter": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "scatter_reduce": [
        MPSSkipInfo(dtypes=COMPLEX_DTYPES, variant="amax"),
        MPSSkipInfo(dtypes=COMPLEX_DTYPES, variant="amin"),
    ],
    "std": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "std_mean": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "take_along_dim": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "tile": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "triangular_solve": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "uniform": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "var": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "var_mean": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
}

"""Failures due to known errors in the MPS backend"""
MPS_DOWNSTREAM_XFAILLIST = {
    "arange": MPSSkipInfo(dtypes=[torch.uint8]),
    # TODO: remove this once downstream function 'aten::_linalg_svd.U' have been implemented
    "linalg.matrix_rank": MPSSkipInfo(),
    # MPS returns incorrect results before MacOS15
    "nanquantile": MPSSkipInfo(TEST_OUTPUT_MATCH, dtypes=[torch.float32], upper=15.0),
    "quantile": MPSSkipInfo(TEST_OUTPUT_MATCH, dtypes=[torch.float32], upper=15.0),
    "nn.functional.conv3d": MPSSkipInfo(TEST_OUTPUT_MATCH, upper=15.0),
}

"""XFails specific to MacOS 13"""
MACOS_13_XFAILLIST = {
    # FFT and BFloat16 support was added in MacOS 14
    "bfloat16": MPSSkipInfo(UNIMPLEMENTED, upper=14.0),
    "fft.fft": MPSSkipInfo(UNIMPLEMENTED, upper=14.0),
    "fft.fft2": MPSSkipInfo(UNIMPLEMENTED, upper=14.0),
    "fft.fftn": MPSSkipInfo(UNIMPLEMENTED, upper=14.0),
    "fft.hfft": MPSSkipInfo(UNIMPLEMENTED, upper=14.0),
    "fft.hfft2": MPSSkipInfo(UNIMPLEMENTED, upper=14.0),
    "fft.hfftn": MPSSkipInfo(UNIMPLEMENTED, upper=14.0),
    "fft.ifft": MPSSkipInfo(UNIMPLEMENTED, upper=14.0),
    "fft.ifft2": MPSSkipInfo(UNIMPLEMENTED, upper=14.0),
    "fft.ifftn": MPSSkipInfo(UNIMPLEMENTED, upper=14.0),
    "fft.ihfft": MPSSkipInfo(UNIMPLEMENTED, upper=14.0),
    "fft.ihfft2": MPSSkipInfo(UNIMPLEMENTED, upper=14.0),
    "fft.ihfftn": MPSSkipInfo(UNIMPLEMENTED, upper=14.0),
    "fft.irfft": MPSSkipInfo(UNIMPLEMENTED, upper=14.0),
    "fft.irfft2": MPSSkipInfo(UNIMPLEMENTED, upper=14.0),
    "fft.irfftn": MPSSkipInfo(UNIMPLEMENTED, upper=14.0),
    "fft.rfft": MPSSkipInfo(UNIMPLEMENTED, upper=14.0),
    "fft.rfft2": MPSSkipInfo(UNIMPLEMENTED, upper=14.0),
    "fft.rfftn": MPSSkipInfo(UNIMPLEMENTED, upper=14.0),
    "stft": MPSSkipInfo(UNIMPLEMENTED, upper=14.0),
    "__rmatmul__": MPSSkipInfo(UNIMPLEMENTED, dtypes=[torch.int64], upper=14.0),
    # Precision issues
    "atan2": MPSSkipInfo(
        TEST_OUTPUT_MATCH,
        dtypes=[torch.uint8, torch.int8, torch.int16, torch.int32, torch.bool],
        upper=14.0,
    ),
    "cdist": MPSSkipInfo(TEST_OUTPUT_MATCH, upper=14.0),
    "cumsum": MPSSkipInfo(TEST_OUTPUT_MATCH, dtypes=[torch.int64], upper=14.0),
    "matmul": MPSSkipInfo(TEST_OUTPUT_MATCH, upper=14.0, dtypes=[torch.int64]),
    "linalg.vander": MPSSkipInfo(TEST_OUTPUT_MATCH, dtypes=[torch.int64], upper=14.0),
    "masked.cumsum": MPSSkipInfo(TEST_OUTPUT_MATCH, dtypes=[torch.int64], upper=14.0),
    "masked.log_softmax": MPSSkipInfo(TEST_OUTPUT_GRAD_MATCH, upper=14.0),
    "masked.softmin": MPSSkipInfo(TEST_OUTPUT_GRAD_MATCH, upper=14.0),
    "masked.softmax": MPSSkipInfo(TEST_OUTPUT_GRAD_MATCH, upper=14.0),
    "cumulative_trapezoid": MPSSkipInfo(
        TEST_OUTPUT_MATCH, dtypes=[torch.int64], upper=14.0
    ),
    "dot": MPSSkipInfo(TEST_OUTPUT_MATCH, dtypes=[torch.int64], upper=14.0),
    "nn.functional.max_pool2d": MPSSkipInfo(
        TEST_OUTPUT_MATCH,
        upper=14.0,
        dtypes=[torch.uint8],
    ),
    "tan": MPSSkipInfo(
        TEST_OUTPUT_MATCH, TEST_OUTPUT_GRAD_MATCH, upper=14.0, dtypes=[torch.float32]
    ),
    # Cumprod int64 support added in MacOS 13.3
    "cumprod": MPSSkipInfo(UNIMPLEMENTED, upper=13.3, dtypes=[torch.int64]),
    "masked.cumprod": MPSSkipInfo(UNIMPLEMENTED, upper=13.3, dtypes=[torch.int64]),
    # isin non-float support added in MacOS 14.0
    "isin": MPSSkipInfo(
        TEST_OUTPUT_MATCH,
        dtypes=[torch.int64, torch.int8, torch.uint8, torch.int16, torch.int32],
        upper=14.0,
    ),
    # Hard crash on MacOS 13 - failed assertion `destination datatype must be fp32'
    "nn.functional.pairwise_distance": MPSSkipInfo(
        TEST_OUTPUT_GRAD_MATCH,
        skip=unittest.skip("Hard crash on MacOS13"),
        dtypes=[torch.float16],
        upper=14.0,
    ),
    "nn.functional.conv1d": MPSSkipInfo(
        TEST_OUTPUT_GRAD_MATCH,
        skip=unittest.skip("Hard crash on MacOS13"),
        dtypes=[torch.float16],
        upper=14.0,
    ),
    "nn.functional.conv2d": MPSSkipInfo(
        TEST_OUTPUT_GRAD_MATCH,
        skip=unittest.skip("Hard crash on MacOS13"),
        dtypes=[torch.float16],
        upper=14.0,
    ),
    "nn.functional.conv3d": MPSSkipInfo(
        TEST_OUTPUT_GRAD_MATCH,
        skip=unittest.skip("Hard crash on MacOS13"),
        dtypes=[torch.float16, torch.float32],
        upper=14.0,
    ),
    "nn.functional.conv_transpose1d": MPSSkipInfo(
        TEST_OUTPUT_GRAD_MATCH,
        skip=unittest.skip("Hard crash on MacOS13"),
        dtypes=[torch.float16],
        upper=14.0,
    ),
    "nn.functional.conv_transpose2d": MPSSkipInfo(
        TEST_OUTPUT_GRAD_MATCH,
        skip=unittest.skip("Hard crash on MacOS13"),
        dtypes=[torch.float16],
        upper=14.0,
    ),
    "nn.functional.conv_transpose3d": MPSSkipInfo(
        TEST_OUTPUT_GRAD_MATCH,
        skip=unittest.skip("Hard crash on MacOS13"),
        dtypes=[torch.float16, torch.float32],
        upper=14.0,
    ),
}

"""Other uncategorized xfails"""
OTHER_XFAILLIST = {
    # Since CPU is not using argsort with stable=True, these cases result in undefined behaviour.
    "argsort": MPSSkipInfo(
        dtypes=[torch.float16, torch.int8, torch.uint8, torch.bool, torch.bfloat16]
    ),
    # Same issue as `argsort` with duplicate indices. This test checks both the sorted values and the indices.
    # The values of the sorted tensor match the CPU, but in case of the returned indices this results in undefined behaviour.
    "sort": MPSSkipInfo(
        dtypes=[torch.int8, torch.uint8, torch.bool, torch.float16, torch.bfloat16]
    ),
    # topk fails with duplicate indices
    "topk": MPSSkipInfo(
        dtypes=[
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
            torch.int8,
            torch.bfloat16,
        ]
    ),
    # Failures due to random output that they generate using
    # Philox engine causing mismatch with CPU results
    "multinomial": MPSSkipInfo(
        dtypes=[torch.float16, torch.float32, torch.bfloat16],  # random results
    ),
    "uniform": MPSSkipInfo(dtypes=[torch.float16, torch.float32, torch.bfloat16]),
    "rand_like": MPSSkipInfo(dtypes=[torch.float16, torch.float32, torch.bfloat16]),
    "randint": MPSSkipInfo(),
    "randint_like": MPSSkipInfo(),
    "randn": MPSSkipInfo(),
    "randn_like": MPSSkipInfo(),
    "bernoulli": MPSSkipInfo(dtypes=[torch.float16, torch.float32, torch.bfloat16]),
    "exponential": MPSSkipInfo(dtypes=[torch.float16, torch.float32, torch.bfloat16]),
    "nn.functional.feature_alpha_dropout": MPSSkipInfo(
        dtypes=[torch.float16, torch.float32, torch.bfloat16],
        variant="with_train",
    ),
    "normal": MPSSkipInfo(dtypes=[torch.float16, torch.float32, torch.bfloat16]),
    "nn.functional.alpha_dropout": MPSSkipInfo(
        dtypes=[torch.float16, torch.float32, torch.bfloat16]
    ),
    "nn.functional.dropout": MPSSkipInfo(
        dtypes=[torch.float16, torch.float32, torch.bfloat16]
    ),
    "nn.functional.dropout2d": MPSSkipInfo(
        dtypes=[torch.float16, torch.float32, torch.bfloat16]
    ),
    "nn.functional.dropout3d": MPSSkipInfo(
        dtypes=[torch.float16, torch.float32, torch.bfloat16]
    ),
    # See https://github.com/pytorch/pytorch/issues/111479
    "nn.functional.multi_head_attention_forward": MPSSkipInfo(
        dtypes=[torch.float32, torch.float16, torch.bfloat16]
    ),
    "index_put": MPSSkipInfo(
        dtypes=[
            torch.bool,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int64,
            torch.float16,
            torch.bfloat16,
        ]
    ),
    # zero to negative integer powers are undefined
    "__rpow__": MPSSkipInfo(
        dtypes=[torch.int8, torch.int16, torch.int32, torch.int64],
    ),
    "resize_": MPSSkipInfo(dtypes=[torch.float16, torch.float32, torch.bfloat16]),
    "resize_as_": MPSSkipInfo(dtypes=[torch.float16, torch.float32, torch.bfloat16]),
    # before macOS 13.2 it falls back to cpu and pass the forward pass
    "grid_sampler_2d": MPSSkipInfo(
        dtypes=[
            torch.float32,
            torch.float16,
            torch.bfloat16,
        ],  # Unsupported Border padding mode
        lower=13.2,
    ),
    # CPU Errors:
    "addr": MPSSkipInfo(
        dtypes=[
            torch.bool,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
            torch.int8,
        ],
    ),  # "addmv_impl_cpu" not implemented for 'Half'
    "as_strided": MPSSkipInfo(
        variant="partial_views"
    ),  # cpu result off, showing random values
    # random results
    # mps vs cpu:
    # Mismatched elements: 40 / 96 (41.7%)
    # Greatest absolute difference: 17.892311096191406 at index (1, 0, 2) (up to 1e-05 allowed)
    # Greatest relative difference: inf at index (1, 0, 0) (up to 1.3e-06 allowed)
    # cuda(2.0.0.dev20230301+cu117) vs cpu:
    # Mismatched elements: 56 / 96 (58.3%)
    # Greatest absolute difference: 17.892311096191406 at index (1, 0, 2) (up to 1e-05 allowed)
    # Greatest relative difference: inf at index (1, 0, 0) (up to 1.3e-06 allowed)
    "nn.functional.scaled_dot_product_attention": MPSSkipInfo(
        dtypes=[torch.float32, torch.float16, torch.bfloat16]
    ),
    # float output for float16 input on MPS
    "logit": MPSSkipInfo(dtypes=[torch.float16, torch.bfloat16]),
    # Fill tensors with uninitialized data, causing mismatch with CPU.
    # They occasionally match, thus skipping them.
    # See https://github.com/pytorch/pytorch/issues/100175
    "new_empty": MPSSkipInfo(),
    "new_empty_strided": MPSSkipInfo(),
    "empty_strided": MPSSkipInfo(),
    # CPU: empty is returning all 0's and there is a mismatch with MPS
    # allocation (MacOS 13). According to
    # https://pytorch.org/docs/2.0/generated/torch.empty.html
    "empty": MPSSkipInfo(),
    "empty_like": MPSSkipInfo(),
    "empty_permuted": MPSSkipInfo(TEST_OUTPUT_MATCH, TEST_OUTPUT_GRAD_MATCH),
    # round has precision issues with float16 dtypes
    "round": [
        MPSSkipInfo(
            TEST_OUTPUT_MATCH, variant=" ", dtypes=[torch.float16, torch.bfloat16]
        ),
        MPSSkipInfo(TEST_OUTPUT_GRAD_MATCH, variant=" ", dtypes=[torch.float16]),
        MPSSkipInfo(TEST_OUTPUT_MATCH, variant="decimals_0", dtypes=[torch.bfloat16]),
        MPSSkipInfo(TEST_OUTPUT_MATCH, variant="decimals_3", dtypes=[torch.float16]),
        MPSSkipInfo(
            TEST_OUTPUT_MATCH, variant="decimals_neg_3", dtypes=[torch.float16]
        ),
    ],
}

ERRORINPUT_XFAILLIST = {
    # Exceptions are not raised
    "__rmod__": MPSSkipInfo(TEST_ERROR_INPUTS),
    "__rsub__": MPSSkipInfo(TEST_ERROR_INPUTS),
    "__rpow__": MPSSkipInfo(TEST_ERROR_INPUTS),
    "bernoulli": MPSSkipInfo(TEST_ERROR_INPUTS),
    "clamp_max": MPSSkipInfo(TEST_ERROR_INPUTS),
    "clamp_min": MPSSkipInfo(TEST_ERROR_INPUTS),
    "masked_scatter": MPSSkipInfo(TEST_ERROR_INPUTS),
    # unsupported float64 dtype
    "cat": MPSSkipInfo(TEST_ERROR_INPUTS),
    "complex": MPSSkipInfo(TEST_ERROR_INPUTS),
    "multinomial": MPSSkipInfo(TEST_ERROR_INPUTS),
    "nn.functional.conv1d": MPSSkipInfo(TEST_ERROR_INPUTS),
    "nn.functional.conv2d": MPSSkipInfo(TEST_ERROR_INPUTS),
    "nn.functional.conv3d": MPSSkipInfo(TEST_ERROR_INPUTS),
    "gather": MPSSkipInfo(TEST_ERROR_INPUTS),
    "scatter": MPSSkipInfo(TEST_ERROR_INPUTS),
    "scatter_add": MPSSkipInfo(TEST_ERROR_INPUTS),
    # MPS does not support tensor dimensions > 16
    "amax": MPSSkipInfo(TEST_ERROR_INPUTS),
    "amin": MPSSkipInfo(TEST_ERROR_INPUTS),
    "aminmax": MPSSkipInfo(TEST_ERROR_INPUTS),
    # memory overlapping checks
    "index_select": MPSSkipInfo(TEST_ERROR_INPUTS),
    # unimplemented
    "logcumsumexp": MPSSkipInfo(TEST_ERROR_INPUTS),
}


MPS_OPINFO: dict[str, list[MPSSkipInfo]] = {}


def append_skips(skip_list: dict) -> None:
    for op_name, skip in skip_list.items():
        if not isinstance(skip, Iterable):
            skip = [skip]
        if op_name in MPS_OPINFO:
            MPS_OPINFO[op_name] += skip
        else:
            MPS_OPINFO[op_name] = skip


append_skips(UNIMPLEMENTED_XFAILIST)
append_skips(XFAILLIST_GRAD)
append_skips(COMPLEX_XFAILLIST)
append_skips(OTHER_XFAILLIST)
append_skips(MPS_DOWNSTREAM_XFAILLIST)
append_skips(MACOS_13_XFAILLIST)
append_skips(ERRORINPUT_XFAILLIST)


def mps_op_db(op_db: list[OpInfo]) -> list[OpInfo]:
    """Utility function for OpInfo tests, updates the op_db with xfails defined in MPS_OPINFO_SKIPLIST"""

    for op in op_db:
        if op.name in MPS_OPINFO:
            if not isinstance(MPS_OPINFO[op.name], Iterable):
                skips: list[MPSSkipInfo] = [MPS_OPINFO[op.name]]  # type: ignore[list-item]
            else:
                skips: list[MPSSkipInfo] = MPS_OPINFO[op.name]  # type: ignore[no-redef]

            for skip in skips:
                # If the SkipInfo specified an OS range or a test variant, make sure it is valid
                if (
                    (not skip.upper or MACOS_VERSION < skip.upper)
                    and (not skip.lower or MACOS_VERSION >= skip.lower)
                    and (
                        (not skip.variant or op.variant_test_name == skip.variant)
                        or (op.variant_test_name == "" and skip.variant == " ")
                    )
                ):
                    if skip.tests == []:
                        decorator = DecorateInfo(
                            skip.skip,
                            skip.test_class,
                            None,
                            dtypes=skip.dtypes,
                        )
                        op.decorators = op.decorators + (decorator,)
                    else:
                        for test in skip.tests:
                            decorator = DecorateInfo(
                                skip.skip,
                                skip.test_class,
                                test,
                                dtypes=skip.dtypes,
                            )
                            op.decorators = op.decorators + (decorator,)
            
            if MACOS_VERSION < 14.0:
                # Skip complex64 dtypes before MacOS14
                decorator = DecorateInfo(
                    unittest.skip("Complex dtypes not supported prior to MacOS14"),
                    dtypes=[torch.complex64],
                )
                op.decorators = op.decorators + (decorator,)

    return op_db
