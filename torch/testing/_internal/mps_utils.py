import unittest
import warnings
from typing import Callable, Dict, Iterable, List, Optional, Union

import torch
from torch.testing._internal.common_utils import MACOS_VERSION
from torch.testing._internal.opinfo.core import DecorateInfo, OpInfo


UNIMPLEMENTED = "unimplemented"
TEST_OUTPUT_MATCH = "test_output_match"
TEST_OUTPUT_GRAD_MATCH = "test_output_grad_match"
TEST_ERROR_INPUTS = "test_error_inputs"


COMMON = "TestCommon"


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
        dtypes: Optional[Union[torch.dtype, List[torch.dtype]]] = None,
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
        self.tests: List[str] = []
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
            self.tests = []
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
    "round": [
        MPSSkipInfo(UNIMPLEMENTED, variant="decimals_neg_3"),
        MPSSkipInfo(UNIMPLEMENTED, variant="decimals_3"),
        MPSSkipInfo(UNIMPLEMENTED, variant="decimals_0"),
        # round not working properly for float16 and bfloat16
        MPSSkipInfo(UNIMPLEMENTED, dtypes=[torch.float16, torch.bfloat16]),
    ],
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
    "linalg.cholesky_ex": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.cond": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.det": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.eigh": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.eigvalsh": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.householder_product": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.ldl_factor": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.ldl_factor_ex": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.ldl_solve": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.lstsq": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.lstsqgrad_oriented": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.lu": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.lu_factor_ex": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.lu_solve": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.matrix_norm": MPSSkipInfo(UNIMPLEMENTED, dtypes=[torch.float32]),
    "linalg.norm": MPSSkipInfo(UNIMPLEMENTED, dtypes=[torch.float32]),
    "linalg.normsubgradients_at_zero": MPSSkipInfo(
        UNIMPLEMENTED, dtypes=[torch.float32]
    ),
    "linalg.qr": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.slogdet": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.solve": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.solve_ex": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.svdvals": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.tensorsolve": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.vecdot": MPSSkipInfo(UNIMPLEMENTED),
    "logcumsumexp": MPSSkipInfo(UNIMPLEMENTED),
    "logdet": MPSSkipInfo(UNIMPLEMENTED),
    "lu": MPSSkipInfo(UNIMPLEMENTED, upper=15.0),
    "lu_solve": MPSSkipInfo(UNIMPLEMENTED),
    "lu_unpack": MPSSkipInfo(UNIMPLEMENTED),
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
    "sinc": MPSSkipInfo(UNIMPLEMENTED),
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
    "special.spherical_bessel_j0": MPSSkipInfo(UNIMPLEMENTED),
    "special.xlog1py": MPSSkipInfo(UNIMPLEMENTED),
    "special.zeta": MPSSkipInfo(UNIMPLEMENTED),
    "svd_lowrank": MPSSkipInfo(UNIMPLEMENTED),
    "symeig": MPSSkipInfo(UNIMPLEMENTED),
    "take": MPSSkipInfo(UNIMPLEMENTED),
    "to": MPSSkipInfo(UNIMPLEMENTED),
    "to_sparse": MPSSkipInfo(UNIMPLEMENTED),
    "unique": MPSSkipInfo(UNIMPLEMENTED),
    "vdot": MPSSkipInfo(UNIMPLEMENTED),
    "segment_reduce_": MPSSkipInfo(UNIMPLEMENTED),
    "_upsample_bilinear2d_aa": MPSSkipInfo(UNIMPLEMENTED),
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
    # round not working properly for float16
    "round": MPSSkipInfo(TEST_OUTPUT_GRAD_MATCH, dtypes=[torch.float16]),
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
    "nn.functional.conv1d": MPSSkipInfo(TEST_OUTPUT_GRAD_MATCH, dtypes=[torch.float16]),
    "nn.functional.conv2d": MPSSkipInfo(TEST_OUTPUT_GRAD_MATCH, dtypes=[torch.float16]),
    "nn.functional.conv3d": MPSSkipInfo(
        TEST_OUTPUT_GRAD_MATCH, dtypes=[torch.float16, torch.float32]
    ),
}


COMPLEX_DTYPES = [torch.complex32, torch.complex64]
"""Ops which do not have support for complex dtypes and are expected to fail"""
COMPLEX_XFAILLIST = {
    "__rdiv__": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "__rmatmul__": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "__rpow__": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "_chunk_cat": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "_unsafe_masked_index": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "_unsafe_masked_index_put_accumulate": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "acos": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "acosh": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "all": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "allclose": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "angle": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "any": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "addbmm": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "addcdiv": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "addcmul": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "addmm": [
        MPSSkipInfo(
            dtypes=COMPLEX_DTYPES,
            variant=" ",
        ),
        MPSSkipInfo(
            dtypes=COMPLEX_DTYPES,
            variant="decomposed",
            upper=15.0,
        ),
    ],
    "addmv": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "addr": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "asin": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "asinh": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "atan": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "atanh": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "baddbmm": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "bfloat16": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "block_diag": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "bmm": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "bool": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "cartesian_prod": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "cat": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "char": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "cholesky": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "column_stack": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "combinations": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "corrcoef": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "constant_pad_nd": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "cos": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "cosh": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "count_nonzero": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "cov": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "cross": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "cumprod": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "cumsum": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "cumulative_trapezoid": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "diff": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "dist": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "div": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "divno_rounding_mode": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "dot": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "dstack": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "einsum": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "eq": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "equal": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "exp2": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "expm1": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "eye": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "fft.fft": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "fft.fft2": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "fft.fftn": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "fft.fftshift": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "fft.ifft": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "fft.ifft2": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "fft.ifftn": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "fft.ifftshift": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "fft.irfftn": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "fft.irfft2": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "fft.irfft": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "fft.hfftn": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "fft.hfft2": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "fft.hfft": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "flip": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "fliplr": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "flipud": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "float": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "gather": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "gradient": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "half": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "hstack": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "index_add": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "index_fill": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "index_put": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "inner": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "int": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "isclose": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "isnan": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "istft": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "ldexp": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "lerp": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "linalg.cholesky": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "linalg.cross": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "linalg.inv": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "linalg.inv_ex": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "linalg.lu_factor": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "linalg.matrix_norm": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "linalg.matrix_power": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "linalg.multi_dot": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "linalg.norm": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "linalg.pinv": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "linalg.solve_triangular": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "linalg.tensorinv": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "linalg.vander": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "linalg.vector_norm": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "linspace": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "linspacetensor_overload": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "log10": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "log1p": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "log2": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "log": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "logical_and": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "logical_not": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "logical_or": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "logical_xor": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "logaddexp": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "logsumexp": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "long": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "masked_fill": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "masked.cumprod": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "masked.cumsum": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "masked.mean": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "masked.normalize": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "masked.prod": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "masked.std": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "masked.sum": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "masked.var": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "masked.logsumexp": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "matmul": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "mean": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "mm": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "mv": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "ne": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "neg": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
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
    "nn.functional.pixel_shuffle": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "nn.functional.pixel_unshuffle": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "nn.functional.rms_norm": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "nn.functional.silu": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "nn.functional.softsign": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "nn.functional.triplet_margin_loss": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "nn.functional.triplet_margin_with_distance_loss": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES
    ),
    "norm": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "normal": MPSSkipInfo(dtypes=COMPLEX_DTYPES, variant="in_place"),
    "pinverse": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "pow": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "prod": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "rand_like": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "reciprocal": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "renorm": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "repeat": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "resize_": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "resize_as_": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "roll": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "rot90": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "rsqrt": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "scatter_add": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "scatter": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "scatter_reduce": [
        MPSSkipInfo(dtypes=COMPLEX_DTYPES, variant="amax"),
        MPSSkipInfo(dtypes=COMPLEX_DTYPES, variant="amin"),
    ],
    "short": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "sigmoid": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "sin": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "sinh": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "sqrt": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "square": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "stack": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "std": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "std_mean": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "stft": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "sum": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "sum_to_size": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "take_along_dim": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "tan": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "tensordot": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "tile": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "trace": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "trapz": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "trapezoid": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "triangular_solve": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "tril": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "triu": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "true_divide": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "uniform": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "var": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "var_mean": MPSSkipInfo(dtypes=COMPLEX_DTYPES),
    "vstack": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "where": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
    "byte": MPSSkipInfo(
        dtypes=COMPLEX_DTYPES,
        upper=15.0,
    ),
}

"""Failures due to known errors in the MPS backend"""
MPS_DOWNSTREAM_XFAILLIST = {
    "arange": MPSSkipInfo(dtypes=[torch.uint8]),
    # TODO: remove this once downstream function 'aten::_linalg_svd.U' have been implemented
    "linalg.matrix_rank": MPSSkipInfo(),
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
    "empty_permuted": MPSSkipInfo(),
}


MPS_OPINFO: Dict[str, List[MPSSkipInfo]] = {}


def append_skips(skip_list: Dict) -> None:
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


def mps_op_db(op_db: List[OpInfo]) -> List[OpInfo]:
    """Utility function for OpInfo tests, updates the op_db with xfails defined in MPS_OPINFO_SKIPLIST"""

    for op in op_db:
        if op.name in MPS_OPINFO:
            if not isinstance(MPS_OPINFO[op.name], Iterable):
                skips: List[MPSSkipInfo] = [MPS_OPINFO[op.name]]  # type: ignore[list-item]
            else:
                skips: List[MPSSkipInfo] = MPS_OPINFO[op.name]  # type: ignore[no-redef]

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

    return op_db
