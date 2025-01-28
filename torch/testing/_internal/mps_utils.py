import unittest
from typing import Callable, Dict, Iterable, List, Optional, Union

import torch
from torch.testing._internal.common_utils import MACOS_VERSION
from torch.testing._internal.opinfo.core import DecorateInfo, OpInfo


UNIMPLEMENTED = "UNIMPLEMENTED"
NONCONTIGUOUS = "test_noncontiguous_samples"
TEST_OUT = "test_out"

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
            raise RuntimeError(
                f"Test is marked as unimplemented on MPS, but instead of NotImplementedError\
                    or TypeError we received {type(e).__name__}:{e} "
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
        test_class: str = COMMON,
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


"""Each op can have multiple skipInfos to account for OS differences & other variations"""
MPS_OPINFO_SKIPLIST: Dict[str, Union[MPSSkipInfo, List[MPSSkipInfo]]] = {
    # UNIMPLEMENTED OPS/DTYPES
    "_refs._conversions.cdouble": MPSSkipInfo(UNIMPLEMENTED),
    "_refs._conversions.double": MPSSkipInfo(UNIMPLEMENTED),
    "_refs.erfc": MPSSkipInfo(UNIMPLEMENTED),
    "_refs.float_power": MPSSkipInfo(UNIMPLEMENTED),
    "_refs.frexp": MPSSkipInfo(UNIMPLEMENTED),
    "_refs.gcd": MPSSkipInfo(UNIMPLEMENTED),
    "_refs.igamma": MPSSkipInfo(UNIMPLEMENTED),
    "_refs.igammac": MPSSkipInfo(UNIMPLEMENTED),
    "_refs.index_copy": MPSSkipInfo(UNIMPLEMENTED),
    "_refs.index_fill": MPSSkipInfo(UNIMPLEMENTED),
    "_refs.index_select": MPSSkipInfo(UNIMPLEMENTED),
    "_refs.lcm": MPSSkipInfo(UNIMPLEMENTED),
    "_refs.logspace": MPSSkipInfo(UNIMPLEMENTED),
    "_refs.nan_to_num": MPSSkipInfo(TEST_OUT),
    "_refs.nn.functional.log_softmax": MPSSkipInfo(UNIMPLEMENTED),
    "_refs.nn.functional.softmax": MPSSkipInfo(UNIMPLEMENTED),
    "_refs.nn.functional.softmin": MPSSkipInfo(UNIMPLEMENTED),
    "_refs.softmax": MPSSkipInfo(UNIMPLEMENTED),
    "_refs.special.bessel_j0": MPSSkipInfo(UNIMPLEMENTED),
    "_refs.special.bessel_j1": MPSSkipInfo(UNIMPLEMENTED),
    "_refs.special.erfcx": MPSSkipInfo(UNIMPLEMENTED),
    "_refs.special.i0e": MPSSkipInfo(UNIMPLEMENTED),
    "_refs.special.i1e": MPSSkipInfo(UNIMPLEMENTED),
    "_refs.special.log_ndtr": MPSSkipInfo(UNIMPLEMENTED),
    "_refs.special.log_softmax": MPSSkipInfo(UNIMPLEMENTED),
    "_refs.special.ndtri": MPSSkipInfo(UNIMPLEMENTED),
    "_refs.special.softmax": MPSSkipInfo(UNIMPLEMENTED),
    "_refs.special.softmin": MPSSkipInfo(UNIMPLEMENTED),
    "_refs.special.spherical_bessel_j0": MPSSkipInfo(UNIMPLEMENTED),
    "_refs.special.zeta": MPSSkipInfo(UNIMPLEMENTED),
    "_refs.to": MPSSkipInfo(UNIMPLEMENTED),
    "_refs.tril_indices": MPSSkipInfo(UNIMPLEMENTED),
    "_refs.triu_indices": MPSSkipInfo(UNIMPLEMENTED),
    "_segment_reduce": MPSSkipInfo(UNIMPLEMENTED),
    "_upsample_bilinear2d_aa": MPSSkipInfo(UNIMPLEMENTED),
    "cauchy": MPSSkipInfo(UNIMPLEMENTED),
    "cdouble": MPSSkipInfo(UNIMPLEMENTED),
    "cholesky": MPSSkipInfo(UNIMPLEMENTED),
    "cholesky_solve": MPSSkipInfo(UNIMPLEMENTED),
    "cummax": MPSSkipInfo(UNIMPLEMENTED),
    "cummin": MPSSkipInfo(UNIMPLEMENTED),
    "double": MPSSkipInfo(UNIMPLEMENTED),
    "erfc": MPSSkipInfo(UNIMPLEMENTED),
    "float_power": MPSSkipInfo(UNIMPLEMENTED),
    "frexp": MPSSkipInfo(UNIMPLEMENTED),
    "gcd": MPSSkipInfo(UNIMPLEMENTED),
    "geometric": MPSSkipInfo(UNIMPLEMENTED),
    "heaviside": MPSSkipInfo(UNIMPLEMENTED),
    "igamma": MPSSkipInfo(UNIMPLEMENTED),
    "igammac": MPSSkipInfo(UNIMPLEMENTED),
    "index_copy": MPSSkipInfo(UNIMPLEMENTED),
    "index_put": MPSSkipInfo(UNIMPLEMENTED),
    "index_reduce": MPSSkipInfo(UNIMPLEMENTED),
    "kthvalue": MPSSkipInfo(UNIMPLEMENTED),
    "lcm": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.cholesky_ex": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.cholesky": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.cond": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.householder_product": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.ldl_factor_ex": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.ldl_factor": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.ldl_solve": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.lstsq": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.lu_factor_ex": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.lu": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.matrix_norm": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.norm": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.qr": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.slogdet": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.svdvals": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.tensorsolve": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.vecdot": MPSSkipInfo(UNIMPLEMENTED),
    "log_normal": MPSSkipInfo(UNIMPLEMENTED),
    "logcumsumexp": MPSSkipInfo(UNIMPLEMENTED),
    "logdet": MPSSkipInfo(UNIMPLEMENTED),
    "logspace": MPSSkipInfo(UNIMPLEMENTED),
    "lu_unpack": MPSSkipInfo(UNIMPLEMENTED),
    "matrix_exp": MPSSkipInfo(UNIMPLEMENTED),
    "mode": MPSSkipInfo(UNIMPLEMENTED),
    "nanmedian": MPSSkipInfo(UNIMPLEMENTED),
    "native_dropout_backward": MPSSkipInfo(UNIMPLEMENTED),
    "nn.functional.adaptive_avg_pool1d": MPSSkipInfo(UNIMPLEMENTED),
    "nn.functional.adaptive_avg_pool2d": MPSSkipInfo(UNIMPLEMENTED),
    "nn.functional.adaptive_avg_pool3d": MPSSkipInfo(UNIMPLEMENTED),
    "nn.functional.adaptive_max_pool3d": MPSSkipInfo(UNIMPLEMENTED),
    "nn.functional.avg_pool3d": MPSSkipInfo(UNIMPLEMENTED),
    "nn.functional.channel_shuffle": MPSSkipInfo(UNIMPLEMENTED),
    "nn.functional.conv_transpose3d": MPSSkipInfo(UNIMPLEMENTED),
    "nn.functional.ctc_loss": MPSSkipInfo(UNIMPLEMENTED),
    "nn.functional.embedding_bag": MPSSkipInfo(UNIMPLEMENTED),
    "nn.functional.fractional_max_pool2d": MPSSkipInfo(UNIMPLEMENTED),
    "nn.functional.fractional_max_pool3d": MPSSkipInfo(UNIMPLEMENTED),
    "nn.functional.hardshrink": MPSSkipInfo(UNIMPLEMENTED),
    "nn.functional.max_pool3d": MPSSkipInfo(UNIMPLEMENTED),
    "nn.functional.max_unpool1d": MPSSkipInfo(UNIMPLEMENTED),
    "nn.functional.max_unpool2d": MPSSkipInfo(UNIMPLEMENTED),
    "nn.functional.max_unpool3d": MPSSkipInfo(UNIMPLEMENTED),
    "nn.functional.multi_margin_loss": MPSSkipInfo(UNIMPLEMENTED),
    "nn.functional.multilabel_margin_loss": MPSSkipInfo(UNIMPLEMENTED),
    "pca_lowrank": MPSSkipInfo(UNIMPLEMENTED),
    "put": MPSSkipInfo(UNIMPLEMENTED),
    "qr": MPSSkipInfo(UNIMPLEMENTED),
    "rsub": MPSSkipInfo(UNIMPLEMENTED),
    "sinc": MPSSkipInfo(UNIMPLEMENTED),
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
    "take": MPSSkipInfo(UNIMPLEMENTED),
    "to": MPSSkipInfo(UNIMPLEMENTED),
    "torch.ops.aten._efficient_attention_forward": MPSSkipInfo(UNIMPLEMENTED),
    "torch.ops.aten._flash_attention_forward": MPSSkipInfo(UNIMPLEMENTED),
    "vdot": MPSSkipInfo(UNIMPLEMENTED),
    # OTHER EXPECTED FAILURES
    "__getitem__": MPSSkipInfo(NONCONTIGUOUS, TEST_OUT),
    "_native_batch_norm_legit": MPSSkipInfo(TEST_OUT),
    "__rmatmul__": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "__rsub__": MPSSkipInfo(NONCONTIGUOUS, TEST_OUT),
    "_chunk_cat": MPSSkipInfo(
        TEST_OUT,
        upper=15.0,
    ),
    "_refs.diag_embed": MPSSkipInfo(
        TEST_OUT,
        upper=15.0,
    ),
    "_refs.linspace": MPSSkipInfo(
        UNIMPLEMENTED,
        variant="tensor_overload",
    ),
    "_refs.log_softmax": MPSSkipInfo(
        UNIMPLEMENTED,
        variant="with_dtype",
    ),
    "_unsafe_masked_index": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "_unsafe_masked_index_put_accumulate": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64, torch.int64],
    ),
    "abs": MPSSkipInfo(
        TEST_OUT,
    ),
    "addbmm": [
        MPSSkipInfo(
            NONCONTIGUOUS,
            dtypes=[torch.complex64],
        ),
        MPSSkipInfo(TEST_OUT),
    ],
    "addcmul": MPSSkipInfo(
        TEST_OUT,
        upper=15.0,
    ),
    "addcdiv": MPSSkipInfo(
        TEST_OUT,
        upper=15.0,
    ),
    "addmm": [
        MPSSkipInfo(
            NONCONTIGUOUS,
            variant="decomposed",
            dtypes=[torch.complex64],
        ),
        MPSSkipInfo(
            NONCONTIGUOUS,
            variant=" ",
            dtypes=[torch.complex64],
        ),
        MPSSkipInfo(
            TEST_OUT,
            upper=15.0,
        ),
    ],
    "addmv": MPSSkipInfo(
        TEST_OUT,
        upper=15.0,
    ),
    "addr": [
        MPSSkipInfo(
            NONCONTIGUOUS,
            dtypes=[torch.complex64, torch.int64],
        ),
        MPSSkipInfo(TEST_OUT),
    ],
    "all": MPSSkipInfo(
        TEST_OUT,
        upper=15.0,
    ),
    "amax": MPSSkipInfo(
        TEST_OUT,
        upper=15.0,
    ),
    "amin": MPSSkipInfo(
        TEST_OUT,
        upper=15.0,
    ),
    "angle": [
        MPSSkipInfo(
            NONCONTIGUOUS,
            dtypes=[torch.int64],
        ),
        MPSSkipInfo(TEST_OUT),
    ],
    "any": MPSSkipInfo(
        TEST_OUT,
        upper=15.0,
    ),
    "argmax": MPSSkipInfo(
        TEST_OUT,
        upper=15.0,
    ),
    "argmin": MPSSkipInfo(
        TEST_OUT,
        upper=15.0,
    ),
    "atan2": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.int64],
    ),
    "baddbmm": [
        MPSSkipInfo(
            NONCONTIGUOUS,
            dtypes=[torch.complex64],
        ),
        MPSSkipInfo(
            TEST_OUT,
            upper=15.0,
        ),
    ],
    "bitwise_and": MPSSkipInfo(
        TEST_OUT,
        lower=15.0,  # Regressed in MacOS15
    ),
    "bitwise_left": MPSSkipInfo(TEST_OUT),
    "bitwise_left_shift": MPSSkipInfo(
        TEST_OUT,
        lower=15.0,  # Regressed in MacOS15
    ),
    "bitwise_not": [
        MPSSkipInfo(
            NONCONTIGUOUS,
            dtypes=[torch.int64],
        ),
        MPSSkipInfo(TEST_OUT, lower=15.0),  # Regressed in MacOS15
    ],
    "bitwise_or": MPSSkipInfo(
        TEST_OUT,
        lower=15.0,  # Regressed in MacOS15
    ),
    "block_diag": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "bitwise_right_shift": MPSSkipInfo(
        TEST_OUT,
        lower=15.0,  # Regressed in MacOS15
    ),
    "bitwise_xor": MPSSkipInfo(
        TEST_OUT,
        lower=15.0,  # Regressed in MacOS15
    ),
    "bmm": [
        MPSSkipInfo(TEST_OUT),
        MPSSkipInfo(
            NONCONTIGUOUS,
            dtypes=[torch.complex64],
            upper=15.0,
        ),
    ],
    "bucketize": MPSSkipInfo(TEST_OUT),
    "cat": MPSSkipInfo(
        TEST_OUT,
        upper=15.0,
    ),
    "cdist": MPSSkipInfo(NONCONTIGUOUS),
    "cholesky_inverse": MPSSkipInfo(NONCONTIGUOUS),
    "clamp": MPSSkipInfo(
        TEST_OUT,
        upper=15.0,
    ),
    "clamp_max": MPSSkipInfo(
        TEST_OUT,
        upper=15.0,
    ),
    "clamp_min": MPSSkipInfo(
        TEST_OUT,
        upper=15.0,
    ),
    "column_stack": MPSSkipInfo(
        TEST_OUT,
        upper=15.0,
    ),
    "conj_physical": MPSSkipInfo(TEST_OUT),
    "cross": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "cumprod": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "cumsum": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "cumulative_trapezoid": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "digamma": [
        MPSSkipInfo(NONCONTIGUOUS),
        MPSSkipInfo(TEST_OUT, lower=15.0),  # Regressed in MacOS15
    ],
    "dist": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "dot": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "dstack": MPSSkipInfo(
        TEST_OUT,
        upper=15.0,
    ),
    "einsum": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
        upper=15.0,
    ),
    "exponential": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.float32],
        upper=15.0,
    ),
    "eye": MPSSkipInfo(
        TEST_OUT,
        upper=15.0,
    ),
    "fft.fft2": MPSSkipInfo(TEST_OUT),
    "fft.fft": MPSSkipInfo(TEST_OUT),
    "fft.fftn": MPSSkipInfo(TEST_OUT),
    "fft.hfft2": MPSSkipInfo(TEST_OUT),
    "fft.hfft": MPSSkipInfo(TEST_OUT),
    "fft.hfftn": MPSSkipInfo(TEST_OUT),
    "fft.ifft2": MPSSkipInfo(TEST_OUT),
    "fft.ifftn": MPSSkipInfo(TEST_OUT),
    "fft.irfft2": MPSSkipInfo(TEST_OUT),
    "fft.irfft": MPSSkipInfo(TEST_OUT),
    "fft.irfftn": MPSSkipInfo(TEST_OUT),
    "fft.rfft2": MPSSkipInfo(TEST_OUT),
    "fft.rfft": MPSSkipInfo(TEST_OUT),
    "fft.rfftn": MPSSkipInfo(TEST_OUT),
    "floor_divide": MPSSkipInfo(TEST_OUT),
    "full_like": MPSSkipInfo(NONCONTIGUOUS),
    "gather": [
        MPSSkipInfo(
            NONCONTIGUOUS,
            dtypes=[torch.complex64],
        ),
        MPSSkipInfo(
            TEST_OUT,
            upper=15.0,
        ),
    ],
    "geqrf": MPSSkipInfo(NONCONTIGUOUS),
    "grid_sampler_2d": MPSSkipInfo(NONCONTIGUOUS),
    "histc": [
        MPSSkipInfo(
            TEST_OUT,
            upper=15.0,
        ),
        MPSSkipInfo(
            NONCONTIGUOUS,
            dtypes=[torch.int64],
        ),
    ],
    "hstack": MPSSkipInfo(
        TEST_OUT,
        upper=15.0,
    ),
    "index_add": [
        MPSSkipInfo(
            NONCONTIGUOUS,
            dtypes=[torch.int64],
        ),
        MPSSkipInfo(
            TEST_OUT,
            upper=15.0,
        ),
    ],
    "index_fill": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64, torch.float32],
    ),
    "index_select": MPSSkipInfo(TEST_OUT),
    "inner": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "isin": MPSSkipInfo(
        TEST_OUT,
        upper=15.0,
    ),
    "isneginf": MPSSkipInfo(
        TEST_OUT,
        upper=15.0,
    ),
    "isposinf": MPSSkipInfo(
        TEST_OUT,
        upper=15.0,
    ),
    "istft": MPSSkipInfo(NONCONTIGUOUS),
    "lerp": [
        MPSSkipInfo(
            TEST_OUT,
            upper=15.0,
        ),
        MPSSkipInfo(
            NONCONTIGUOUS,
            dtypes=[torch.complex64],
        ),
    ],
    "lgamma": [
        MPSSkipInfo(NONCONTIGUOUS),
        MPSSkipInfo(
            TEST_OUT,
            lower=15.0,  # Regressed in MacOS15
        ),
    ],
    "linalg.cross": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "linalg.det": MPSSkipInfo(
        UNIMPLEMENTED,
    ),
    "linalg.eig": MPSSkipInfo(NONCONTIGUOUS),
    "linalg.eigh": MPSSkipInfo(NONCONTIGUOUS),
    "linalg.eigvals": MPSSkipInfo(NONCONTIGUOUS),
    "linalg.eigvalsh": MPSSkipInfo(NONCONTIGUOUS),
    "linalg.inv_ex": [
        MPSSkipInfo(
            NONCONTIGUOUS,
            dtypes=[torch.complex64],
        ),
        MPSSkipInfo(
            NONCONTIGUOUS,
            skip=unittest.skip(
                "Crashes on MPS with err: MPSMatrixDecompositionLU.mm:1146: \
                failed assertion `Number of columns in source exceeds source matrix size.'"
            ),
            dtypes=[torch.float32],
        ),
    ],
    "linalg.inv": [
        MPSSkipInfo(
            NONCONTIGUOUS,
            dtypes=[torch.complex64],
        ),
        MPSSkipInfo(
            NONCONTIGUOUS,
            skip=unittest.skip(
                "Crashes on MPS with err: failed assertion `Number of columns in source exceeds source matrix size.'"
            ),
            dtypes=[torch.float32],
        ),
    ],
    "linalg.lu_factor": [
        MPSSkipInfo(
            NONCONTIGUOUS,
            dtypes=[torch.complex64],
        ),
        MPSSkipInfo(
            NONCONTIGUOUS,
            skip=unittest.skip(
                "Crashes on MPS with err: failed assertion `A command encoder is already encoding to this command buffer'"
            ),
            dtypes=[torch.float32],
        ),
        MPSSkipInfo(TEST_OUT),
    ],
    "linalg.lu_solve": MPSSkipInfo(NONCONTIGUOUS, TEST_OUT),
    "linalg.matrix_power": [
        MPSSkipInfo(
            NONCONTIGUOUS,
            dtypes=[torch.complex64],
        ),
        MPSSkipInfo(
            NONCONTIGUOUS,
            skip=unittest.skip(
                "Crashes on MPS with err: failed assertion `Number of columns in source exceeds source matrix size.'"
            ),
            dtypes=[torch.float32],
        ),
        MPSSkipInfo(TEST_OUT),
    ],
    "linalg.matrix_rank": MPSSkipInfo(NONCONTIGUOUS),
    "linalg.multi_dot": MPSSkipInfo(TEST_OUT),
    "linalg.pinv": MPSSkipInfo(
        NONCONTIGUOUS,
        variant="hermitian",
    ),
    "linalg.solve": MPSSkipInfo(NONCONTIGUOUS, TEST_OUT),
    "linalg.solve_ex": MPSSkipInfo(NONCONTIGUOUS),
    "linalg.solve_triangular": MPSSkipInfo(NONCONTIGUOUS, TEST_OUT),
    "linalg.svd": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "linalg.tensorinv": [
        MPSSkipInfo(
            NONCONTIGUOUS,
            dtypes=[torch.complex64],
        ),
        MPSSkipInfo(
            NONCONTIGUOUS,
            dtypes=[torch.float32],
            skip=unittest.skip(""),
            skip_msg="Crashes on MPS with err: failed assertion `Number of columns in source exceeds source matrix size.'",
        ),
    ],
    "linalg.vander": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "linalg.vector_norm": [
        MPSSkipInfo(
            TEST_OUT,
            upper=15.0,
        ),
        MPSSkipInfo(
            NONCONTIGUOUS,
            dtypes=[torch.complex64],
        ),
    ],
    "log1p": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.int64],
    ),
    "log_softmax": MPSSkipInfo(
        UNIMPLEMENTED,
        variant="with_dtype",
    ),
    "logical_and": MPSSkipInfo(TEST_OUT),
    "logical_not": MPSSkipInfo(TEST_OUT),
    "logical_or": MPSSkipInfo(TEST_OUT),
    "logical_xor": MPSSkipInfo(TEST_OUT),
    "logit": MPSSkipInfo(TEST_OUT),
    "logsumexp": MPSSkipInfo(
        TEST_OUT,
        upper=15.0,
    ),
    "lu": MPSSkipInfo(NONCONTIGUOUS),
    "lu_solve": MPSSkipInfo(NONCONTIGUOUS),
    "masked.cumprod": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "masked.cumsum": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "masked.median": MPSSkipInfo(
        UNIMPLEMENTED,
        dtypes=[torch.float32],
    ),
    "masked.normalize": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "masked_scatter": MPSSkipInfo(NONCONTIGUOUS),
    "matmul": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "max": [
        MPSSkipInfo(
            TEST_OUT,
            variant="reduction_no_dim",
        ),
        MPSSkipInfo(
            TEST_OUT,
            variant="reduction_with_dim",
            upper=15.0,
        ),
    ],
    "mean": MPSSkipInfo(TEST_OUT),
    "min": [
        MPSSkipInfo(
            TEST_OUT,
            variant="reduction_no_dim",
        ),
        MPSSkipInfo(
            TEST_OUT,
            variant="reduction_with_dim",
            upper=15.0,
        ),
    ],
    "mm": [
        MPSSkipInfo(
            NONCONTIGUOUS,
            dtypes=[torch.complex64],
        ),
        MPSSkipInfo(TEST_OUT),
    ],
    "msort": MPSSkipInfo(
        TEST_OUT,
        upper=15.0,
    ),
    "mv": MPSSkipInfo(
        TEST_OUT,
        upper=15.0,
    ),
    "mvlgamma": MPSSkipInfo(TEST_OUT),
    "nan_to_num": MPSSkipInfo(TEST_OUT),
    "nanmean": [
        MPSSkipInfo(
            NONCONTIGUOUS,
            dtypes=[torch.complex64],
        ),
        MPSSkipInfo(TEST_OUT),
    ],
    "nansum": [
        MPSSkipInfo(
            NONCONTIGUOUS,
            dtypes=[torch.complex64],
        ),
        MPSSkipInfo(
            TEST_OUT,
            skip=unittest.skip(
                "Crashes on MPS with error 'Function isNaN_i64_i8 was not found in the library'"
            ),
        ),
    ],
    "nanquantile": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.float32],
        upper=15.0,
    ),
    "native_batch_norm": MPSSkipInfo(TEST_OUT),
    "nn.functional.avg_pool2d": MPSSkipInfo(TEST_OUT),
    "nn.functional.binary_cross_entropy": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.float32],
        upper=15.0,
    ),
    "nn.functional.binary_cross_entropy_with_logits": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.float32],
        upper=15.0,
    ),
    "nn.functional.celu": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.float32],
        upper=15.0,
    ),
    "nn.functional.conv2d": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.float32],
        lower=15.0,  # Regressed in MacOS15
    ),
    "nn.functional.elu": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.float32],
        upper=15.0,
    ),
    "nn.functional.grid_sample": MPSSkipInfo(NONCONTIGUOUS),
    "nn.functional.interpolate": [
        MPSSkipInfo(UNIMPLEMENTED, variant="area"),
        MPSSkipInfo(
            NONCONTIGUOUS,
            variant="nearest-exact",
        ),
        MPSSkipInfo(
            UNIMPLEMENTED,
            variant="trilinear",
        ),
    ],
    "nn.functional.l1_loss": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "nn.functional.linear": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "nn.functional.logsigmoid": [
        MPSSkipInfo(
            NONCONTIGUOUS,
            dtypes=[torch.float32],
            upper=15.0,
        ),
        MPSSkipInfo(TEST_OUT),
    ],
    "nn.functional.max_pool2d": MPSSkipInfo(NONCONTIGUOUS),
    "nn.functional.mish": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.float32],
        upper=15.0,
    ),
    "nn.functional.normalize": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "nn.functional.pairwise_distance": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "nn.functional.pdist": MPSSkipInfo(
        UNIMPLEMENTED,
        dtypes=[torch.float32],
    ),
    "nn.functional.relu": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.float32, torch.int64],
        upper=15.0,
    ),
    "nn.functional.rrelu": MPSSkipInfo(
        UNIMPLEMENTED,
        dtypes=[torch.float32],
    ),
    "nn.functional.scaled_dot_product_attention": MPSSkipInfo(TEST_OUT),
    "nn.functional.selu": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.float32],
        upper=15.0,
    ),
    "nn.functional.silu": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.float32, torch.complex64],
        upper=15.0,
    ),
    "nn.functional.softmin": MPSSkipInfo(
        UNIMPLEMENTED,
        variant="with_dtype",
    ),
    "nn.functional.softplus": MPSSkipInfo(
        TEST_OUT,
        upper=15.0,
    ),
    "nn.functional.softshrink": MPSSkipInfo(
        TEST_OUT,
        upper=15.0,
    ),
    "nn.functional.triplet_margin_loss": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "nn.functional.triplet_margin_with_distance_loss": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "norm": [
        MPSSkipInfo(TEST_OUT),
        MPSSkipInfo(
            NONCONTIGUOUS,
            dtypes=[torch.complex64],
        ),
        MPSSkipInfo(
            UNIMPLEMENTED,
            variant="nuc",
            dtypes=[torch.float32],
        ),
    ],
    "normal": MPSSkipInfo(NONCONTIGUOUS),
    "ones_like": MPSSkipInfo(NONCONTIGUOUS),
    "ormqr": MPSSkipInfo(NONCONTIGUOUS),
    "polygamma": [
        MPSSkipInfo(
            TEST_OUT,
            variant="polygamma_n_0",
            lower=15.0,  # Regressed in MacOS15
        ),
        MPSSkipInfo(NONCONTIGUOUS),
    ],
    "polar": MPSSkipInfo(TEST_OUT),
    "prod": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "quantile": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.float32],
        upper=15.0,
    ),
    "rand_like": MPSSkipInfo(NONCONTIGUOUS),
    "randint_like": MPSSkipInfo(NONCONTIGUOUS),
    "randn": MPSSkipInfo(
        TEST_OUT,
        upper=15.0,
    ),
    "randn_like": MPSSkipInfo(NONCONTIGUOUS),
    "renorm": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "repeat": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "round": MPSSkipInfo(
        UNIMPLEMENTED,
        variant="decimals_0",
    ),
    "searchsorted": MPSSkipInfo(TEST_OUT),
    "scatter_add": [
        MPSSkipInfo(
            NONCONTIGUOUS,
            dtypes=[torch.complex64],
        ),
        MPSSkipInfo(
            TEST_OUT,
            upper=15.0,
        ),
    ],
    "scatter": [
        MPSSkipInfo(
            NONCONTIGUOUS,
            dtypes=[torch.complex64],
        ),
        MPSSkipInfo(
            TEST_OUT,
            upper=15.0,
        ),
    ],
    "scatter_reduce": [
        MPSSkipInfo(
            NONCONTIGUOUS,
            variant="amax",
            dtypes=[torch.int64],
        ),
        MPSSkipInfo(
            NONCONTIGUOUS,
            variant="amin",
            dtypes=[torch.int64],
        ),
        MPSSkipInfo(
            TEST_OUT,
            upper=15.0,
        ),
    ],
    "sigmoid": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.int64],
    ),
    "softmax": MPSSkipInfo(
        UNIMPLEMENTED,
        variant="with_dtype",
    ),
    "sort": MPSSkipInfo(
        TEST_OUT,
        upper=15.0,
    ),
    "special.polygamma": [
        MPSSkipInfo(NONCONTIGUOUS),
        MPSSkipInfo(
            TEST_OUT,
            lower=15.0,  # Regressed in MacOS15
        ),
    ],
    "stack": MPSSkipInfo(
        TEST_OUT,
        upper=15.0,
    ),
    "std": [
        MPSSkipInfo(
            NONCONTIGUOUS,
            dtypes=[torch.complex64],
        ),
        MPSSkipInfo(
            TEST_OUT,
            variant=" ",
        ),
    ],
    "svd": MPSSkipInfo(
        NONCONTIGUOUS,
        variant=" ",
        dtypes=[torch.complex64],
    ),
    "take_along_dim": [
        MPSSkipInfo(
            NONCONTIGUOUS,
            dtypes=[torch.complex64],
        ),
        MPSSkipInfo(
            TEST_OUT,
            upper=15.0,
        ),
    ],
    "tensordot": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "tile": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "to_sparse": MPSSkipInfo(TEST_OUT),
    "topk": MPSSkipInfo(
        TEST_OUT,
        upper=15.0,
    ),
    "trace": MPSSkipInfo(
        NONCONTIGUOUS,
        skip=unittest.skip(
            "Crashes on MPS with err: 'mps.scatter' op operand #0 must be tensor of mps native type values,\
            but got 'tensor<25xcomplex<f32>>"
        ),
    ),
    "triangular_solve": MPSSkipInfo(NONCONTIGUOUS),
    "tril": MPSSkipInfo(
        TEST_OUT,
        upper=15.0,
    ),
    "triu": MPSSkipInfo(
        TEST_OUT,
        upper=15.0,
    ),
    "unfold": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "unfold_copy": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "uniform": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64, torch.float32],
        upper=15.0,
    ),
    "unique": MPSSkipInfo(NONCONTIGUOUS),
    "var": [
        MPSSkipInfo(
            NONCONTIGUOUS,
            dtypes=[torch.complex64],
        ),
        MPSSkipInfo(
            TEST_OUT,
            variant=" ",
        ),
    ],
    "vstack": MPSSkipInfo(
        TEST_OUT,
        upper=15.0,
    ),
    "where": MPSSkipInfo(
        TEST_OUT,
        upper=15.0,
    ),
    "zeros_like": MPSSkipInfo(NONCONTIGUOUS),
}


def mps_op_db(op_db: List[OpInfo]) -> List[OpInfo]:
    """Utility function for OpInfo tests, updates the op_db with xfails defined in MPS_OPINFO_SKIPLIST"""

    for op in op_db:
        if op.name in MPS_OPINFO_SKIPLIST:
            if not isinstance(MPS_OPINFO_SKIPLIST[op.name], Iterable):
                skips: List[MPSSkipInfo] = [MPS_OPINFO_SKIPLIST[op.name]]  # type: ignore[list-item]
            else:
                skips: List[MPSSkipInfo] = MPS_OPINFO_SKIPLIST[op.name]  # type: ignore[no-redef]

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
                    if not isinstance(skip.tests, List):
                        skip.tests = [skip.tests]
                    if skip.tests == []:
                        decorator = DecorateInfo(
                            skip.skip,
                            device_type="mps",
                            dtypes=skip.dtypes,
                        )
                        op.decorators = op.decorators + (decorator,)
                    else:
                        for test in skip.tests:
                            decorator = DecorateInfo(
                                skip.skip,
                                skip.test_class,
                                test,
                                device_type="mps",
                                dtypes=skip.dtypes,
                            )
                            op.decorators = op.decorators + (decorator,)

    return op_db
