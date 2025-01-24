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
            raise unittest.SkipTest("Requires op not currently implemented on MPS") from e
        except TypeError as e:
            raise unittest.SkipTest("Uses dtype not supported on MPS") from e
        except unittest.SkipTest as e:
            # Don't error out on tests that have been explicitly skipped for some other reason
            raise e
        except Exception as e:
            raise RuntimeError(
                f"Test is marked as unimplemented on MPS, but instead of NotImplementedError or TypeError we received {type(e).__name__}:{e} "
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
            self.tests = [NONCONTIGUOUS, TEST_OUT]
            self.skip = xfailUnimplemented

"""Each op can have multiple skipInfos to account for OS differences & other variations"""
MPS_OPINFO_SKIPLIST: Dict[str, Union[MPSSkipInfo, List[MPSSkipInfo]]] = {
    "__getitem__": MPSSkipInfo(NONCONTIGUOUS, TEST_OUT),
    "_native_batch_norm_legit": MPSSkipInfo(TEST_OUT),
    "__rmatmul__": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "__rsub__": MPSSkipInfo(NONCONTIGUOUS, TEST_OUT),
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
    "_refs.linspace": MPSSkipInfo(
        UNIMPLEMENTED,
        variant="tensor_overload",
    ),
    "_refs.log_softmax": MPSSkipInfo(
        UNIMPLEMENTED,
        variant="with_dtype",
    ),
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
    "_unsafe_masked_index": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "_unsafe_masked_index_put_accumulate": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64, torch.int64],
    ),
    "_upsample_bilinear2d_aa": MPSSkipInfo(UNIMPLEMENTED),
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
    "addmm": [
        MPSSkipInfo(
            NONCONTIGUOUS,
            variant="decomposed",
            dtypes=[torch.complex64],
        ),
        MPSSkipInfo(
            NONCONTIGUOUS,
            variant="",
            dtypes=[torch.complex64],
        ),
    ],
    "addr": [
        MPSSkipInfo(
            NONCONTIGUOUS,
            dtypes=[torch.complex64, torch.int64],
        ),
        MPSSkipInfo(TEST_OUT)
    ],
    "angle": [
        MPSSkipInfo(
            NONCONTIGUOUS,
            dtypes=[torch.int64],
        ),
        MPSSkipInfo(TEST_OUT),
    ],
    "atan2": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.int64],
    ),
    "baddbmm": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "bmm": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
        upper=15.0,
        lower=14.0,
    ),
    "bitwise_and": MPSSkipInfo(TEST_OUT),
    "bitwise_left": MPSSkipInfo(TEST_OUT),
    "bitwise_left_shift": MPSSkipInfo(TEST_OUT),
    "bitwise_not": [
        MPSSkipInfo(
            NONCONTIGUOUS,
            dtypes=[torch.int64],
        ),
        MPSSkipInfo(TEST_OUT),
    ],
    "bitwise_or": MPSSkipInfo(TEST_OUT),
    "block_diag": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "bitwise_right_shift": MPSSkipInfo(TEST_OUT),
    "bitwise_xor": MPSSkipInfo(TEST_OUT),
    "bmm": MPSSkipInfo(TEST_OUT),
    "bucketize": MPSSkipInfo(TEST_OUT),
    "cauchy": MPSSkipInfo(UNIMPLEMENTED),
    "cdist": MPSSkipInfo(NONCONTIGUOUS),
    "cdouble": MPSSkipInfo(UNIMPLEMENTED),
    "cholesky_inverse": MPSSkipInfo(NONCONTIGUOUS),
    "cholesky": MPSSkipInfo(UNIMPLEMENTED),
    "cholesky_solve": MPSSkipInfo(UNIMPLEMENTED),
    "conj_physical": MPSSkipInfo(TEST_OUT),
    "cross": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "cummax": MPSSkipInfo(UNIMPLEMENTED),
    "cummin": MPSSkipInfo(UNIMPLEMENTED),
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
    "digamma": MPSSkipInfo(NONCONTIGUOUS, TEST_OUT),
    "dist": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "dot": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "double": MPSSkipInfo(UNIMPLEMENTED),
    "einsum": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
        upper=15.0,
        lower=14.0,
    ),
    "erfc": MPSSkipInfo(UNIMPLEMENTED),
    "exponential": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.float32],
        upper=15.0,
        lower=14.0,
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
    "float_power": MPSSkipInfo(UNIMPLEMENTED),
    "floor_divide": MPSSkipInfo(TEST_OUT),
    "frexp": MPSSkipInfo(UNIMPLEMENTED),
    "full_like": MPSSkipInfo(NONCONTIGUOUS),
    "gather": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "gcd": MPSSkipInfo(UNIMPLEMENTED),
    "geometric": MPSSkipInfo(UNIMPLEMENTED),
    "geqrf": MPSSkipInfo(NONCONTIGUOUS),
    "grid_sampler_2d": MPSSkipInfo(NONCONTIGUOUS),
    "heaviside": MPSSkipInfo(UNIMPLEMENTED),
    "histc": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.int64],
    ),
    "igamma": MPSSkipInfo(UNIMPLEMENTED),
    "igammac": MPSSkipInfo(UNIMPLEMENTED),
    "index_add": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.int64],
    ),
    "index_copy": MPSSkipInfo(UNIMPLEMENTED),
    "index_fill": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64, torch.float32],
    ),
    "index_put": MPSSkipInfo(UNIMPLEMENTED),
    "index_reduce": MPSSkipInfo(UNIMPLEMENTED),
    "index_select": MPSSkipInfo(TEST_OUT),
    "inner": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "istft": MPSSkipInfo(NONCONTIGUOUS),
    "kthvalue": MPSSkipInfo(UNIMPLEMENTED),
    "lcm": MPSSkipInfo(UNIMPLEMENTED),
    "lerp": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "lgamma": MPSSkipInfo(NONCONTIGUOUS, TEST_OUT),
    "linalg.cholesky_ex": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.cholesky": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.cond": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.cross": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "linalg.det": MPSSkipInfo(
        UNIMPLEMENTED,
        variant="",
    ),
    "linalg.eig": MPSSkipInfo(NONCONTIGUOUS),
    "linalg.eigh": MPSSkipInfo(NONCONTIGUOUS),
    "linalg.eigvals": MPSSkipInfo(NONCONTIGUOUS),
    "linalg.eigvalsh": MPSSkipInfo(NONCONTIGUOUS),
    "linalg.householder_product": MPSSkipInfo(UNIMPLEMENTED),
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
    "linalg.ldl_factor_ex": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.ldl_factor": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.ldl_solve": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.lstsq": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.lu_factor_ex": MPSSkipInfo(UNIMPLEMENTED),
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
    "linalg.lu": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.lu_solve": MPSSkipInfo(NONCONTIGUOUS, TEST_OUT),
    "linalg.matrix_norm": MPSSkipInfo(UNIMPLEMENTED),
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
    "linalg.norm": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.pinv": MPSSkipInfo(
        NONCONTIGUOUS,
        variant="hermitian",
    ),
    "linalg.qr": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.slogdet": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.solve": MPSSkipInfo(NONCONTIGUOUS, TEST_OUT),
    "linalg.solve_ex": MPSSkipInfo(NONCONTIGUOUS),
    "linalg.solve_triangular": MPSSkipInfo(NONCONTIGUOUS, TEST_OUT),
    "linalg.svd": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "linalg.svdvals": MPSSkipInfo(UNIMPLEMENTED),
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
    "linalg.tensorsolve": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.vander": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "linalg.vecdot": MPSSkipInfo(UNIMPLEMENTED),
    "linalg.vector_norm": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "log1p": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.int64],
    ),
    "log_normal": MPSSkipInfo(UNIMPLEMENTED),
    "log_softmax": MPSSkipInfo(
        UNIMPLEMENTED,
        variant="with_dtype",
    ),
    "logcumsumexp": MPSSkipInfo(UNIMPLEMENTED),
    "logdet": MPSSkipInfo(UNIMPLEMENTED),
    "logical_and": MPSSkipInfo(TEST_OUT),
    "logical_not": MPSSkipInfo(TEST_OUT),
    "logical_or": MPSSkipInfo(TEST_OUT),
    "logical_xor": MPSSkipInfo(TEST_OUT),
    "logit": MPSSkipInfo(TEST_OUT),
    "logspace": MPSSkipInfo(UNIMPLEMENTED),
    "lu": MPSSkipInfo(NONCONTIGUOUS),
    "lu_solve": MPSSkipInfo(NONCONTIGUOUS),
    "lu_unpack": MPSSkipInfo(UNIMPLEMENTED),
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
    "matrix_exp": MPSSkipInfo(UNIMPLEMENTED),
    "max": MPSSkipInfo(
        TEST_OUT,
        variant="reduction_no_dim",
    ),
    "mean": MPSSkipInfo(TEST_OUT),
    "min": MPSSkipInfo(
        TEST_OUT,
        variant="reduction_no_dim",
    ),
    "mm": [
        MPSSkipInfo(
            NONCONTIGUOUS,
            dtypes=[torch.complex64],
        ),
        MPSSkipInfo(TEST_OUT),
    ],
    "mode": MPSSkipInfo(UNIMPLEMENTED),
    "mvlgamma": MPSSkipInfo(TEST_OUT),
    "nan_to_num": MPSSkipInfo(TEST_OUT),
    "nanmean": [
        MPSSkipInfo(
            NONCONTIGUOUS,
            dtypes=[torch.complex64],
        ),
        MPSSkipInfo(TEST_OUT),
    ],
    "nanmedian": MPSSkipInfo(UNIMPLEMENTED),
    "nansum": [
        MPSSkipInfo(
            NONCONTIGUOUS,
            dtypes=[torch.complex64],
        ),
        MPSSkipInfo(
            TEST_OUT,
            skip=unittest.skip("Crashes on MPS with error 'Function isNaN_i64_i8 was not found in the library'"),
        ),
    ],
    "nanquantile": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.float32],
        upper=15.0,
        lower=14.0,
    ),
    "native_batch_norm": MPSSkipInfo(TEST_OUT),
    "native_dropout_backward": MPSSkipInfo(UNIMPLEMENTED),
    "nn.functional.adaptive_avg_pool1d": MPSSkipInfo(UNIMPLEMENTED),
    "nn.functional.adaptive_avg_pool2d": MPSSkipInfo(UNIMPLEMENTED),
    "nn.functional.adaptive_avg_pool3d": MPSSkipInfo(UNIMPLEMENTED),
    "nn.functional.adaptive_max_pool3d": MPSSkipInfo(UNIMPLEMENTED),
    "nn.functional.avg_pool2d": MPSSkipInfo(TEST_OUT),
    "nn.functional.avg_pool3d": MPSSkipInfo(UNIMPLEMENTED),
    "nn.functional.binary_cross_entropy": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.float32],
        upper=15.0,
        lower=14.0,
    ),
    "nn.functional.binary_cross_entropy_with_logits": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.float32],
        upper=15.0,
        lower=14.0,
    ),
    "nn.functional.celu": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.float32],
        upper=15.0,
        lower=14.0,
    ),
    "nn.functional.channel_shuffle": MPSSkipInfo(UNIMPLEMENTED),
    "nn.functional.conv2d": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.float32],
        lower=15.0,  # Regressed in MacOS15
    ),
    "nn.functional.conv_transpose3d": MPSSkipInfo(UNIMPLEMENTED),
    "nn.functional.ctc_loss": MPSSkipInfo(UNIMPLEMENTED),
    "nn.functional.elu": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.float32],
        upper=15.0,
        lower=14.0,
    ),
    "nn.functional.embedding_bag": MPSSkipInfo(UNIMPLEMENTED),
    "nn.functional.fractional_max_pool2d": MPSSkipInfo(UNIMPLEMENTED),
    "nn.functional.fractional_max_pool3d": MPSSkipInfo(UNIMPLEMENTED),
    "nn.functional.grid_sample": MPSSkipInfo(NONCONTIGUOUS),
    "nn.functional.hardshrink": MPSSkipInfo(UNIMPLEMENTED),
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
            lower=14.0,
        ),
        MPSSkipInfo(TEST_OUT),
    ],
    "nn.functional.max_pool2d": MPSSkipInfo(NONCONTIGUOUS),
    "nn.functional.max_pool3d": MPSSkipInfo(UNIMPLEMENTED),
    "nn.functional.max_unpool1d": MPSSkipInfo(UNIMPLEMENTED),
    "nn.functional.max_unpool2d": MPSSkipInfo(UNIMPLEMENTED),
    "nn.functional.max_unpool3d": MPSSkipInfo(UNIMPLEMENTED),
    "nn.functional.mish": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.float32],
        upper=15.0,
        lower=14.0,
    ),
    "nn.functional.multi_margin_loss": MPSSkipInfo(UNIMPLEMENTED),
    "nn.functional.multilabel_margin_loss": MPSSkipInfo(UNIMPLEMENTED),
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
        lower=14.0,
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
        lower=14.0,
    ),
    "nn.functional.silu": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.float32, torch.complex64],
        upper=15.0,
        lower=14.0,
    ),
    "nn.functional.softmin": MPSSkipInfo(
        UNIMPLEMENTED,
        variant="with_dtype",
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
    "pca_lowrank": MPSSkipInfo(UNIMPLEMENTED),
    "polygamma": MPSSkipInfo(
        NONCONTIGUOUS, TEST_OUT,
        variant="polygamma_n_0",
    ),
    "polar": MPSSkipInfo(TEST_OUT),
    "prod": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "put": MPSSkipInfo(UNIMPLEMENTED),
    "qr": MPSSkipInfo(UNIMPLEMENTED),
    "quantile": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.float32],
        upper=15.0,
        lower=14.0,
    ),
    "rand_like": MPSSkipInfo(NONCONTIGUOUS),
    "randint_like": MPSSkipInfo(NONCONTIGUOUS),
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
    "rsub": MPSSkipInfo(UNIMPLEMENTED),
    "searchsorted": MPSSkipInfo(TEST_OUT),
    "scatter_add": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "scatter": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
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
    ],
    "sigmoid": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.int64],
    ),
    "sinc": MPSSkipInfo(UNIMPLEMENTED),
    "softmax": MPSSkipInfo(
        UNIMPLEMENTED,
        variant="with_dtype",
    ),
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
    "special.polygamma": MPSSkipInfo(NONCONTIGUOUS, TEST_OUT),
    "special.scaled_modified_bessel_k0": MPSSkipInfo(UNIMPLEMENTED),
    "special.scaled_modified_bessel_k1": MPSSkipInfo(UNIMPLEMENTED),
    "special.spherical_bessel_j0": MPSSkipInfo(UNIMPLEMENTED),
    "special.xlog1py": MPSSkipInfo(UNIMPLEMENTED),
    "special.zeta": MPSSkipInfo(UNIMPLEMENTED),
    "std": MPSSkipInfo(UNIMPLEMENTED),
    "svd": MPSSkipInfo(
        NONCONTIGUOUS,
        variant="",
        dtypes=[torch.complex64],
    ),
    "svd_lowrank": MPSSkipInfo(UNIMPLEMENTED),
    "take_along_dim": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "take": MPSSkipInfo(UNIMPLEMENTED),
    "tensordot": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "tile": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "to": MPSSkipInfo(UNIMPLEMENTED),
    "torch.ops.aten._efficient_attention_forward": MPSSkipInfo(UNIMPLEMENTED),
    "torch.ops.aten._flash_attention_forward": MPSSkipInfo(UNIMPLEMENTED),
    "trace": MPSSkipInfo(
        NONCONTIGUOUS,
        skip=unittest.skip(
            "Crashes on MPS with err: 'mps.scatter' op operand #0 must be tensor of mps native type values,\
            but got 'tensor<25xcomplex<f32>>"
        ),
    ),
    "triangular_solve": MPSSkipInfo(NONCONTIGUOUS),
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
        lower=14.0,
    ),
    "unique": MPSSkipInfo(NONCONTIGUOUS),
    "var": MPSSkipInfo(UNIMPLEMENTED),
    "vdot": MPSSkipInfo(UNIMPLEMENTED),
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
                        or (op.variant_test_name is None and skip.variant == "")
                    )
                ):
                    if not isinstance(skip.tests, List):
                        skip.tests = [skip.tests]
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
