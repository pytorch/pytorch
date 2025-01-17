import unittest
from typing import Dict, Iterable, List, Optional, Union

import torch
from torch.testing._internal.common_utils import MACOS_VERSION
from torch.testing._internal.opinfo.core import DecorateInfo, OpInfo


NONCONTIGUOUS = "test_noncontiguous_samples"


class MPSSkipInfo:
    def __init__(
        self,
        tests: Union[str | List[str]],
        variant: Optional[str] = None,
        dtypes: Optional[Union[torch.dtype | List[torch.dtype]]] = None,
        skip: bool = False,
        skip_msg: str = "Skipped!",
        upper: Optional[float] = None,
        lower: Optional[float] = None,
    ):
        """Basic struct for tracking MPS OpInfo xfails

        tests: Test(s) to apply this xfail info to
        variant: Variant name. Set to empty str ("") to explicitly specify the non-variant case
        If set to None, will instead apply to all variants of the test
        dtypes: If none specified, xfails all dtype variants
        skip: If True, skip instead of xfailing this test
        upper: Upper bound MacOS version this xfail applies to (exclusive)
        lower: Lower bound MacOS version this xfail applies to (inclusive)
        """
        self.tests = tests
        self.variant = variant
        self.dtypes = dtypes
        self.skip = skip
        self.skip_msg = skip_msg
        self.upper = upper
        self.lower = lower


"""Each op can have multiple skipInfos to account for OS differences & other variations"""
MPS_OPINFO_SKIPLIST: Dict[str, Union[MPSSkipInfo, List[MPSSkipInfo]]] = {
    "__getitem__": MPSSkipInfo(NONCONTIGUOUS),
    "__rmatmul__": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "__rsub__": MPSSkipInfo(NONCONTIGUOUS),
    "_segment_reduce": MPSSkipInfo(NONCONTIGUOUS),
    "_unsafe_masked_index": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "_unsafe_masked_index_put_accumulate": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64, torch.int64],
    ),
    "_upsample_bilinear2d_aa": MPSSkipInfo(NONCONTIGUOUS),
    "addbmm": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
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
    "addr": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64, torch.int64],
    ),
    "angle": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.int64],
    ),
    "atan2": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.int64],
    ),
    "baddbmm": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "bitwise_not": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.int64],
    ),
    "block_diag": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "cauchy": MPSSkipInfo(NONCONTIGUOUS),
    "cdist": MPSSkipInfo(NONCONTIGUOUS),
    "cdouble": MPSSkipInfo(NONCONTIGUOUS),
    "cholesky_inverse": MPSSkipInfo(NONCONTIGUOUS),
    "cholesky": MPSSkipInfo(NONCONTIGUOUS),
    "cholesky_solve": MPSSkipInfo(NONCONTIGUOUS),
    "cross": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "cummax": MPSSkipInfo(NONCONTIGUOUS),
    "cummin": MPSSkipInfo(NONCONTIGUOUS),
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
    "digamma": MPSSkipInfo(NONCONTIGUOUS),
    "dist": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "dot": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "double": MPSSkipInfo(NONCONTIGUOUS),
    "erfc": MPSSkipInfo(NONCONTIGUOUS),
    "float_power": MPSSkipInfo(NONCONTIGUOUS),
    "frexp": MPSSkipInfo(NONCONTIGUOUS),
    "full_like": MPSSkipInfo(NONCONTIGUOUS),
    "gather": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "gcd": MPSSkipInfo(NONCONTIGUOUS),
    "geometric": MPSSkipInfo(NONCONTIGUOUS),
    "geqrf": MPSSkipInfo(NONCONTIGUOUS),
    "grid_sampler_2d": MPSSkipInfo(NONCONTIGUOUS),
    "heaviside": MPSSkipInfo(NONCONTIGUOUS),
    "histc": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.int64],
    ),
    "igamma": MPSSkipInfo(NONCONTIGUOUS),
    "igammac": MPSSkipInfo(NONCONTIGUOUS),
    "index_add": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.int64],
    ),
    "index_copy": MPSSkipInfo(NONCONTIGUOUS),
    "index_fill": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64, torch.float32],
    ),
    "index_put": MPSSkipInfo(NONCONTIGUOUS),
    "index_reduce": MPSSkipInfo(NONCONTIGUOUS),
    "inner": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "istft": MPSSkipInfo(NONCONTIGUOUS),
    "kthvalue": MPSSkipInfo(NONCONTIGUOUS),
    "lcm": MPSSkipInfo(NONCONTIGUOUS),
    "lerp": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "lgamma": MPSSkipInfo(NONCONTIGUOUS),
    "linalg.cholesky_ex": MPSSkipInfo(NONCONTIGUOUS),
    "linalg.cholesky": MPSSkipInfo(NONCONTIGUOUS),
    "linalg.cond": MPSSkipInfo(NONCONTIGUOUS),
    "linalg.cross": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "linalg.det": MPSSkipInfo(NONCONTIGUOUS),
    "linalg.eig": MPSSkipInfo(NONCONTIGUOUS),
    "linalg.eigh": MPSSkipInfo(NONCONTIGUOUS),
    "linalg.eigvals": MPSSkipInfo(NONCONTIGUOUS),
    "linalg.eigvalsh": MPSSkipInfo(NONCONTIGUOUS),
    "linalg.householder_product": MPSSkipInfo(NONCONTIGUOUS),
    "linalg.inv_ex": [
        MPSSkipInfo(
            NONCONTIGUOUS,
            dtypes=[torch.complex64],
        ),
        MPSSkipInfo(
            NONCONTIGUOUS,
            skip=True,
            skip_msg="Crashes on MPS with err: MPSMatrixDecompositionLU.mm:1146: \
                failed assertion `Number of columns in source exceeds source matrix size.'",
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
            skip=True,
            skip_msg="Crashes on MPS with err: failed assertion `Number of columns in source exceeds source matrix size.'",
            dtypes=[torch.float32],
        ),
    ],
    "linalg.ldl_factor_ex": MPSSkipInfo(NONCONTIGUOUS),
    "linalg.ldl_factor": MPSSkipInfo(NONCONTIGUOUS),
    "linalg.ldl_solve": MPSSkipInfo(NONCONTIGUOUS),
    "linalg.lstsq": MPSSkipInfo(NONCONTIGUOUS),
    "linalg.lu_factor_ex": MPSSkipInfo(NONCONTIGUOUS),
    "linalg.lu_factor": [
        MPSSkipInfo(
            NONCONTIGUOUS,
            dtypes=[torch.complex64],
        ),
        MPSSkipInfo(
            NONCONTIGUOUS,
            skip=True,
            skip_msg="Crashes on MPS with err: failed assertion `A command encoder is already encoding to this command buffer'",
            dtypes=[torch.float32],
        ),
    ],
    "linalg.lu": MPSSkipInfo(NONCONTIGUOUS),
    "linalg.lu_solve": MPSSkipInfo(NONCONTIGUOUS),
    "linalg.matrix_norm": MPSSkipInfo(NONCONTIGUOUS),
    "linalg.matrix_power": [
        MPSSkipInfo(
            NONCONTIGUOUS,
            dtypes=[torch.complex64],
        ),
        MPSSkipInfo(
            NONCONTIGUOUS,
            skip=True,
            dtypes=[torch.float32],
            skip_msg="Crashes on MPS with err: failed assertion `Number of columns in source exceeds source matrix size.'",
        ),
    ],
    "linalg.matrix_rank": MPSSkipInfo(NONCONTIGUOUS),
    "linalg.norm": MPSSkipInfo(NONCONTIGUOUS),
    "linalg.pinv": MPSSkipInfo(
        NONCONTIGUOUS,
        variant="hermitian",
    ),
    "linalg.qr": MPSSkipInfo(NONCONTIGUOUS),
    "linalg.slogdet": MPSSkipInfo(NONCONTIGUOUS),
    "linalg.solve": MPSSkipInfo(NONCONTIGUOUS),
    "linalg.solve_ex": MPSSkipInfo(NONCONTIGUOUS),
    "linalg.solve_triangular": MPSSkipInfo(NONCONTIGUOUS),
    "linalg.svd": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "linalg.svdvals": MPSSkipInfo(NONCONTIGUOUS),
    "linalg.tensorinv": [
        MPSSkipInfo(
            NONCONTIGUOUS,
            dtypes=[torch.complex64],
        ),
        MPSSkipInfo(
            NONCONTIGUOUS,
            dtypes=[torch.float32],
            skip=True,
            skip_msg="Crashes on MPS with err: failed assertion `Number of columns in source exceeds source matrix size.'",
        ),
    ],
    "linalg.tensorsolve": MPSSkipInfo(NONCONTIGUOUS),
    "linalg.vander": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "linalg.vecdot": MPSSkipInfo(NONCONTIGUOUS),
    "linalg.vector_norm": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "log1p": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.int64],
    ),
    "log_normal": MPSSkipInfo(NONCONTIGUOUS),
    "log_softmax": MPSSkipInfo(
        NONCONTIGUOUS,
        variant="with_dtype",
    ),
    "logcumsumexp": MPSSkipInfo(NONCONTIGUOUS),
    "logdet": MPSSkipInfo(NONCONTIGUOUS),
    "logspace": MPSSkipInfo(NONCONTIGUOUS),
    "lu": MPSSkipInfo(NONCONTIGUOUS),
    "lu_solve": MPSSkipInfo(NONCONTIGUOUS),
    "lu_unpack": MPSSkipInfo(NONCONTIGUOUS),
    "masked.cumprod": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "masked.cumsum": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "masked.median": MPSSkipInfo(
        NONCONTIGUOUS,
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
    "matrix_exp": MPSSkipInfo(NONCONTIGUOUS),
    "mm": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "mode": MPSSkipInfo(NONCONTIGUOUS),
    "nanmean": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "nanmedian": MPSSkipInfo(NONCONTIGUOUS),
    "nansum": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "native_dropout_backward": MPSSkipInfo(NONCONTIGUOUS),
    "nn.functional.adaptive_avg_pool1d": MPSSkipInfo(NONCONTIGUOUS),
    "nn.functional.adaptive_avg_pool2d": MPSSkipInfo(NONCONTIGUOUS),
    "nn.functional.adaptive_avg_pool3d": MPSSkipInfo(NONCONTIGUOUS),
    "nn.functional.adaptive_max_pool3d": MPSSkipInfo(NONCONTIGUOUS),
    "nn.functional.avg_pool3d": MPSSkipInfo(NONCONTIGUOUS),
    "nn.functional.channel_shuffle": MPSSkipInfo(NONCONTIGUOUS),
    "nn.functional.conv2d": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.float32],
    ),
    "nn.functional.conv_transpose3d": MPSSkipInfo(NONCONTIGUOUS),
    "nn.functional.ctc_loss": MPSSkipInfo(NONCONTIGUOUS),
    "nn.functional.embedding_bag": MPSSkipInfo(NONCONTIGUOUS),
    "nn.functional.fractional_max_pool2d": MPSSkipInfo(NONCONTIGUOUS),
    "nn.functional.fractional_max_pool3d": MPSSkipInfo(NONCONTIGUOUS),
    "nn.functional.grid_sample": MPSSkipInfo(NONCONTIGUOUS),
    "nn.functional.hardshrink": MPSSkipInfo(NONCONTIGUOUS),
    "nn.functional.interpolate": [
        MPSSkipInfo(NONCONTIGUOUS, variant="area"),
        MPSSkipInfo(
            NONCONTIGUOUS,
            variant="nearest-exact",
        ),
        MPSSkipInfo(
            NONCONTIGUOUS,
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
    "nn.functional.max_pool2d": MPSSkipInfo(NONCONTIGUOUS),
    "nn.functional.max_pool3d": MPSSkipInfo(NONCONTIGUOUS),
    "nn.functional.max_unpool1d": MPSSkipInfo(NONCONTIGUOUS),
    "nn.functional.max_unpool2d": MPSSkipInfo(NONCONTIGUOUS),
    "nn.functional.max_unpool3d": MPSSkipInfo(NONCONTIGUOUS),
    "nn.functional.multi_margin_loss": MPSSkipInfo(NONCONTIGUOUS),
    "nn.functional.multilabel_margin_loss": MPSSkipInfo(NONCONTIGUOUS),
    "nn.functional.normalize": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "nn.functional.pairwise_distance": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "nn.functional.pdist": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.float32],
    ),
    "nn.functional.rrelu": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.float32],
    ),
    "nn.functional.softmin": MPSSkipInfo(
        NONCONTIGUOUS,
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
        MPSSkipInfo(
            NONCONTIGUOUS,
            dtypes=[torch.complex64],
        ),
        MPSSkipInfo(
            NONCONTIGUOUS,
            variant="nuc",
            dtypes=[torch.float32],
        ),
    ],
    "normal": MPSSkipInfo(NONCONTIGUOUS),
    "ones_like": MPSSkipInfo(NONCONTIGUOUS),
    "ormqr": MPSSkipInfo(NONCONTIGUOUS),
    "pca_lowrank": MPSSkipInfo(NONCONTIGUOUS),
    "polygamma": MPSSkipInfo(
        NONCONTIGUOUS,
        variant="polygamma_n_0",
    ),
    "prod": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "put": MPSSkipInfo(NONCONTIGUOUS),
    "qr": MPSSkipInfo(NONCONTIGUOUS),
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
        NONCONTIGUOUS,
        variant="decimals_0",
    ),
    "rsub": MPSSkipInfo(NONCONTIGUOUS),
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
    "sinc": MPSSkipInfo(NONCONTIGUOUS),
    "softmax": MPSSkipInfo(
        NONCONTIGUOUS,
        variant="with_dtype",
    ),
    "special.airy_ai": MPSSkipInfo(NONCONTIGUOUS),
    "special.bessel_j0": MPSSkipInfo(NONCONTIGUOUS),
    "special.bessel_j1": MPSSkipInfo(NONCONTIGUOUS),
    "special.bessel_y0": MPSSkipInfo(NONCONTIGUOUS),
    "special.bessel_y1": MPSSkipInfo(NONCONTIGUOUS),
    "special.chebyshev_polynomial_t": MPSSkipInfo(NONCONTIGUOUS),
    "special.chebyshev_polynomial_u": MPSSkipInfo(NONCONTIGUOUS),
    "special.entr": MPSSkipInfo(NONCONTIGUOUS),
    "special.erfcx": MPSSkipInfo(NONCONTIGUOUS),
    "special.hermite_polynomial_h": MPSSkipInfo(NONCONTIGUOUS),
    "special.hermite_polynomial_he": MPSSkipInfo(NONCONTIGUOUS),
    "special.i0e": MPSSkipInfo(NONCONTIGUOUS),
    "special.i1e": MPSSkipInfo(NONCONTIGUOUS),
    "special.laguerre_polynomial_l": MPSSkipInfo(NONCONTIGUOUS),
    "special.log_ndtr": MPSSkipInfo(NONCONTIGUOUS),
    "special.modified_bessel_i0": MPSSkipInfo(NONCONTIGUOUS),
    "special.modified_bessel_i1": MPSSkipInfo(NONCONTIGUOUS),
    "special.modified_bessel_k0": MPSSkipInfo(NONCONTIGUOUS),
    "special.modified_bessel_k1": MPSSkipInfo(NONCONTIGUOUS),
    "special.ndtri": MPSSkipInfo(NONCONTIGUOUS),
    "special.polygamma": MPSSkipInfo(NONCONTIGUOUS),
    "special.scaled_modified_bessel_k0": MPSSkipInfo(NONCONTIGUOUS),
    "special.scaled_modified_bessel_k1": MPSSkipInfo(NONCONTIGUOUS),
    "special.spherical_bessel_j0": MPSSkipInfo(NONCONTIGUOUS),
    "special.xlog1py": MPSSkipInfo(NONCONTIGUOUS),
    "special.zeta": MPSSkipInfo(NONCONTIGUOUS),
    "std": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "svd": MPSSkipInfo(
        NONCONTIGUOUS,
        variant="",
        dtypes=[torch.complex64],
    ),
    "svd_lowrank": MPSSkipInfo(NONCONTIGUOUS),
    "take_along_dim": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "take": MPSSkipInfo(NONCONTIGUOUS),
    "tensordot": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "tile": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "to": MPSSkipInfo(NONCONTIGUOUS),
    "torch.ops.aten._efficient_attention_forward": MPSSkipInfo(NONCONTIGUOUS),
    "trace": MPSSkipInfo(
        NONCONTIGUOUS,
        skip=True,
        skip_msg="Crashes on MPS with err: 'mps.scatter' op operand #0 must be tensor of mps native type values,\
            but got 'tensor<25xcomplex<f32>>",
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
    "unique": MPSSkipInfo(NONCONTIGUOUS),
    "var": MPSSkipInfo(
        NONCONTIGUOUS,
        dtypes=[torch.complex64],
    ),
    "vdot": MPSSkipInfo(NONCONTIGUOUS),
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
                        test_module = "TestCommon"

                        decorator = DecorateInfo(
                            unittest.skip(skip.skip_msg)
                            if skip.skip
                            else unittest.expectedFailure,
                            test_module,
                            test,
                            device_type="mps",
                            dtypes=skip.dtypes,
                        )
                        op.decorators = op.decorators + (decorator,)

    return op_db
