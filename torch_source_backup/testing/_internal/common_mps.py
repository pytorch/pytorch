import unittest
from collections.abc import Sequence
from typing import Optional

import torch

from .common_utils import MACOS_VERSION
from .opinfo.core import DecorateInfo, OpInfo


if torch.backends.mps.is_available():

    def mps_ops_modifier(
        ops: Sequence[OpInfo],
        device_type: Optional[str] = None,
        xfail_exclusion: Optional[list[str]] = None,
    ) -> Sequence[OpInfo]:
        if xfail_exclusion is None:
            xfail_exclusion = []

        # Supported complex OPS
        SUPPORTED_COMPLEX_OPS = {
            "__radd__",
            "__rmul__",
            "__rsub__",
            "__getitem__",
            "_unsafe_masked_index",
            "abs",
            "add",
            "alias_copy",
            "argwhere",
            "atleast_1d",
            "atleast_2d",
            "atleast_3d",
            "as_strided",
            "as_strided_copy",
            "as_strided_scatter",
            "asin",
            "acos",
            "atan",
            "broadcast_tensors",
            "broadcast_to",
            "chalf",
            "cfloat",
            "chunk",
            "clone",
            "conj",
            "conj_physical",
            "contiguous",
            "cos",
            "cosh",
            "diag",
            "diag_embed",
            "diagflat",
            "diagonal",
            "diagonal_copy",
            "diagonal_scatter",
            "divno_rounding_mode",
            "dsplit",
            "empty",
            "empty_permuted",
            "empty_strided",
            "exp",
            "expm1",
            "exp2",
            "expand",
            "expand_as",
            "expand_copy",
            "flatten",
            "fill",
            "full",
            "full_like",
            "H",
            "hsplit",
            "imag",
            "index_copy",
            "index_select",
            "isfinite",
            "isinf",
            "isreal",
            "item",
            "kron",
            "linalg.diagonal",
            "linalg.svd",
            "log10",
            "log1p",
            "log2",
            "log",
            "mH",
            "mT",
            "masked_fill",
            "masked_scatter",
            "masked_select",
            "meshgridlist_of_tensors",
            "meshgridvariadic_tensors",
            "movedim",
            "mul",
            "narrow",
            "narrow_copy",
            "neg",
            "new_full",
            "new_ones",
            "new_zeros",
            "nn.functional.conv1d",
            "nn.functional.conv2d",
            "nn.functional.conv_transpose1d",
            "nn.functional.conv_transpose2d",
            "nn.functional.conv_transpose3d",
            "nn.functional.feature_alpha_dropoutwithout_train",
            "nn.functional.padcircular",
            "nn.functional.softsign",
            "nn.functional.tanhshrink",
            "nn.functional.unfold",
            "nonzero",
            "ones",
            "ones_like",
            "outer",
            "permute",
            "permute_copy",
            "positive",
            "randn",
            "ravel",
            "real",
            "repeat_interleave",
            "reshape_as",
            "reshape",
            "resolve_conj",
            "resolve_neg",
            "rsqrt",
            "rsub",
            "scalar_tensor",
            "select",
            "sgn",
            "sigmoid",
            "sin",
            "sinc",
            "sinh",
            "slice",
            "special.spherical_bessel_j0",
            "special.entr",
            "special.xlog1py",
            "special.zeta",
            "split",
            "split_with_sizes",
            "split_with_sizes_copy",
            "splitlist_args",
            "sqrt",
            "squeeze",
            "squeeze_copy",
            "squeezemultiple",
            "sub",
            "svd",
            "t",
            "t_copy",
            "tanh",
            "tan",
            "tensor_split",
            "transpose",
            "transpose_copy",
            "tril",
            "triu",
            "true_divide",
            "T",
            "unbind",
            "unbind_copy",
            "unflatten",
            "unfold",
            "unfold_copy",
            "unsafe_chunk",
            "unsafe_split",
            "unsqueeze",
            "unsqueeze_copy",
            "view_as",
            "view_as_real",
            "view",
            "view_copy",
            "vsplit",
            "zero_",
            "zeros",
            "zeros_like",
        }

        AFTER_MACOS_14_0_SUPPORTED_COMPLEX_OPS = {
            "__rdiv__",
            "__rmatmul__",
            "_chunk_cat",
            "acosh",
            "all",
            "allclose",
            "angle",
            "any",
            "addcdiv",
            "addcmul",
            "addmmdecomposed",
            "addmv",
            "atanh",
            "bfloat16",
            "bmm",
            "bool",
            "cartesian_prod",
            "cat",
            "char",
            "column_stack",
            "combinations",
            "corrcoef",
            "constant_pad_nd",
            "cov",
            "count_nonzero",
            "diff",
            "div",
            "dot",
            "dstack",
            "einsum",
            "eq",
            "equal",
            "eye",
            "fft.fft",
            "fft.fft2",
            "fft.fftn",
            "fft.fftshift",
            "fft.ifft",
            "fft.ifft2",
            "fft.ifftn",
            "fft.ifftshift",
            "fft.irfftn",
            "fft.irfft2",
            "fft.irfft",
            "fft.hfftn",
            "fft.hfft2",
            "fft.hfft",
            "flip",
            "fliplr",
            "flipud",
            "float",
            "gradient",
            "half",
            "hstack",
            "inner",
            "int",
            "isclose",
            "isnan",
            "ldexp",
            "lerp",
            "linalg.multi_dot",
            "linalg.pinv",
            "linspace",
            "linspacetensor_overload",
            "logical_and",
            "logical_not",
            "logical_or",
            "logical_xor",
            "logsumexp",
            "long",
            "masked.mean",
            "masked.prod",
            "masked.std",
            "masked.sum",
            "masked.var",
            "masked.logsumexp",
            "matmul",
            "mean",
            "mm",
            "mv",
            "ne",
            "nn.functional.padconstant",
            "nn.functional.padreflect",
            "nn.functional.padreplicate",
            "nn.functional.pixel_shuffle",
            "nn.functional.pixel_unshuffle",
            "nn.functional.rms_norm",
            "pinverse",
            "prod",
            "reciprocal",
            "roll",
            "rot90",
            "short",
            "sinh",
            "sqrt",
            "square",
            "stack",
            "stft",
            "sum",
            "sum_to_size",
            "tensordot",
            "trace",
            "trapz",
            "trapezoid",
            "vstack",
            "where",
            "byte",
        }
        # Those ops worked on MacOS12, but broken on MacOS13, see https://github.com/pytorch/pytorch/issues/85758
        MACOS_BEFORE_13_3_XFAILLIST = {
            # Failures due to precision issues (due to fast-math). These has been fixed in MacOS 13.3+
            "cdist": [torch.float32],
            # CPU Error: cpu not giving nan for x/0.0
            "atan2": [
                torch.bool,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
                torch.int8,
            ],
            # test blow pass on macOS 12 as it falls back to cpu
            # Argsort case using duplicate indices (undefined behaviour):
            #  - CPU output: tensor([2546, 6917, 3181,  ..., 7128, 5133,   30], device='cpu')
            #  - MPS output: tensor([2546, 6917, 3181,  ..., 7128,   30, 5133], device='mps:0')
            # Elements from index 30 and 5133 are both equal.
            # Since CPU is not using argsort with stable=True, these cases result in undefined behaviour.
            "argsort": [torch.float16, torch.int8, torch.uint8, torch.bool],
            # Same issue as `argsort` with duplicate indices. This test checks both the sorted values and the indices.
            # The values of the sorted tensor match the CPU,
            # but in case of the returned indices this results in undefined behaviour.
            "sort": [torch.int8, torch.uint8, torch.bool, torch.float16],
            # Unsupported dtypes
            "cumsum": [torch.int64],
            "cumprod": [torch.int64],
            "cumulative_trapezoid": [torch.int64],
            "masked.cumsum": [torch.int64],
            "masked.cumprod": [torch.int64],
            "linalg.vander": [torch.int64],
            # Fail with `Expected 1.0 but got nan.` for empty tensors
            # Caused by sample input at index 23: SampleInput(
            #     input=Tensor[size=(), device="mps:0", dtype=torch.float32],
            #     args=(0),
            #     kwargs={'mask': 'Tensor[size=(), device="mps:0", dtype=torch.bool]'},
            #     broadcasts_input=False, name='')
            "masked.softmin": [torch.float32, torch.float16],
            "masked.softmax": [torch.float32, torch.float16],
            "masked.log_softmax": [torch.float32, torch.float16],
        }

        MACOS_AFTER_13_1_XFAILLIST = {
            # before macOS 13.2 it falls back to cpu and pass the forward pass
            "grid_sampler_2d": [
                torch.float32,
                torch.float16,
                torch.bfloat16,
            ],  # Unsupported Border padding mode
        }

        MACOS_13_3_XFAILLIST = {
            # Failure due to precision issue for fp16
            # on both cpu and mps there are test cases that might produce inf result
            # 'nn.functional.pairwise_distance': [torch.float16],
            # test blow pass on macOS 12 as it falls back to cpu
            # Argsort case using duplicate indices (undefined behaviour):
            #  - CPU output: tensor([2546, 6917, 3181,  ..., 7128, 5133,   30], device='cpu')
            #  - MPS output: tensor([2546, 6917, 3181,  ..., 7128,   30, 5133], device='mps:0')
            # Elements from index 30 and 5133 are both equal.
            # Since CPU is not using argsort with stable=True, these cases result in undefined behaviour.
            "argsort": [
                torch.float16,
                torch.int8,
                torch.uint8,
                torch.bool,
                torch.bfloat16,
            ],
            # Same issue as `argsort` with duplicate indices. This test checks both the sorted values and the indices.
            # The values of the sorted tensor match the CPU,
            # but in case of the returned indices this results in undefined behaviour.
            "sort": [
                torch.int8,
                torch.uint8,
                torch.bool,
                torch.float16,
                torch.bfloat16,
            ],
        }

        MACOS_BEFORE_14_4_XFAILLIST = {
            # These ops work fine in 14.4 but fail in 14.2 or 13.x
            "fft.hfft2": [torch.complex64],
        }

        # Those ops are not expected to work
        UNIMPLEMENTED_XFAILLIST = {
            # Failures due to lack of op implementation on MPS backend
            "logspace": None,
            "logspacetensor_overload": None,
            "linalg.eig": None,
            "linalg.eigvals": None,
            "put": None,
            "cauchy_": None,
            "cauchy": None,
            "cholesky_inverse": None,
            "cholesky_solve": None,
            "frexp": None,
            "gcd": None,
            "geqrf": None,
            "nn.functional.grid_sample": None,  # Unsupported Border padding mode
            "heaviside": None,
            "igamma": None,
            "igammac": None,
            "index_reduceprod": None,
            "index_reducemean": None,
            "index_reduceamax": None,
            "index_reduceamin": None,
            "kthvalue": None,
            "lcm": None,
            "linalg.cond": None,
            "linalg.eigh": None,
            "linalg.eigvalsh": None,
            "linalg.householder_product": None,
            "linalg.ldl_factor": None,
            "linalg.ldl_factor_ex": None,
            "linalg.ldl_solve": None,
            "linalg.lstsq": None,
            "linalg.lstsqgrad_oriented": None,
            "linalg.lu": None,
            "linalg.lu_solve": None,
            "linalg.matrix_norm": [torch.float32],
            "linalg.norm": [torch.float32],
            "linalg.normsubgradients_at_zero": [torch.float32],
            "linalg.qr": None,
            "linalg.svdvals": None,
            "linalg.vecdot": None,
            "logcumsumexp": None,
            "lu_solve": None,
            "masked.median": None,
            "matrix_exp": None,
            "mode": None,
            "native_dropout_backward": None,
            "normnuc": None,
            "nn.functional.fractional_max_pool2d": None,
            "nn.functional.fractional_max_pool3d": None,
            "nn.functional.adaptive_avg_pool3d": None,
            "nn.functional.adaptive_max_pool3d": None,
            "nn.functional.interpolatearea": None,
            "nn.functional.interpolatebicubic": [torch.uint8],
            "nn.functional.max_unpool1dgrad": None,
            "nn.functional.max_unpool2dgrad": None,
            "nn.functional.max_unpool3dgrad": None,
            "nn.functional.avg_pool3d": None,
            "nn.functional.ctc_loss": None,
            "nn.functional.embedding_bag": None,
            "nn.functional.max_pool3d": None,
            "nn.functional.max_unpool1d": None,
            "nn.functional.max_unpool2d": None,
            "nn.functional.max_unpool3d": None,
            "nn.functional.multi_margin_loss": None,
            "nn.functional.multilabel_margin_loss": None,
            "nn.functional.pdist": None,
            "nn.functional.rrelu": None,
            "nn.functional.norm": None,
            "ormqr": None,
            "pca_lowrank": None,
            "qr": None,
            "scatter_reduceamax": [torch.int32, torch.int64]
            if MACOS_VERSION < 15.0
            else [torch.int64],
            "scatter_reduceamin": [torch.int32, torch.int64]
            if MACOS_VERSION < 15.0
            else [torch.int64],
            "segment_reduce": None,
            "_segment.reduce": None,
            "segment.reduce": None,
            "segment_reduce_offsets": None,
            "_segment_reduce_offsets": None,
            "_segment_reduce_lengths": None,
            "_segment_reducelengths": None,
            "_segment_reduceoffsets": None,
            "sparse.mm": None,
            "sparse.sampled_addmm": None,
            "sparse.mmreduce": None,
            "special.airy_ai": None,
            "special.erfcx": None,
            "special.laguerre_polynomial_l": None,
            "special.log_ndtr": None,
            "special.ndtri": None,
            "svd_lowrank": None,
            "symeig": None,
            "take": None,
            "to": None,
            "to_sparse": None,
            "unique": None,
            "vdot": None,
            "segment_reduce_": None,
            "_upsample_bilinear2d_aa": [torch.uint8],  # uint8 is for CPU only
            "_upsample_bicubic2d_aa": [torch.uint8],  # uint8 is for CPU only
            "geometric": None,
            "geometric_": None,
            "log_normal_": None,
            "log_normal": None,
            "cdouble": None,
            "double": None,
            "nn.functional.softminwith_dtype": None,
            "log_softmaxwith_dtype": None,
            "softmaxwith_dtype": None,
            "float_power": None,
            "linalg.matrix_rankhermitian": None,
            "linalg.pinvhermitian": None,
            "nonzero_static": None,
            # MPS: input sizes must be divisible by output sizes
            "nn.functional.adaptive_avg_pool1d": None,
            "nn.functional.adaptive_avg_pool2d": None,
            # Convolution for integral types is not supported on MPS
            "nn.functional.conv1d": [torch.int64],
            "nn.functional.conv2d": [torch.int64],
            "nn.functional.conv3d": [torch.int64],
            "nn.functional.conv_transpose1d": [torch.int64],
            "nn.functional.conv_transpose2d": [torch.int64, torch.bfloat16],
            "nn.functional.conv_transpose3d": [
                torch.int64,
                torch.bfloat16,
                torch.float16,
            ],
            # Unsupported dtypes
            "dot": [torch.int64] if MACOS_VERSION < 14.0 else [],
            "histc": [torch.float16, torch.bfloat16],
            "index_add": [torch.int64],
            # GEMM on MPS is not supported for integral types
            "nn.functional.linear": [
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
                torch.int8,
            ],
            "addmmdecomposed": [
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
                torch.int8,
            ],
            "addbmm": [torch.int16, torch.int32, torch.int64, torch.uint8, torch.int8],
            "addmm": [torch.int16, torch.int32, torch.int64, torch.uint8, torch.int8],
            "baddbmm": [torch.int16, torch.int32, torch.int64, torch.uint8, torch.int8],
            "mat": [torch.int16, torch.int32, torch.int64, torch.uint8, torch.int8],
            "matmul": [torch.int64] if MACOS_VERSION < 14.0 else [],
            "__rmatmul__": [torch.int64] if MACOS_VERSION < 14.0 else [],
            # returned output on CPU is float64
            "bincount": [
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
                torch.int8,
            ],
            # round not working properly for float16 and bfloat16
            "round": [torch.float16, torch.bfloat16],
            "rounddecimals_0": [torch.bfloat16],
            # atomic operations not supported
            "_unsafe_masked_index_put_accumulate": [
                torch.int8,
                torch.uint8,
                torch.int16,
                torch.int64,
            ],
        }

        if MACOS_VERSION < 14.0:
            # FFT and BFloat16 support was added in MacOS 14
            UNIMPLEMENTED_XFAILLIST.update(
                {
                    "bfloat16": None,
                    "fft.fft": None,
                    "fft.fft2": None,
                    "fft.fftn": None,
                    "fft.hfft": None,
                    "fft.hfft2": None,
                    "fft.hfftn": None,
                    "fft.ifft": None,
                    "fft.ifft2": None,
                    "fft.ifftn": None,
                    "fft.ihfft": None,
                    "fft.ihfft2": None,
                    "fft.ihfftn": None,
                    "fft.irfft": None,
                    "fft.irfft2": None,
                    "fft.irfftn": None,
                    "fft.rfft": None,
                    "fft.rfft2": None,
                    "fft.rfftn": None,
                    "stft": None,
                    # Error in TestConsistencyCPU.test_output_match_isin_cpu fails for integers,
                    # not reproducible in later OS. Added assert to op if used in < 14.0
                    "isin": [
                        torch.int64,
                        torch.int32,
                        torch.int16,
                        torch.uint8,
                        torch.int8,
                    ],
                    "nn.functional.max_pool2d": [torch.uint8],
                }
            )

        if MACOS_VERSION < 15.0:
            UNIMPLEMENTED_XFAILLIST.update(
                {
                    "quantile": None,
                    "nanquantile": None,
                }
            )

        UNDEFINED_XFAILLIST = {
            # Top 60 operators
            # topk fails with duplicate indices
            "topk": [
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
                torch.int8,
            ],
            # Failures due to random output that they generate using
            # Philox engine causing mismatch with CPU results
            "multinomial": [
                torch.float16,
                torch.float32,
                torch.bfloat16,
            ],  # random results
            "uniform": [torch.float16, torch.float32, torch.bfloat16],
            "rand_like": [torch.float16, torch.float32, torch.bfloat16],
            "randint": None,
            "randint_like": None,
            "randn": None,
            "randn_like": None,
            "bernoulli": [torch.float16, torch.float32, torch.bfloat16],
            "exponential": [torch.float16, torch.float32, torch.bfloat16],
            "nn.functional.feature_alpha_dropoutwith_train": [
                torch.float16,
                torch.float32,
                torch.bfloat16,
            ],
            "normal": [torch.float16, torch.float32, torch.bfloat16],
            "normalin_place": [torch.float16, torch.float32, torch.bfloat16],
            "normalnumber_mean": [torch.float16, torch.float32, torch.bfloat16],
            "nn.functional.alpha_dropout": [
                torch.float16,
                torch.float32,
                torch.bfloat16,
            ],
            "nn.functional.dropout": [torch.float16, torch.float32, torch.bfloat16],
            "nn.functional.dropout2d": [torch.float16, torch.float32, torch.bfloat16],
            "nn.functional.dropout3d": [torch.float16, torch.float32, torch.bfloat16],
            # See https://github.com/pytorch/pytorch/issues/111479
            "nn.functional.multi_head_attention_forward": [
                torch.float32,
                torch.float16,
                torch.bfloat16,
            ],
            "index_put": [
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int64,
            ],
            # zero to negative integer powers are undefined
            "__rpow__": [torch.int8, torch.int16, torch.int32, torch.int64],
            "resize_": [torch.float16, torch.float32, torch.bfloat16],
            "resize_as_": [torch.float16, torch.float32, torch.bfloat16],
            # CPU Errors:
            "addr": [
                torch.bool,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
                torch.int8,
            ],  # "addmv_impl_cpu" not implemented for 'Half'
            "as_stridedpartial_views": None,  # cpu result off, showing random values
            # random results
            # mps vs cpu:
            # Mismatched elements: 40 / 96 (41.7%)
            # Greatest absolute difference: 17.892311096191406 at index (1, 0, 2) (up to 1e-05 allowed)
            # Greatest relative difference: inf at index (1, 0, 0) (up to 1.3e-06 allowed)
            # cuda(2.0.0.dev20230301+cu117) vs cpu:
            # Mismatched elements: 56 / 96 (58.3%)
            # Greatest absolute difference: 17.892311096191406 at index (1, 0, 2) (up to 1e-05 allowed)
            # Greatest relative difference: inf at index (1, 0, 0) (up to 1.3e-06 allowed)
            "nn.functional.scaled_dot_product_attention": [
                torch.float32,
                torch.float16,
                torch.bfloat16,
            ],
        }

        ON_MPS_XFAILLIST = {
            # Failures due to lack of implementation of downstream functions on MPS backend
            # TODO: remove these once downstream function 'aten::_linalg_svd.U' have been implemented
            "linalg.matrix_rank": None,
            # Exception: Caused by `torch.arange(-8.001, -4.0, dtype=torch.uint8, device="mps")`
            "arange": [torch.uint8],
        }

        EMPTY_OPS_SKIPLIST = {
            # Fill tensors with uninitialized data, causing mismatch with CPU.
            # They occasionally match, thus skipping them.
            # See https://github.com/pytorch/pytorch/issues/100175
            "new_empty": None,
            "new_empty_strided": None,
            "empty_strided": None,
            # CPU: empty is returning all 0's and there is a mismatch with MPS
            # allocation (MacOS 13). According to
            # https://pytorch.org/docs/2.0/generated/torch.empty.html
            "empty": None,
            "empty_like": None,
            "empty_permuted": None,
        }

        SKIPLIST = {
            # Unsupported
            # This doesn't work on M1, but is partially working on M2 with the exception of torch.float16
            "nn.functional.conv3d": None,
        }

        def addDecorator(op: OpInfo, d: DecorateInfo) -> None:
            if device_type is not None:
                d.device_type = device_type

            op.decorators = op.decorators + (d,)

        for op in ops:
            key = op.name + op.variant_test_name
            if key in EMPTY_OPS_SKIPLIST:
                addDecorator(
                    op,
                    DecorateInfo(
                        unittest.skip("Skipping empty ops."),
                        dtypes=EMPTY_OPS_SKIPLIST[key],
                    ),
                )
            if key in SKIPLIST:
                addDecorator(
                    op, DecorateInfo(unittest.skip("Skipped!"), dtypes=SKIPLIST[key])
                )
            for xfaillist in [
                UNIMPLEMENTED_XFAILLIST,
                UNDEFINED_XFAILLIST,
                ON_MPS_XFAILLIST,
            ]:
                if key in xfaillist and key not in xfail_exclusion:
                    addDecorator(
                        op,
                        DecorateInfo(unittest.expectedFailure, dtypes=xfaillist[key]),
                    )

            if (
                key in MACOS_BEFORE_14_4_XFAILLIST
                and key not in xfail_exclusion
                and (MACOS_VERSION < 14.4)
            ):
                addDecorator(
                    op,
                    DecorateInfo(
                        unittest.expectedFailure,
                        dtypes=MACOS_BEFORE_14_4_XFAILLIST[key],
                    ),
                )

            if (
                key in MACOS_BEFORE_13_3_XFAILLIST
                and key not in xfail_exclusion
                and (torch.backends.mps.is_macos13_or_newer() and MACOS_VERSION < 13.3)
            ):
                addDecorator(
                    op,
                    DecorateInfo(
                        unittest.expectedFailure,
                        dtypes=MACOS_BEFORE_13_3_XFAILLIST[key],
                    ),
                )

            if (
                key in MACOS_AFTER_13_1_XFAILLIST
                and key not in xfail_exclusion
                and torch.backends.mps.is_macos13_or_newer(2)
            ):
                addDecorator(
                    op,
                    DecorateInfo(
                        unittest.expectedFailure, dtypes=MACOS_AFTER_13_1_XFAILLIST[key]
                    ),
                )

            if (
                key in MACOS_13_3_XFAILLIST
                and key not in xfail_exclusion
                and (MACOS_VERSION >= 13.3)
            ):
                addDecorator(
                    op,
                    DecorateInfo(
                        unittest.expectedFailure, dtypes=MACOS_13_3_XFAILLIST[key]
                    ),
                )

            # If ops is not supported for complex types, expect it to fail
            if key not in SUPPORTED_COMPLEX_OPS and (
                key not in AFTER_MACOS_14_0_SUPPORTED_COMPLEX_OPS
                or MACOS_VERSION < 14.0
            ):
                addDecorator(
                    op,
                    DecorateInfo(
                        unittest.expectedFailure,
                        dtypes=[torch.complex32, torch.complex64],
                    ),
                )

        return ops

    def mps_ops_grad_modifier(ops: Sequence[OpInfo]) -> Sequence[OpInfo]:
        XFAILLIST_GRAD = {
            # Unimplemented ops
            "_segment_reduce": [torch.float16, torch.float32],
            "_chunk_cat": [torch.float16, torch.float32],
            "_upsample_bilinear2d_aa": None,  # `_upsample_bilinear2d_aa_backward_out` not implemented for MPS
            "_upsample_bicubic2d_aa": None,  # `_upsample_bilinear2d_aa_backward_out` not implemented for MPS
            "sparse.mmreduce": [torch.float32],  # csr not supported
            "unique_consecutive": [torch.float16, torch.float32],
            "scalar_tensor": [torch.float16, torch.float32],
            "cdist": [torch.float32],
            "masked.scatter": [torch.float16, torch.float32],
            "index_fill": [torch.float16, torch.float32],  # missing `aten::_unique`.
            "linalg.solve": [torch.float16, torch.float32],  # missing `aten::lu_solve`.
            "linalg.solve_ex": [
                torch.float16,
                torch.float32,
            ],  # missing `aten::lu_solve`.
            "linalg.tensorsolve": [
                torch.float16,
                torch.float32,
            ],  # missing `aten::lu_solve`.
            "linalg.det": [torch.float16, torch.float32],  # missing aten::lu_solve.out
            "linalg.slogdet": [
                torch.float16,
                torch.float32,
            ],  # missing aten::lu_solve.out
            "logdet": [torch.float16, torch.float32],  # missing aten::lu_solve.out
            "aminmax": [torch.float32, torch.float16],
            "special.i1": [torch.float16],  # "i1_backward" not implemented for 'Half'
            "special.i1e": [torch.float16],  # "i1e_backward" not implemented for 'Half'
            # Correctness issues
            "atanh": [torch.float32],
            # Random output
            "exponential": [torch.float16, torch.float32],
            # CPU errors
            # derivative for zeta is not implemented
            "special.zeta": None,
            # derivative for aten::nextafter is not implemented on CPU
            "nextafter": None,
            # derivative for aten::floor_divide is not implemented on CPU
            "floor_divide": [torch.float16, torch.float32],
            # derivative for aten::narrow_copy is not implemented on CPU
            "narrow_copy": [torch.float16, torch.float32],
            # derivative for aten::_histogramdd_from_bin_cts is not implemented on CPU
            "histogramdd": [torch.float16, torch.float32],
            # derivative for aten::histogram is not implemented
            "histogram": [torch.float16, torch.float32],
            # 'bool' object is not iterable
            "allclose": [torch.float16, torch.float32],
            "equal": [torch.float16, torch.float32],
            # 'float' object is not iterable
            "item": [torch.float16, torch.float32],
            # "smooth_l1_backward_cpu_out" not implemented for 'Half'
            "nn.functional.smooth_l1_loss": [torch.float16],
            # cpu error: grad requires non-empty inputs
            "randn": [torch.float16, torch.float32],
            "signal.windows.bartlett": [torch.float32],
            "signal.windows.blackman": [torch.float32],
            "signal.windows.cosine": [torch.float32],
            "signal.windows.exponential": [torch.float32],
            "signal.windows.gaussian": [torch.float32],
            "signal.windows.general_cosine": [torch.float32],
            "signal.windows.general_hamming": [torch.float32],
            "signal.windows.hamming": [torch.float32],
            "signal.windows.hann": [torch.float32],
            "signal.windows.kaiser": [torch.float32],
            "signal.windows.nuttall": [torch.float32],
            "eye": [torch.float16, torch.float32],
            # round not working properly for float16
            "round": [torch.float16],
            # topk fails with duplicate indices
            "topk": [torch.float16],
        }

        MACOS_BEFORE_13_3_XFAILLIST_GRAD = {
            # Failures due to precision issues (may be fast-math). These has been fixed in MacOS 14
            "masked.softmin": [torch.float32, torch.float16],
            "masked.softmax": [torch.float32, torch.float16],
            "masked.log_softmax": [torch.float32, torch.float16],
            "atanh": [torch.float16],
            "triangular_solve": [torch.float32],
            # Unsupported Border padding mode, forward pass success as fallback to cpu
            "grid_sampler_2d": [torch.float32, torch.float16, torch.bfloat16],
            # Same issue as `argsort` and `sort` with duplicate elements (undefined behaviour).
            # Forward pass is passing since `msort` doesn't return the indices, just the values, which match the CPU.
            # On the backward pass for `sort` both are used (values and indices), thus resulting in a issmatch between CPU and MPS.
            # Running `msort` with stable `sort` passes.
            "msort": [torch.float16],
        }

        SKIPLIST_GRAD = {
            "nn.functional.pairwise_distance": [torch.float16],
            # failed assertion `destination datatype must be fp32'
            "nn.functional.conv1d": [torch.float16],
            "nn.functional.conv2d": [torch.float16],
            "nn.functional.conv3d": [torch.float16],
            "nn.functional.conv_transpose1d": [torch.float16],
            "nn.functional.conv_transpose2d": [torch.float16],
            "nn.functional.conv_transpose3d": [torch.float16],
        }

        MACOS_13_3_XFAILLIST_GRAD = {
            # Same issue as `argsort` and `sort` with duplicate elements (undefined behaviour).
            # Forward pass is passing since `msort` doesn't return the indices, just the values, which match the CPU.
            # On the backward pass for `sort` both are used (values and indices), thus resulting in a issmatch between CPU and MPS.
            # Running `msort` with stable `sort` passes.
            "msort": [torch.float16],
        }

        ON_MPS_XFAILLIST = {
            # Failures due to lack of implementation of downstream functions on MPS backend
            # TODO: remove these once downstream function 'aten::_linalg_svd.U' have been implemented
            "linalg.matrix_rank": None,
            # Exception: Caused by sample input at index 3 on MPS
            "nn.functional.conv3d": [torch.float32],
        }

        def addDecorator(op: OpInfo, d: DecorateInfo) -> None:
            op.decorators = op.decorators + (d,)

        for op in ops:
            key = op.name + op.variant_test_name
            if key in XFAILLIST_GRAD:
                addDecorator(
                    op,
                    DecorateInfo(unittest.expectedFailure, dtypes=XFAILLIST_GRAD[key]),
                )

            if key in SKIPLIST_GRAD:
                addDecorator(op, DecorateInfo(unittest.skip, dtypes=SKIPLIST_GRAD[key]))

            if key in ON_MPS_XFAILLIST:
                addDecorator(
                    op,
                    DecorateInfo(
                        unittest.expectedFailure, dtypes=ON_MPS_XFAILLIST[key]
                    ),
                )

            if key in MACOS_BEFORE_13_3_XFAILLIST_GRAD and (
                torch.backends.mps.is_macos13_or_newer() and MACOS_VERSION < 13.3
            ):
                addDecorator(
                    op,
                    DecorateInfo(
                        unittest.expectedFailure,
                        dtypes=MACOS_BEFORE_13_3_XFAILLIST_GRAD[key],
                    ),
                )

            if key in MACOS_13_3_XFAILLIST_GRAD and (MACOS_VERSION >= 13.3):
                addDecorator(
                    op,
                    DecorateInfo(
                        unittest.expectedFailure, dtypes=MACOS_13_3_XFAILLIST_GRAD[key]
                    ),
                )
        return ops

    def mps_ops_error_inputs_modifier(ops: Sequence[OpInfo]) -> Sequence[OpInfo]:
        # Error input samples do not take a dtype argument.
        XFAILLIST = {
            # Exceptions are not raised
            "__rmod__",
            "__rsub__",
            "__rpow__",
            "bernoulli",
            "clamp_max",
            "clamp_min",
            "masked_scatter",
            # unsupported float64 dtype
            "cat",
            "complex",
            "multinomial",
            "nn.functional.conv1d",
            "nn.functional.conv2d",
            "nn.functional.conv3d",
            "gather",
            "scatter",
            "scatter_add",
            # MPS does not support tensor dimensions > 16
            "amax",
            "amin",
            "aminmax",
            # memory overlapping checks
            "index_select",
            # unimplemented
            "logcumsumexp",
        }

        def addDecorator(op: OpInfo, d: DecorateInfo) -> None:
            op.decorators = op.decorators + (d,)

        for op in ops:
            key = op.name + op.variant_test_name
            if key in XFAILLIST:
                addDecorator(op, DecorateInfo(unittest.expectedFailure))

        return ops
