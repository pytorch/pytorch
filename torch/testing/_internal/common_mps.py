import unittest
from collections.abc import Sequence

import torch

from .common_utils import MACOS_VERSION
from .opinfo.core import DecorateInfo, OpInfo


if torch.backends.mps.is_available():

    def mps_ops_modifier(
        ops: Sequence[OpInfo],
        device_type: str = "mps",
        xfail_exclusion: list[str] | None = None,
        sparse: bool = False,
    ) -> Sequence[OpInfo]:
        if xfail_exclusion is None:
            xfail_exclusion = []

        # Complex OPS that are NOT supported on MPS. Any op that is tested with
        # complex dtypes and is absent from this list is expected to pass; the
        # ops below are xfailed for complex32/complex64. Drill this list down as
        # complex support is added.
        UNSUPPORTED_COMPLEX_OPS = {
            "addr",
            "cholesky_inverse",
            "float_power",
            "geqrf",
            "linalg.eig",
            "linalg.eigvals",
            "linalg.inv",
            "linalg.inv_ex",
            "linalg.ldl_factor",
            "linalg.ldl_factor_ex",
            "linalg.ldl_solve",
            "linalg.matrix_power",
            "linalg.matrix_sqrth",
            "linalg.solve_triangular",
            "linalg.tensorinv",
            "log_softmaxwith_dtype",
            "nn.functional.channel_shuffle",
            "nn.functional.conv3d",
            "nn.functional.padreplicate_negative",
            "ormqr",
            "renorm",
            "sparse.sampled_addmm",
            "to_sparse",
            "triangular_solve",
        }

        MACOS_BEFORE_14_4_XFAILLIST = {
            # These ops work fine in 14.4 but fail in 14.2 or 13.x
            "fft.hfft2": [torch.complex64],
        }

        MACOS_BEFORE_15_0_XFAILLIST = {
            # matrix_exp is disabled on MPS before macOS 15 (TORCH_CHECK): MPSGraph
            # complex matmul is numerically unreliable there and breaks the
            # scale-and-square recurrence, so the op raises for every dtype.
            "matrix_exp": None,
        }

        # Those ops are not expected to work
        UNIMPLEMENTED_XFAILLIST: dict[str, list | None] = {
            # Failures due to lack of op implementation on MPS backend
            "linalg.eig": None,
            "linalg.eigvals": None,
            "hash_tensor": None,
            "heaviside": None,
            # "kthvalue": None,
            "linalg.ldl_factor": None,
            "linalg.ldl_factor_ex": None,
            "linalg.ldl_solve": None,
            "linalg.matrix_sqrth": None,
            "max_pool2d_with_indices_backward": [
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            ],
            "median": [torch.bool],
            "mode": None,
            "nanmedian": [torch.bool],
            "native_batch_norm": [
                torch.uint8,
                torch.bool,
                torch.int8,
                torch.int16,
                torch.int32,
            ],
            "nn.functional.avg_pool1d": [
                torch.int16,
                torch.int32,
                torch.uint8,
                torch.bool,
                torch.int8,
            ],
            "nn.functional.avg_pool2d": [
                torch.int16,
                torch.int32,
                torch.uint8,
                torch.bool,
                torch.int8,
            ],
            "nn.functional.avg_pool3d": [
                torch.int16,
                torch.int32,
                torch.uint8,
                torch.int8,
            ],
            "nn.functional.batch_norm": [
                torch.uint8,
                torch.bool,
                torch.int8,
                torch.int16,
                torch.int32,
            ],
            "nn.functional.fractional_max_pool2d": None,
            "nn.functional.fractional_max_pool3d": None,
            "nn.functional.glu": [
                torch.int32,
                torch.uint8,
                torch.bool,
                torch.int8,
                torch.int16,
            ],
            "nn.functional.huber_loss": [
                torch.uint8,
                torch.bool,
                torch.int8,
                torch.int16,
                torch.int32,
            ],
            "nn.functional.adaptive_avg_pool3d": None,
            "nn.functional.adaptive_max_pool1d": [
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
                torch.bool,
                torch.int8,
            ],
            "nn.functional.adaptive_max_pool2d": [
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
                torch.bool,
                torch.int8,
            ],
            "nn.functional.adaptive_max_pool3d": None,
            "nn.functional.interpolatearea": None,
            "nn.functional.interpolatebicubic": [torch.uint8],
            "nn.functional.local_response_norm": [
                torch.int8,
                torch.int16,
                torch.int32,
                torch.uint8,
                torch.bool,
            ],
            "nn.functional.max_pool1d": [
                torch.uint8,
                torch.bool,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            ],
            "nn.functional.max_pool2d": [torch.bool],
            "nn.functional.max_pool3d": [torch.bool],
            "nn.functional.max_unpool1d": [
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
                torch.bool,
                torch.int8,
            ],
            "nn.functional.max_unpool1dgrad": [
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
                torch.bool,
                torch.int8,
            ],
            "nn.functional.max_unpool2d": [
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
                torch.bool,
                torch.int8,
            ],
            "nn.functional.max_unpool2dgrad": [
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
                torch.bool,
                torch.int8,
            ],
            "nn.functional.max_unpool3d": [
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
                torch.bool,
                torch.int8,
            ],
            "nn.functional.max_unpool3dgrad": [
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
                torch.bool,
                torch.int8,
            ],
            "nn.functional.mish": [
                torch.int32,
                torch.uint8,
                torch.bool,
                torch.int8,
                torch.int16,
            ],
            "nn.functional.multi_margin_loss": None,
            "nn.functional.multilabel_margin_loss": [
                torch.int8,
                torch.uint8,
                torch.int32,
                torch.int16,
                torch.float32,
            ],
            "nn.functional.multilabel_soft_margin_loss": [
                torch.int8,
                torch.uint8,
                torch.int32,
                torch.int16,
            ],
            "nn.functional.nll_loss": [
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
                torch.bool,
                torch.int8,
            ],
            "nn.functional.padreplicate_negative": [torch.bool],
            "nn.functional.pdist": None,
            "nn.functional.rrelu": None,
            "nn.functional.silu": [
                torch.int16,
                torch.int32,
                torch.uint8,
                torch.int8,
            ],
            "nn.functional.softplus": [
                torch.int32,
                torch.uint8,
                torch.bool,
                torch.int8,
                torch.int16,
            ],
            "ormqr": None,
            "rounddecimals_0": [
                torch.uint8,
                torch.int8,
                torch.int64,
                torch.int32,
                torch.int16,
            ],
            # int64 lacks atomic_binary_op in Metal; the old MPSGraph path cast
            # to int32 (silently lossy). amin/amax for int64 go through the
            # sign-flip encode + ulong atomic_min/max bracket and work fine.
            # bool prod/mean are excluded via dtypesIfMPS in the OpInfo itself.
            "scatter_reduceprod": [torch.int64],
            "_segment_reducelengths": None,
            "_segment_reduceoffsets": None,
            "sparse.sampled_addmm": None,
            "sparse.mmreduce": None,
            "special.legendre_polynomial_p": None,
            "special.log_ndtr": None,
            "special.ndtri": None,
            "stft": [torch.float16, torch.bfloat16],
            "svd_lowrank": None,
            "to": None,
            "_upsample_bilinear2d_aa": [torch.uint8],  # uint8 is for CPU only
            "_upsample_bicubic2d_aa": [torch.uint8],  # uint8 is for CPU only
            "cdouble": None,
            "double": None,
            "log_softmaxwith_dtype": [
                torch.uint8,
                torch.int8,
                torch.int32,
                torch.int16,
                torch.int64,
                torch.float32,
            ],
            "float_power": None,
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
            # _mps_linear rejects non-float inputs; unlike mm/matmul it has no
            # integral Metal GEMM fallback.
            "nn.functional.linear": [
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
                torch.int8,
            ],
            # returned output on CPU is float64
            "bincount": [
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
                torch.int8,
            ],
        }
        UNIMPLEMENTED_XFAILLIST_SPARSE: dict[str, list | None] = {
            "logspace": None,
            "logspacetensor_overload": None,
            "linalg.eig": None,
            "linalg.eigvals": None,
            "put": None,
        }

        if sparse:
            UNIMPLEMENTED_XFAILLIST.update(UNIMPLEMENTED_XFAILLIST_SPARSE)

        UNDEFINED_XFAILLIST: dict[str, list | None] = {
            # Top 60 operators
            # PCA singular vectors are sign-ambiguous; the new Metal randn in
            # #182386 shifted the sequence so seeded sample inputs land on
            # different sign choices than CPU.
            "pca_lowrank": [torch.float32],
            # logcumsumexp on complex inputs disagrees with CPU at branch
            # cuts (off by 2*pi); shifted RNG exposed a sample on the cut.
            "logcumsumexp": [torch.complex64],
            # See https://github.com/pytorch/pytorch/issues/111479
            "nn.functional.multi_head_attention_forward": [
                torch.float32,
                torch.float16,
                torch.bfloat16,
            ],
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

        ON_MPS_XFAILLIST: dict[str, list | None] = {
            # Failure due to precision issue for fp16
            # on both cpu and mps there are test cases that might produce inf result
            # 'nn.functional.pairwise_distance': [torch.float16],
            # test below pass on macOS 12 as it falls back to cpu
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
            # MPS uses float32 intermediates (opmath_t) while CPU uses native
            # half/bfloat16 precision, causing unbounded divergence.
            # Half precision is covered by test_grid_sampler_3d_half_precision.
            "nn.functional.grid_sample": [torch.float16, torch.bfloat16],
        }

        def addDecorator(op: OpInfo, d: DecorateInfo) -> None:
            if device_type is not None:
                d.device_type = device_type

            op.decorators = op.decorators + (d,)

        for op in ops:
            key = op.name + op.variant_test_name
            addDecorator(
                op,
                DecorateInfo(
                    unittest.expectedFailure,
                    dtypes=[
                        torch.double,
                        torch.cdouble,
                    ],
                ),
            )
            if sparse:
                # Skipped due to test_sparse_zero_dims test in test_sparse.py which allocates empty tensor
                # which leads to unexpected success with it
                addDecorator(
                    op,
                    DecorateInfo(
                        unittest.skip(
                            "Skipped due to MPS not supporting complex128 tensors"
                        ),
                        dtypes=[
                            torch.complex128,
                        ],
                    ),
                )
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
                key in MACOS_BEFORE_15_0_XFAILLIST
                and key not in xfail_exclusion
                and (MACOS_VERSION < 15.0)
            ):
                addDecorator(
                    op,
                    DecorateInfo(
                        unittest.expectedFailure,
                        dtypes=MACOS_BEFORE_15_0_XFAILLIST[key],
                    ),
                )

            # If op is not supported for complex types, expect it to fail
            if key in UNSUPPORTED_COMPLEX_OPS:
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
            "sparse.mmreduce": [torch.float32],  # csr not supported
            "linalg.householder_product": None,
            "linalg.lstsq": [torch.float32],
            "linalg.lstsqgrad_oriented": [torch.float32],
            # Correctness issues
            # Same issue as `argsort` and `sort` with duplicate elements (undefined behaviour).
            # Forward pass is passing since `msort` doesn't return the indices, just the values, which match the CPU.
            # On the backward pass for `sort` both are used (values and indices), thus resulting in a mismatch between CPU and MPS.
            # Running `msort` with stable `sort` passes.
            "msort": [torch.float16],
            # Random ops are routed to `_assert_random_op_match` for the
            # forward leg of `test_output_grad_match`; the gradients (on the
            # `mean`/`std` parameters of `normal`, etc.) are deterministic
            # given the inputs and shouldn't need broad xfailing.
            #
            # Dropout family is the exception: backward reuses the forward's
            # random mask, and since MPS and CPU draw different masks the
            # backward gradients legitimately diverge. xfail the grad leg.
            "nn.functional.dropout": [torch.float16, torch.float32],
            "nn.functional.dropout2d": [torch.float16, torch.float32],
            "nn.functional.dropout3d": [torch.float16, torch.float32],
            "nn.functional.alpha_dropout": [torch.float16, torch.float32],
            "nn.functional.feature_alpha_dropoutwith_train": [
                torch.float16,
                torch.float32,
            ],
            # PCA singular vectors are sign-ambiguous - same root cause as the
            # forward leg above. RNG shift lands seeded samples on different
            # sign choices.
            "pca_lowrank": [torch.float32],
            # CPU errors
            # 'bool' object is not iterable
            "allclose": [torch.float16, torch.float32],
            "equal": [torch.float16, torch.float32],
            # 'float' object is not iterable
            "item": [torch.float16, torch.float32],
            # Could not run 'aten::uniform_' with arguments from the 'SparseCPU' backend
            "to_sparse": None,
        }

        SKIPLIST_GRAD = {
            # topk index gather is flaky on fp16 - whether duplicates appear
            # in the seeded sample input depends on prior test order's RNG
            # draws, so we skip rather than xfail.
            "topk": [torch.float16],
            "nn.functional.pairwise_distance": [torch.float16],
            # failed assertion `destination datatype must be fp32'
            "nn.functional.conv1d": [torch.float16],
            "nn.functional.conv2d": [torch.float16],
            "nn.functional.conv3d": [torch.float16],
            "nn.functional.conv_transpose1d": [torch.float16],
            "nn.functional.conv_transpose2d": [torch.float16],
            "nn.functional.conv_transpose3d": [torch.float16],
        }

        ON_MPS_XFAILLIST = {
            # Exception: Caused by sample input at index 3 on MPS
            "nn.functional.conv3d": [torch.float32],
        }

        MACOS_BEFORE_15_0_XFAILLIST_GRAD = {
            # matrix_exp is disabled on MPS before macOS 15 (TORCH_CHECK), so the
            # forward leg of the grad test raises for every dtype.
            "matrix_exp": None,
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

            if key in MACOS_BEFORE_15_0_XFAILLIST_GRAD and MACOS_VERSION < 15.0:
                addDecorator(
                    op,
                    DecorateInfo(
                        unittest.expectedFailure,
                        dtypes=MACOS_BEFORE_15_0_XFAILLIST_GRAD[key],
                    ),
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

        return ops

    def mps_ops_error_inputs_modifier(ops: Sequence[OpInfo]) -> Sequence[OpInfo]:
        # Error input samples do not take a dtype argument.
        XFAILLIST = {
            # Exceptions are not raised
            "__rmod__",
            "__rsub__",
            "__rpow__",
            "clamp_max",
            "clamp_min",
            "masked_scatter",
            # MPS does not support tensor dimensions > 16
            "amax",
            "amin",
            "aminmax",
        }

        def addDecorator(op: OpInfo, d: DecorateInfo) -> None:
            op.decorators = op.decorators + (d,)

        for op in ops:
            key = op.name + op.variant_test_name
            if key in XFAILLIST:
                addDecorator(op, DecorateInfo(unittest.expectedFailure))

        return ops
else:

    def mps_ops_modifier(
        ops: Sequence[OpInfo],
        device_type: str = "mps",
        xfail_exclusion: list[str] | None = None,
        sparse: bool = False,
    ) -> Sequence[OpInfo]:
        return ops

    def mps_ops_grad_modifier(ops: Sequence[OpInfo]) -> Sequence[OpInfo]:
        return ops

    def mps_ops_error_inputs_modifier(ops: Sequence[OpInfo]) -> Sequence[OpInfo]:
        return ops
