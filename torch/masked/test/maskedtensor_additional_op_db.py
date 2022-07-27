# Copyright (c) Meta Platforms, Inc. and affiliates

import itertools
import operator
import unittest
from functools import partial

import numpy as np
import scipy
import torch
from common_utils import _create_random_mask
from torch.testing import (
    all_types_and,
    all_types_and_complex_and,
    floating_and_complex_types_and,
    floating_types,
    floating_types_and,
    integral_types,
    integral_types_and,
)
from torch.testing._internal.common_device_type import (
    precisionOverride,
    tol,
    toleranceOverride,
)
from torch.testing._internal.common_methods_invocations import (
    _NOTHING,
    BinaryUfuncInfo,
    DecorateInfo,
    error_inputs_neg,
    np_sinc_with_fp16_as_fp32,
    np_unary_ufunc_integer_promotion_wrapper,
    OpInfo,
    reference_lgamma,
    reference_sgn,
    reference_sigmoid,
    reference_sign,
    sample_inputs_add_sub,
    sample_inputs_comparison_ops,
    sample_inputs_elementwise_binary,
    sample_inputs_i0_i1,
    sample_inputs_logit,
    SampleInput,
    UnaryUfuncInfo,
)
from torch.testing._internal.common_utils import make_tensor, TEST_SCIPY


# Reasonable testing sizes for dimensions
L = 20
M = 10
S = 5


def sample_inputs_unary(
    op_info, device, dtype, requires_grad, op_kwargs=None, **kwargs
):
    if not op_kwargs:
        op_kwargs = {}

    low, high = op_info.domain
    low = low if low is None else low + op_info._domain_eps
    high = high if high is None else high - op_info._domain_eps

    if op_info.supports_sparse_csr:
        # Tensors with dim=2 for sparse CSR testing
        yield SampleInput(
            make_tensor(
                (L, L),
                device=device,
                dtype=dtype,
                low=low,
                high=high,
                requires_grad=requires_grad,
            ),
            kwargs=op_kwargs,
        )
    else:
        # Creates a 1D, empty, and scalar tensor
        for shape in ((L,), (1, 0, 3), ()):
            yield SampleInput(
                make_tensor(
                    shape,
                    device=device,
                    dtype=dtype,
                    low=low,
                    high=high,
                    requires_grad=requires_grad,
                ),
                kwargs=op_kwargs,
            )


def sample_inputs_clamp_scalar(op_info, device, dtype, requires_grad, **kwargs):
    shapes = [(2, 3, 2), (2, 0, 3)]

    if dtype is torch.uint8:
        min_max_vals = ((2, 5), (3, 7))
    else:
        min_max_vals = ((0, 1), (-1, 1))

    output = []
    for shape, vals in itertools.product(shapes, min_max_vals):
        tensor = make_tensor(
            shape,
            dtype=dtype,
            device=device,
            low=None,
            high=None,
            requires_grad=requires_grad,
        )
        min_val, max_val = vals
        mask = _create_random_mask(shape, device)
        output.append(
            SampleInput(
                tensor.clone().requires_grad_(requires_grad),
                args=vals,
                kwargs={"mask": mask},
            )
        )

    return output


additional_op_db = []

additional_op_db.extend(
    [
        UnaryUfuncInfo(
            "abs",
            aliases=("absolute",),
            ref=np.abs,
            dtypes=all_types_and_complex_and(torch.half, torch.bfloat16),
            dtypesIfCUDA=floating_and_complex_types_and(torch.half, torch.bfloat16),
            sample_inputs_func=sample_inputs_unary,
            supports_inplace_autograd=False,
            supports_fwgrad_bwgrad=True,
            assert_autodiffed=True,
            supports_sparse_csr=True,
            supports_forward_ad=True,
        ),
        UnaryUfuncInfo(
            "acos",
            aliases=("arccos",),
            ref=np.arccos,
            domain=(-1, 1),
            dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
            dtypesIfCUDA=all_types_and_complex_and(
                torch.bool, torch.half, torch.bfloat16
            ),
            sample_inputs_func=sample_inputs_unary,
            assert_autodiffed=True,
            supports_forward_ad=True,
            supports_fwgrad_bwgrad=True,
            decorators=(
                precisionOverride(
                    {torch.float16: 1e-2, torch.bfloat16: 1e-1, torch.complex64: 1e-2}
                ),
            ),
        ),
        UnaryUfuncInfo(
            "acosh",
            aliases=("arccosh",),
            ref=np.arccosh,
            domain=(1, None),
            dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
            dtypesIfCUDA=all_types_and_complex_and(
                torch.bool, torch.half, torch.bfloat16
            ),
            sample_inputs_func=sample_inputs_unary,
            # "rsqrt_cuda" not implemented for 'BFloat16'
            backward_dtypesIfCUDA=all_types_and_complex_and(
                torch.bool, torch.half, torch.bfloat16
            ),
            decorators=(precisionOverride({torch.bfloat16: 5e-2}),),
            supports_inplace_autograd=False,
            supports_forward_ad=True,
            supports_fwgrad_bwgrad=True,
        ),
        UnaryUfuncInfo(
            "angle",
            ref=np.angle,
            dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
            dtypesIfCUDA=all_types_and_complex_and(torch.bool),
            sample_inputs_func=sample_inputs_unary,
            decorators=(
                precisionOverride({torch.float16: 1e-2, torch.bfloat16: 1e-2}),
            ),
            supports_forward_ad=True,
            supports_fwgrad_bwgrad=True,
            supports_sparse_csr=True,
            supports_complex_to_float=True,
        ),
        UnaryUfuncInfo(
            "asin",
            aliases=("arcsin",),
            ref=np.arcsin,
            domain=(-1, 1),
            supports_sparse=True,
            supports_sparse_csr=True,
            supports_forward_ad=True,
            supports_fwgrad_bwgrad=True,
            dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
            dtypesIfCUDA=all_types_and_complex_and(
                torch.bool, torch.half, torch.bfloat16
            ),
            sample_inputs_func=sample_inputs_unary,
            assert_autodiffed=True,
            decorators=[
                DecorateInfo(
                    toleranceOverride({torch.float16: tol(atol=1e-05, rtol=1e-03)}),
                    "TestUnaryUfuncs",
                    device_type="cuda",
                ),
                precisionOverride({torch.bfloat16: 1e-2}),
            ],
        ),
        # NOTE: derivative for inplace asinh is not implemented
        UnaryUfuncInfo(
            "asinh",
            aliases=("arcsinh",),
            ref=np.arcsinh,
            dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
            dtypesIfCUDA=all_types_and_complex_and(
                torch.bool, torch.half, torch.bfloat16
            ),
            sample_inputs_func=sample_inputs_unary,
            decorators=(precisionOverride({torch.bfloat16: 5e-2}),),
            supports_inplace_autograd=False,
            supports_forward_ad=True,
            supports_fwgrad_bwgrad=True,
            supports_sparse=True,
            supports_sparse_csr=True,
        ),
        UnaryUfuncInfo(
            "atan",
            aliases=("arctan",),
            ref=np.arctan,
            dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
            dtypesIfCUDA=all_types_and_complex_and(
                torch.bool, torch.half, torch.bfloat16
            ),
            sample_inputs_func=sample_inputs_unary,
            assert_autodiffed=True,
            supports_forward_ad=True,
            supports_fwgrad_bwgrad=True,
            supports_sparse=True,
            supports_sparse_csr=True,
            decorators=(precisionOverride({torch.bfloat16: 1e-2}),),
        ),
        UnaryUfuncInfo(
            "atanh",
            aliases=("arctanh",),
            ref=np.arctanh,
            domain=(-1, 1),
            dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
            dtypesIfCUDA=all_types_and_complex_and(
                torch.bool, torch.half, torch.bfloat16
            ),
            sample_inputs_func=sample_inputs_unary,
            decorators=(precisionOverride({torch.bfloat16: 1e-2}),),
            supports_inplace_autograd=False,
            supports_forward_ad=True,
            supports_fwgrad_bwgrad=True,
            supports_sparse=True,
            supports_sparse_csr=True,
        ),
        UnaryUfuncInfo(
            "bitwise_not",
            ref=np.bitwise_not,
            dtypes=integral_types_and(torch.bool),
            supports_autograd=False,
        ),
        UnaryUfuncInfo(
            "ceil",
            ref=np.ceil,
            dtypes=floating_types_and(torch.bfloat16),
            dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
            sample_inputs_func=sample_inputs_unary,
            supports_forward_ad=True,
            supports_fwgrad_bwgrad=True,
            supports_sparse=True,
            supports_sparse_csr=True,
            assert_autodiffed=True,
        ),
        # NOTE: clamp has separate opinfos for scalar min/max (unary op) vs. tensors
        # Our test assumes that clamp is the unary function
        # OpInfo('clamp',
        #        aliases=('clip',),
        #        dtypes=all_types_and(torch.bfloat16),
        #        dtypesIfCUDA=all_types_and(torch.half, torch.bfloat16),
        #        sample_inputs_func=sample_inputs_clamp,
        #        assert_autodiffed=True,
        #        supports_forward_ad=True,
        #        supports_fwgrad_bwgrad=True),
        UnaryUfuncInfo(
            "clamp",
            variant_test_name="scalar",
            aliases=("clip",),
            decorators=(
                precisionOverride({torch.bfloat16: 7e-2, torch.float16: 1e-2}),
            ),
            ref=np.clip,
            dtypes=all_types_and(torch.bfloat16),
            dtypesIfCUDA=all_types_and(torch.half, torch.bfloat16),
            sample_inputs_func=sample_inputs_clamp_scalar,
            assert_autodiffed=True,
            supports_forward_ad=True,
            supports_fwgrad_bwgrad=True,
        ),
        UnaryUfuncInfo(
            "conj_physical",
            ref=np.conj,
            dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half),
            sample_inputs_func=sample_inputs_unary,
            supports_forward_ad=True,
            supports_fwgrad_bwgrad=True,
            supports_sparse=True,
            supports_sparse_csr=True,
        ),
        UnaryUfuncInfo(
            "cos",
            ref=np.cos,
            dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
            dtypesIfCUDA=all_types_and_complex_and(
                torch.bool, torch.half, torch.bfloat16
            ),
            sample_inputs_func=sample_inputs_unary,
            assert_autodiffed=True,
            handles_large_floats=False,
            supports_forward_ad=True,
            supports_fwgrad_bwgrad=True,
            decorators=(precisionOverride({torch.bfloat16: 1e-2}),),
        ),
        UnaryUfuncInfo(
            "cosh",
            ref=np_unary_ufunc_integer_promotion_wrapper(np.cosh),
            dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
            dtypesIfCUDA=all_types_and_complex_and(
                torch.bool, torch.half, torch.bfloat16
            ),
            sample_inputs_func=sample_inputs_unary,
            assert_autodiffed=True,
            supports_forward_ad=True,
            supports_fwgrad_bwgrad=True,
        ),
        UnaryUfuncInfo(
            "digamma",
            ref=scipy.special.digamma if TEST_SCIPY else _NOTHING,
            aliases=(
                "special.psi",
                "special.digamma",
            ),
            decorators=(precisionOverride({torch.float16: 5e-1}),),
            dtypes=all_types_and(torch.bool, torch.bfloat16),
            dtypesIfCUDA=all_types_and(torch.bool, torch.half),
            sample_inputs_func=sample_inputs_unary,
            supports_forward_ad=True,
            supports_fwgrad_bwgrad=True,
        ),
        UnaryUfuncInfo(
            "erf",
            ref=scipy.special.erf if TEST_SCIPY else _NOTHING,
            aliases=("special.erf",),
            decorators=(
                precisionOverride({torch.float16: 1e-2, torch.bfloat16: 1e-2}),
            ),
            dtypes=all_types_and(torch.bool, torch.bfloat16),
            dtypesIfCUDA=all_types_and(torch.bool, torch.half, torch.bfloat16),
            sample_inputs_func=sample_inputs_unary,
            assert_autodiffed=True,
            assert_jit_shape_analysis=True,
            supports_sparse=True,
            supports_sparse_csr=True,
            supports_forward_ad=True,
            supports_fwgrad_bwgrad=True,
        ),
        UnaryUfuncInfo(
            "erfc",
            ref=scipy.special.erfc if TEST_SCIPY else _NOTHING,
            aliases=("special.erfc",),
            decorators=(
                precisionOverride({torch.float16: 1e-2, torch.bfloat16: 1e-2}),
            ),
            dtypes=all_types_and(torch.bool, torch.bfloat16),
            dtypesIfCUDA=all_types_and(torch.bool, torch.half, torch.bfloat16),
            sample_inputs_func=sample_inputs_unary,
            assert_autodiffed=True,
            supports_forward_ad=True,
            supports_fwgrad_bwgrad=True,
        ),
        UnaryUfuncInfo(
            "erfinv",
            ref=scipy.special.erfinv if TEST_SCIPY else _NOTHING,
            aliases=("special.erfinv",),
            decorators=(
                precisionOverride(
                    {torch.float16: 1e-2, torch.bfloat16: 1e-2, torch.float32: 1e-4}
                ),
            ),
            dtypes=all_types_and(torch.bool, torch.bfloat16),
            dtypesIfCUDA=all_types_and(torch.bool, torch.half),
            sample_inputs_func=sample_inputs_unary,
            supports_sparse_csr=True,
            supports_forward_ad=True,
            supports_fwgrad_bwgrad=True,
            domain=(-1, 1),
        ),
        UnaryUfuncInfo(
            "exp",
            ref=np_unary_ufunc_integer_promotion_wrapper(np.exp),
            dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
            dtypesIfCUDA=all_types_and_complex_and(
                torch.bool, torch.half, torch.bfloat16
            ),
            sample_inputs_func=sample_inputs_unary,
            assert_autodiffed=True,
            supports_forward_ad=True,
            supports_fwgrad_bwgrad=True,
        ),
        UnaryUfuncInfo(
            "exp2",
            aliases=("special.exp2",),
            ref=np_unary_ufunc_integer_promotion_wrapper(np.exp2),
            dtypes=all_types_and(torch.bool, torch.half, torch.bfloat16),
            dtypesIfCUDA=all_types_and(torch.bool, torch.half, torch.bfloat16),
            sample_inputs_func=sample_inputs_unary,
            supports_forward_ad=True,
            supports_fwgrad_bwgrad=True,
        ),
        UnaryUfuncInfo(
            "expm1",
            aliases=("special.expm1",),
            ref=np_unary_ufunc_integer_promotion_wrapper(np.expm1),
            dtypes=all_types_and(torch.bool, torch.bfloat16),
            dtypesIfCUDA=all_types_and(torch.bool, torch.half, torch.bfloat16),
            sample_inputs_func=sample_inputs_unary,
            supports_forward_ad=True,
            supports_fwgrad_bwgrad=True,
            supports_sparse=True,
            supports_sparse_csr=True,
            assert_autodiffed=True,
        ),
        UnaryUfuncInfo(
            "trunc",
            aliases=("fix",),
            ref=np.trunc,
            dtypes=floating_types_and(torch.bfloat16),
            dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16),
            sample_inputs_func=sample_inputs_unary,
            supports_forward_ad=True,
            supports_fwgrad_bwgrad=True,
            supports_sparse=True,
            supports_sparse_csr=True,
            assert_autodiffed=True,
        ),
        UnaryUfuncInfo(
            "floor",
            ref=np.floor,
            dtypes=floating_types_and(torch.bfloat16),
            dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
            sample_inputs_func=sample_inputs_unary,
            supports_forward_ad=True,
            supports_fwgrad_bwgrad=True,
            supports_sparse=True,
            supports_sparse_csr=True,
            assert_autodiffed=True,
        ),
        UnaryUfuncInfo(
            "frac",
            ref=lambda x: np.modf(x)[0],
            dtypes=floating_types_and(torch.bfloat16, torch.float16),
            dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16),
            sample_inputs_func=sample_inputs_unary,
            assert_autodiffed=True,
            supports_forward_ad=True,
            supports_fwgrad_bwgrad=True,
        ),
        UnaryUfuncInfo(
            "lgamma",
            ref=reference_lgamma if TEST_SCIPY else _NOTHING,
            aliases=("special.gammaln",),
            decorators=(precisionOverride({torch.float16: 7e-1}),),
            dtypes=all_types_and(torch.bool, torch.bfloat16),
            dtypesIfCUDA=all_types_and(torch.bool, torch.half),
            sample_inputs_func=sample_inputs_unary,
            supports_forward_ad=True,
            supports_fwgrad_bwgrad=True,
        ),
        UnaryUfuncInfo(
            "log",
            ref=np.log,
            domain=(0, None),
            dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
            dtypesIfCUDA=all_types_and_complex_and(
                torch.bool, torch.half, torch.bfloat16
            ),
            sample_inputs_func=sample_inputs_unary,
            assert_autodiffed=True,
            supports_forward_ad=True,
            supports_fwgrad_bwgrad=True,
            decorators=(precisionOverride({torch.bfloat16: 5e-2}),),
        ),
        UnaryUfuncInfo(
            "log10",
            ref=np.log10,
            domain=(0, None),
            decorators=(precisionOverride({torch.bfloat16: 5e-2}),),
            dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
            assert_autodiffed=True,
            dtypesIfCUDA=all_types_and_complex_and(
                torch.bool, torch.half, torch.bfloat16
            ),
            sample_inputs_func=sample_inputs_unary,
            supports_forward_ad=True,
            supports_fwgrad_bwgrad=True,
        ),
        UnaryUfuncInfo(
            "log1p",
            ref=np.log1p,
            aliases=("special.log1p",),
            domain=(-1, None),
            dtypes=all_types_and(torch.bool, torch.bfloat16),
            dtypesIfCUDA=all_types_and(torch.bool, torch.half, torch.bfloat16),
            sample_inputs_func=sample_inputs_unary,
            decorators=(precisionOverride({torch.bfloat16: 1e-1}),),
            supports_forward_ad=True,
            supports_fwgrad_bwgrad=True,
            supports_sparse=True,
            supports_sparse_csr=True,
            assert_autodiffed=True,
        ),
        UnaryUfuncInfo(
            "log2",
            ref=np.log2,
            domain=(0, None),
            dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
            dtypesIfCUDA=all_types_and_complex_and(
                torch.bool, torch.half, torch.bfloat16
            ),
            sample_inputs_func=sample_inputs_unary,
            assert_autodiffed=True,
            supports_forward_ad=True,
            supports_fwgrad_bwgrad=True,
            decorators=(precisionOverride({torch.bfloat16: 1e-1}),),
        ),
        UnaryUfuncInfo(
            "logit",
            ref=scipy.special.logit if TEST_SCIPY else _NOTHING,
            domain=(0, 1),
            aliases=("special.logit",),
            supports_forward_ad=True,
            supports_fwgrad_bwgrad=True,
            decorators=(
                precisionOverride({torch.bfloat16: 5e-1, torch.float16: 5e-1}),
            ),
            dtypes=all_types_and(torch.bool, torch.bfloat16),
            dtypesIfCUDA=all_types_and(torch.bool, torch.half, torch.bfloat16),
            sample_inputs_func=sample_inputs_logit,
        ),
        UnaryUfuncInfo(
            "i0",
            ref=np_unary_ufunc_integer_promotion_wrapper(scipy.special.i0)
            if TEST_SCIPY
            else _NOTHING,
            aliases=("special.i0",),
            decorators=(
                precisionOverride({torch.bfloat16: 3e-1, torch.float16: 5e-1}),
            ),
            backward_dtypes=floating_types(),
            backward_dtypesIfCUDA=floating_types(),
            dtypesIfROCM=floating_types(),
            dtypes=all_types_and(torch.bool, torch.bfloat16),
            dtypesIfCUDA=all_types_and(torch.bool, torch.half, torch.bfloat16),
            sample_inputs_func=sample_inputs_i0_i1,
            supports_forward_ad=True,
            supports_fwgrad_bwgrad=True,
        ),
        UnaryUfuncInfo(
            "isnan",
            ref=np.isnan,
            dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
            sample_inputs_func=sample_inputs_unary,
            supports_out=False,
            supports_sparse=True,
            supports_sparse_csr=True,
            supports_autograd=False,
        ),
        UnaryUfuncInfo(
            "nan_to_num",
            ref=np.nan_to_num,
            dtypes=all_types_and(torch.half, torch.bool, torch.bfloat16),
            dtypesIfCUDA=all_types_and(torch.half, torch.bool, torch.bfloat16),
            sample_inputs_func=sample_inputs_unary,
            supports_forward_ad=True,
            supports_fwgrad_bwgrad=True,
            supports_sparse=True,
            # Passing numpy_kwargs via sample_kwargs, as numpy does comparison
            # with BFloat16 in float, since it currently doesn't support BFloat16.
            # Ref: https://github.com/pytorch/pytorch/issues/57982#issuecomment-839150556
            sample_kwargs=lambda device, dtype, input: (
                {},
                {
                    "posinf": torch.finfo(torch.bfloat16).max,
                    "neginf": torch.finfo(torch.bfloat16).min,
                },
            )
            if dtype is torch.bfloat16
            else ({}, {}),
        ),
        UnaryUfuncInfo(
            "neg",
            aliases=("negative",),
            ref=np.negative,
            dtypes=all_types_and_complex_and(torch.half, torch.bfloat16),
            sample_inputs_func=sample_inputs_unary,
            error_inputs_func=error_inputs_neg,
            supports_forward_ad=True,
            supports_fwgrad_bwgrad=True,
            supports_sparse=True,
            supports_sparse_csr=True,
            assert_autodiffed=True,
        ),
        UnaryUfuncInfo(
            "positive",
            ref=np.positive,
            dtypes=all_types_and_complex_and(torch.half, torch.bfloat16),
            sample_inputs_func=sample_inputs_unary,
            supports_out=False,
            supports_forward_ad=True,
            supports_fwgrad_bwgrad=True,
        ),
        UnaryUfuncInfo(
            "rad2deg",
            ref=np.degrees,
            decorators=(
                precisionOverride({torch.bfloat16: 7e-1, torch.float16: 7e-1}),
            ),
            dtypes=all_types_and(torch.bool, torch.half, torch.bfloat16),
            sample_inputs_func=sample_inputs_unary,
            supports_forward_ad=True,
            supports_fwgrad_bwgrad=True,
        ),
        UnaryUfuncInfo(
            "reciprocal",
            ref=np_unary_ufunc_integer_promotion_wrapper(np.reciprocal),
            dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
            sample_inputs_func=sample_inputs_unary,
            assert_autodiffed=True,
            supports_forward_ad=True,
            supports_fwgrad_bwgrad=True,
        ),
        # To test reference numerics against multiple values of argument `decimals`,
        # we make multiple OpInfo entries with each entry corresponding to different value of decimals.
        UnaryUfuncInfo(
            "round",
            ref=np.round,
            aliases=("special.round",),
            dtypes=floating_types_and(torch.bfloat16),
            dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
            sample_inputs_func=sample_inputs_unary,
            supports_forward_ad=True,
            supports_fwgrad_bwgrad=True,
            supports_sparse=True,
            supports_sparse_csr=True,
            assert_autodiffed=True,
        ),
        UnaryUfuncInfo(
            "rsqrt",
            ref=lambda x: np.reciprocal(np.sqrt(x)),
            domain=(0, None),
            dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
            dtypesIfCUDA=all_types_and_complex_and(
                torch.bool, torch.half, torch.bfloat16
            ),
            sample_inputs_func=sample_inputs_unary,
            decorators=(precisionOverride({torch.half: 5e-2}),),
            assert_autodiffed=True,
            supports_forward_ad=True,
            supports_fwgrad_bwgrad=True,
        ),
        UnaryUfuncInfo(
            "sigmoid",
            aliases=("special.expit", "nn.functional.sigmoid"),
            ref=reference_sigmoid if TEST_SCIPY else _NOTHING,
            decorators=(
                precisionOverride(
                    {torch.float16: 1e-2, torch.complex64: 1e-1, torch.bfloat16: 1e-2}
                ),
            ),
            dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
            dtypesIfCUDA=all_types_and_complex_and(
                torch.bool, torch.half, torch.bfloat16
            ),
            sample_inputs_func=sample_inputs_unary,
            supports_forward_ad=True,
            supports_fwgrad_bwgrad=True,
            assert_autodiffed=True,
        ),
        UnaryUfuncInfo(
            "sign",
            ref=reference_sign,
            dtypes=all_types_and(torch.bool, torch.bfloat16, torch.half),
            dtypesIfCUDA=all_types_and(torch.bool, torch.bfloat16, torch.half),
            sample_inputs_func=sample_inputs_unary,
            supports_forward_ad=True,
            supports_fwgrad_bwgrad=True,
            supports_sparse=True,
            supports_sparse_csr=True,
        ),
        UnaryUfuncInfo(
            "sgn",
            ref=reference_sgn,
            dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half),
            sample_inputs_func=sample_inputs_unary,
            supports_forward_ad=True,
            supports_fwgrad_bwgrad=True,
            supports_sparse=True,
            supports_sparse_csr=True,
        ),
        UnaryUfuncInfo(
            "signbit",
            ref=np.signbit,
            dtypes=all_types_and(torch.bool, torch.bfloat16, torch.half),
            sample_inputs_func=sample_inputs_unary,
            supports_sparse=True,
            supports_sparse_csr=True,
            supports_autograd=False,
        ),
        UnaryUfuncInfo(
            "sin",
            ref=np.sin,
            dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
            dtypesIfCUDA=all_types_and_complex_and(
                torch.bool, torch.half, torch.bfloat16
            ),
            sample_inputs_func=sample_inputs_unary,
            assert_autodiffed=True,
            handles_large_floats=False,
            supports_sparse=True,
            supports_sparse_csr=True,
            supports_forward_ad=True,
            supports_fwgrad_bwgrad=True,
            skips=(
                DecorateInfo(
                    unittest.skip("Skipped! sparse backward not supported"),
                    "TestSparseUnaryUfuncs",
                    "test_sparse_fn_grad",
                ),
            ),
            decorators=(precisionOverride({torch.bfloat16: 1e-2}),),
        ),
        UnaryUfuncInfo(
            "sinc",
            ref=np_sinc_with_fp16_as_fp32,
            aliases=("special.sinc",),
            dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
            dtypesIfCUDA=all_types_and_complex_and(
                torch.bool, torch.half, torch.bfloat16
            ),
            sample_inputs_func=sample_inputs_unary,
            handles_large_floats=False,
            supports_forward_ad=True,
            supports_fwgrad_bwgrad=True,
            decorators=(
                precisionOverride({torch.bfloat16: 1e-2, torch.float16: 1e-2}),
            ),
        ),
        UnaryUfuncInfo(
            "sinh",
            ref=np_unary_ufunc_integer_promotion_wrapper(np.sinh),
            dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
            dtypesIfCUDA=all_types_and_complex_and(
                torch.bool, torch.half, torch.bfloat16
            ),
            sample_inputs_func=sample_inputs_unary,
            assert_autodiffed=True,
            supports_forward_ad=True,
            supports_fwgrad_bwgrad=True,
            supports_sparse=True,
            supports_sparse_csr=True,
            decorators=(precisionOverride({torch.float16: 1e-2}),),
        ),
        UnaryUfuncInfo(
            "sqrt",
            ref=np.sqrt,
            supports_sparse=True,
            domain=(0, None),
            dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
            dtypesIfCUDA=all_types_and_complex_and(
                torch.bool, torch.half, torch.bfloat16
            ),
            sample_inputs_func=sample_inputs_unary,
            assert_autodiffed=True,
            supports_forward_ad=True,
            supports_sparse_csr=True,
            supports_fwgrad_bwgrad=True,
            decorators=(precisionOverride({torch.bfloat16: 7e-2}),),
        ),
        UnaryUfuncInfo(
            "square",
            ref=np.square,
            dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
            sample_inputs_func=sample_inputs_unary,
            decorators=(
                precisionOverride({torch.complex64: 3e-4, torch.bfloat16: 3e-1}),
            ),
            supports_forward_ad=True,
            supports_fwgrad_bwgrad=True,
        ),
        UnaryUfuncInfo(
            "tan",
            ref=np.tan,
            dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
            dtypesIfCUDA=all_types_and_complex_and(
                torch.bool, torch.half, torch.bfloat16
            ),
            sample_inputs_func=sample_inputs_unary,
            assert_autodiffed=True,
            supports_forward_ad=True,
            supports_fwgrad_bwgrad=True,
            supports_sparse=True,
            supports_sparse_csr=True,
        ),
        UnaryUfuncInfo(
            "tanh",
            ref=np.tanh,
            aliases=("nn.functional.tanh",),
            decorators=(precisionOverride({torch.bfloat16: 1e-2}),),
            dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
            dtypesIfCUDA=all_types_and_complex_and(
                torch.bool, torch.half, torch.bfloat16
            ),
            sample_inputs_func=sample_inputs_unary,
            # "tanh_backward_cpu" not implemented for 'BFloat16'
            backward_dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
            assert_autodiffed=True,
            assert_jit_shape_analysis=True,
            supports_forward_ad=True,
            supports_fwgrad_bwgrad=True,
            supports_sparse=True,
            supports_sparse_csr=True,
        ),
    ]
)


additional_op_db.extend(
    [
        BinaryUfuncInfo(
            "add",
            # NumPy has no builtin reference for the alpha kwarg, but it is easy enough to emulate
            ref=lambda input, other, *, alpha=1: np.add(input, other)
            if alpha == 1
            else np.add(input, np.multiply(alpha, other)),
            dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
            assert_autodiffed=True,
            sample_inputs_func=sample_inputs_add_sub,
            supports_inplace_autograd=False,
            supports_fwgrad_bwgrad=True,
            supports_forward_ad=True,
            supports_two_python_scalars=True,
        ),
        BinaryUfuncInfo(
            "atan2",
            aliases=("arctan2",),
            dtypes=all_types_and(torch.bool),
            dtypesIfCUDA=all_types_and(torch.bool, torch.half, torch.bfloat16),
            supports_forward_ad=True,
            supports_fwgrad_bwgrad=True,
            promotes_int_to_float=True,
            supports_rhs_python_scalar=False,
        ),
        BinaryUfuncInfo(
            "bitwise_and",
            dtypes=integral_types_and(torch.bool),
            supports_autograd=False,
        ),
        BinaryUfuncInfo(
            "bitwise_or",
            ref=np.bitwise_or,
            dtypes=integral_types_and(torch.bool),
            supports_autograd=False,
        ),
        BinaryUfuncInfo(
            "bitwise_xor",
            ref=np.bitwise_xor,
            dtypes=integral_types_and(torch.bool),
            supports_autograd=False,
        ),
        BinaryUfuncInfo(
            "bitwise_left_shift",
            op=torch.bitwise_left_shift,
            dtypes=integral_types(),
            dtypesIfCUDA=integral_types(),
            operator_variant=operator.lshift,
            inplace_operator_variant=operator.ilshift,
            supports_autograd=False,
            supports_one_python_scalar=True,
            rhs_make_tensor_kwargs=dict(low=0),
            skips=(
                DecorateInfo(
                    unittest.skip("Skipped!"), "TestBinaryUfuncs", "test_type_promotion"
                ),
            ),
        ),
        BinaryUfuncInfo(
            "bitwise_right_shift",
            op=torch.bitwise_right_shift,
            dtypes=integral_types(),
            dtypesIfCUDA=integral_types(),
            operator_variant=operator.rshift,
            inplace_operator_variant=operator.irshift,
            supports_autograd=False,
            supports_one_python_scalar=True,
            rhs_make_tensor_kwargs=dict(low=0),
            skips=(
                DecorateInfo(
                    unittest.skip("Skipped!"), "TestBinaryUfuncs", "test_type_promotion"
                ),
            ),
        ),
        BinaryUfuncInfo(
            "div",
            aliases=("divide",),
            variant_test_name="no_rounding_mode",
            dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
            supports_forward_ad=True,
            promotes_int_to_float=True,
            supports_fwgrad_bwgrad=True,
            supports_two_python_scalars=True,
            assert_autodiffed=True,
            rhs_make_tensor_kwargs={"exclude_zero": True},
        ),
        BinaryUfuncInfo(
            "div",
            aliases=("divide",),
            variant_test_name="trunc_rounding",
            dtypes=all_types_and(torch.half, torch.bfloat16),
            sample_inputs_func=partial(
                sample_inputs_elementwise_binary,
                sample_kwargs={"rounding_mode": "trunc"},
            ),
            supports_forward_ad=True,
            promotes_int_to_float=True,
            supports_fwgrad_bwgrad=True,
            supports_two_python_scalars=True,
            assert_autodiffed=True,
            rhs_make_tensor_kwargs={"exclude_zero": True},
        ),
        BinaryUfuncInfo(
            "div",
            aliases=("divide",),
            variant_test_name="floor_rounding",
            dtypes=all_types_and(torch.half, torch.bfloat16),
            sample_inputs_func=partial(
                sample_inputs_elementwise_binary,
                sample_kwargs={"rounding_mode": "floor"},
            ),
            supports_forward_ad=True,
            promotes_int_to_float=True,
            supports_fwgrad_bwgrad=True,
            supports_two_python_scalars=True,
            assert_autodiffed=True,
            rhs_make_tensor_kwargs={"exclude_zero": True},
        ),
        BinaryUfuncInfo(
            "floor_divide",
            dtypes=all_types_and(torch.half, torch.bfloat16),
            supports_autograd=False,
            rhs_make_tensor_kwargs={"exclude_zero": True},
            supports_two_python_scalars=True,
        ),
        BinaryUfuncInfo(
            "fmod",
            ref=np.fmod,
            dtypes=all_types_and(torch.float16),
            supports_forward_ad=True,
            supports_fwgrad_bwgrad=True,
            assert_autodiffed=None,
            rhs_make_tensor_kwargs={"exclude_zero": True},
        ),
        OpInfo(
            "logaddexp",
            dtypes=floating_types_and(torch.bfloat16),
            dtypesIfCUDA=floating_types_and(torch.bfloat16),
            dtypesIfROCM=floating_types_and(torch.bfloat16),
            supports_forward_ad=True,
            supports_fwgrad_bwgrad=True,
            sample_inputs_func=lambda op_info, device, dtype, requires_grad=False, **kwargs: (
                SampleInput(
                    make_tensor(
                        (S, S), dtype=dtype, device=device, requires_grad=requires_grad
                    ),
                    args=(
                        make_tensor(
                            (S, S),
                            dtype=dtype,
                            device=device,
                            requires_grad=requires_grad,
                        ),
                    ),
                ),
            ),
        ),
        OpInfo(
            "logaddexp2",
            dtypes=floating_types_and(torch.bfloat16),
            dtypesIfCUDA=floating_types_and(torch.bfloat16),
            dtypesIfROCM=floating_types_and(torch.bfloat16),
            supports_forward_ad=True,
            supports_fwgrad_bwgrad=True,
            sample_inputs_func=lambda op_info, device, dtype, requires_grad=False, **kwargs: (
                SampleInput(
                    make_tensor(
                        (S, S), dtype=dtype, device=device, requires_grad=requires_grad
                    ),
                    args=(
                        make_tensor(
                            (S, S),
                            dtype=dtype,
                            device=device,
                            requires_grad=requires_grad,
                        ),
                    ),
                ),
            ),
        ),
        BinaryUfuncInfo(
            "mul",
            aliases=("multiply",),
            dtypes=all_types_and_complex_and(torch.float16, torch.bfloat16, torch.bool),
            assert_autodiffed=True,
            supports_forward_ad=True,
            supports_fwgrad_bwgrad=True,
        ),
        BinaryUfuncInfo(
            "nextafter",
            dtypes=floating_types_and(torch.bfloat16),
            supports_autograd=False,
            supports_rhs_python_scalar=False,
        ),
        BinaryUfuncInfo(
            "remainder",
            ref=np.remainder,
            dtypes=all_types_and(torch.float16, torch.bfloat16),
            dtypesIfCUDA=all_types_and(torch.float16, torch.bfloat16),
            supports_forward_ad=True,
            supports_fwgrad_bwgrad=True,
            assert_autodiffed=None,
            supports_one_python_scalar=True,
            rhs_make_tensor_kwargs={"exclude_zero": True},
        ),
        BinaryUfuncInfo(
            "sub",
            # NumPy has no builtin reference for the alpha kwarg, but it is easy enough to emulate
            ref=lambda input, other, *, alpha=1: np.subtract(
                input, np.multiply(alpha, other)
            ),
            aliases=("subtract",),
            dtypes=all_types_and_complex_and(torch.bfloat16, torch.float16),
            assert_autodiffed=True,
            supports_forward_ad=True,
            supports_fwgrad_bwgrad=True,
            sample_inputs_func=sample_inputs_add_sub,
            supports_inplace_autograd=False,
            supports_two_python_scalars=True,
        ),
        BinaryUfuncInfo(
            "true_divide",
            dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
            supports_forward_ad=True,
            promotes_int_to_float=True,
            supports_fwgrad_bwgrad=True,
            supports_two_python_scalars=True,
            rhs_make_tensor_kwargs={"exclude_zero": True},
        ),
        BinaryUfuncInfo(
            "eq",
            dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
            always_returns_bool=True,
            supports_autograd=False,
            sample_inputs_func=sample_inputs_comparison_ops,
        ),
        BinaryUfuncInfo(
            "ne",
            aliases=("not_equal",),
            dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
            always_returns_bool=True,
            supports_autograd=False,
        ),
        BinaryUfuncInfo(
            "le",
            aliases=("less_equal",),
            dtypes=all_types_and(torch.bool, torch.bfloat16, torch.float16),
            always_returns_bool=True,
            supports_autograd=False,
        ),
        BinaryUfuncInfo(
            "ge",
            aliases=("greater_equal",),
            dtypes=all_types_and(torch.bool, torch.bfloat16, torch.float16),
            always_returns_bool=True,
            supports_autograd=False,
        ),
        BinaryUfuncInfo(
            "gt",
            aliases=("greater",),
            dtypes=all_types_and(torch.bool, torch.bfloat16, torch.float16),
            always_returns_bool=True,
            supports_autograd=False,
        ),
        BinaryUfuncInfo(
            "lt",
            aliases=("less",),
            dtypes=all_types_and(torch.bool, torch.bfloat16, torch.float16),
            always_returns_bool=True,
            supports_autograd=False,
        ),
        BinaryUfuncInfo(
            "maximum",
            dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool),
            supports_forward_ad=True,
            supports_fwgrad_bwgrad=True,
            ref=np.maximum,
            supports_rhs_python_scalar=False,
        ),
        BinaryUfuncInfo(
            "minimum",
            dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool),
            supports_forward_ad=True,
            supports_fwgrad_bwgrad=True,
            ref=np.minimum,
            supports_rhs_python_scalar=False,
        ),
        BinaryUfuncInfo(
            "fmax",
            op=torch.fmax,
            dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool),
            supports_forward_ad=True,
            supports_fwgrad_bwgrad=True,
            supports_rhs_python_scalar=False,
        ),
        BinaryUfuncInfo(
            "fmin",
            op=torch.fmin,
            dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool),
            supports_forward_ad=True,
            supports_fwgrad_bwgrad=True,
            supports_rhs_python_scalar=False,
        ),
    ]
)
