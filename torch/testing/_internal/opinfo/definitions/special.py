# mypy: ignore-errors

import unittest
from functools import partial
from itertools import product
from typing import List

import numpy as np

import torch
from torch.testing import make_tensor
from torch.testing._internal.common_device_type import (
    precisionOverride,
    tol,
    toleranceOverride,
)
from torch.testing._internal.common_dtype import all_types_and, floating_types
from torch.testing._internal.common_utils import TEST_SCIPY, torch_to_numpy_dtype_dict
from torch.testing._internal.opinfo.core import (
    BinaryUfuncInfo,
    DecorateInfo,
    L,
    NumericsFilter,
    OpInfo,
    S,
    SampleInput,
    UnaryUfuncInfo,
)
from torch.testing._internal.opinfo.refs import (
    ElementwiseBinaryPythonRefInfo,
    ElementwiseUnaryPythonRefInfo,
)
from torch.testing._internal.opinfo.utils import (
    np_unary_ufunc_integer_promotion_wrapper,
)


if TEST_SCIPY:
    import scipy.special


# TODO: Consolidate `i0e` with sample_inputs_unary when `make_tensor`,
#       supports `exclude` argument.
#       For more context: https://github.com/pytorch/pytorch/pull/56352#discussion_r633277617
def sample_inputs_i0_i1(op_info, device, dtype, requires_grad, **kwargs):
    exclude_zero = requires_grad and op_info.op == torch.special.i0e
    make_arg = partial(
        make_tensor,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
        exclude_zero=exclude_zero,
    )
    yield SampleInput(make_arg((S,)))
    yield SampleInput(make_arg(()))

    if requires_grad and not exclude_zero:
        # Special Case for gradient
        # Sample with `0` in the input
        t = make_arg((S,))
        t[0] = 0

        yield SampleInput(t)


def sample_inputs_polygamma(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(
        make_tensor,
        device=device,
        # TODO: eliminate low after gh-106692 is fixed:
        low=(1 if dtype in {torch.int32, torch.int64} else None),
        dtype=dtype,
        requires_grad=requires_grad,
    )
    tensor_shapes = ((S, S), ())
    ns = (1, 2, 3, 4, 5)

    for shape, n in product(tensor_shapes, ns):
        yield SampleInput(make_arg(shape), args=(n,))


def reference_polygamma(x, n):
    # WEIRD `scipy.special.polygamma` behavior
    # >>> scipy.special.polygamma(0, np.array(501, dtype=np.float32)).dtype
    # dtype('float64')
    # >>> scipy.special.polygamma(0, np.array([501], dtype=np.float32)).dtype
    # dtype('float32')
    #
    # Thus we cast output to the default torch dtype or preserve double
    result_dtype = torch_to_numpy_dtype_dict[torch.get_default_dtype()]
    if x.dtype == np.double:
        result_dtype = np.double
    return scipy.special.polygamma(n, x).astype(result_dtype)


def sample_inputs_entr(op_info, device, dtype, requires_grad, **kwargs):
    low, _ = op_info.domain

    if requires_grad:
        low = 0 + op_info._domain_eps

    make_arg = partial(
        make_tensor, dtype=dtype, device=device, low=low, requires_grad=requires_grad
    )
    yield SampleInput(make_arg((L,)))
    yield SampleInput(make_arg(()))


def sample_inputs_erfcx(op_info, device, dtype, requires_grad, **kwargs):
    for shape in ((L,), (1, 0, 3), ()):
        yield SampleInput(
            make_tensor(
                shape,
                device=device,
                dtype=dtype,
                low=-5,
                requires_grad=requires_grad,
            ),
        )


op_db: List[OpInfo] = [
    UnaryUfuncInfo(
        "special.i0e",
        aten_name="special_i0e",
        ref=scipy.special.i0e if TEST_SCIPY else None,
        decorators=(precisionOverride({torch.bfloat16: 3e-1, torch.float16: 3e-1}),),
        dtypes=all_types_and(torch.bool, torch.half, torch.bfloat16),
        sample_inputs_func=sample_inputs_i0_i1,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
    ),
    UnaryUfuncInfo(
        "special.i1",
        aten_name="special_i1",
        ref=np_unary_ufunc_integer_promotion_wrapper(scipy.special.i1)
        if TEST_SCIPY
        else None,
        dtypes=all_types_and(torch.bool, torch.half, torch.bfloat16),
        backward_dtypes=floating_types(),
        sample_inputs_func=sample_inputs_i0_i1,
        decorators=(
            DecorateInfo(
                toleranceOverride(
                    {
                        torch.float32: tol(atol=1e-4, rtol=0),
                        torch.bool: tol(atol=1e-4, rtol=0),
                    }
                )
            ),
        ),
        skips=(
            DecorateInfo(
                unittest.skip("Incorrect result!"),
                "TestUnaryUfuncs",
                "test_reference_numerics_large",
                dtypes=(torch.int8,),
            ),
        ),
        supports_fwgrad_bwgrad=True,
        supports_forward_ad=True,
    ),
    UnaryUfuncInfo(
        "special.i1e",
        aten_name="special_i1e",
        ref=scipy.special.i1e if TEST_SCIPY else None,
        dtypes=all_types_and(torch.bool, torch.half, torch.bfloat16),
        backward_dtypes=floating_types(),
        sample_inputs_func=sample_inputs_i0_i1,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
    ),
    UnaryUfuncInfo(
        "special.ndtr",
        aten_name="special_ndtr",
        decorators=(precisionOverride({torch.bfloat16: 5e-3, torch.float16: 5e-4}),),
        ref=scipy.special.ndtr if TEST_SCIPY else None,
        dtypes=all_types_and(torch.bool, torch.half, torch.bfloat16),
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        skips=(
            # Dispatch stub: unsupported device typemeta
            DecorateInfo(
                unittest.expectedFailure,
                "TestFwdGradients",
                "test_fn_fwgrad_bwgrad",
                device_type="meta",
            ),
        ),
    ),
    # A separate OpInfo entry for special.polygamma is needed to reorder the arguments
    # for the alias. See the discussion here: https://github.com/pytorch/pytorch/pull/59691#discussion_r650261939
    UnaryUfuncInfo(
        "special.polygamma",
        op=lambda x, n, **kwargs: torch.special.polygamma(n, x, **kwargs),
        variant_test_name="special_polygamma_n_0",
        ref=reference_polygamma if TEST_SCIPY else None,
        dtypes=all_types_and(torch.bool, torch.half, torch.bfloat16),
        dtypesIfGPU=all_types_and(torch.bool, torch.half, torch.bfloat16),
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        sample_inputs_func=sample_inputs_polygamma,
        skips=(
            # lambda impl
            DecorateInfo(
                unittest.expectedFailure, "TestJit", "test_variant_consistency_jit"
            ),
            DecorateInfo(
                unittest.expectedFailure,
                "TestNormalizeOperators",
                "test_normalize_operator_exhaustive",
            ),
        ),
        sample_kwargs=lambda device, dtype, input: ({"n": 0}, {"n": 0}),
        # polygamma functions have multiple singularities at x having non-positive integer value
        reference_numerics_filter=NumericsFilter(
            condition=lambda x: (x < 0.1) & ((x - x.round()).abs() < 1e-4), safe_val=1
        ),
    ),
    BinaryUfuncInfo(
        "special.xlog1py",
        aten_name="special_xlog1py",
        dtypes=all_types_and(torch.bool, torch.half, torch.bfloat16),
        promotes_int_to_float=True,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        supports_one_python_scalar=True,
        # We don't test -1 as the gradient will be NaN and it'll break
        rhs_make_tensor_kwargs=dict(low=-0.99),
    ),
    BinaryUfuncInfo(
        "special.zeta",
        aten_name="special_zeta",
        dtypes=all_types_and(torch.bool),
        promotes_int_to_float=True,
        supports_autograd=False,
        supports_one_python_scalar=True,
        skips=(
            # Reference reference_inputs nans and infs on cuda and nan, inf, 0., -inf for cpu
            DecorateInfo(unittest.expectedFailure, "TestCommon", "test_compare_cpu"),
        ),
    ),
    # TODO: FIXME
    # OpInfo entry to verify the gradient formula of `other`/`q`
    # BinaryUfuncInfo('special.zeta',
    #                 op=lambda q, x, **kwargs: torch.special.zeta(x, q, **kwargs),
    #                 aten_name='special_zeta',
    #                 variant_test_name='grad',
    #                 dtypes=all_types_and(torch.bool),
    #                 promotes_int_to_float=True,
    #                 supports_autograd=True,
    #                 supports_rhs_python_scalar=False,
    #                 decorators=[
    #                     # Derivative wrt first tensor not implemented
    #                     DecorateInfo(unittest.expectedFailure, "TestCommon",
    #                                  "test_floating_inputs_are_differentiable")
    #                 ],
    #                 skips=(
    #                     # Lambda doesn't work in JIT test
    #                     # AssertionError: JIT Test does not execute any logic
    #                     DecorateInfo(unittest.skip("Skipped!"), "TestJit", "test_variant_consistency_jit"),
    #                 )),
    UnaryUfuncInfo(
        "special.entr",
        ref=scipy.special.entr if TEST_SCIPY else None,
        aten_name="special_entr",
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        decorators=(precisionOverride({torch.float16: 1e-1, torch.bfloat16: 1e-1}),),
        dtypes=all_types_and(torch.bool, torch.half, torch.bfloat16),
        skips=(
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestUnaryUfuncs",
                "test_reference_numerics_large",
                dtypes=[torch.bfloat16, torch.float16],
            ),
        ),
        supports_inplace_autograd=False,
        sample_inputs_func=sample_inputs_entr,
    ),
    UnaryUfuncInfo(
        "special.ndtri",
        ref=scipy.special.ndtri if TEST_SCIPY else None,
        domain=(0, 1),
        aten_name="special_ndtri",
        dtypes=all_types_and(torch.bool),
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
    ),
    UnaryUfuncInfo(
        "special.log_ndtr",
        aten_name="special_log_ndtr",
        ref=scipy.special.log_ndtr if TEST_SCIPY else None,
        dtypes=all_types_and(torch.bool),
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
    ),
    UnaryUfuncInfo(
        "special.erfcx",
        ref=scipy.special.erfcx if TEST_SCIPY else None,
        aten_name="special_erfcx",
        decorators=(
            toleranceOverride(
                {
                    torch.float32: tol(atol=0, rtol=4e-6),
                }
            ),
        ),
        dtypes=all_types_and(torch.bool),
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        sample_inputs_func=sample_inputs_erfcx,
    ),
    UnaryUfuncInfo(
        "special.airy_ai",
        decorators=(
            precisionOverride(
                {
                    torch.float32: 1e-03,
                    torch.float64: 1e-05,
                },
            ),
        ),
        dtypes=all_types_and(torch.bool),
        ref=lambda x: scipy.special.airy(x)[0] if TEST_SCIPY else None,
        skips=(
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestUnaryUfuncs",
                "test_reference_numerics_large",
            ),
        ),
        supports_autograd=False,
    ),
    UnaryUfuncInfo(
        "special.bessel_j0",
        decorators=(
            precisionOverride(
                {
                    torch.float32: 1e-04,
                    torch.float64: 1e-05,
                },
            ),
        ),
        dtypes=all_types_and(torch.bool),
        ref=scipy.special.j0 if TEST_SCIPY else None,
        supports_autograd=False,
    ),
    UnaryUfuncInfo(
        "special.bessel_j1",
        decorators=(
            precisionOverride(
                {
                    torch.float32: 1e-04,
                    torch.float64: 1e-05,
                },
            ),
        ),
        dtypes=all_types_and(torch.bool),
        ref=scipy.special.j1 if TEST_SCIPY else None,
        supports_autograd=False,
    ),
    UnaryUfuncInfo(
        "special.bessel_y0",
        decorators=(
            precisionOverride(
                {
                    torch.float32: 1e-04,
                    torch.float64: 1e-05,
                },
            ),
        ),
        dtypes=all_types_and(torch.bool),
        ref=scipy.special.y0 if TEST_SCIPY else None,
        supports_autograd=False,
    ),
    UnaryUfuncInfo(
        "special.bessel_y1",
        decorators=(
            precisionOverride(
                {
                    torch.float32: 1e-04,
                    torch.float64: 1e-05,
                },
            ),
        ),
        dtypes=all_types_and(torch.bool),
        ref=scipy.special.y1 if TEST_SCIPY else None,
        supports_autograd=False,
    ),
    BinaryUfuncInfo(
        "special.chebyshev_polynomial_t",
        dtypes=all_types_and(torch.bool),
        promotes_int_to_float=True,
        skips=(
            DecorateInfo(unittest.skip("Skipped!"), "TestCudaFuserOpInfo"),
            DecorateInfo(unittest.skip("Skipped!"), "TestNNCOpInfo"),
            DecorateInfo(
                unittest.skip("testing takes an unreasonably long time, #79528"),
                "TestCommon",
                "test_compare_cpu",
            ),
        ),
        supports_one_python_scalar=True,
        supports_autograd=False,
    ),
    BinaryUfuncInfo(
        "special.chebyshev_polynomial_u",
        dtypes=all_types_and(torch.bool),
        promotes_int_to_float=True,
        skips=(
            DecorateInfo(unittest.skip("Skipped!"), "TestCudaFuserOpInfo"),
            DecorateInfo(unittest.skip("Skipped!"), "TestNNCOpInfo"),
            DecorateInfo(
                unittest.skip("testing takes an unreasonably long time, #79528"),
                "TestCommon",
                "test_compare_cpu",
            ),
        ),
        supports_one_python_scalar=True,
        supports_autograd=False,
    ),
    BinaryUfuncInfo(
        "special.chebyshev_polynomial_v",
        dtypes=all_types_and(torch.bool),
        promotes_int_to_float=True,
        skips=(
            DecorateInfo(
                unittest.skip(
                    "Skipping - testing takes an unreasonably long time, #79528"
                )
            ),
            DecorateInfo(unittest.skip("Skipped!"), "TestCudaFuserOpInfo"),
            DecorateInfo(unittest.skip("Skipped!"), "TestNNCOpInfo"),
        ),
        supports_one_python_scalar=True,
        supports_autograd=False,
    ),
    BinaryUfuncInfo(
        "special.chebyshev_polynomial_w",
        dtypes=all_types_and(torch.bool),
        promotes_int_to_float=True,
        skips=(
            DecorateInfo(
                unittest.skip(
                    "Skipping - testing takes an unreasonably long time, #79528"
                )
            ),
            DecorateInfo(unittest.skip("Skipped!"), "TestCudaFuserOpInfo"),
            DecorateInfo(unittest.skip("Skipped!"), "TestNNCOpInfo"),
        ),
        supports_one_python_scalar=True,
        supports_autograd=False,
    ),
    BinaryUfuncInfo(
        "special.hermite_polynomial_h",
        dtypes=all_types_and(torch.bool),
        promotes_int_to_float=True,
        skips=(
            DecorateInfo(unittest.skip("Skipped!"), "TestCudaFuserOpInfo"),
            DecorateInfo(unittest.skip("Skipped!"), "TestNNCOpInfo"),
            # Greatest absolute difference: inf
            DecorateInfo(unittest.expectedFailure, "TestCommon", "test_compare_cpu"),
        ),
        supports_one_python_scalar=True,
        supports_autograd=False,
    ),
    BinaryUfuncInfo(
        "special.hermite_polynomial_he",
        dtypes=all_types_and(torch.bool),
        promotes_int_to_float=True,
        skips=(
            DecorateInfo(unittest.skip("Skipped!"), "TestCudaFuserOpInfo"),
            DecorateInfo(unittest.skip("Skipped!"), "TestNNCOpInfo"),
            DecorateInfo(
                unittest.skip("testing takes an unreasonably long time, #79528"),
                "TestCommon",
                "test_compare_cpu",
            ),
        ),
        supports_one_python_scalar=True,
        supports_autograd=False,
    ),
    BinaryUfuncInfo(
        "special.laguerre_polynomial_l",
        dtypes=all_types_and(torch.bool),
        promotes_int_to_float=True,
        skips=(
            DecorateInfo(unittest.skip("Skipped!"), "TestCudaFuserOpInfo"),
            DecorateInfo(unittest.skip("Skipped!"), "TestNNCOpInfo"),
            DecorateInfo(
                unittest.skip("testing takes an unreasonably long time, #79528"),
                "TestCommon",
                "test_compare_cpu",
            ),
        ),
        supports_one_python_scalar=True,
        supports_autograd=False,
    ),
    BinaryUfuncInfo(
        "special.legendre_polynomial_p",
        dtypes=all_types_and(torch.bool),
        promotes_int_to_float=True,
        skips=(
            DecorateInfo(
                unittest.skip(
                    "Skipping - testing takes an unreasonably long time, #79528"
                )
            ),
            DecorateInfo(unittest.skip("Skipped!"), "TestCudaFuserOpInfo"),
            DecorateInfo(unittest.skip("Skipped!"), "TestNNCOpInfo"),
            DecorateInfo(
                unittest.skip("testing takes an unreasonably long time, #79528"),
                "TestCommon",
                "test_compare_cpu",
            ),
        ),
        supports_one_python_scalar=True,
        supports_autograd=False,
    ),
    UnaryUfuncInfo(
        "special.modified_bessel_i0",
        decorators=(
            precisionOverride(
                {
                    torch.float32: 1e-03,
                    torch.float64: 1e-05,
                },
            ),
        ),
        dtypes=all_types_and(torch.bool),
        ref=scipy.special.i0 if TEST_SCIPY else None,
        supports_autograd=False,
    ),
    UnaryUfuncInfo(
        "special.modified_bessel_i1",
        decorators=(
            precisionOverride(
                {
                    torch.float32: 1e-03,
                    torch.float64: 1e-05,
                },
            ),
        ),
        dtypes=all_types_and(torch.bool),
        ref=scipy.special.i1 if TEST_SCIPY else None,
        supports_autograd=False,
    ),
    UnaryUfuncInfo(
        "special.modified_bessel_k0",
        decorators=(
            precisionOverride(
                {
                    torch.float32: 1e-03,
                    torch.float64: 1e-05,
                },
            ),
        ),
        dtypes=all_types_and(torch.bool),
        ref=scipy.special.k0 if TEST_SCIPY else None,
        supports_autograd=False,
    ),
    UnaryUfuncInfo(
        "special.modified_bessel_k1",
        decorators=(
            precisionOverride(
                {
                    torch.float32: 1e-03,
                    torch.float64: 1e-05,
                },
            ),
        ),
        dtypes=all_types_and(torch.bool),
        ref=scipy.special.k1 if TEST_SCIPY else None,
        supports_autograd=False,
    ),
    UnaryUfuncInfo(
        "special.scaled_modified_bessel_k0",
        decorators=(
            toleranceOverride(
                {
                    torch.float32: tol(atol=1e-03, rtol=1e-03),
                    torch.float64: tol(atol=1e-05, rtol=1e-03),
                }
            ),
        ),
        dtypes=all_types_and(torch.bool),
        ref=scipy.special.k0e if TEST_SCIPY else None,
        supports_autograd=False,
    ),
    UnaryUfuncInfo(
        "special.scaled_modified_bessel_k1",
        decorators=(
            toleranceOverride(
                {
                    torch.float32: tol(atol=1e-03, rtol=1e-03),
                    torch.float64: tol(atol=1e-05, rtol=1e-03),
                }
            ),
        ),
        dtypes=all_types_and(torch.bool),
        ref=scipy.special.k1e if TEST_SCIPY else None,
        supports_autograd=False,
    ),
    BinaryUfuncInfo(
        "special.shifted_chebyshev_polynomial_t",
        dtypes=all_types_and(torch.bool),
        promotes_int_to_float=True,
        skips=(
            DecorateInfo(
                unittest.skip(
                    "Skipping - testing takes an unreasonably long time, #79528"
                )
            ),
            DecorateInfo(unittest.skip("Skipped!"), "TestCudaFuserOpInfo"),
            DecorateInfo(unittest.skip("Skipped!"), "TestNNCOpInfo"),
            DecorateInfo(
                unittest.skip("testing takes an unreasonably long time, #79528"),
                "TestCommon",
                "test_compare_cpu",
            ),
        ),
        supports_one_python_scalar=True,
        supports_autograd=False,
    ),
    BinaryUfuncInfo(
        "special.shifted_chebyshev_polynomial_u",
        dtypes=all_types_and(torch.bool),
        promotes_int_to_float=True,
        skips=(
            DecorateInfo(
                unittest.skip(
                    "Skipping - testing takes an unreasonably long time, #79528"
                )
            ),
            DecorateInfo(unittest.skip("Skipped!"), "TestCudaFuserOpInfo"),
            DecorateInfo(unittest.skip("Skipped!"), "TestNNCOpInfo"),
            DecorateInfo(
                unittest.skip("testing takes an unreasonably long time, #79528"),
                "TestCommon",
                "test_compare_cpu",
            ),
        ),
        supports_one_python_scalar=True,
        supports_autograd=False,
    ),
    BinaryUfuncInfo(
        "special.shifted_chebyshev_polynomial_v",
        dtypes=all_types_and(torch.bool),
        promotes_int_to_float=True,
        skips=(
            DecorateInfo(
                unittest.skip(
                    "Skipping - testing takes an unreasonably long time, #79528"
                )
            ),
            DecorateInfo(unittest.skip("Skipped!"), "TestCudaFuserOpInfo"),
            DecorateInfo(unittest.skip("Skipped!"), "TestNNCOpInfo"),
            DecorateInfo(
                unittest.skip("testing takes an unreasonably long time, #79528"),
                "TestCommon",
                "test_compare_cpu",
            ),
        ),
        supports_one_python_scalar=True,
        supports_autograd=False,
    ),
    BinaryUfuncInfo(
        "special.shifted_chebyshev_polynomial_w",
        dtypes=all_types_and(torch.bool),
        promotes_int_to_float=True,
        skips=(
            DecorateInfo(
                unittest.skip(
                    "Skipping - testing takes an unreasonably long time, #79528"
                )
            ),
            DecorateInfo(unittest.skip("Skipped!"), "TestCudaFuserOpInfo"),
            DecorateInfo(unittest.skip("Skipped!"), "TestNNCOpInfo"),
            DecorateInfo(
                unittest.skip("testing takes an unreasonably long time, #79528"),
                "TestCommon",
                "test_compare_cpu",
            ),
        ),
        supports_one_python_scalar=True,
        supports_autograd=False,
    ),
    UnaryUfuncInfo(
        "special.spherical_bessel_j0",
        decorators=(
            toleranceOverride(
                {
                    torch.float32: tol(atol=1e-03, rtol=1e-03),
                    torch.float64: tol(atol=1e-05, rtol=1e-03),
                }
            ),
        ),
        dtypes=all_types_and(torch.bool),
        ref=lambda x: scipy.special.spherical_jn(0, x) if TEST_SCIPY else None,
        supports_autograd=False,
    ),
]

python_ref_db: List[OpInfo] = [
    #
    # Elementwise Unary Special OpInfos
    #
    ElementwiseUnaryPythonRefInfo(
        "_refs.special.bessel_j0",
        torch_opinfo_name="special.bessel_j0",
        op_db=op_db,
        decorators=(
            precisionOverride(
                {
                    torch.float32: 1e-04,
                    torch.float64: 1e-05,
                },
            ),
        ),
    ),
    ElementwiseUnaryPythonRefInfo(
        "_refs.special.bessel_j1",
        torch_opinfo_name="special.bessel_j1",
        op_db=op_db,
        decorators=(
            precisionOverride(
                {
                    torch.float32: 1e-04,
                    torch.float64: 1e-05,
                },
            ),
        ),
    ),
    ElementwiseUnaryPythonRefInfo(
        "_refs.special.entr",
        torch_opinfo_name="special.entr",
        op_db=op_db,
        decorators=(precisionOverride({torch.float16: 1e-1, torch.bfloat16: 1e-1}),),
        skips=(
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestUnaryUfuncs",
                "test_reference_numerics_large",
                dtypes=[torch.bfloat16, torch.float16],
            ),
        ),
    ),
    ElementwiseUnaryPythonRefInfo(
        "_refs.special.erfcx",
        torch_opinfo_name="special.erfcx",
        op_db=op_db,
        decorators=(
            toleranceOverride(
                {
                    torch.float32: tol(atol=0, rtol=4e-6),
                }
            ),
        ),
    ),
    ElementwiseUnaryPythonRefInfo(
        "_refs.special.i0e",
        torch_opinfo_name="special.i0e",
        op_db=op_db,
        decorators=(precisionOverride({torch.bfloat16: 3e-1, torch.float16: 3e-1}),),
    ),
    ElementwiseUnaryPythonRefInfo(
        "_refs.special.i1",
        torch_opinfo_name="special.i1",
        op_db=op_db,
        decorators=(
            DecorateInfo(
                toleranceOverride(
                    {
                        torch.float32: tol(atol=1e-4, rtol=0),
                        torch.bool: tol(atol=1e-4, rtol=0),
                    }
                )
            ),
        ),
        skips=(
            DecorateInfo(
                unittest.skip("Incorrect result!"),
                "TestUnaryUfuncs",
                "test_reference_numerics_large",
                dtypes=(torch.int8,),
            ),
        ),
    ),
    ElementwiseUnaryPythonRefInfo(
        "_refs.special.i1e",
        torch_opinfo_name="special.i1e",
        op_db=op_db,
    ),
    ElementwiseUnaryPythonRefInfo(
        "_refs.special.log_ndtr",
        torch_opinfo_name="special.log_ndtr",
        op_db=op_db,
    ),
    ElementwiseUnaryPythonRefInfo(
        "_refs.special.ndtr",
        torch_opinfo_name="special.ndtr",
        op_db=op_db,
    ),
    ElementwiseUnaryPythonRefInfo(
        "_refs.special.ndtri",
        torch_opinfo_name="special.ndtri",
        op_db=op_db,
    ),
    ElementwiseUnaryPythonRefInfo(
        "_refs.special.spherical_bessel_j0",
        torch_opinfo_name="special.spherical_bessel_j0",
        op_db=op_db,
        decorators=(
            toleranceOverride(
                {
                    torch.float32: tol(atol=1e-03, rtol=1e-03),
                    torch.float64: tol(atol=1e-05, rtol=1e-03),
                }
            ),
        ),
    ),
    #
    # Elementwise Binary Special OpInfos
    #
    ElementwiseBinaryPythonRefInfo(
        "_refs.special.zeta",
        torch_opinfo_name="special.zeta",
        supports_one_python_scalar=True,
        op_db=op_db,
        skips=(
            # Reference reference_inputs nans and infs on cuda and nan, inf, 0., -inf for cpu
            DecorateInfo(unittest.expectedFailure, "TestCommon", "test_compare_cpu"),
        ),
    ),
]
