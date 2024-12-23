# Owner(s): ["module: tests"]
# ruff: noqa: F841

import itertools
import math
import operator
import random
import sys
import warnings
from functools import partial
from itertools import chain, product
from numbers import Number

import numpy as np

import torch
import torch.autograd.forward_ad as fwAD
from torch import inf, nan
from torch.testing import make_tensor
from torch.testing._internal.common_device_type import (
    deviceCountAtLeast,
    dtypes,
    dtypesIfCPU,
    dtypesIfCUDA,
    expectedFailureMeta,
    instantiate_device_type_tests,
    onlyCPU,
    onlyCUDA,
    onlyNativeDeviceTypes,
    OpDTypes,
    ops,
    precisionOverride,
    skipIf,
    skipMeta,
)
from torch.testing._internal.common_dtype import (
    all_types_and,
    all_types_and_complex_and,
    complex_types,
    floating_and_complex_types,
    floating_types_and,
    get_all_int_dtypes,
    get_all_math_dtypes,
    integral_types,
    integral_types_and,
)
from torch.testing._internal.common_methods_invocations import (
    binary_ufuncs,
    binary_ufuncs_and_refs,
    generate_elementwise_binary_broadcasting_tensors,
    generate_elementwise_binary_extremal_value_tensors,
    generate_elementwise_binary_large_value_tensors,
    generate_elementwise_binary_small_value_tensors,
    generate_elementwise_binary_tensors,
    generate_elementwise_binary_with_scalar_and_type_promotion_samples,
    generate_elementwise_binary_with_scalar_samples,
)
from torch.testing._internal.common_utils import (
    gradcheck,
    iter_indices,
    numpy_to_torch_dtype_dict,
    run_tests,
    set_default_dtype,
    skipIfTorchDynamo,
    slowTest,
    TEST_SCIPY,
    TestCase,
    torch_to_numpy_dtype_dict,
    xfailIfTorchDynamo,
)


if TEST_SCIPY:
    import scipy.integrate
    import scipy.special


# TODO: update to use opinfos consistently
class TestBinaryUfuncs(TestCase):
    # Generic tests for elementwise binary (AKA binary universal (u) functions (funcs))
    # TODO: below contiguous tensor results are compared with a variety of noncontiguous results.
    #   It would be interesting to have the lhs and rhs have different discontiguities.

    # Helper for comparing torch tensors and NumPy arrays
    # TODO: should this or assertEqual also validate that strides are equal?
    def assertEqualHelper(
        self, actual, expected, msg, *, dtype, exact_dtype=True, **kwargs
    ):
        assert isinstance(actual, torch.Tensor)

        # Some NumPy functions return scalars, not arrays
        if isinstance(expected, Number):
            self.assertEqual(actual.item(), expected, msg=msg, **kwargs)
        elif isinstance(expected, np.ndarray):
            # Handles exact dtype comparisons between arrays and tensors
            if exact_dtype:
                # Allows array dtype to be float32 when comparing with bfloat16 tensors
                #   since NumPy doesn't support the bfloat16 dtype
                # Also ops like scipy.special.erf, scipy.special.erfc, etc, promote float16
                # to float32
                if expected.dtype == np.float32:
                    assert actual.dtype in (
                        torch.float16,
                        torch.bfloat16,
                        torch.float32,
                    )
                else:
                    assert expected.dtype == torch_to_numpy_dtype_dict[actual.dtype]

            self.assertEqual(
                actual,
                torch.from_numpy(expected).to(actual.dtype),
                msg,
                exact_device=False,
                **kwargs,
            )
        else:
            self.assertEqual(actual, expected, msg, exact_device=False, **kwargs)

    # Tests that the function and its (array-accepting) reference produce the same
    #   values on given tensors
    def _test_reference_numerics(self, dtype, op, gen, equal_nan=True):
        def _helper_reference_numerics(
            expected, actual, msg, exact_dtype, equal_nan=True
        ):
            if not torch.can_cast(
                numpy_to_torch_dtype_dict[expected.dtype.type], dtype
            ):
                exact_dtype = False

            if dtype is torch.bfloat16 and expected.dtype == np.float32:
                # Ref: https://github.com/pytorch/pytorch/blob/master/torch/testing/_internal/common_utils.py#L1149
                self.assertEqualHelper(
                    actual,
                    expected,
                    msg,
                    dtype=dtype,
                    exact_dtype=exact_dtype,
                    rtol=16e-3,
                    atol=1e-5,
                )
            else:
                self.assertEqualHelper(
                    actual,
                    expected,
                    msg,
                    dtype=dtype,
                    equal_nan=equal_nan,
                    exact_dtype=exact_dtype,
                )

        for sample in gen:
            # Each sample input acquired from the generator is just one lhs tensor
            #   and one rhs tensor
            l = sample.input
            r = sample.args[0]

            numpy_sample = sample.numpy()
            l_numpy = numpy_sample.input
            r_numpy = numpy_sample.args[0]
            actual = op(l, r)
            expected = op.ref(l_numpy, r_numpy)

            # Dtype promo rules have changed since NumPy 2.
            # Specialize the backward-incompatible cases.
            if (
                np.__version__ > "2"
                and op.name in ("sub", "_refs.sub")
                and isinstance(l_numpy, np.ndarray)
            ):
                expected = expected.astype(l_numpy.dtype)

            # Crafts a custom error message for smaller, printable tensors
            def _numel(x):
                if isinstance(x, torch.Tensor):
                    return x.numel()
                # Assumes x is a scalar
                return 1

            if _numel(l) <= 100 and _numel(r) <= 100:
                msg = (
                    "Failed to produce expected results! Input lhs tensor was"
                    f" {l}, rhs tensor was {r}, torch result is {actual}, and reference result is"
                    f" {expected}."
                )
            else:
                msg = None

            exact_dtype = True
            if isinstance(actual, torch.Tensor):
                _helper_reference_numerics(
                    expected, actual, msg, exact_dtype, equal_nan
                )
            else:
                for x, y in zip(expected, actual):
                    # testing multi-outputs results
                    _helper_reference_numerics(x, y, msg, exact_dtype, equal_nan)

    # The following tests only apply to elementwise binary operators with references
    binary_ufuncs_with_references = list(
        filter(lambda op: op.ref is not None and op.ref is not None, binary_ufuncs)
    )

    @ops(binary_ufuncs_with_references)
    def test_reference_numerics(self, device, dtype, op):
        gen = generate_elementwise_binary_tensors(op, device=device, dtype=dtype)
        self._test_reference_numerics(dtype, op, gen, equal_nan=True)

    @ops(binary_ufuncs_with_references)
    def test_reference_numerics_small_values(self, device, dtype, op):
        if dtype is torch.bool:
            self.skipTest("Doesn't support bool!")

        gen = generate_elementwise_binary_small_value_tensors(
            op, device=device, dtype=dtype
        )
        self._test_reference_numerics(dtype, op, gen, equal_nan=True)

    @ops(
        binary_ufuncs_with_references,
        allowed_dtypes=(
            torch.int16,
            torch.int32,
            torch.int64,
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
            torch.complex64,
            torch.complex128,
        ),
    )
    def test_reference_numerics_large_values(self, device, dtype, op):
        gen = generate_elementwise_binary_large_value_tensors(
            op, device=device, dtype=dtype
        )
        self._test_reference_numerics(dtype, op, gen, equal_nan=True)

    @ops(
        binary_ufuncs_with_references,
        allowed_dtypes=(
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
            torch.complex64,
            torch.complex128,
        ),
    )
    def test_reference_numerics_extremal_values(self, device, dtype, op):
        gen = generate_elementwise_binary_extremal_value_tensors(
            op, device=device, dtype=dtype
        )
        self._test_reference_numerics(dtype, op, gen, equal_nan=True)

    # tests broadcasting and noncontiguous broadcasting behavior
    @ops(
        binary_ufuncs_with_references,
        allowed_dtypes=(
            torch.long,
            torch.float32,
        ),
    )
    def test_broadcasting(self, device, dtype, op):
        gen = generate_elementwise_binary_broadcasting_tensors(
            op, device=device, dtype=dtype
        )
        self._test_reference_numerics(dtype, op, gen, equal_nan=True)

    @ops(
        binary_ufuncs_with_references,
        allowed_dtypes=(torch.long, torch.float32, torch.complex64),
    )
    def test_scalar_support(self, device, dtype, op):
        gen = generate_elementwise_binary_with_scalar_samples(
            op, device=device, dtype=dtype
        )
        self._test_reference_numerics(dtype, op, gen, equal_nan=True)
        gen = generate_elementwise_binary_with_scalar_and_type_promotion_samples(
            op, device=device, dtype=dtype
        )
        self._test_reference_numerics(dtype, op, gen, equal_nan=True)

    @ops(binary_ufuncs)
    def test_contig_vs_every_other(self, device, dtype, op):
        lhs = make_tensor(
            (1026,), device=device, dtype=dtype, **op.lhs_make_tensor_kwargs
        )
        rhs = make_tensor(
            (1026,), device=device, dtype=dtype, **op.rhs_make_tensor_kwargs
        )

        lhs_non_contig = lhs[::2]
        rhs_non_contig = rhs[::2]

        self.assertTrue(lhs.is_contiguous())
        self.assertTrue(rhs.is_contiguous())

        self.assertFalse(lhs_non_contig.is_contiguous())
        self.assertFalse(rhs_non_contig.is_contiguous())

        expected = op(lhs, rhs)[::2]
        actual = op(lhs_non_contig, rhs_non_contig)
        self.assertEqual(expected, actual)

    @ops(binary_ufuncs)
    def test_contig_vs_transposed(self, device, dtype, op):
        lhs = make_tensor(
            (789, 357), device=device, dtype=dtype, **op.lhs_make_tensor_kwargs
        )
        rhs = make_tensor(
            (789, 357), device=device, dtype=dtype, **op.rhs_make_tensor_kwargs
        )

        lhs_non_contig = lhs.T
        rhs_non_contig = rhs.T

        self.assertTrue(lhs.is_contiguous())
        self.assertTrue(rhs.is_contiguous())

        self.assertFalse(lhs_non_contig.is_contiguous())
        self.assertFalse(rhs_non_contig.is_contiguous())

        expected = op(lhs, rhs).T
        actual = op(lhs_non_contig, rhs_non_contig)
        self.assertEqual(expected, actual)

    @ops(binary_ufuncs)
    def test_non_contig(self, device, dtype, op):
        shapes = ((5, 7), (1024,))
        for shape in shapes:
            lhs = make_tensor(
                shape, dtype=dtype, device=device, **op.lhs_make_tensor_kwargs
            )
            rhs = make_tensor(
                shape, dtype=dtype, device=device, **op.rhs_make_tensor_kwargs
            )

            lhs_non_contig = torch.empty(shape + (2,), device=device, dtype=dtype)[
                ..., 0
            ]
            lhs_non_contig.copy_(lhs)

            rhs_non_contig = torch.empty(shape + (2,), device=device, dtype=dtype)[
                ..., 0
            ]
            rhs_non_contig.copy_(rhs)

            self.assertTrue(lhs.is_contiguous())
            self.assertTrue(rhs.is_contiguous())

            self.assertFalse(lhs_non_contig.is_contiguous())
            self.assertFalse(rhs_non_contig.is_contiguous())

            expected = op(lhs, rhs)
            actual = op(lhs_non_contig, rhs_non_contig)
            self.assertEqual(expected, actual)

    @ops(binary_ufuncs)
    def test_non_contig_index(self, device, dtype, op):
        shape = (2, 2, 1, 2)
        lhs = make_tensor(
            shape, dtype=dtype, device=device, **op.lhs_make_tensor_kwargs
        )
        rhs = make_tensor(
            shape, dtype=dtype, device=device, **op.rhs_make_tensor_kwargs
        )

        lhs_non_contig = lhs[:, 1, ...]
        lhs = lhs_non_contig.contiguous()

        rhs_non_contig = rhs[:, 1, ...]
        rhs = rhs_non_contig.contiguous()

        self.assertTrue(lhs.is_contiguous())
        self.assertTrue(rhs.is_contiguous())

        self.assertFalse(lhs_non_contig.is_contiguous())
        self.assertFalse(rhs_non_contig.is_contiguous())

        expected = op(lhs, rhs)
        actual = op(lhs_non_contig, rhs_non_contig)
        self.assertEqual(expected, actual)

    @ops(binary_ufuncs)
    def test_non_contig_expand(self, device, dtype, op):
        shapes = [(1, 3), (1, 7), (5, 7)]
        for shape in shapes:
            lhs = make_tensor(
                shape, dtype=dtype, device=device, **op.lhs_make_tensor_kwargs
            )
            rhs = make_tensor(
                shape, dtype=dtype, device=device, **op.rhs_make_tensor_kwargs
            )

            lhs_non_contig = lhs.clone().expand(3, -1, -1)
            rhs_non_contig = rhs.clone().expand(3, -1, -1)

            self.assertTrue(lhs.is_contiguous())
            self.assertTrue(rhs.is_contiguous())

            self.assertFalse(lhs_non_contig.is_contiguous())
            self.assertFalse(rhs_non_contig.is_contiguous())

            expected = op(lhs, rhs)
            actual = op(lhs_non_contig, rhs_non_contig)
            for i in range(3):
                self.assertEqual(expected, actual[i])

    @ops(binary_ufuncs)
    def test_contig_size1(self, device, dtype, op):
        shape = (5, 100)
        lhs = make_tensor(
            shape, dtype=dtype, device=device, **op.lhs_make_tensor_kwargs
        )
        rhs = make_tensor(
            shape, dtype=dtype, device=device, **op.rhs_make_tensor_kwargs
        )

        lhs = lhs[:1, :50]
        lhs_alt = torch.empty(lhs.size(), device=device, dtype=dtype)
        lhs_alt.copy_(lhs)

        rhs = rhs[:1, :50]
        rhs_alt = torch.empty(rhs.size(), device=device, dtype=dtype)
        rhs_alt.copy_(rhs)

        self.assertTrue(lhs.is_contiguous())
        self.assertTrue(rhs.is_contiguous())

        self.assertTrue(lhs_alt.is_contiguous())
        self.assertTrue(rhs_alt.is_contiguous())

        expected = op(lhs, rhs)
        actual = op(lhs_alt, rhs_alt)
        self.assertEqual(expected, actual)

    @ops(binary_ufuncs)
    def test_contig_size1_large_dim(self, device, dtype, op):
        shape = (5, 2, 3, 1, 4, 5, 3, 2, 1, 2, 3, 4)
        lhs = make_tensor(
            shape, dtype=dtype, device=device, **op.lhs_make_tensor_kwargs
        )
        rhs = make_tensor(
            shape, dtype=dtype, device=device, **op.rhs_make_tensor_kwargs
        )

        lhs = lhs[:1, :, :, :, :, :, :, :, :, :, :, :]
        lhs_alt = torch.empty(lhs.size(), device=device, dtype=dtype)
        lhs_alt.copy_(lhs)

        rhs = rhs[:1, :, :, :, :, :, :, :, :, :, :, :]
        rhs_alt = torch.empty(rhs.size(), device=device, dtype=dtype)
        rhs_alt.copy_(rhs)

        self.assertTrue(lhs.is_contiguous())
        self.assertTrue(rhs.is_contiguous())

        self.assertTrue(lhs_alt.is_contiguous())
        self.assertTrue(rhs_alt.is_contiguous())

        expected = op(lhs, rhs)
        actual = op(lhs_alt, rhs_alt)
        self.assertEqual(expected, actual)

    @ops(binary_ufuncs)
    def test_batch_vs_slicing(self, device, dtype, op):
        shape = (32, 512)
        lhs = make_tensor(
            shape, dtype=dtype, device=device, **op.lhs_make_tensor_kwargs
        )
        rhs = make_tensor(
            shape, dtype=dtype, device=device, **op.rhs_make_tensor_kwargs
        )

        expected = op(lhs, rhs)

        actual = []
        for idx in range(32):
            actual.append(op(lhs[idx], rhs[idx]))
        actual = torch.stack(actual)

        self.assertEqual(expected, actual)

    # Tests that elementwise binary operators participate in type promotion properly
    # NOTE: because the cross-product of all possible type promotion tests is huge, this
    #   just spot checks some handwritten cases.
    # NOTE: It may be possible to refactor this test into something simpler
    @ops(binary_ufuncs_and_refs, dtypes=OpDTypes.none)
    def test_type_promotion(self, device, op):
        supported_dtypes = op.supported_dtypes(torch.device(device).type)

        make_lhs = partial(
            make_tensor, (5,), device=device, **op.lhs_make_tensor_kwargs
        )
        make_rhs = partial(
            make_tensor, (5,), device=device, **op.rhs_make_tensor_kwargs
        )

        make_rhs_scalar_tensor = partial(
            make_tensor, (), device="cpu", **op.rhs_make_tensor_kwargs
        )

        def _supported(dtypes):
            return all(x in supported_dtypes for x in dtypes)

        # int x int type promotion
        if _supported((torch.int16, torch.int32, torch.int64)):
            lhs_i16 = make_lhs(dtype=torch.int16)
            lhs_i32 = make_lhs(dtype=torch.int32)
            lhs_i64 = make_lhs(dtype=torch.int64)

            rhs_i16 = make_rhs(dtype=torch.int16)
            rhs_i32 = make_rhs(dtype=torch.int32)
            rhs_i64 = make_rhs(dtype=torch.int64)

            if op.promotes_int_to_float:
                default_dtype = torch.get_default_dtype()
                self.assertEqual(op(lhs_i16, rhs_i32).dtype, default_dtype)
                self.assertEqual(
                    op(lhs_i16, rhs_i32),
                    op(lhs_i16.to(default_dtype), rhs_i32.to(default_dtype)),
                )

                self.assertEqual(op(lhs_i32, rhs_i64).dtype, default_dtype)
                self.assertEqual(
                    op(lhs_i32, rhs_i64),
                    op(lhs_i32.to(default_dtype), rhs_i64.to(default_dtype)),
                )
            elif op.always_returns_bool:
                self.assertEqual(op(lhs_i16, rhs_i32).dtype, torch.bool)
                self.assertEqual(op(lhs_i32, rhs_i64).dtype, torch.bool)
            else:  # standard type promotion
                self.assertEqual(op(lhs_i16, rhs_i32).dtype, torch.int32)
                self.assertEqual(
                    op(lhs_i16, rhs_i32), op(lhs_i16.to(torch.int32), rhs_i32)
                )

                self.assertEqual(op(lhs_i32, rhs_i64).dtype, torch.int64)
                self.assertEqual(
                    op(lhs_i32, rhs_i64), op(lhs_i32.to(torch.int64), rhs_i64)
                )

            if op.supports_out:
                if not op.promotes_int_to_float:
                    # Integers can be safely cast to other integer types
                    out = torch.empty_like(lhs_i64)
                    self.assertEqual(op(lhs_i16, rhs_i32, out=out).dtype, torch.int64)
                    self.assertEqual(op(lhs_i16, rhs_i32), out, exact_dtype=False)

                    out = torch.empty_like(lhs_i16)
                    self.assertEqual(op(lhs_i32, rhs_i64, out=out).dtype, torch.int16)
                else:
                    # Float outs cannot be safely cast to integer types
                    with self.assertRaisesRegex(RuntimeError, "can't be cast"):
                        op(lhs_i16, rhs_i32, out=torch.empty_like(lhs_i64))

                if not op.always_returns_bool:
                    # Neither integer nor float outs can be cast to bool
                    with self.assertRaisesRegex(RuntimeError, "can't be cast"):
                        op(
                            lhs_i16,
                            rhs_i32,
                            out=torch.empty_like(lhs_i64, dtype=torch.bool),
                        )

                # All these output types can be cast to any float or complex type
                out = torch.empty_like(lhs_i64, dtype=torch.float16)
                self.assertEqual(op(lhs_i16, rhs_i32, out=out).dtype, torch.float16)

                out = torch.empty_like(lhs_i64, dtype=torch.bfloat16)
                self.assertEqual(op(lhs_i16, rhs_i32, out=out).dtype, torch.bfloat16)

                out = torch.empty_like(lhs_i64, dtype=torch.float32)
                self.assertEqual(op(lhs_i16, rhs_i32, out=out).dtype, torch.float32)
                self.assertEqual(op(lhs_i16, rhs_i32), out, exact_dtype=False)

                out = torch.empty_like(lhs_i64, dtype=torch.complex64)
                self.assertEqual(op(lhs_i16, rhs_i32, out=out).dtype, torch.complex64)
                self.assertEqual(op(lhs_i16, rhs_i32), out, exact_dtype=False)

        # float x float type promotion
        if _supported((torch.float32, torch.float64)):
            lhs_f32 = make_lhs(dtype=torch.float32)
            lhs_f64 = make_lhs(dtype=torch.float64)

            rhs_f32 = make_rhs(dtype=torch.float32)
            rhs_f64 = make_rhs(dtype=torch.float64)

            if op.always_returns_bool:
                self.assertEqual(op(lhs_f32, rhs_f64).dtype, torch.bool)
            else:  # normal float type promotion
                self.assertEqual(op(lhs_f32, rhs_f64).dtype, torch.float64)
                self.assertEqual(
                    op(lhs_f32, rhs_f64), op(lhs_f32.to(torch.float64), rhs_f64)
                )

            if op.supports_out:
                # All these output types can be cast to any float or complex type
                out = torch.empty_like(lhs_f64, dtype=torch.float16)
                self.assertEqual(op(lhs_f32, rhs_f64, out=out).dtype, torch.float16)

                out = torch.empty_like(lhs_f64, dtype=torch.bfloat16)
                self.assertEqual(op(lhs_f32, rhs_f64, out=out).dtype, torch.bfloat16)
                self.assertEqual(op(lhs_f32, rhs_f64), out, exact_dtype=False)

                out = torch.empty_like(lhs_f64, dtype=torch.float32)
                self.assertEqual(op(lhs_f32, rhs_f64, out=out).dtype, torch.float32)
                self.assertEqual(op(lhs_f32, rhs_f64), out, exact_dtype=False)

                out = torch.empty_like(lhs_f64, dtype=torch.complex64)
                self.assertEqual(op(lhs_f32, rhs_f64, out=out).dtype, torch.complex64)
                self.assertEqual(op(lhs_f32, rhs_f64), out, exact_dtype=False)

                if not op.always_returns_bool:
                    # float outs can't be cast to an integer dtype
                    with self.assertRaisesRegex(RuntimeError, "can't be cast"):
                        op(
                            lhs_f32,
                            rhs_f64,
                            out=torch.empty_like(lhs_f64, dtype=torch.int64),
                        )
                else:
                    # bool outs can be cast to an integer dtype
                    out = torch.empty_like(lhs_f64, dtype=torch.int64)
                    self.assertEqual(op(lhs_f32, rhs_f64, out=out).dtype, torch.int64)
                    self.assertEqual(op(lhs_f32, rhs_f64), out, exact_dtype=False)

        # complex x complex type promotion
        if _supported((torch.complex64, torch.complex128)):
            lhs_c64 = make_lhs(dtype=torch.complex64)
            lhs_c128 = make_lhs(dtype=torch.complex128)

            rhs_c64 = make_rhs(dtype=torch.complex64)
            rhs_c128 = make_rhs(dtype=torch.complex128)

            if op.always_returns_bool:
                self.assertEqual(op(lhs_c64, lhs_c128).dtype, torch.bool)
            else:  # normal complex type promotion
                self.assertEqual(op(lhs_c64, rhs_c128).dtype, torch.complex128)
                self.assertEqual(
                    op(lhs_c64, rhs_c128), op(lhs_c64.to(torch.complex128), rhs_c128)
                )

            if op.supports_out:
                # All these output types can be cast to any or complex type
                out = torch.empty_like(lhs_c64, dtype=torch.complex64)

                self.assertEqual(op(lhs_c64, rhs_c128, out=out).dtype, torch.complex64)
                result = op(lhs_c64, rhs_c128)
                self.assertEqual(result, out.to(result.dtype))

                if not op.always_returns_bool:
                    # complex outs can't be cast to float types
                    with self.assertRaisesRegex(RuntimeError, "can't be cast"):
                        op(
                            lhs_c64,
                            rhs_c128,
                            out=torch.empty_like(lhs_c64, dtype=torch.float64),
                        )
                    # complex outs can't be cast to an integer dtype
                    with self.assertRaisesRegex(RuntimeError, "can't be cast"):
                        op(
                            lhs_c64,
                            rhs_c128,
                            out=torch.empty_like(lhs_c64, dtype=torch.int64),
                        )
                else:
                    # bool outs can be cast to a float type
                    out = torch.empty_like(lhs_c64, dtype=torch.float64)
                    self.assertEqual(
                        op(lhs_c64, rhs_c128, out=out).dtype, torch.float64
                    )
                    self.assertEqual(op(lhs_c64, rhs_c128), out, exact_dtype=False)

                    # bool outs can be cast to an integer dtype
                    out = torch.empty_like(lhs_f64, dtype=torch.int64)
                    self.assertEqual(op(lhs_f32, rhs_f64, out=out).dtype, torch.int64)
                    self.assertEqual(op(lhs_f32, rhs_f64), out, exact_dtype=False)

        # int x float type promotion
        # Note: float type is the result dtype
        if _supported((torch.long, torch.float32)):
            lhs_i64 = make_lhs(dtype=torch.int64)
            rhs_f32 = make_rhs(dtype=torch.float32)

            result = op(lhs_i64, rhs_f32)
            expected_dtype = torch.float32 if not op.always_returns_bool else torch.bool
            self.assertEqual(result.dtype, expected_dtype)

        # float x complex type promotion
        # Note: complex type with highest "value type" is the result dtype
        if _supported((torch.float64, torch.complex64)):
            lhs_f64 = make_lhs(dtype=torch.float64)
            rhs_c64 = make_rhs(dtype=torch.complex64)

            result = op(lhs_f64, rhs_c64)
            expected_dtype = (
                torch.complex128 if not op.always_returns_bool else torch.bool
            )
            self.assertEqual(result.dtype, expected_dtype)

        # int x float scalar type promotion
        # Note: default float dtype is the result dtype
        if _supported((torch.int64, torch.float32)) and op.supports_rhs_python_scalar:
            lhs_i64 = make_lhs(dtype=torch.int64)
            rhs_f_scalar = 1.0

            result = op(lhs_i64, rhs_f_scalar)
            expected_dtype = (
                torch.get_default_dtype() if not op.always_returns_bool else torch.bool
            )
            self.assertEqual(result.dtype, expected_dtype)

            # repeats with a scalar float tensor, which should set the dtype
            rhs_f32_scalar_tensor = make_rhs_scalar_tensor(dtype=torch.float32)
            result = op(lhs_i64, rhs_f32_scalar_tensor)
            expected_dtype = torch.float32 if not op.always_returns_bool else torch.bool
            self.assertEqual(result.dtype, expected_dtype)

            # Additional test with double
            if _supported((torch.float64,)):
                rhs_f64_scalar_tensor = make_rhs_scalar_tensor(dtype=torch.float64)
                result = op(lhs_i64, rhs_f64_scalar_tensor)
                expected_dtype = (
                    torch.float64 if not op.always_returns_bool else torch.bool
                )
                self.assertEqual(result.dtype, expected_dtype)

        # float x complex scalar type promotion
        # Note: result dtype is complex with highest "value type" among all tensors
        if (
            _supported((torch.float32, torch.complex64))
            and op.supports_rhs_python_scalar
        ):
            lhs_f32 = make_lhs(dtype=torch.float32)
            rhs_c_scalar = complex(1, 1)

            result = op(lhs_f32, rhs_c_scalar)
            expected_dtype = (
                torch.complex64 if not op.always_returns_bool else torch.bool
            )
            self.assertEqual(result.dtype, expected_dtype)

            # repeats with a scalar complex tensor
            rhs_c64_scalar_tensor = make_rhs_scalar_tensor(dtype=torch.complex64)
            result = op(lhs_f32, rhs_c64_scalar_tensor)
            expected_dtype = (
                torch.complex64 if not op.always_returns_bool else torch.bool
            )
            self.assertEqual(result.dtype, expected_dtype)

            # Additional test with complexdouble
            if _supported((torch.complex128,)):
                rhs_c128_scalar_tensor = make_rhs_scalar_tensor(dtype=torch.complex128)
                result = op(lhs_f32, rhs_c128_scalar_tensor)
                # Value type of 1D+ Tensor (lhs_f32) takes priority over scalar tensor (rhs_c128).
                expected_dtype = (
                    torch.complex64 if not op.always_returns_bool else torch.bool
                )
                self.assertEqual(result.dtype, expected_dtype)

        # float x float scalar tensor
        # Note: result dtype is the type of the float tensor
        if _supported((torch.float32, torch.float64)) and op.supports_rhs_python_scalar:
            lhs_f32 = make_lhs(dtype=torch.float32)
            rhs_f64_scalar_tensor = make_rhs_scalar_tensor(dtype=torch.float64)

            result = op(lhs_f32, rhs_f64_scalar_tensor)
            expected_dtype = torch.float32 if not op.always_returns_bool else torch.bool
            self.assertEqual(result.dtype, expected_dtype)

        # complex x complex scalar tensor
        # Note: result dtype is the type of the complex tensor
        if (
            _supported((torch.complex64, torch.complex128))
            and op.supports_rhs_python_scalar
        ):
            lhs_c64 = make_lhs(dtype=torch.complex64)
            rhs_c128_scalar_tensor = make_rhs_scalar_tensor(dtype=torch.complex128)

            result = op(lhs_c64, rhs_c128_scalar_tensor)
            expected_dtype = (
                torch.complex64 if not op.always_returns_bool else torch.bool
            )
            self.assertEqual(result.dtype, expected_dtype)

        # scalar  x scalar
        # Note: result dtype is default float type
        if op.supports_two_python_scalars and _supported((torch.long, torch.float32)):
            rhs_f_scalar = 2.0
            for lhs in (1, 1.0):
                result = op(lhs, rhs_f_scalar)
                expected_dtype = (
                    torch.get_default_dtype()
                    if not op.always_returns_bool
                    else torch.bool
                )
                self.assertEqual(result.dtype, expected_dtype)

    # TODO: move to error input test
    @ops(binary_ufuncs, allowed_dtypes=(torch.float32,))
    def test_not_broadcastable(self, device, dtype, op):
        for shape_lhs, shape_rhs in (
            ((2,), (3,)),
            ((3, 1), (2, 1)),
            ((1, 3, 2), (3,)),
            ((3, 1, 2), (2, 1, 2)),
        ):
            lhs = make_tensor(
                shape_lhs, device=device, dtype=dtype, **op.lhs_make_tensor_kwargs
            )
            rhs = make_tensor(
                shape_rhs, device=device, dtype=dtype, **op.rhs_make_tensor_kwargs
            )

            try:
                broadcasted_shape = op(lhs, rhs).shape
            except RuntimeError:
                continue

            msg = (
                f"On {device}, torch.{op.name} broadcasts inputs shapes {shape_lhs} and {shape_rhs} into "
                f"{broadcasted_shape}, although they are not broadcastable."
            )
            raise AssertionError(msg)

    def test_add_broadcast_empty(self, device):
        # empty + empty
        self.assertRaises(
            RuntimeError,
            lambda: torch.randn(5, 0, device=device) + torch.randn(0, 5, device=device),
        )
        self.assertEqual(
            torch.randn(5, 0, device=device),
            torch.randn(0, device=device) + torch.randn(5, 0, device=device),
        )
        self.assertEqual(
            torch.randn(5, 0, 0, device=device),
            torch.randn(0, device=device) + torch.randn(5, 0, 1, device=device),
        )

        # scalar + empty
        self.assertEqual(
            torch.randn(5, 0, 6, device=device),
            torch.randn((), device=device) + torch.randn(5, 0, 6, device=device),
        )

        # non-empty, empty
        self.assertEqual(
            torch.randn(0, device=device),
            torch.randn(0, device=device) + torch.randn(1, device=device),
        )
        self.assertEqual(
            torch.randn(0, 7, 0, 6, 5, 0, 7, device=device),
            torch.randn(0, 7, 0, 6, 5, 0, 1, device=device)
            + torch.randn(1, 1, 5, 1, 7, device=device),
        )
        self.assertRaises(
            RuntimeError,
            lambda: torch.randn(7, 0, device=device) + torch.randn(2, 1, device=device),
        )

    def test_addcmul_scalars_as_floats(self, device):
        # zero-dim variables that don't require grad should bind to scalar arguments
        x = torch.tensor(2.0)
        y = torch.tensor(3.0, device=device)
        # 3 + (3 * 3) * 2
        self.assertEqual(y.addcmul(y, y, value=x), 21)

        x = torch.tensor(2.0, requires_grad=True)
        self.assertRaises(Exception, lambda: y.addcmul(y, y, value=x))

    # Tests that the binary operators and, or, and xor (as well as their reflected and inplace versions)
    # work properly (AKA &, ||, ^ and &=, |=, ^=)
    @dtypes(*integral_types_and(torch.bool))
    def test_bitwise_ops(self, device, dtype):
        # Tensor x Tensor and Tensor x Scalar ops
        ops = (
            operator.and_,
            operator.iand,
            operator.or_,
            operator.ior,
            operator.xor,
            operator.ixor,
        )
        inplace_ops = (operator.iand, operator.ior, operator.ixor)
        shapes = ((5,), (15, 15), (500, 500))

        for op, shape in itertools.product(ops, shapes):
            # Tests tensor x tensor case
            a = make_tensor(shape, device=device, dtype=dtype)
            b = make_tensor(shape, device=device, dtype=dtype)
            a_np = a.cpu().clone().numpy()
            b_np = b.cpu().clone().numpy()
            self.assertEqual(op(a, b), op(a_np, b_np))

            # Tests tensor x scalar case
            a = make_tensor(shape, device=device, dtype=dtype)
            b_scalar = make_tensor((), device="cpu", dtype=dtype).item()
            a_np = a.cpu().clone().numpy()
            self.assertEqual(op(a, b_scalar), op(a_np, b_scalar))

            # Tests scalar x tensor case
            a_scalar = make_tensor((), device="cpu", dtype=dtype).item()
            b = make_tensor(shape, device=device, dtype=dtype)
            b_np = b.cpu().clone().numpy()
            self.assertEqual(op(a_scalar, b), op(a_scalar, b_np))

            # Tests scalar x tensor case (for ops which aren't inplace)
            if op in inplace_ops:
                # Tests tensor x tensor case
                a = make_tensor(shape, device=device, dtype=dtype)
                b = make_tensor(shape, device=device, dtype=dtype)
                a_np = a.cpu().clone().numpy()
                b_np = b.cpu().clone().numpy()
                op(a, b)
                op(a_np, b_np)
                self.assertEqual(a, a_np)

                # Tests tensor x scalar case
                a = make_tensor(shape, device=device, dtype=dtype)
                b_scalar = make_tensor((), device="cpu", dtype=dtype).item()
                a_np = a.cpu().clone().numpy()
                op(a, b_scalar)
                op(a_np, b_scalar)
                self.assertEqual(a, a_np)

    def test_inplace_division(self, device):
        t = torch.rand(5, 5, device=device)
        id_before = id(t)
        t /= 2
        id_after = id(t)
        self.assertEqual(id_before, id_after)

    @dtypes(*all_types_and(torch.half, torch.bfloat16))
    def test_div_rounding_modes(self, device, dtype):
        if dtype.is_floating_point:
            low, high = -10.0, 10.0
        else:
            info = torch.iinfo(dtype)
            low, high = info.min, info.max

        a = make_tensor((100,), dtype=dtype, device=device, low=low, high=high)
        b = make_tensor((100,), dtype=dtype, device=device, low=low, high=high)

        # Avoid division by zero so we can test (a / b) * b == a
        if dtype.is_floating_point:
            eps = 0.1
            b[(-eps < b) & (b < eps)] = eps
        else:
            b[b == 0] = 1

        if not dtype.is_floating_point:
            # floor(a / b) * b can be < a, so fixup slightly to avoid underflow
            a = torch.where(a < 0, a + b, a)

        d_true = torch.divide(a, b, rounding_mode=None)
        self.assertTrue(d_true.is_floating_point())
        self.assertEqual(d_true * b, a.to(d_true.dtype))

        d_floor = torch.divide(a, b, rounding_mode="floor")
        if dtype not in (torch.bfloat16, torch.half):
            self.assertEqual(d_floor * b + torch.remainder(a, b), a)
        else:
            self.assertEqual(
                d_floor * b + torch.remainder(a.float(), b.float()),
                a,
                exact_dtype=False,
            )

        d_trunc = torch.divide(a, b, rounding_mode="trunc")
        rounding_unsupported = (
            dtype == torch.half
            and device != "cuda"
            or dtype == torch.bfloat16
            and device != "cpu"
        )
        d_ref = d_true.float() if rounding_unsupported else d_true
        self.assertEqual(d_trunc, d_ref.trunc().to(dtype))

    @dtypes(*floating_types_and(torch.bfloat16, torch.float16))
    def test_floor_div_extremal(self, device, dtype):
        for num, denom, shape in itertools.product(
            [torch.finfo(dtype).max * 0.7],
            [0.5, -0.5, 0.0],
            [(), (32,)],  # Scalar and vectorized
        ):
            a = torch.full(shape, num, dtype=dtype, device=device)
            b = torch.full(shape, denom, dtype=dtype, device=device)

            ref = np.floor_divide(num, denom).item()
            if ref > torch.finfo(dtype).max:
                ref = np.inf
            elif ref < torch.finfo(dtype).min:
                ref = -np.inf
            expect = torch.full(shape, ref, dtype=dtype, device=device)
            actual = torch.div(a, b, rounding_mode="floor")
            self.assertEqual(expect, actual)

    @dtypes(torch.bfloat16, torch.half, torch.float32, torch.float64)
    def test_div_rounding_nonfinite(self, device, dtype):
        # Compare division of special floating point values against NumPy
        num = torch.tensor(
            [1.0, -1.0, 0, 0.1, -0.1, np.pi, -np.pi, np.inf, -np.inf, np.nan],
            dtype=dtype,
            device=device,
        )
        # Divide by zero is tested separately
        denom = num[num != 0]

        a, b = num[None, :].clone(), denom[:, None].clone()

        # Compare bfloat16 against NumPy float
        exact_dtype = dtype != torch.bfloat16
        if exact_dtype:
            an, bn = a.cpu().numpy(), b.cpu().numpy()
        else:
            an, bn = a.float().cpu().numpy(), b.float().cpu().numpy()

        for mode, np_ref in ((None, np.true_divide), ("floor", np.floor_divide)):
            expect = np_ref(an, bn)
            kwargs = dict(rounding_mode=mode) if mode is not None else {}
            with set_default_dtype(torch.double):
                actual = torch.divide(a, b, **kwargs)
            self.assertEqual(
                actual,
                torch.from_numpy(expect),
                exact_device=False,
                exact_dtype=exact_dtype,
            )

        # Compare contiguous (likely vectorized) against non-contiguous (not vectorized)
        a_noncontig = torch.empty([2 * i for i in a.shape], dtype=dtype, device=device)[
            ::2, ::2
        ]
        a_noncontig[:] = a
        b_noncontig = torch.empty([2 * i for i in b.shape], dtype=dtype, device=device)[
            ::2, ::2
        ]
        b_noncontig[:] = b

        for rounding_mode in (None, "trunc", "floor"):
            expect = torch.divide(a_noncontig, b_noncontig, rounding_mode=rounding_mode)
            actual = torch.divide(a, b, rounding_mode=rounding_mode)
            self.assertEqual(actual, expect)

    @dtypes(torch.bfloat16, torch.half, torch.float32, torch.float64)
    def test_divide_by_zero_rounding(self, device, dtype):
        a = torch.tensor(
            [1.0, -1.0, 0, 0.1, -0.1, np.pi, -np.pi, np.inf, -np.inf, np.nan],
            dtype=dtype,
        )
        exact_dtype = dtype != torch.bfloat16
        if exact_dtype:
            an = a.cpu().numpy()
        else:
            an = a.float().cpu().numpy()

        zero = torch.zeros_like(a)

        # NOTE: NumPy's floor_divide rounding changed in 1.20.0 to be consistent with divide
        expect = np.divide(an, 0)
        for rounding_mode in (None, "floor"):
            # CPU scalar
            actual = torch.divide(a, 0, rounding_mode=rounding_mode)
            self.assertEqual(actual, expect, exact_dtype=exact_dtype)
            # Device tensor
            actual = torch.divide(a, zero, rounding_mode=rounding_mode)
            self.assertEqual(actual, expect, exact_dtype=exact_dtype)

    @dtypes(*all_types_and(torch.half))
    def test_div_rounding_numpy(self, device, dtype):
        info = torch.finfo(dtype) if dtype.is_floating_point else torch.iinfo(dtype)
        low, high = info.min, info.max

        # Compare division of random values against NumPy
        a = make_tensor((4096,), dtype=dtype, device=device, low=low, high=high)
        b = make_tensor((4096,), dtype=dtype, device=device, low=low, high=high)

        # Avoid division by zero which raises for integers and, for floats,
        # NumPy 1.20 changed floor_divide to follow IEEE rules for inf/nan
        # after dividing by zero.
        b[b == 0] = 1

        # Compare bfloat16 against NumPy float
        exact_dtype = dtype != torch.bfloat16

        if exact_dtype:
            an, bn = a.cpu().numpy(), b.cpu().numpy()
        else:
            an, bn = a.float().cpu().numpy(), b.float().cpu().numpy()

        for mode, np_ref in (
            (None, np.true_divide),
            ("floor", np.floor_divide),
            ("trunc", lambda a, b: np.trunc(np.true_divide(a, b)).astype(a.dtype)),
        ):
            expect = torch.from_numpy(np_ref(an, bn))

            kwargs = dict(rounding_mode=mode) if mode is not None else {}
            # Contiguous (likely vectorized)
            with set_default_dtype(torch.double):
                actual = torch.divide(a, b, **kwargs)
            self.assertEqual(
                actual, expect, exact_device=False, exact_dtype=exact_dtype
            )

            # Non-contiguous (not vectorized)
            expect = expect[::2]
            with set_default_dtype(torch.double):
                actual = torch.divide(a[::2], b[::2], **kwargs)

            self.assertEqual(
                actual, expect, exact_device=False, exact_dtype=exact_dtype
            )

    @dtypes(*complex_types())
    def test_complex_div_underflow_overflow(self, device, dtype):
        # test to make sure the complex division does not produce underflow or overflow
        # in the intermediate of its calculations
        # NOTE: the calculation still produces an error if the number is greater than
        # finfo.max / 2, but hopefully people realized that it's a dangerous region to work with
        finfo = torch.finfo(dtype)
        nom_lst = [
            complex(finfo.min / 2, finfo.min / 2),
            complex(finfo.max / 2, finfo.max / 2),
            complex(finfo.tiny, finfo.tiny),
            complex(finfo.tiny, 0.0),
            complex(0.0, 0.0),
        ]
        denom_lst = [
            complex(finfo.min / 2, finfo.min / 2),
            complex(finfo.max / 2, finfo.max / 2),
            complex(finfo.tiny, finfo.tiny),
            complex(0.0, finfo.tiny),
            complex(finfo.tiny, finfo.tiny),
        ]
        expected_lst = [
            complex(1.0, 0.0),
            complex(1.0, 0.0),
            complex(1.0, 0.0),
            complex(0.0, -1.0),
            complex(0.0, 0.0),
        ]
        nom = torch.tensor(nom_lst, dtype=dtype, device=device)
        denom = torch.tensor(denom_lst, dtype=dtype, device=device)
        expected = torch.tensor(expected_lst, dtype=dtype, device=device)
        res = nom / denom
        self.assertEqual(res, expected)

    # Tests that trying to add, inplace, a CUDA tensor to a CPU tensor
    #   throws the correct error message
    @onlyCUDA
    def test_cross_device_inplace_error_msg(self, device):
        a = torch.tensor(2.0)
        b = torch.tensor(2.0, device=device)
        with self.assertRaisesRegex(
            RuntimeError, "Expected all tensors to be on the same device"
        ):
            a += b

    # TODO: refactor this test into a more generic one, it's parked here currently
    @onlyNativeDeviceTypes
    def test_out_resize_warning(self, device):
        a = torch.tensor((1, 2, 3), device=device, dtype=torch.float32)
        b = torch.tensor((4, 5, 6), device=device, dtype=torch.float32)

        unary_inputs = (a,)
        binary_inputs = (a, b)
        unary_ops = (torch.ceil, torch.exp)
        binary_ops = (torch.add, torch.sub)
        for op in unary_ops + binary_ops:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                inputs = unary_inputs if op in unary_ops else binary_inputs

                # No warnings
                op(*inputs, out=torch.empty(3, device=device))
                op(*inputs, out=torch.empty(0, device=device))
                self.assertEqual(len(w), 0)

                # Cases that throw warnings
                op(*inputs, out=torch.empty(2, device=device))
                self.assertEqual(len(w), 1)
        # test that multi-d out doesn't trigger segfault
        arg1 = (torch.ones(2, 1, device=device), torch.ones(1, device=device))
        arg2 = (torch.ones(2, device=device), torch.ones(1, 1, device=device))
        outs = (
            torch.ones(2, 1, 1, 1, device=device),
            torch.ones(2, 2, 2, 2, device=device),
        )

        for a1, a2, o in zip(arg1, arg2, outs):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                torch.mul(a1, a2, out=o)
                self.assertEqual(len(w), 1)

    # Verifies that the inplace dunders (like idiv) actually are in place
    @expectedFailureMeta  # UserWarning not triggered
    @onlyNativeDeviceTypes
    def test_inplace_dunders(self, device):
        t = torch.randn((1,), device=device)
        expected = t.data_ptr()
        t += 1
        t -= 1
        t *= 1
        t /= 1
        t **= 1
        t //= 1
        t %= 1
        self.assertEqual(expected, t.data_ptr())

    def check_internal_mem_overlap(
        self, inplace_op, num_inputs, dtype, device, expected_failure=False
    ):
        if isinstance(inplace_op, str):
            inplace_op = getattr(torch.Tensor, inplace_op)
        input = torch.randn(1, dtype=dtype, device=device).expand(3, 3)
        inputs = [input] + [torch.randn_like(input) for i in range(num_inputs - 1)]
        if not expected_failure:
            with self.assertRaisesRegex(RuntimeError, "single memory location"):
                inplace_op(*inputs)
        else:
            with self.assertRaises(AssertionError):
                with self.assertRaisesRegex(RuntimeError, "single memory location"):
                    inplace_op(*inputs)

    def unary_check_input_output_mem_overlap(
        self, data, sz, op, expected_failure=False
    ):
        def _test(op, output, input):
            output_exp = torch.empty_like(output)
            op(input, out=output_exp)
            self.assertEqual(op(input, out=output), output_exp, msg=op.__name__)

        # output is identical to input:
        _test(op, output=data[0:sz], input=data[0:sz])
        # output and input are independent:
        _test(op, output=data[0:sz], input=data[sz : 2 * sz])
        # output partially overlaps with input:
        if not expected_failure:
            with self.assertRaisesRegex(RuntimeError, "unsupported operation"):
                _test(op, data[0:sz], data[1 : sz + 1])
        else:
            with self.assertRaises(AssertionError):
                with self.assertRaisesRegex(RuntimeError, "unsupported operation"):
                    _test(op, data[0:sz], data[1 : sz + 1])

    def binary_check_input_output_mem_overlap(self, op, device, expected_failure=False):
        sz = 3
        data = torch.randn(2 * sz, device=device)
        other = torch.randn(sz, device=device)

        self.unary_check_input_output_mem_overlap(
            data,
            sz,
            lambda input, out: op(other, input, out=out),
            expected_failure=expected_failure,
        )

        self.unary_check_input_output_mem_overlap(
            data,
            sz,
            lambda input, out: op(input, other, out=out),
            expected_failure=expected_failure,
        )

    # https://github.com/pytorch/pytorch/issues/126474
    @xfailIfTorchDynamo
    @dtypes(torch.double)
    def test_binary_op_mem_overlap(self, device, dtype):
        ops = [
            ("add", True, True, "cpu"),
            ("add", True, True, "cuda"),
            ("mul", True, True, "cpu"),
            ("mul", True, True, "cuda"),
            ("sub", True, True, "cpu"),
            ("sub", True, True, "cuda"),
            ("div", True, True, "cpu"),
            ("div", True, True, "cuda"),
            ("pow", True, True, "cpu"),
            ("pow", True, True, "cuda"),
            ("fmod", True, True, "cpu"),
            ("fmod", True, True, "cuda"),
            ("atan2", True, True, "cpu"),
            ("atan2", True, True, "cuda"),
            ("hypot", True, True, "cpu"),
            ("hypot", True, True, "cuda"),
            ("igamma", True, True, "cpu"),
            ("igamma", True, True, "cuda"),
            ("igammac", True, True, "cpu"),
            ("igammac", True, True, "cuda"),
            ("nextafter", True, True, "cpu"),
            ("nextafter", True, True, "cuda"),
            ("le", True, True, "cpu"),
            ("le", True, True, "cuda"),
            ("lt", True, True, "cpu"),
            ("lt", True, True, "cuda"),
            ("ge", True, True, "cpu"),
            ("ge", True, True, "cuda"),
            ("gt", True, True, "cpu"),
            ("gt", True, True, "cuda"),
            ("eq", True, True, "cpu"),
            ("eq", True, True, "cuda"),
            ("ne", True, True, "cpu"),
            ("ne", True, True, "cuda"),
            ("logical_and", True, True, "cpu"),
            ("logical_and", True, True, "cuda"),
            ("logical_or", True, True, "cpu"),
            ("logical_or", True, True, "cuda"),
            ("logical_xor", True, True, "cpu"),
            ("logical_xor", True, True, "cuda"),
        ]

        for (
            fn,
            has_input_output_mem_overlap_check,
            has_internal_mem_overlap_check,
            dev,
        ) in ops:
            if dev != device:
                continue
            out_op = getattr(torch, fn)
            inplace_op = getattr(torch.Tensor, fn + "_")
            self.check_internal_mem_overlap(
                inplace_op,
                2,
                dtype,
                device,
                expected_failure=not has_internal_mem_overlap_check,
            )

            self.binary_check_input_output_mem_overlap(
                out_op, device, expected_failure=not has_input_output_mem_overlap_check
            )

    def _do_pow_for_exponents(self, m1, exponents, pow_fn, atol):
        for num in exponents:
            if (
                isinstance(num, int)
                and num < 0
                and not m1.is_floating_point()
                and not m1.is_complex()
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"Integers to negative integer powers are not allowed\.",
                ):
                    torch.pow(m1[4], num)
            else:
                # base - tensor, exponent - number
                # contiguous
                res1 = torch.pow(m1[4], num)
                res2 = res1.clone().zero_()
                # `math.pow` has issues with complex exponentiation so we need to resort to normal `pow`.
                for i in range(res2.size(0)):
                    res2[i] = pow_fn(m1[4][i], num)
                rtol = 0 if atol is not None else None
                self.assertEqual(res1, res2, atol=atol, rtol=rtol)

                # non-contiguous
                res1 = torch.pow(m1[:, 4], num)
                res2 = res1.clone().zero_()
                for i in range(res2.size(0)):
                    res2[i] = pow_fn(m1[i, 4], num)
                self.assertEqual(res1, res2, atol=atol, rtol=rtol)

                # scalar ** tensor to enforce correct handling of dtypes for __rpow__().
                expected_dtype = torch.result_type(num, m1)
                res1 = num ** m1[4]
                res2 = (
                    torch.tensor(num, dtype=expected_dtype, device=m1.device) ** m1[4]
                )
                self.assertEqual(res1, res2)
                self.assertEqual(res1.dtype, expected_dtype)

    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16))
    def test_pow(self, device, dtype):
        m1 = torch.empty(0, dtype=dtype, device=device)
        if m1.is_floating_point() or m1.is_complex():
            m1 = (
                make_tensor((100, 100), low=0, high=1, dtype=dtype, device=device) + 0.5
            )
        else:
            # math.pow will overflow and throw exceptions for large integers
            range_high = 4 if dtype in (torch.int8, torch.uint8) else 10
            m1 = make_tensor(
                (100, 100), low=1, high=range_high, dtype=dtype, device=device
            )

        exponents = [-2.8, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 3.3, True, False]
        complex_exponents = [
            -2.5j,
            -1.0j,
            0j,
            1.0j,
            2.5j,
            1.0 + 1.0j,
            -1.0 - 1.5j,
            3.3j,
        ]
        if m1.is_complex():
            self._do_pow_for_exponents(m1, exponents + complex_exponents, pow, 10e-4)
        else:
            self._do_pow_for_exponents(m1, exponents, math.pow, None)
            will_raise_error = (
                dtype is torch.half and torch.device(device).type == "cpu"
            )
            if will_raise_error:
                # On CPU,
                # Half Tensor with complex exponents leads to computation dtype
                # of ComplexHalf for which this ops is not supported yet
                with self.assertRaisesRegex(
                    RuntimeError, "not implemented for 'ComplexHalf'"
                ):
                    self._do_pow_for_exponents(m1, complex_exponents, pow, 10e-4)
            else:
                self._do_pow_for_exponents(m1, complex_exponents, pow, 10e-4)

        # base - number, exponent - tensor
        # contiguous
        res1 = torch.pow(3, m1[4])
        res2 = res1.clone().zero_()
        for i in range(res2.size(0)):
            res2[i] = pow(3, m1[4, i])
        self.assertEqual(res1, res2)

        # non-contiguous
        res1 = torch.pow(3, m1[:, 4])
        res2 = res1.clone().zero_()
        for i in range(res2.size(0)):
            res2[i] = pow(3, m1[i][4])
        self.assertEqual(res1, res2)

    # TODO: refactor all these tests using opinfos properly
    def _test_pow(self, base, exponent, np_exponent=None):
        if np_exponent is None:
            np_exponent = exponent

        def to_np(value):
            if isinstance(value, torch.Tensor):
                return value.cpu().numpy()
            return value

        try:
            np_res = np.power(to_np(base), to_np(np_exponent))
            expected = (
                torch.from_numpy(np_res)
                if isinstance(np_res, np.ndarray)
                else torch.tensor(np_res, dtype=base.dtype)
            )
        except ValueError as e:
            err_msg = "Integers to negative integer powers are not allowed."
            self.assertEqual(str(e), err_msg)
            out = torch.empty_like(base)
            test_cases = [
                lambda: base.pow(exponent),
                lambda: base.pow_(exponent),
                lambda: torch.pow(base, exponent),
                lambda: torch.pow(base, exponent, out=out),
            ]
            for test_case in test_cases:
                self.assertRaisesRegex(RuntimeError, err_msg, test_case)
        else:
            if isinstance(base, torch.Tensor):
                actual = base.pow(exponent)
                self.assertEqual(actual, expected.to(actual))
                actual = base.clone()
                # When base is a 0-dim cpu tensor and exp is a cuda tensor, we exp `pow` to work but `pow_` to fail, since
                # `pow` will try to create the output tensor on a cuda device, but `pow_` needs to use the cpu tensor as the output
                if (
                    isinstance(exponent, torch.Tensor)
                    and base.dim() == 0
                    and base.device.type == "cpu"
                    and exponent.device.type == "cuda"
                ):
                    regex = "Expected all tensors to be on the same device, but found at least two devices, cuda.* and cpu!"
                    self.assertRaisesRegex(RuntimeError, regex, base.pow_, exponent)
                elif torch.can_cast(torch.result_type(base, exponent), base.dtype):
                    actual2 = actual.pow_(exponent)
                    self.assertEqual(actual, expected)
                    self.assertEqual(actual2, expected)
                else:
                    self.assertRaisesRegex(
                        RuntimeError,
                        r"result type \w+ can't be cast to the desired output type \w+",
                        lambda: actual.pow_(exponent),
                    )

            actual = torch.pow(base, exponent)
            self.assertEqual(actual, expected.to(actual))

            actual2 = torch.pow(base, exponent, out=actual)
            self.assertEqual(actual, expected.to(actual))
            self.assertEqual(actual2, expected.to(actual))

    # We can potentially merge this into OpInfo, but one blocker is that the
    # first input must be a scalar. It is not as simple as just wrapping this in
    # a lambada that switches the inputs, because we also want to test samples inputs
    # where the second input is a scalar. The wrapper would need some more logic.
    def test_pow_scalar_base(self, device):
        a = (
            torch.arange(1, 13, dtype=torch.double, device=device)
            .view(3, 4)
            .requires_grad_()
        )
        gradcheck(lambda a: torch.pow(2, a), (a,))

    # Tests pow() for integral, floating-type tensors, with integral, floating-type
    # exponents (tensor or scalar), respectively. noncontiguous tensors are also tested.
    def test_int_and_float_pow(self, device):
        def _test_int_and_float_pow(dt, low, high, dev):
            test_cases = (
                ((4, 4), 0, (4, 1)),
                ((3, 1), 4, (3, 1)),
                ((2,), 4, (1,)),
                ((1,), 2, ()),
                ((513, 513), 4, (513,)),
                ((5, 5, 5), 5, (5,)),
                ((), 2, ()),
            )
            for base_shape, exp_scalar, exp_shape in test_cases:
                base_tensor = make_tensor(
                    base_shape, dtype=dt, device=dev, low=low, high=high
                )
                # int tensors don't take negative exponents
                if dt in [
                    torch.uint8,
                    torch.int8,
                    torch.int16,
                    torch.int32,
                    torch.int64,
                ]:
                    exp_tensor = make_tensor(
                        exp_shape, dtype=dt, device=dev, low=0, high=high
                    )
                else:
                    exp_tensor = make_tensor(
                        exp_shape, dtype=dt, device=dev, low=low, high=high
                    )
                self._test_pow(base_tensor, exp_scalar)
                self._test_pow(base_tensor, exp_tensor)
                # test non-contiguous tensors as well
                base_tensor = make_tensor(
                    base_shape,
                    dtype=dt,
                    device=dev,
                    low=low,
                    high=high,
                    noncontiguous=True,
                )
                if dt in [
                    torch.uint8,
                    torch.int8,
                    torch.int16,
                    torch.int32,
                    torch.int64,
                ]:
                    exp_tensor = make_tensor(
                        exp_shape,
                        dtype=dt,
                        device=dev,
                        low=0,
                        high=high,
                        noncontiguous=True,
                    )
                else:
                    exp_tensor = make_tensor(
                        exp_shape,
                        dtype=dt,
                        device=dev,
                        low=low,
                        high=high,
                        noncontiguous=True,
                    )
                self._test_pow(base_tensor, exp_scalar)
                self._test_pow(base_tensor, exp_tensor)

        _test_int_and_float_pow(torch.int8, -2, 2, device)
        _test_int_and_float_pow(torch.uint8, 0, 3, device)
        _test_int_and_float_pow(torch.int16, -5, 5, device)
        _test_int_and_float_pow(torch.int64, -10, 10, device)
        _test_int_and_float_pow(torch.int32, -10, 10, device)
        _test_int_and_float_pow(torch.float16, 0.0, 5.0, device)
        _test_int_and_float_pow(torch.float32, 0.0, 10.0, device)
        _test_int_and_float_pow(torch.float64, 0.0, 10.0, device)
        # pow's output would have some NaNs as well
        _test_int_and_float_pow(torch.float32, -10.0, 10.0, device)
        _test_int_and_float_pow(torch.float64, -10.0, 10.0, device)

    # Tests that a Runtime error occurs when a base tensor cannot be resized
    # by pow's inplace variant due to PyTorch's broadcasting semantics.
    def test_pow_inplace_resizing_exception(self, device):
        test_cases = (
            ((), (3,)),
            ((2,), (2, 1)),
            ((2, 1), (2, 2)),
            ((2, 2), (2, 1, 1)),
        )
        test_inputs = [
            (
                make_tensor(
                    base_size, dtype=torch.float64, device=device, high=10.0, low=0.0
                ),
                make_tensor(
                    exp_size, dtype=torch.float64, device=device, high=10.0, low=0.0
                ),
            )
            for base_size, exp_size in test_cases
        ]
        for base, exponent in test_inputs:
            regex = "doesn't match the broadcast shape"
            self.assertRaisesRegex(RuntimeError, regex, base.pow_, exponent)

    def test_int_tensor_pow_neg_ints(self, device):
        ints = [
            torch.iinfo(torch.int32).min,
            -3,
            -2,
            -1,
            0,
            1,
            2,
            3,
            torch.iinfo(torch.int32).max,
        ]
        neg_ints = [torch.iinfo(torch.int32).min, -3, -2, -1]
        tensor = torch.tensor(ints, dtype=torch.int32, device=device)
        for pow in neg_ints:
            self._test_pow(tensor, pow)

    def test_long_tensor_pow_floats(self, device):
        ints = [0, 1, 23, 4567]
        floats = [0.0, 1 / 3, 1 / 2, 1.0, 3 / 2, 2.0]
        tensor = torch.tensor(ints, dtype=torch.int64, device=device)
        for pow in floats:
            self._test_pow(tensor, pow)

    @dtypes(*[torch.float32, torch.float64])
    def test_float_scalar_pow_float_tensor(self, device, dtype):
        floats = [2.0, -3 / 2, -1.0, -1 / 2, -1 / 3, 0.0, 1 / 3, 1 / 2, 1.0, 3 / 2, 2.0]
        exponent_shapes = (
            (1,),
            (2, 2),
            (2, 1),
            (2, 2, 2),
        )
        tensors = [
            make_tensor(shape, dtype=dtype, device=device, low=0)
            for shape in exponent_shapes
        ]
        floats_tensor = torch.tensor(floats, dtype=dtype, device=device)
        for base in floats:
            self._test_pow(base, floats_tensor)
            for tensor in tensors:
                self._test_pow(base, tensor)

    @onlyCUDA
    def test_cuda_tensor_pow_scalar_tensor(self, device):
        cuda_tensors = [
            torch.randn((3, 3), device=device),
            torch.tensor(3.0, device=device),
        ]
        scalar_tensors = [
            torch.tensor(5.0, device="cpu"),
            torch.tensor(-3),
            torch.tensor(1),
        ]
        for base, exp in product(cuda_tensors, scalar_tensors):
            self._test_pow(base, exp)

    @onlyCUDA
    def test_cpu_tensor_pow_cuda_scalar_tensor(self, device):
        cuda_tensors = [
            torch.tensor(5.0, device="cuda"),
            torch.tensor(-3, device="cuda"),
        ]
        for exp in cuda_tensors:
            base = torch.randn((3, 3), device="cpu")
            regex = "Expected all tensors to be on the same device, but found at least two devices, cuda.* and cpu!"
            self.assertRaisesRegex(RuntimeError, regex, torch.pow, base, exp)
        for exp in cuda_tensors:
            # Binary ops with a cpu + cuda tensor are allowed if the cpu tensor has 0 dimension
            base = torch.tensor(3.0, device="cpu")
            self._test_pow(base, exp)

    @onlyCUDA
    @dtypes(torch.complex64, torch.complex128)
    def test_pow_cuda_complex_extremal_failing(self, device, dtype):
        t = torch.tensor(complex(-1.0, float("inf")), dtype=dtype, device=device)
        with self.assertRaises(AssertionError):
            cuda_out = t.pow(2)
            cpu_out = t.cpu().pow(2)
            self.assertEqual(cpu_out, cuda_out)

    @skipIfTorchDynamo()
    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half))
    def test_complex_scalar_pow_tensor(self, device, dtype):
        complexes = [0.5j, 1.0 + 1.0j, -1.5j, 2.2 - 1.6j, 1 + 0j]
        first_exp = make_tensor((100,), dtype=dtype, device=device, low=-2, high=2)
        second_exp = make_tensor(
            (100,), dtype=dtype, device=device, low=-2, high=2, noncontiguous=True
        )
        first_exp[0] = first_exp[10] = first_exp[20] = 0
        second_exp[0] = second_exp[10] = second_exp[20] = 0
        for base in complexes:
            # On CPU,
            # Half Tensor with complex base leads to computation dtype
            # of ComplexHalf for which this ops is not supported yet
            # NOTE: pow has fast-path when base is 1 which supports
            # ComplexHalf
            will_raise_error = (
                torch.device(device).type == "cpu"
                and dtype is torch.half
                and base != (1 + 0j)
            )
            if will_raise_error:
                with self.assertRaisesRegex(
                    RuntimeError, "not implemented for 'ComplexHalf'"
                ):
                    self._test_pow(base, first_exp)
                    self._test_pow(base, second_exp)
            else:
                self._test_pow(base, first_exp)
                self._test_pow(base, second_exp)

    @onlyNativeDeviceTypes
    @skipMeta
    def test_pow_scalar_type_promotion(self, device):
        # Test against a scalar and non-scalar input
        inputs = [17, [17]]
        for input in inputs:
            # We expect the computation to be performed in uint8 (overflowing to 0), and then cast to int64
            input_tensor_uint8 = torch.tensor(input, dtype=torch.uint8, device=device)
            out_uint8_computation = torch.pow(
                2,
                input_tensor_uint8,
                out=torch.tensor(0, dtype=torch.int64, device=device),
            )

            # Computation should run in int64, and not overflow
            input_tensor_int64 = torch.tensor(input, dtype=torch.int64, device=device)
            out_int64_computation = torch.pow(
                2,
                input_tensor_int64,
                out=torch.tensor(0, dtype=torch.int64, device=device),
            )

            self.assertNotEqual(out_uint8_computation, out_int64_computation)
            self.assertEqual(
                out_uint8_computation.to(dtype=torch.uint8),
                out_int64_computation.to(dtype=torch.uint8),
            )

    def test_tensor_pow_tensor(self, device):
        def rotate(l, n):
            return l[-n:] + l[:-n]

        def test_tensor_pow_tensor(values, torch_type, numpy_type):
            vals_tensor = torch.tensor(values, dtype=torch_type, device=device)
            for i in range(len(values)):
                pows = rotate(values, i)
                pows_tensor = torch.tensor(pows, dtype=torch_type, device=device)
                self._test_pow(vals_tensor, pows_tensor)

        ints = [0, 1, 2, 3]
        test_tensor_pow_tensor(ints, torch.uint8, np.uint8)
        test_tensor_pow_tensor(ints, torch.int8, np.int8)
        test_tensor_pow_tensor(ints, torch.int16, np.int16)
        test_tensor_pow_tensor(ints, torch.int32, np.int32)
        test_tensor_pow_tensor(ints, torch.int64, np.int64)

        floats = [-3.0, -2.0, -1.0, -1 / 2, -1 / 3, 0.0, 1 / 3, 1 / 2, 1.0, 2.0, 3.0]
        test_tensor_pow_tensor(floats, torch.float16, np.float16)
        test_tensor_pow_tensor(floats, torch.float32, np.float32)
        test_tensor_pow_tensor(floats, torch.float64, np.float64)

    def test_logical_xor_with_nontrivial_alignment(self, device):
        # test tensor that is not aligned to multiple of 16 bytes
        size = 128
        a = torch.randn(size, device=device) > 0
        b = torch.randn(size, device=device) > 0
        c = torch.randn(size, device=device) > 0
        non_trivial_alignment = [1, 2, 4, 8, 15]
        for i in non_trivial_alignment:
            for j in non_trivial_alignment:
                for k in non_trivial_alignment:
                    a_ = a[i : 100 + i]
                    b_ = b[j : 100 + j]
                    c_ = c[k : 100 + k]
                    torch.logical_xor(a_, b_, out=c_)
                    for x, y, z in zip(a_.tolist(), b_.tolist(), c_.tolist()):
                        self.assertEqual(x ^ y, z)

    @dtypes(torch.float)
    def test_add_with_tail(self, device, dtype):
        # test tensor where there is a tail which is not a multiple
        # of GPU warp size
        for tail_size in [1, 63, 67, 130]:
            size = 4096 + tail_size
            a = torch.randn(size, device=device, dtype=dtype)
            b = torch.randn(size, device=device, dtype=dtype)
            c = a + b
            for x, y, z in zip(a.tolist(), b.tolist(), c.tolist()):
                self.assertEqual(x + y, z)

    # Tests that CUDA tensors on different devices cannot be used in the same
    # binary operation, and that CUDA "scalars" cannot be used in the same
    # binary operation as non-scalar CPU tensors.
    @deviceCountAtLeast(2)
    @onlyCUDA
    def test_cross_device_binary_ops(self, devices):
        vals = (1.0, (2.0,))
        cpu_tensor = torch.randn(2, 2)

        def do_test(op, a, b):
            with self.assertRaisesRegex(RuntimeError, "Expected all tensors.+"):
                op(a, b)
            with self.assertRaisesRegex(RuntimeError, "Expected all tensors.+"):
                op(b, a)
            with self.assertRaisesRegex(RuntimeError, "Expected all tensors.+"):
                op(a, cpu_tensor)
            with self.assertRaisesRegex(RuntimeError, "Expected all tensors.+"):
                op(cpu_tensor, a)

        for op in (
            operator.add,
            torch.add,
            operator.sub,
            torch.sub,
            operator.mul,
            torch.mul,
            operator.truediv,
            torch.true_divide,
            operator.floordiv,
            torch.floor_divide,
        ):
            for a, b in product(vals, vals):
                a = torch.tensor(a, device=devices[0])
                b = torch.tensor(b, device=devices[1])

            do_test(op, a, b)

    # This test ensures that a scalar Tensor can be safely used
    # in a binary operation in conjunction with a Tensor on all
    # available CUDA devices
    @deviceCountAtLeast(2)
    @onlyCUDA
    def test_binary_op_scalar_device_unspecified(self, devices):
        scalar_val = torch.tensor(1.0)
        for default_device in devices:
            with torch.cuda.device(default_device):
                for device in devices:
                    device_obj = torch.device(device)
                    x = torch.rand(3, device=device)
                    y0 = x * scalar_val
                    self.assertEqual(y0.device, device_obj)
                    y1 = scalar_val * x
                    self.assertEqual(y1.device, device_obj)
                    self.assertEqual(y0, y1)

    def test_div_and_floordiv_vs_python(self, device):
        # Tests torch division ops which can handle both arguments being
        #   scalars.
        def _scalar_helper(python_op, torch_op):
            for a, b in product(range(-10, 10), range(-10, 10)):
                for op in (lambda x: x * 0.5, lambda x: math.floor(x)):
                    a = op(a)
                    b = op(b)

                    # Skips zero divisors
                    if b == 0:
                        continue

                    expected = python_op(a, b)

                    for op in (operator.truediv, torch.true_divide):
                        actual_scalar = torch_op(a, b)

                        a_t = torch.tensor(a, device=device)
                        b_t = torch.tensor(b, device=device)

                        actual_tensor = torch_op(a_t, b_t)
                        actual_first_tensor = torch_op(a_t, b)
                        actual_second_tensor = torch_op(a, b_t)

                        self.assertEqual(actual_scalar, expected)
                        self.assertEqual(actual_tensor.item(), expected)
                        self.assertEqual(actual_first_tensor, actual_tensor)
                        self.assertEqual(actual_second_tensor, actual_tensor)

        _scalar_helper(operator.truediv, operator.truediv)
        _scalar_helper(operator.truediv, torch.true_divide)
        _scalar_helper(lambda a, b: math.floor(a / b), operator.floordiv)
        _scalar_helper(lambda a, b: math.floor(a / b), torch.floor_divide)

    @onlyNativeDeviceTypes
    @skipIfTorchDynamo("Not a suitable test for TorchDynamo")
    def test_div_and_floordiv_script_vs_python(self, device):
        # Creates jitted functions of two tensors
        def _wrapped_div(a, b):
            return a / b

        def _wrapped_floordiv(a, b):
            return a // b

        scripted_div = torch.jit.script(_wrapped_div)
        scripted_floordiv = torch.jit.script(_wrapped_floordiv)
        for a, b in product(range(-10, 10), range(-10, 10)):
            for op in (lambda x: x * 0.5, lambda x: math.floor(x)):
                a = op(a)
                b = op(b)

                # Skips zero divisors
                if b == 0:
                    continue

                expected_div = a / b
                expected_floordiv = math.floor(a / b)
                a_t = torch.tensor(a, device=device)
                b_t = torch.tensor(b, device=device)

                self.assertEqual(scripted_div(a_t, b_t), expected_div)
                self.assertEqual(scripted_floordiv(a_t, b_t), expected_floordiv)

        # Creates jitted functions of one tensor
        def _wrapped_div_scalar(a):
            return a / 5

        # NOTE: the JIT implements division as torch.reciprocal(a) * 5
        def _wrapped_rdiv_scalar(a):
            return 5 / a

        def _wrapped_floordiv_scalar(a):
            return a // 5

        # NOTE: this fails if the input is not an integer tensor
        # See https://github.com/pytorch/pytorch/issues/45199
        def _wrapped_rfloordiv_scalar(a):
            return 5 // a

        scripted_div_scalar = torch.jit.script(_wrapped_div_scalar)
        scripted_rdiv_scalar = torch.jit.script(_wrapped_rdiv_scalar)
        scripted_floordiv_scalar = torch.jit.script(_wrapped_floordiv_scalar)
        scripted_rfloordiv_scalar = torch.jit.script(_wrapped_rfloordiv_scalar)

        for a in range(-10, 10):
            for op in (lambda x: x * 0.5, lambda x: math.floor(x)):
                a = op(a)

                a_t = torch.tensor(a, device=device)

                self.assertEqual(a / 5, scripted_div_scalar(a_t))

                # Skips zero divisors
                if a == 0:
                    continue

                self.assertEqual(5 / a, scripted_rdiv_scalar(a_t))

                # Handles Issue 45199 (see comment above)
                if a_t.is_floating_point():
                    with self.assertRaises(RuntimeError):
                        scripted_rfloordiv_scalar(a_t)
                else:
                    # This should emit a UserWarning, why doesn't it?
                    # See issue gh-52387
                    self.assertEqual(5 // a, scripted_rfloordiv_scalar(a_t))

    @onlyNativeDeviceTypes
    @skipIfTorchDynamo("Not a suitable test for TorchDynamo")
    def test_idiv_and_ifloordiv_vs_python(self, device):
        def _wrapped_idiv_tensor(a, b):
            a /= b
            return a

        def _wrapped_idiv_scalar(a):
            a /= 5
            return a

        def _wrapped_true_divide__tensor(a, b):
            a.true_divide_(b)
            return a

        def _wrapped_true_divide__scalar(a):
            a.true_divide_(5)
            return a

        def _wrapped_floor_divide__tensor(a, b):
            a.floor_divide_(b)
            return a

        def _wrapped_floor_divide__scalar(a):
            a.floor_divide_(5)
            return a

        # The following functions are unsupported by the JIT
        def _wrapped_ifloordiv_tensor(a, b):
            a //= b
            return a

        def _wrapped_ifloordiv_scalar(a):
            a //= 5
            return a

        with self.assertRaises(torch.jit.frontend.NotSupportedError):
            scripted_ifloordiv_tensor = torch.jit.script(_wrapped_ifloordiv_tensor)

        with self.assertRaises(torch.jit.frontend.NotSupportedError):
            scripted_ifloordiv_scalar = torch.jit.script(_wrapped_ifloordiv_scalar)

        scripted_idiv_tensor = torch.jit.script(_wrapped_idiv_tensor)
        scripted_idiv_scalar = torch.jit.script(_wrapped_idiv_scalar)
        scripted_true_divide__tensor = torch.jit.script(_wrapped_true_divide__tensor)
        scripted_true_divide__scalar = torch.jit.script(_wrapped_true_divide__scalar)
        scripted_floor_divide__tensor = torch.jit.script(_wrapped_floor_divide__tensor)
        scripted_floor_divide__scalar = torch.jit.script(_wrapped_floor_divide__scalar)

        for a, b in product(range(-10, 10), range(-10, 10)):
            for op in (lambda x: x * 0.5, lambda x: math.floor(x)):
                a = op(a)
                b = op(b)

                # Skips zero divisors
                if b == 0:
                    continue

                expected_idiv = a / b
                expected_ifloordiv = a // b

                a_t = torch.tensor(a, device=device)
                b_t = torch.tensor(b, device=device)

                if a_t.is_floating_point():
                    tmp0 = a_t.clone()
                    tmp0 /= b

                    tmp1 = a_t.clone()
                    tmp1 /= b_t

                    self.assertEqual(tmp0.item(), expected_idiv)
                    self.assertEqual(tmp1.item(), expected_idiv)
                    self.assertEqual(
                        scripted_true_divide__tensor(a_t.clone(), b_t).item(),
                        expected_idiv,
                    )
                    self.assertEqual(
                        scripted_true_divide__scalar(a_t.clone()).item(), a / 5
                    )
                else:
                    tmp = a_t.clone()
                    with self.assertRaises(RuntimeError):
                        tmp /= b
                    with self.assertRaises(RuntimeError):
                        tmp /= b_t
                    with self.assertRaises(RuntimeError):
                        scripted_true_divide__tensor(tmp, b_t)
                    with self.assertRaises(RuntimeError):
                        scripted_true_divide__scalar(tmp)

                if not a_t.is_floating_point() and b_t.is_floating_point():
                    # Inplace modification fails because a float tensor is required
                    #   if the divisor is a float tensor
                    a_t.clone().floor_divide_(b_t)
                    scripted_floor_divide__tensor(a_t.clone(), b_t)
                    tmp = a_t.clone()
                    tmp //= b_t
                else:
                    # Inplace modification is OK when both or neither tensor is
                    #   a float tensor
                    self.assertEqual(
                        a_t.clone().floor_divide_(b_t).item(), expected_ifloordiv
                    )
                    self.assertEqual(
                        scripted_floor_divide__tensor(a_t.clone(), b_t).item(),
                        expected_ifloordiv,
                    )
                    tmp = a_t.clone()
                    tmp //= b_t
                    self.assertEqual(tmp.item(), expected_ifloordiv)

                self.assertEqual(scripted_floor_divide__scalar(a_t), math.floor(a / 5))

    # Tests binary op equivalence with Python builtin ops
    # Also tests that reverse operations are equivalent to forward ops
    # NOTE: division ops are tested separately above
    def test_binary_ops_with_scalars(self, device):
        for python_op, torch_op in (
            (operator.add, torch.add),
            (operator.sub, torch.sub),
            (operator.mul, torch.mul),
            (operator.truediv, torch.div),
        ):
            for a, b in product(range(-10, 10), range(-10, 10)):
                for op in (lambda x: x * 0.5, lambda x: math.floor(x)):
                    a = op(a)
                    b = op(b)

                    # Skips zero divisors
                    if b == 0 or a == 0:
                        continue

                    a_tensor = torch.tensor(a, device=device)
                    b_tensor = torch.tensor(b, device=device)
                    a_tensor_cpu = a_tensor.cpu()
                    b_tensor_cpu = b_tensor.cpu()
                    vals = (a, b, a_tensor, b_tensor, a_tensor_cpu, b_tensor_cpu)

                    for args in product(vals, vals):
                        first, second = args

                        first_scalar = (
                            first
                            if not isinstance(first, torch.Tensor)
                            else first.item()
                        )
                        second_scalar = (
                            second
                            if not isinstance(second, torch.Tensor)
                            else second.item()
                        )
                        expected = python_op(first_scalar, second_scalar)

                        self.assertEqual(expected, python_op(first, second))
                        self.assertEqual(expected, torch_op(first, second))

    @dtypes(
        *product(
            all_types_and(torch.half, torch.bfloat16, torch.bool),
            all_types_and(torch.half, torch.bfloat16, torch.bool),
        )
    )
    def test_maximum_minimum_type_promotion(self, device, dtypes):
        a = torch.tensor((0, 1), device=device, dtype=dtypes[0])
        b = torch.tensor((1, 0), device=device, dtype=dtypes[1])
        for op in (
            torch.maximum,
            torch.max,
            torch.fmax,
            torch.minimum,
            torch.min,
            torch.fmin,
        ):
            result = op(a, b)
            self.assertEqual(result.dtype, torch.result_type(a, b))

    @dtypes(*integral_types_and(torch.bool))
    def test_maximum_minimum_int_and_bool(self, device, dtype):
        ops = (
            (torch.maximum, torch.max, np.maximum),
            (torch.minimum, torch.min, np.minimum),
            (torch.fmax, None, np.fmax),
            (torch.fmin, None, np.fmin),
        )
        rng = np.random.default_rng()
        a_np = np.array(
            rng.integers(-100, 100, size=10), dtype=torch_to_numpy_dtype_dict[dtype]
        )
        b_np = np.array(
            rng.integers(-100, 100, size=10), dtype=torch_to_numpy_dtype_dict[dtype]
        )

        for torch_op, alias, numpy_op in ops:
            a_tensor = torch.from_numpy(a_np).to(device=device, dtype=dtype)
            b_tensor = torch.from_numpy(b_np).to(device=device, dtype=dtype)
            tensor_result = torch_op(a_tensor, b_tensor)

            out = torch.empty_like(a_tensor)
            torch_op(a_tensor, b_tensor, out=out)

            numpy_result = numpy_op(a_np, b_np)

            if alias is not None:
                alias_result = alias(a_tensor, b_tensor)
                self.assertEqual(alias_result, tensor_result)

            self.assertEqual(tensor_result, numpy_result)
            self.assertEqual(out, numpy_result)

    @precisionOverride({torch.bfloat16: 1e-2})
    @dtypes(*(floating_types_and(torch.half, torch.bfloat16)))
    def test_maximum_minimum_float(self, device, dtype):
        ops = (
            (torch.maximum, torch.max, np.maximum),
            (torch.minimum, torch.min, np.minimum),
            (torch.fmax, None, np.fmax),
            (torch.fmin, None, np.fmin),
        )

        if dtype == torch.bfloat16:
            a_np = np.random.randn(10).astype(np.float64)
            b_np = np.random.randn(10).astype(np.float64)
        else:
            a_np = np.random.randn(10).astype(torch_to_numpy_dtype_dict[dtype])
            b_np = np.random.randn(10).astype(torch_to_numpy_dtype_dict[dtype])

        for torch_op, alias, numpy_op in ops:
            numpy_result = numpy_op(a_np, b_np)

            a_tensor = torch.from_numpy(a_np).to(device=device, dtype=dtype)
            b_tensor = torch.from_numpy(b_np).to(device=device, dtype=dtype)
            tensor_result = torch_op(a_tensor, b_tensor)
            out = torch.empty_like(a_tensor)
            torch_op(a_tensor, b_tensor, out=out)

            if alias is not None:
                alias_result = alias(a_tensor, b_tensor)
                self.assertEqual(alias_result, tensor_result, exact_dtype=False)

            self.assertEqual(tensor_result, numpy_result, exact_dtype=False)
            self.assertEqual(out, numpy_result, exact_dtype=False)

    @dtypes(*(floating_types_and(torch.half, torch.bfloat16)))
    def test_maximum_minimum_float_nan_and_inf(self, device, dtype):
        # np.maximum and np.minimum functions compare input arrays element-wisely.
        # if one of the elements being compared is a NaN, then that element is returned.
        ops = (
            (torch.maximum, torch.max, np.maximum),
            (torch.minimum, torch.min, np.minimum),
            (torch.fmax, None, np.fmax),
            (torch.fmin, None, np.fmin),
        )
        a_vals = (
            float("inf"),
            -float("inf"),
            float("nan"),
            float("inf"),
            float("nan"),
            float("nan"),
            1,
            float("nan"),
        )
        b_vals = (
            -float("inf"),
            float("inf"),
            float("inf"),
            float("nan"),
            float("nan"),
            0,
            float("nan"),
            -5,
        )
        if dtype == torch.bfloat16:
            a_np = np.array(a_vals, dtype=np.float64)
            b_np = np.array(b_vals, dtype=np.float64)
        else:
            a_np = np.array(a_vals, dtype=torch_to_numpy_dtype_dict[dtype])
            b_np = np.array(b_vals, dtype=torch_to_numpy_dtype_dict[dtype])

        for torch_op, alias, numpy_op in ops:
            numpy_result = numpy_op(a_np, b_np)

            a_tensor = torch.from_numpy(a_np).to(device=device, dtype=dtype)
            b_tensor = torch.from_numpy(b_np).to(device=device, dtype=dtype)
            tensor_result = torch_op(a_tensor, b_tensor)

            out = torch.empty_like(a_tensor)
            torch_op(a_tensor, b_tensor, out=out)

            if alias is not None:
                alias_result = alias(a_tensor, b_tensor)
                self.assertEqual(alias_result, tensor_result)

            if dtype == torch.bfloat16:
                self.assertEqual(tensor_result, numpy_result, exact_dtype=False)
                self.assertEqual(out, numpy_result, exact_dtype=False)
            else:
                self.assertEqual(tensor_result, numpy_result)
                self.assertEqual(out, numpy_result)

    @dtypes(
        *product(
            complex_types(),
            all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool),
        )
    )
    def test_maximum_minimum_complex(self, device, dtypes):
        for torch_op in (
            torch.maximum,
            torch.minimum,
            torch.max,
            torch.min,
            torch.fmax,
            torch.fmin,
        ):
            with self.assertRaisesRegex(RuntimeError, ".+not implemented for.+"):
                torch_op(
                    torch.ones(1, device=device, dtype=dtypes[0]),
                    torch.ones(1, device=device, dtype=dtypes[1]),
                )

            with self.assertRaisesRegex(RuntimeError, ".+not implemented for.+"):
                torch_op(
                    torch.ones(1, device=device, dtype=dtypes[1]),
                    torch.ones(1, device=device, dtype=dtypes[0]),
                )

    @onlyCUDA
    def test_maximum_minimum_cross_device(self, device):
        a = torch.tensor((1, 2, -1))
        b = torch.tensor((3, 0, 4), device=device)
        ops = (torch.maximum, torch.minimum)

        for torch_op in ops:
            with self.assertRaisesRegex(
                RuntimeError, "Expected all tensors to be on the same device"
            ):
                torch_op(a, b)

            with self.assertRaisesRegex(
                RuntimeError, "Expected all tensors to be on the same device"
            ):
                torch_op(b, a)

        # test cuda tensor and cpu scalar
        ops = ((torch.maximum, np.maximum), (torch.minimum, np.minimum))
        a_np = np.array(1)
        b_np = np.array([3, 0, 4])

        for torch_op, numpy_op in ops:
            a_tensor = torch.from_numpy(a_np)
            b_tensor = torch.from_numpy(b_np).to(device=device)
            tensor_result_1 = torch_op(a_tensor, b_tensor)
            numpy_result_1 = numpy_op(a_np, b_np)
            tensor_result_2 = torch_op(b_tensor, a_tensor)
            numpy_result_2 = numpy_op(b_np, a_np)

            self.assertEqual(tensor_result_1, numpy_result_1)
            self.assertEqual(tensor_result_2, numpy_result_2)

    @dtypes(
        *product(
            floating_types_and(torch.half, torch.bfloat16),
            floating_types_and(torch.half, torch.bfloat16),
        )
    )
    def test_maximum_and_minimum_subgradient(self, device, dtypes):
        def run_test(f, a, b, expected_a_grad, expected_b_grad):
            a = torch.tensor(a, requires_grad=True, device=device, dtype=dtypes[0])
            b = torch.tensor(b, requires_grad=True, device=device, dtype=dtypes[1])
            z = f(a, b)
            z.sum().backward()
            self.assertEqual(a.grad, expected_a_grad)
            self.assertEqual(b.grad, expected_b_grad)

        run_test(
            torch.maximum,
            [0.0, 1.0, 2.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.5, 1.0],
            [1.0, 0.5, 0.0],
        )
        run_test(
            torch.minimum,
            [0.0, 1.0, 2.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.5, 0.0],
            [0.0, 0.5, 1.0],
        )

    def test_maximum_minimum_forward_ad_float32(self, device):
        # TODO: This should really be covered by OpInfo but it isn't. The problem
        # is that our gradient tests test using float64 but it should also test
        # float32
        x = torch.randn(3, device=device, dtype=torch.float32)
        y = torch.randn(3, device=device, dtype=torch.float32)
        tx = torch.randn(3, device=device, dtype=torch.float32)
        ty = torch.randn(3, device=device, dtype=torch.float32)

        with fwAD.dual_level():
            x_dual = fwAD.make_dual(x, tx)
            y_dual = fwAD.make_dual(y, ty)
            result = torch.maximum(x_dual, y_dual)
            _, result_tangent = fwAD.unpack_dual(result)

        expected = torch.where(x > y, tx, ty)
        self.assertEqual(result_tangent, expected)

        with fwAD.dual_level():
            x_dual = fwAD.make_dual(x, tx)
            y_dual = fwAD.make_dual(y, ty)
            result = torch.minimum(x_dual, y_dual)
            _, result_tangent = fwAD.unpack_dual(result)

        expected = torch.where(x < y, tx, ty)
        self.assertEqual(result_tangent, expected)

    # TODO: tests like this should be generic
    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @dtypes(torch.float, torch.double)
    def test_mul_intertype_scalar(self, device, dtype):
        x = torch.tensor(1.5, dtype=dtype, device=device)
        y = torch.tensor(3, dtype=torch.int32, device=device)

        self.assertEqual(x * y, 4.5)
        self.assertEqual(y * x, 4.5)

        with self.assertRaisesRegex(
            RuntimeError, "can't be cast to the desired output type"
        ):
            y *= x
        x *= y
        self.assertEqual(x, 4.5)

    @onlyCPU
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool))
    def test_sub(self, device, dtype):
        if dtype in integral_types():
            # Before Python 3.10, floats were implicitly converted to ints, but with
            #   DeprecationWarning: an integer is required (got type float).
            #   Implicit conversion to integers using __int__ is deprecated,
            #   and may be removed in a future version of Python.
            # Since Python 3.10, that attempt gives an error.
            m1 = torch.tensor([2, 4], dtype=dtype, device=device)
            m2 = torch.tensor([1, 2], dtype=dtype, device=device)
            diff = torch.tensor([1, 2], dtype=dtype)
        else:
            m1 = torch.tensor([2.34, 4.44], dtype=dtype, device=device)
            m2 = torch.tensor([1.23, 2.33], dtype=dtype, device=device)
            diff = torch.tensor([1.11, 2.11], dtype=dtype)

        if dtype == torch.bool:
            self.assertRaises(RuntimeError, lambda: m1 - m2)
        elif dtype == torch.bfloat16 or dtype == torch.half:
            # bfloat16 has a lower precision so we have to have a separate check for it
            self.assertEqual(m1 - m2, diff, atol=0.01, rtol=0)
        else:
            self.assertEqual(m1 - m2, diff)

    # TODO: what is this test testing?
    @onlyCPU
    @dtypes(torch.float)
    def test_csub(self, device, dtype):
        # with a tensor
        a = torch.randn(100, 90, dtype=dtype, device=device)
        b = a.clone().normal_()

        res_add = torch.add(a, b, alpha=-1)
        res_csub = a.clone()
        res_csub.sub_(b)
        self.assertEqual(res_add, res_csub)

        # with a scalar
        a = torch.randn(100, 100, dtype=dtype, device=device)

        scalar = 123.5
        res_add = torch.add(a, -scalar)
        res_csub = a.clone()
        res_csub.sub_(scalar)
        self.assertEqual(res_add, res_csub)

    # TODO: reconcile with minimum/maximum tests
    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @dtypes(torch.float, torch.double)
    def test_min_max_binary_op_nan(self, device, dtype):
        a = torch.rand(1000, dtype=dtype, device=device)
        b = torch.rand(1000, dtype=dtype, device=device)

        # 0:250: a -- nan, b -- not nan
        a[:250] = float("nan")
        # 250:500: a -- not nan, b -- nan
        b[250:500] = float("nan")
        # 500:750: a and b both nan
        a[500:750] = float("nan")
        b[500:750] = float("nan")
        # 750:1000: neither nan

        ma = torch.max(a, b)
        mi = torch.min(a, b)

        for i in range(750):
            self.assertTrue(
                torch.isnan(ma[i]),
                f"max(a, b): {ma[i]}, a: {a[i]}, b: {b[i]}",
            )
            self.assertTrue(
                torch.isnan(mi[i]),
                f"min(a, b): {mi[i]}, a: {a[i]}, b: {b[i]}",
            )

        for i in range(750, 1000):
            self.assertFalse(
                torch.isnan(ma[i]),
                f"max(a, b): {ma[i]}, a: {a[i]}, b: {b[i]}",
            )
            self.assertFalse(
                torch.isnan(mi[i]),
                f"min(a, b): {mi[i]}, a: {a[i]}, b: {b[i]}",
            )

    @dtypes(
        *product(
            all_types_and(torch.half, torch.bfloat16, torch.bool),
            all_types_and(torch.half, torch.bfloat16, torch.bool),
        )
    )
    def test_copysign(self, device, dtypes):
        def _test_copysign_numpy(a, b):
            torch_result = torch.copysign(a, b)

            if a.dtype == torch.bfloat16:
                np_a = a.to(torch.float).cpu().numpy()
            else:
                np_a = a.cpu().numpy()

            if b.dtype == torch.bfloat16:
                np_b = b.to(torch.float).cpu().numpy()
            else:
                np_b = b.cpu().numpy()
            expected = torch.from_numpy(np.copysign(np_a, np_b))
            # To handle inconsistencies of type promotion between PyTorch and Numpy
            # Applied for both arguments having integral precision and bfloat16
            types = integral_types_and(torch.bool, torch.bfloat16)
            if a.dtype in types or b.dtype in types:
                promoted_type = torch.promote_types(torch_result.dtype, expected.dtype)
                torch_result = torch_result.to(promoted_type)
                expected = expected.to(promoted_type)

            # Verify Value
            self.assertEqual(torch_result, expected)
            # Verify Sign
            # Use double copysign to verify the correctnes of 0.0 and -0.0, since
            # it always True for self.assertEqual(0.0 == -0.0). So, we use 1 as the
            # magnitude to verify the sign between torch and numpy results, elementwise.
            # Special case: NaN conversions between FP32 and FP16 is not bitwise
            # equivalent to pass this assertion.
            if a.dtype != torch.float16 and b.dtype != torch.float16:
                self.assertEqual(
                    torch.copysign(torch.tensor(1.0), torch_result),
                    torch.copysign(torch.tensor(1.0), expected),
                )

        # Compare Result with NumPy
        # Type promotion
        a = make_tensor((10, 10), device=device, dtype=dtypes[0], low=-9, high=9)
        b = make_tensor((10, 10), device=device, dtype=dtypes[1], low=-9, high=9)
        _test_copysign_numpy(a, b)

        # Broadcast
        a = make_tensor((10, 1, 10), device=device, dtype=dtypes[0], low=-9, high=9)
        b = make_tensor((10, 10), device=device, dtype=dtypes[1], low=-9, high=9)
        _test_copysign_numpy(a, b)

        a = make_tensor((10, 10), device=device, dtype=dtypes[0], low=-9, high=9)
        b = make_tensor((10, 1, 10), device=device, dtype=dtypes[1], low=-9, high=9)
        _test_copysign_numpy(a, b)

        # 0.0/-0.0/inf/-inf/nan
        cases = [0.0, -0.0, float("inf"), float("-inf"), float("nan")]
        # torch.bfloat16 can not hold '-nan'
        # torch.half can not hold '-nan' on CUDA
        types = [torch.float32, torch.float64]
        if device == "cpu":
            types.append(torch.float16)
        if dtypes[0] in types:
            b = make_tensor((10, 10), device=device, dtype=dtypes[1], low=-9, high=9)
            for case in cases:
                _test_copysign_numpy(
                    torch.tensor([case], device=device, dtype=dtypes[0]), b
                )

        if dtypes[1] in floating_types_and(torch.half, torch.bfloat16):
            a = make_tensor((10, 10), device=device, dtype=dtypes[0], low=-9, high=9)
            for case in cases:
                _test_copysign_numpy(
                    a, torch.tensor([case], device=device, dtype=dtypes[1])
                )

    @dtypes(
        *product(
            floating_types_and(torch.half, torch.bfloat16),
            floating_types_and(torch.half, torch.bfloat16),
        )
    )
    def test_copysign_subgradient(self, device, dtypes):
        # Input is 0.0
        x = torch.tensor(
            [0.0, 0.0, 0.0], dtype=dtypes[0], device=device, requires_grad=True
        )
        y = torch.tensor(
            [-1.0, 0.0, 1.0], dtype=dtypes[1], device=device, requires_grad=True
        )
        out = torch.copysign(x, y)
        out.sum().backward()
        self.assertEqual(x.grad.tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(y.grad.tolist(), [0.0] * 3)

        # Input is -0.0
        x = torch.tensor(
            [-0.0, -0.0, -0.0], dtype=dtypes[0], device=device, requires_grad=True
        )
        y = torch.tensor(
            [-1.0, 0.0, 1.0], dtype=dtypes[1], device=device, requires_grad=True
        )
        out = torch.copysign(x, y)
        out.sum().backward()
        self.assertEqual(x.grad.tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(y.grad.tolist(), [0.0] * 3)

        # Other is 0.0
        x = torch.tensor(
            [-1.0, 0.0, 1.0], dtype=dtypes[0], device=device, requires_grad=True
        )
        y = torch.tensor(
            [0.0, 0.0, 0.0], dtype=dtypes[1], device=device, requires_grad=True
        )
        out = torch.copysign(x, y)
        out.sum().backward()
        self.assertEqual(x.grad.tolist(), [-1.0, 0.0, 1.0])
        self.assertEqual(y.grad.tolist(), [0.0] * 3)

        # Other is -0.0
        x = torch.tensor(
            [-1.0, 0.0, 1.0], dtype=dtypes[0], device=device, requires_grad=True
        )
        y = torch.tensor(
            [-0.0, -0.0, -0.0], dtype=dtypes[1], device=device, requires_grad=True
        )
        out = torch.copysign(x, y)
        out.sum().backward()
        self.assertEqual(x.grad.tolist(), [1.0, 0.0, -1.0])
        self.assertEqual(y.grad.tolist(), [0.0] * 3)

    @dtypes(torch.bfloat16, torch.float)
    def test_div(self, device, dtype):
        for op, method, inplace in (
            (torch.div, torch.Tensor.div, torch.Tensor.div_),
            (torch.true_divide, torch.Tensor.true_divide, torch.Tensor.true_divide_),
        ):
            m1 = torch.randn(10, 10, dtype=torch.float, device=device).to(dtype=dtype)
            res1 = m1.clone()
            inplace(res1[:, 3], 2)
            res2 = m1.clone()
            for i in range(m1.size(0)):
                res2[i, 3] = res2[i, 3] / 2
            self.assertEqual(res1, res2)

            if dtype == torch.bfloat16:
                a1 = torch.tensor([4.2, 6.2], dtype=dtype, device=device)
                a2 = torch.tensor([2.0, 2.0], dtype=dtype, device=device)
                self.assertEqual(
                    op(a1, a2),
                    torch.tensor([2.1, 3.1], dtype=dtype, device=device),
                    atol=0.01,
                    rtol=0,
                )
                self.assertEqual(method(a1, a2), op(a1, a2))

    @dtypes(torch.bfloat16, torch.float)
    def test_true_divide_out(self, device, dtype):
        a1 = torch.tensor([4.2, 6.2], dtype=dtype, device=device)
        a2 = torch.tensor([2.0, 2.0], dtype=dtype, device=device)
        res = torch.empty_like(a1)
        self.assertEqual(
            torch.true_divide(a1, a2, out=res),
            torch.tensor([2.1, 3.1], dtype=dtype, device=device),
            atol=0.01,
            rtol=0,
        )

    @dtypes(torch.half)
    def test_divmul_scalar(self, device, dtype):
        x = torch.tensor(100.0, device=device, dtype=dtype)
        x_ref = x.float()
        scale = 1e5
        res = x.div(scale)
        expected = x_ref.div(scale)
        self.assertEqual(res, expected.to(dtype), atol=0.0, rtol=0.0)
        x = torch.tensor(1e-5, device=device, dtype=dtype)
        x_ref = x.float()
        res = x.mul(scale)
        expected = x_ref.mul(scale)
        self.assertEqual(res, expected.to(dtype), atol=0.0, rtol=0.0)
        res = scale * x
        self.assertEqual(res, expected.to(dtype), atol=0.0, rtol=0.0)

    @dtypesIfCUDA(
        *set(get_all_math_dtypes("cuda")) - {torch.complex64, torch.complex128}
    )
    @dtypes(*set(get_all_math_dtypes("cpu")) - {torch.complex64, torch.complex128})
    def test_floor_divide_tensor(self, device, dtype):
        x = torch.randn(10, device=device).mul(30).to(dtype)
        y = torch.arange(1, 11, dtype=dtype, device=device)

        z = x // y
        z_alt = torch.floor(x.double() / y.double()).to(dtype)

        self.assertEqual(z.dtype, x.dtype)
        self.assertEqual(z, z_alt)

    @dtypesIfCUDA(
        *set(get_all_math_dtypes("cuda")) - {torch.complex64, torch.complex128}
    )
    @dtypes(*set(get_all_math_dtypes("cpu")) - {torch.complex64, torch.complex128})
    def test_floor_divide_scalar(self, device, dtype):
        x = torch.randn(100, device=device).mul(10).to(dtype)

        z = x // 3
        z_alt = torch.tensor(
            [math.floor(v.item() / 3.0) for v in x], dtype=x.dtype, device=device
        )

        self.assertEqual(z.dtype, x.dtype)
        self.assertEqual(z, z_alt)

    @onlyCPU
    @dtypes(*get_all_math_dtypes("cpu"))
    def test_rdiv(self, device, dtype):
        if dtype is torch.float16:
            return
        elif dtype.is_complex:
            x = torch.rand(100, dtype=dtype, device=device).add(1).mul(4)
        else:
            x = torch.rand(100, device=device).add(1).mul(4).to(dtype)
        y = 30 / x
        z = torch.tensor([30 / v.item() for v in x], device=device)
        self.assertEqual(y, z, exact_dtype=False)

    @dtypes(*floating_types_and(torch.half))
    def test_fmod_remainder_by_zero_float(self, device, dtype):
        fn_list = (torch.fmod, torch.remainder)
        for fn in fn_list:
            # check floating-point tensor fmod/remainder to zero is nan on both CPU and GPU
            x = make_tensor((10, 10), device=device, dtype=dtype, low=-9, high=9)
            zero = torch.zeros_like(x)
            self.assertTrue(torch.all(fn(x, 0.0).isnan()))
            self.assertTrue(torch.all(fn(x, zero).isnan()))

    @onlyNativeDeviceTypes  # Check Issue https://github.com/pytorch/pytorch/issues/48130
    @dtypes(*integral_types())
    def test_fmod_remainder_by_zero_integral(self, device, dtype):
        fn_list = (torch.fmod, torch.remainder)
        for fn in fn_list:
            # check integral tensor fmod/remainder to zero
            x = make_tensor((10, 10), device=device, dtype=dtype, low=-9, high=9)
            zero = torch.zeros_like(x)
            # RuntimeError on CPU
            if self.device_type == "cpu":
                with self.assertRaisesRegex(RuntimeError, "ZeroDivisionError"):
                    fn(x, zero)
            elif torch.version.hip is not None:
                # ROCm behavior: x % 0 is a no-op; x is returned
                self.assertEqual(fn(x, zero), x)
            else:
                # CUDA behavior: Different value for different dtype
                # Due to it's an undefined behavior, CUDA returns a pattern of all 1s
                # for integral dividend (other than int64) divided by zero. For int64,
                # CUDA returns all 1s for negative dividend, half 1s for positive dividend.
                # uint8: 0xff -> 255
                # int32: 0xffffffff -> -1
                if dtype == torch.int64:
                    self.assertEqual(fn(x, zero) == 4294967295, x >= 0)
                    self.assertEqual(fn(x, zero) == -1, x < 0)
                else:
                    value = 255 if dtype == torch.uint8 else -1
                    self.assertTrue(torch.all(fn(x, zero) == value))

    @dtypes(*all_types_and(torch.half))
    def test_fmod_remainder(self, device, dtype):
        # Use numpy as reference
        def _helper(x, mod, fns_list):
            for fn, inplace_fn, ref_fn in fns_list:
                np_x = x.cpu().numpy() if torch.is_tensor(x) else x
                np_mod = mod.cpu().numpy() if torch.is_tensor(mod) else mod
                exp = ref_fn(np_x, np_mod)
                exp = torch.from_numpy(exp)
                res = fn(x, mod)

                self.assertEqual(res, exp, exact_dtype=False)

                if torch.is_tensor(x):
                    # out
                    out = torch.empty(0, device=device, dtype=res.dtype)
                    fn(x, mod, out=out)
                    self.assertEqual(out, exp, exact_dtype=False)
                    self.assertEqual(out.size(), torch.Size([10, 10]))
                    # in-place (Type cast runtime error)
                    try:
                        inplace_fn(x, mod)
                        self.assertEqual(x, exp, exact_dtype=False)
                    except RuntimeError as e:
                        self.assertRegex(
                            str(e),
                            "result type (Half|Float|Double) "
                            "can't be cast to the desired output "
                            "type (Byte|Char|Short|Int|Long)",
                        )

        x = make_tensor((10, 10), device=device, dtype=dtype, low=-9, high=9)
        # mod with same dtype as x
        mod = make_tensor((10, 10), device=device, dtype=dtype, low=-9, high=9)
        # Exclude 0
        mod[mod == 0] = 1

        # Mods: Integer, Float, Tensor, Non-contiguous Tensor
        mods = [3, 2.3, mod, mod.t()]
        # mod with floating-point dtype
        if dtype in integral_types():
            mod_float = make_tensor(
                (10, 10), device=device, dtype=torch.float, low=-9, high=9
            )
            mod[mod == 0] = 1
            mods.append(mod_float)

        for dividend, mod in product([x, x.t()], mods):
            _helper(
                dividend,
                mod,
                (
                    (torch.fmod, torch.Tensor.fmod_, np.fmod),
                    (torch.remainder, torch.Tensor.remainder_, np.remainder),
                ),
            )

        # Tests for torch.remainder(scalar, tensor)
        for dividend, mod in product([5, 3.14], mods):
            if torch.is_tensor(mod):
                _helper(
                    dividend,
                    mod,
                    ((torch.remainder, torch.Tensor.remainder_, np.remainder),),
                )

    @dtypes(torch.float, torch.double)
    def test_remainder_fmod_large_dividend(self, device, dtype):
        alarge = 1e9
        pi = 3.14159265358979
        for avalue in [alarge, -alarge]:
            for bvalue in [pi, -pi]:
                a = torch.tensor([avalue], dtype=dtype, device=device)
                b = torch.tensor([bvalue], dtype=dtype, device=device)
                c = torch.remainder(a, b)
                d = torch.fmod(a, b)
                self.assertTrue(
                    (b[0] > 0) == (c[0] > 0)
                )  # remainder has same sign as divisor
                self.assertTrue(
                    (a[0] > 0) == (d[0] > 0)
                )  # fmod has same sign as dividend
                self.assertTrue(
                    abs(c[0]) < abs(b[0])
                )  # remainder is within range of divisor
                self.assertTrue(
                    abs(d[0]) < abs(b[0])
                )  # fmod is within range of divisor
                if (a[0] > 0) == (b[0] > 0):
                    self.assertTrue(c[0] == d[0])  # remainder is same as fmod
                else:
                    self.assertTrue(
                        abs(c[0] - d[0]) == abs(b[0])
                    )  # differ by one divisor

    @dtypesIfCPU(torch.bfloat16, torch.half, torch.float32, torch.float64)
    @dtypes(torch.float32, torch.float64)
    def test_hypot(self, device, dtype):
        inputs = [
            (
                torch.randn(10, device=device).to(dtype),
                torch.randn(10, device=device).to(dtype),
            ),
            (
                torch.randn((3, 3, 3), device=device).to(dtype),
                torch.randn((3, 3, 3), device=device).to(dtype),
            ),
            (
                torch.randn((10, 1), device=device).to(dtype),
                torch.randn((10, 1), device=device).to(dtype).transpose(0, 1),
            ),
            (
                torch.randint(100, (10,), device=device, dtype=torch.long),
                torch.randn(10, device=device).to(dtype),
            ),
        ]
        for input in inputs:
            actual = torch.hypot(input[0], input[1])
            if dtype in [torch.bfloat16, torch.half]:
                expected = torch.sqrt(input[0] * input[0] + input[1] * input[1])
            else:
                expected = np.hypot(input[0].cpu().numpy(), input[1].cpu().numpy())
            self.assertEqual(actual, expected, exact_dtype=False)

    @onlyNativeDeviceTypes
    @dtypes(torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    def test_gcd(self, device, dtype):
        # Tests gcd(0, 0), gcd(0, a) cases
        t1 = torch.tensor([0, 10, 0], dtype=dtype, device=device)
        t2 = torch.tensor([0, 0, 10], dtype=dtype, device=device)
        actual = torch.gcd(t1, t2)
        expected = np.gcd([0, 10, 0], [0, 0, 10])
        self.assertEqual(actual, expected, exact_dtype=False)

        if dtype == torch.uint8:
            # Test unsigned integers with potential sign issues (i.e., uint8 with value >= 128)
            a = torch.tensor([190, 210], device=device, dtype=dtype)
            b = torch.tensor([190, 220], device=device, dtype=dtype)
            actual = torch.gcd(a, b)
            expected = torch.tensor([190, 10], device=device, dtype=dtype)
            self.assertEqual(actual, expected)
        else:
            # Compares with NumPy
            a = torch.randint(-20, 20, (1024,), device=device, dtype=dtype)
            b = torch.randint(-20, 20, (1024,), device=device, dtype=dtype)
            actual = torch.gcd(a, b)
            expected = np.gcd(a.cpu().numpy(), b.cpu().numpy())
            self.assertEqual(actual, expected)

    @onlyNativeDeviceTypes
    @dtypes(torch.int16, torch.int32, torch.int64)
    def test_lcm(self, device, dtype):
        # Tests lcm(0, 0), lcm(0, a) cases
        t1 = torch.tensor([0, 10, 0], dtype=dtype, device=device)
        t2 = torch.tensor([0, 0, 10], dtype=dtype, device=device)
        actual = torch.lcm(t1, t2)
        expected = np.lcm([0, 10, 0], [0, 0, 10])
        self.assertEqual(actual, expected, exact_dtype=False)

        # Compares with NumPy
        a = torch.randint(-20, 20, (1024,), device=device, dtype=dtype)
        b = torch.randint(-20, 20, (1024,), device=device, dtype=dtype)
        actual = torch.lcm(a, b)
        expected = np.lcm(a.cpu().numpy(), b.cpu().numpy())
        self.assertEqual(actual, expected, exact_dtype=False)

    @onlyNativeDeviceTypes
    @dtypesIfCPU(torch.float32, torch.float64, torch.float16)
    @dtypes(torch.float32, torch.float64)
    def test_nextafter(self, device, dtype):
        # Test special cases
        t1 = torch.tensor([0, 0, 10], device=device, dtype=dtype)
        t2 = torch.tensor([inf, -inf, 10], device=device, dtype=dtype)
        actual = torch.nextafter(t1, t2)
        expected = np.nextafter(t1.cpu().numpy(), t2.cpu().numpy())
        self.assertEqual(actual, expected, atol=0, rtol=0)

        actual = torch.nextafter(t2, t1)
        expected = np.nextafter(t2.cpu().numpy(), t1.cpu().numpy())
        self.assertEqual(actual, expected, atol=0, rtol=0)

        t1 = torch.tensor([0, nan], device=device, dtype=dtype)
        t2 = torch.tensor([nan, 0], device=device, dtype=dtype)
        self.assertTrue(torch.nextafter(t1, t2).isnan().all())

        a = torch.randn(100, device=device, dtype=dtype)
        b = torch.randn(100, device=device, dtype=dtype)
        actual = torch.nextafter(a, b)
        expected = np.nextafter(a.cpu().numpy(), b.cpu().numpy())
        self.assertEqual(actual, expected, atol=0, rtol=0)

    @onlyNativeDeviceTypes
    @dtypes(torch.bfloat16)
    def test_nextafter_bfloat16(self, device, dtype):
        nan = float("nan")
        inf = float("inf")
        cases = (
            # (from, to, expected)
            (0, 1, 9.183549615799121e-41),
            (0, -1, -9.183549615799121e-41),
            (1, -2, 0.99609375),
            (1, 0, 0.99609375),
            (1, 2, 1.0078125),
            (-1, -2, -1.0078125),
            (-1, 0, -0.99609375),
            (2, -1, 1.9921875),
            (2, 1, 1.9921875),
            (20, 3000, 20.125),
            (20, -3000, 19.875),
            (3000, -20, 2992.0),
            (-3000, 20, -2992.0),
            (65536, 0, 65280.0),
            (65536, inf, 66048.0),
            (-65536, 0, -65280.0),
            (-65536, -inf, -66048.0),
            (nan, 0, nan),
            (0, nan, nan),
            (nan, nan, nan),
            (nan, inf, nan),
            (inf, nan, nan),
            (inf, -inf, 3.3895313892515355e38),
            (-inf, inf, -3.3895313892515355e38),
            (inf, 0, 3.3895313892515355e38),
            (0, inf, 9.183549615799121e-41),
            (-inf, 0, -3.3895313892515355e38),
            (0, -inf, -9.183549615799121e-41),
        )

        for from_v, to_v, expected in cases:
            from_t = torch.tensor([from_v], device=device, dtype=dtype)
            to_t = torch.tensor([to_v], device=device, dtype=dtype)
            actual = torch.nextafter(from_t, to_t).item()
            self.assertEqual(actual, expected, atol=0, rtol=0)

    def _test_cop(self, torchfn, mathfn, dtype, device):
        def reference_implementation(res2):
            for i, j in iter_indices(sm1):
                idx1d = i * sm1.size(0) + j
                res2[i, j] = mathfn(sm1[i, j], sm2[idx1d])
            return res2

        # contiguous
        m1 = torch.randn(10, 10, 10, dtype=dtype, device=device)
        m2 = torch.randn(10, 10 * 10, dtype=dtype, device=device)
        sm1 = m1[4]
        sm2 = m2[4]

        res1 = torchfn(sm1, sm2.view(10, 10))
        res2 = reference_implementation(res1.clone())
        self.assertEqual(res1, res2)

        # non-contiguous
        m1 = torch.randn(10, 10, 10, dtype=dtype, device=device)
        m2 = torch.randn(10 * 10, 10 * 10, dtype=dtype, device=device)
        sm1 = m1[:, 4]
        sm2 = m2[:, 4]
        # view as sm1.size()
        sm2.set_(
            sm2.storage(),
            sm2.storage_offset(),
            sm1.size(),
            (sm2.stride()[0] * 10, sm2.stride()[0]),
        )
        res1 = torchfn(sm1, sm2)
        # reference_implementation assumes 1-d sm2
        sm2.set_(
            sm2.storage(), sm2.storage_offset(), m2[:, 4].size(), m2[:, 4].stride()
        )
        res2 = reference_implementation(res1.clone())
        self.assertEqual(res1, res2)

    @onlyCPU
    @dtypes(torch.float)
    def test_cdiv(self, device, dtype):
        self._test_cop(torch.div, operator.truediv, dtype, device)

    @onlyCPU
    @dtypes(torch.float)
    def test_cremainder(self, device, dtype):
        self._test_cop(torch.remainder, operator.mod, dtype, device)

    @onlyCPU
    @dtypes(torch.float)
    def test_cmul(self, device, dtype):
        self._test_cop(torch.mul, operator.mul, dtype, device)

    @onlyCPU
    @dtypes(torch.float)
    def test_cpow(self, device, dtype):
        self._test_cop(
            torch.pow, lambda x, y: nan if x < 0 else math.pow(x, y), dtype, device
        )

    @onlyCPU
    @dtypes(torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    def test_floor_divide_zero(self, device, dtype):
        a = torch.tensor([0, 1], dtype=dtype, device=device)
        b = torch.tensor([0, 1], dtype=dtype, device=device)
        with self.assertRaisesRegex(RuntimeError, "ZeroDivisionError"):
            with self.assertWarnsOnceRegex(UserWarning, "floor_divide"):
                a // b

    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool))
    def test_muldiv_scalar(self, device, dtype):
        x = make_tensor((10, 3), dtype=dtype, device=device, low=None, high=None)
        s = make_tensor((1,), dtype=dtype, device="cpu", low=None, high=None).item()
        y = torch.full_like(x, s)
        self.assertEqual(x * s, x * y)
        self.assertEqual(s * x, y * x)
        self.assertEqual(x / s, x / y)
        self.assertEqual(s / x, y / x)

    # TODO: update make_tensor to support extremal additions and remove this in favor of make_tensor
    def _generate_input(self, shape, dtype, device, with_extremal):
        if shape == ():
            x = torch.tensor((), dtype=dtype, device=device)
        else:
            if dtype.is_floating_point or dtype.is_complex:
                # work around torch.randn not being implemented for bfloat16
                if dtype == torch.bfloat16:
                    x = torch.randn(*shape, device=device) * random.randint(30, 100)
                    x = x.to(torch.bfloat16)
                else:
                    x = torch.randn(
                        *shape, dtype=dtype, device=device
                    ) * random.randint(30, 100)
                x[torch.randn(*shape) > 0.5] = 0
                if with_extremal and dtype.is_floating_point:
                    # Use extremal values
                    x[torch.randn(*shape) > 0.5] = float("nan")
                    x[torch.randn(*shape) > 0.5] = float("inf")
                    x[torch.randn(*shape) > 0.5] = float("-inf")
                elif with_extremal and dtype.is_complex:
                    x[torch.randn(*shape) > 0.5] = complex("nan")
                    x[torch.randn(*shape) > 0.5] = complex("inf")
                    x[torch.randn(*shape) > 0.5] = complex("-inf")
            elif dtype == torch.bool:
                x = torch.zeros(shape, dtype=dtype, device=device)
                x[torch.randn(*shape) > 0.5] = True
            else:
                x = torch.randint(15, 100, shape, dtype=dtype, device=device)

        return x

    @dtypes(
        *tuple(
            itertools.combinations_with_replacement(
                all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool), 2
            )
        )
    )
    def test_comparison_ops_type_promotion_and_broadcasting(self, device, dtypes):
        # issue #42660
        # testing all combinations of broadcasting and type promotion
        # with a range of dtypes and input shapes, and with extremal values
        def compare_with_numpy_bin_op(torch_fn, np_fn, x, y, out=None):
            # working around the fact that numpy doesn't support bfloat16
            # by letting numpy treat them as float32's
            x_np = x if x.dtype != torch.bfloat16 else x.to(torch.float32)
            y_np = (
                y.cpu().numpy()
                if y.dtype != torch.bfloat16
                else y.to(torch.float32).cpu().numpy()
            )
            self.compare_with_numpy(
                lambda inp: torch_fn(inp, y, out=out) if out else torch_fn(inp, y),
                lambda inp: np_fn(inp, y_np, out=out) if out else np_fn(inp, y_np),
                x_np,
            )

        complex_op_denylist = [
            torch.lt,
            torch.le,
            torch.gt,
            torch.ge,
        ]  # complex not supported
        input_sizes = [(1,), (10,), (10, 1), (1, 10), (4, 10), (64, 10), (12, 3)]
        op_pairs = [
            (torch.lt, np.less),
            (torch.le, np.less_equal),
            (torch.gt, np.greater),
            (torch.ge, np.greater_equal),
            (torch.eq, np.equal),
            (torch.ne, np.not_equal),
            (torch.logical_and, np.logical_and),
            (torch.logical_or, np.logical_or),
            (torch.logical_xor, np.logical_xor),
        ]

        for size1 in input_sizes:
            size2 = (2,) + size1  # perform broadcasting
            for with_extremal in [False, True]:
                a = self._generate_input(size1, dtypes[0], device, with_extremal)
                b = self._generate_input(size2, dtypes[1], device, with_extremal)
                for torch_op, numpy_op in op_pairs:
                    if (
                        dtypes[0].is_complex or dtypes[1].is_complex
                    ) and torch_op in complex_op_denylist:
                        continue
                    # functional version of op
                    compare_with_numpy_bin_op(torch_op, numpy_op, a, b)

                    # functional comparison ops always return bool tensors
                    self.assertEqual(torch_op(a, b).dtype, torch.bool)

                    # out version of op
                    out = torch.zeros(
                        1, dtype=torch.complex128
                    )  # all casts to complex128 are safe
                    compare_with_numpy_bin_op(torch_op, numpy_op, a, b, out=out)

    @onlyNativeDeviceTypes
    @dtypes(torch.int8, torch.int16, torch.int32, torch.int64)
    def test_signed_shift(self, device, dtype):
        "Ensure that signed integer bit shifting works as expected."
        a = torch.tensor([-10, 10], device=device, dtype=dtype)  # [11...1110110, 1010]
        expected_l = torch.tensor(
            [-40, 40], device=device, dtype=dtype
        )  # [11...11011000, 101000]
        self.assertEqual(a << 2, expected_l)
        self.compare_with_numpy(lambda x: x << 2, lambda x: np.left_shift(x, 2), a)
        expected_r = torch.tensor(
            [-5, 5], device=device, dtype=dtype
        )  # [1111...111011, 101]
        self.assertEqual(a >> 1, expected_r)
        self.compare_with_numpy(lambda x: x >> 1, lambda x: np.right_shift(x, 1), a)

    @onlyNativeDeviceTypes
    @dtypes(*get_all_int_dtypes())
    def test_shift_limits(self, device, dtype):
        "Ensure that integer bit shifting works as expected with out-of-limits shift values."
        # Issue #70904
        iinfo = torch.iinfo(dtype)
        bits = iinfo.bits
        low = iinfo.min
        high = iinfo.max
        exact_dtype = (
            dtype != torch.uint8
        )  # numpy changes dtype from uint8 to int16 for some out-of-limits shift values
        for input in (
            torch.tensor(
                [-1, 0, 1], device=device, dtype=dtype
            ),  # small for non-vectorized operation
            torch.tensor(
                [low, high], device=device, dtype=dtype
            ),  # small for non-vectorized operation
            make_tensor(
                (64, 64, 64), low=low, high=high, device=device, dtype=dtype
            ),  # large for vectorized operation
        ):
            shift_left_expected = torch.zeros_like(input)
            shift_right_expected = torch.clamp(input, -1, 0)
            # NumPy 2 does not support negative shift values.
            if np.__version__ > "2":
                iterator = range(bits, 100)
            else:
                iterator = chain(range(-100, -1), range(bits, 100))
            for shift in iterator:
                shift_left = input << shift
                self.assertEqual(shift_left, shift_left_expected, msg=f"<< {shift}")
                self.compare_with_numpy(
                    lambda x: x << shift,
                    lambda x: np.left_shift(x, shift),
                    input,
                    exact_dtype=exact_dtype,
                    msg=f"<< {shift}",
                )
                shift_right = input >> shift
                self.assertEqual(shift_right, shift_right_expected, msg=f">> {shift}")
                self.compare_with_numpy(
                    lambda x: x >> shift,
                    lambda x: np.right_shift(x, shift),
                    input,
                    exact_dtype=exact_dtype,
                    msg=f">> {shift}",
                )

    @onlyNativeDeviceTypes
    @dtypes(
        *list(
            product(
                all_types_and(torch.half, torch.bfloat16, torch.bool),
                all_types_and(torch.half, torch.bfloat16, torch.bool),
            )
        )
    )
    def test_heaviside(self, device, dtypes):
        input_dtype = dtypes[0]
        values_dtype = dtypes[1]

        rng = np.random.default_rng()
        input = np.array(
            rng.integers(-10, 10, size=10),
            dtype=torch_to_numpy_dtype_dict[
                input_dtype if (input_dtype != torch.bfloat16) else torch.float64
            ],
        )
        input[0] = input[3] = input[7] = 0
        values = np.array(
            rng.integers(-10, 10, size=10),
            dtype=torch_to_numpy_dtype_dict[
                values_dtype if (values_dtype != torch.bfloat16) else torch.float64
            ],
        )
        np_result = torch.from_numpy(np.heaviside(input, values)).to(
            device=device, dtype=input_dtype
        )

        input = torch.from_numpy(input).to(device=device, dtype=input_dtype)
        values = torch.from_numpy(values).to(device=device, dtype=values_dtype)
        out = torch.empty_like(input)

        if input_dtype == values_dtype:
            torch_result = torch.heaviside(input, values)
            self.assertEqual(np_result, torch_result)

            torch_result = input.heaviside(values)
            self.assertEqual(np_result, torch_result)

            torch.heaviside(input, values, out=out)
            self.assertEqual(np_result, out)

            input.heaviside_(values)
            self.assertEqual(np_result, input)
        else:
            with self.assertRaisesRegex(
                RuntimeError,
                "heaviside is not yet implemented for tensors with different dtypes.",
            ):
                torch.heaviside(input, values)
            with self.assertRaisesRegex(
                RuntimeError,
                "heaviside is not yet implemented for tensors with different dtypes.",
            ):
                input.heaviside(values)
            with self.assertRaisesRegex(
                RuntimeError,
                "heaviside is not yet implemented for tensors with different dtypes.",
            ):
                torch.heaviside(input, values, out=out)
            with self.assertRaisesRegex(
                RuntimeError,
                "heaviside is not yet implemented for tensors with different dtypes.",
            ):
                input.heaviside_(values)

    @onlyCUDA
    def test_heaviside_cross_device(self, device):
        x = torch.tensor([-9, 5, 0, 6, -2, 2], device=device)
        y = torch.tensor(0)
        result = torch.heaviside(x, y)
        expect = torch.tensor([0, 1, 0, 1, 0, 1], device=device)
        self.assertEqual(result, expect)

        result = torch.heaviside(y, x)
        expect = torch.tensor([-9, 5, 0, 6, -2, 2], device=device)
        self.assertEqual(result, expect)

        x = torch.tensor([-9, 5, 0, 6, -2, 2])
        y = torch.tensor(0, device=device)
        with self.assertRaisesRegex(
            RuntimeError, "Expected all tensors to be on the same device"
        ):
            torch.heaviside(x, y)

        with self.assertRaisesRegex(
            RuntimeError, "Expected all tensors to be on the same device"
        ):
            torch.heaviside(y, x)

    @dtypes(*list(product(complex_types(), complex_types())))
    def test_heaviside_complex(self, device, dtypes):
        input_dtype = dtypes[0]
        values_dtype = dtypes[1]

        data = (complex(0, -6), complex(-1, 3), complex(1, 1))
        input = torch.tensor(data, device=device, dtype=input_dtype)
        values = torch.tensor(data, device=device, dtype=values_dtype)
        out = torch.empty_like(input)
        real = input.real

        with self.assertRaisesRegex(
            RuntimeError, "heaviside is not yet implemented for complex tensors."
        ):
            torch.heaviside(input, real)
        with self.assertRaisesRegex(
            RuntimeError, "heaviside is not yet implemented for complex tensors."
        ):
            real.heaviside(values)
        with self.assertRaisesRegex(
            RuntimeError, "heaviside is not yet implemented for complex tensors."
        ):
            input.heaviside_(values)
        with self.assertRaisesRegex(
            RuntimeError, "heaviside is not yet implemented for complex tensors."
        ):
            torch.heaviside(real, real, out=out)

    def _test_logical(self, device, dtypes, op, a_, b_, expected_res_):
        expected_res = torch.tensor(expected_res_, dtype=dtypes[0], device=device)
        a = torch.tensor(a_, dtype=dtypes[0], device=device)
        b = torch.tensor(b_, dtype=dtypes[1], device=device)

        # new tensor
        self.assertEqual(expected_res.bool(), getattr(a, op)(b))
        # out
        c = torch.empty(0, dtype=torch.bool, device=device)
        getattr(torch, op)(a, b, out=c)
        self.assertEqual(expected_res.bool(), c)

        getattr(a, op + "_")(b)
        self.assertEqual(expected_res, a)

    @dtypes(
        *product(
            all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool),
            all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool),
        )
    )
    def test_logical_xor(self, device, dtypes):
        self._test_logical(
            device, dtypes, "logical_xor", [10, 0, 1, 0], [1, 0, 0, 10], [0, 0, 1, 1]
        )

    @dtypes(
        *product(
            all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool),
            all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool),
        )
    )
    def test_logical_and(self, device, dtypes):
        self._test_logical(
            device, dtypes, "logical_and", [10, 0, 1, 0], [1, 0, 0, 10], [1, 0, 0, 0]
        )

    @dtypes(
        *product(
            all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool),
            all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool),
        )
    )
    def test_logical_or(self, device, dtypes):
        self._test_logical(
            device, dtypes, "logical_or", [10, 0, 1, 0], [1, 0, 0, 10], [1, 0, 1, 1]
        )

    def test_remainder_overflow(self, device):
        # Check Integer Overflows
        x = torch.tensor(23500, dtype=torch.int64, device=device)
        q = 392486996410368
        self.assertEqual(x % q, x)
        self.assertEqual(-x % q, q - x)
        self.assertEqual(x % -q, x - q)
        self.assertEqual(-x % -q, -x)

    def test_rpow(self, device):
        m = torch.randn(10, 10, device=device)
        self.assertEqual(torch.pow(2, m), 2**m)

        # test with scalar
        m = torch.randn(1, device=device).squeeze()
        assert m.dim() == 0, "m is intentionally a scalar"
        self.assertEqual(torch.pow(2, m), 2**m)

    def test_ldexp(self, device):
        # random values
        mantissas = torch.randn(64, device=device)
        exponents = torch.randint(-31, 31, (64,), device=device, dtype=torch.int32)

        # basic test
        np_outcome = np.ldexp(mantissas.cpu().numpy(), exponents.cpu().numpy())
        pt_outcome_1 = torch.ldexp(mantissas, exponents)
        pt_outcome_2 = mantissas.ldexp(exponents)
        self.assertEqual(np_outcome, pt_outcome_1.cpu())
        self.assertEqual(np_outcome, pt_outcome_2.cpu())
        mantissas.ldexp_(exponents)
        self.assertEqual(np_outcome, mantissas.cpu())

        # test bounds
        mantissas = torch.tensor(
            [float("inf"), float("-inf"), float("inf"), float("nan")], device=device
        )
        exponents = torch.randint(0, 31, (4,), device=device, dtype=torch.int32)
        np_outcome = np.ldexp(mantissas.cpu().numpy(), exponents.cpu().numpy())
        pt_outcome = torch.ldexp(mantissas, exponents)
        self.assertEqual(np_outcome, pt_outcome.cpu())

        # test half dtype behavior
        mantissas = torch.randn(64, device=device, dtype=torch.half)
        exponents = torch.randint(-5, 5, (64,), device=device)
        self.assertEqual(torch.ldexp(mantissas, exponents).dtype, torch.half)

        # test float64 computation
        mantissas = torch.tensor([1], dtype=torch.float64, device=device)
        exponents = torch.tensor([128], dtype=torch.int64, device=device)
        expected = torch.pow(
            torch.full((1,), 2, device=device, dtype=torch.float64), 128
        )
        self.assertEqual(torch.ldexp(mantissas, exponents), expected)

    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    def test_lerp(self, device, dtype):
        start_end_weight_shapes = [(), (5,), (5, 5)]
        for shapes in product(
            start_end_weight_shapes, start_end_weight_shapes, start_end_weight_shapes
        ):
            start = torch.randn(shapes[0], device=device, dtype=dtype)
            end = torch.randn(shapes[1], device=device, dtype=dtype)

            # Tensor weights
            weights = [
                torch.randn(shapes[2], device=device, dtype=dtype),
                random.random(),
                torch.randn([], device="cpu", dtype=dtype),
            ]
            if dtype.is_complex:
                weights += [complex(0, 1), complex(0.4, 1.2)]

            for weight in weights:
                actual = torch.lerp(start, end, weight)
                actual_method = start.lerp(end, weight)
                self.assertEqual(actual, actual_method)
                actual_out = torch.tensor(1.0, dtype=dtype, device=device)
                torch.lerp(start, end, weight, out=actual_out)
                self.assertEqual(actual, actual_out)
                expected = start + weight * (end - start)
                self.assertEqual(expected, actual)

    @onlyCUDA
    @dtypes(torch.half, torch.bfloat16)
    def test_lerp_lowp(self, device, dtype):
        xvals = (0.0, -30000.0)
        yvals = (0.1, -20000.0)
        xs = [torch.full((4,), xval, device=device, dtype=dtype) for xval in xvals]
        ys = [torch.full((4,), yval, device=device, dtype=dtype) for yval in yvals]
        weights = [70000, torch.full((4,), 8, device=device, dtype=dtype)]
        for x, y, w in zip(xs, ys, weights):
            xref = x.float()
            yref = y.float()
            wref = w.float() if isinstance(w, torch.Tensor) else w
            actual = torch.lerp(x, y, w)
            expected = torch.lerp(xref, yref, wref).to(dtype)
            self.assertEqual(actual, expected, atol=0.0, rtol=0.0)

    @onlyCPU
    @dtypes(torch.half, torch.bfloat16)
    def test_lerp_lowp_cpu(self, device, dtype):
        xvals = (0.0, -30000.0)
        yvals = (0.1, -20000.0)
        for shape in [(4,), (20,), (3, 10, 10)]:
            xs = [torch.full(shape, xval, device=device, dtype=dtype) for xval in xvals]
            ys = [torch.full(shape, yval, device=device, dtype=dtype) for yval in yvals]
            weights = [70000, torch.full(shape, 8, device=device, dtype=dtype)]
            for x, y, w in zip(xs, ys, weights):
                xref = x.float()
                yref = y.float()
                wref = w.float() if isinstance(w, torch.Tensor) else w
                actual = torch.lerp(x, y, w)
                expected = torch.lerp(xref, yref, wref).to(dtype)
                self.assertEqual(actual, expected, atol=0.0, rtol=0.0)

    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    def test_lerp_tensor_type_promotion(self, device, dtype):
        start = make_tensor((5, 5), dtype=dtype, device=device, low=1, high=100)
        end = make_tensor((5, 5), dtype=dtype, device=device, low=1, high=100)
        weight = make_tensor((5, 5), dtype=torch.float, device=device, low=1, high=100)

        actual = torch.lerp(start, end, weight)
        expected = start + weight.to(dtype) * (end - start)
        self.assertEqual(expected, actual)

    def _test_logaddexp(self, device, dtype, base2):
        if base2:
            ref_func = np.logaddexp2
            our_func = torch.logaddexp2
        elif dtype in (torch.complex64, torch.complex128):
            # numpy has not implemented logaddexp for complex
            def complex_logaddexp(x1, x2):
                x = np.stack((x1, x2))
                amax = np.amax(x, axis=0)
                amax[~np.isfinite(amax)] = 0
                return np.log(np.sum(np.exp(x - amax), axis=0)) + np.squeeze(amax)

            ref_func = complex_logaddexp
            our_func = torch.logaddexp
        else:
            ref_func = np.logaddexp
            our_func = torch.logaddexp

        def _test_helper(a, b):
            if dtype == torch.bfloat16:
                ref = ref_func(a.cpu().float().numpy(), b.cpu().float().numpy())
                v = our_func(a, b)
                self.assertEqual(ref, v.float(), atol=0.01, rtol=0.01)
            else:
                ref = ref_func(a.cpu().numpy(), b.cpu().numpy())
                v = our_func(a, b)
                self.assertEqual(ref, v)

        # simple test
        a = torch.randn(64, 2, dtype=dtype, device=device) - 0.5
        b = torch.randn(64, 2, dtype=dtype, device=device) - 0.5
        _test_helper(a, b)
        _test_helper(a[:3], b[:3])

        # large value test for numerical stability
        a *= 10000
        b *= 10000
        _test_helper(a, b)
        _test_helper(a[:3], b[:3])

        a = torch.tensor(
            [float("inf"), float("-inf"), float("inf"), float("nan")],
            dtype=dtype,
            device=device,
        )
        b = torch.tensor(
            [float("inf"), float("-inf"), float("-inf"), float("nan")],
            dtype=dtype,
            device=device,
        )
        _test_helper(a, b)

    @skipIfTorchDynamo()  # complex infs/nans differ under Dynamo/Inductor
    @dtypesIfCUDA(torch.float32, torch.float64, torch.bfloat16)
    @dtypes(
        torch.float32, torch.float64, torch.bfloat16, torch.complex64, torch.complex128
    )
    def test_logaddexp(self, device, dtype):
        if sys.version_info >= (3, 12) and dtype in (torch.complex64, torch.complex128):
            return self.skipTest("complex flaky in 3.12")
        self._test_logaddexp(device, dtype, base2=False)

    @dtypes(torch.float32, torch.float64, torch.bfloat16)
    def test_logaddexp2(self, device, dtype):
        self._test_logaddexp(device, dtype, base2=True)

    def test_add(self, device):
        dtypes = floating_and_complex_types()
        for dtype in dtypes:
            # [res] torch.add([res,] tensor1, tensor2)
            m1 = torch.randn(100, 100, dtype=dtype, device=device)
            v1 = torch.randn(100, dtype=dtype, device=device)

            # contiguous
            res1 = torch.add(m1[4], v1)
            res2 = res1.clone().zero_()
            for i in range(m1.size(1)):
                res2[i] = m1[4, i] + v1[i]
            self.assertEqual(res1, res2)

            m1 = torch.randn(100, 100, device=device)
            v1 = torch.randn(100, device=device)

            # non-contiguous
            res1 = torch.add(m1[:, 4], v1)
            res2 = res1.clone().zero_()
            for i in range(m1.size(0)):
                res2[i] = m1[i, 4] + v1[i]
            self.assertEqual(res1, res2)

            # [res] torch.add([res,] tensor, value)
            m1 = torch.randn(10, 10, device=device)

            # contiguous
            res1 = m1.clone()
            res1[3].add_(2)
            res2 = m1.clone()
            for i in range(m1.size(1)):
                res2[3, i] = res2[3, i] + 2
            self.assertEqual(res1, res2)

            # non-contiguous
            m1 = torch.randn(10, 10, device=device)
            res1 = m1.clone()
            res1[:, 3].add_(2)
            res2 = m1.clone()
            for i in range(m1.size(0)):
                res2[i, 3] = res2[i, 3] + 2
            self.assertEqual(res1, res2)

            # inter-type
            m1 = torch.randn(10, 10, dtype=dtype, device=device)
            self.assertEqual(m1 + 3, m1 + torch.tensor(3))
            self.assertEqual(3 + m1, torch.tensor(3) + m1)

            # contiguous + non-contiguous
            m1 = torch.randn(10, 10, dtype=dtype, device=device)
            m2 = torch.randn(10, 10, dtype=dtype, device=device).t()
            res = m1 + m2
            self.assertTrue(res.is_contiguous())
            self.assertEqual(res, m1 + m2.contiguous())

            # 1d + empty
            m1 = torch.tensor([1.0], dtype=dtype, device=device)
            m2 = torch.tensor([], dtype=dtype, device=device)
            self.assertEqual(m1 + m2, [])

        # inter-type unint8
        one = torch.tensor(1, dtype=torch.uint8, device=device)
        self.assertEqual(torch.add(one, 1), 2)
        self.assertEqual(torch.add(one, 1).dtype, torch.uint8)

        # bool
        m1 = torch.tensor(
            [True, False, False, True, False, False], dtype=torch.bool, device=device
        )
        m2 = torch.tensor(
            [True, True, False, False, False, True], dtype=torch.bool, device=device
        )
        expected = torch.tensor(
            [True, True, False, True, False, True], dtype=torch.bool, device=device
        )
        self.assertEqual(m1 + m2, expected)

        # fused multiply add
        a = torch.zeros(2, 3, dtype=torch.bool, device=device)
        res = torch.add(a, a, alpha=0)
        expected = torch.zeros(2, 3, device=device).bool()
        self.assertEqual(res, expected)

        # bfloat16
        m1 = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
        m2 = torch.tensor([3.0, 4.0], dtype=torch.bfloat16)
        self.assertEqual(m1 + m2, torch.tensor([4.0, 6.0], dtype=torch.bfloat16))

        # different alpha types
        m1 = torch.tensor([2 + 3j, 4 + 5j], dtype=torch.complex64, device=device)
        m2 = torch.tensor([4 + 5j, 2 + 3j], dtype=torch.complex64, device=device)
        # add complex numbers with float alpha
        res = torch.add(m1, m2, alpha=0.1)
        expected = torch.tensor(
            [2.4000 + 3.5000j, 4.2000 + 5.3000j], dtype=torch.complex64, device=device
        )
        self.assertEqual(res, expected)

        # add complex numbers with complex alpha
        res = torch.add(m1, m2, alpha=complex(0.1, 0.2))
        expected = torch.tensor(
            [1.4000 + 4.3000j, 3.6000 + 5.7000j], dtype=torch.complex64, device=device
        )
        self.assertEqual(res, expected)

        # add complex numbers with integer alpha
        res = torch.add(m1, m2, alpha=2)
        expected = torch.tensor(
            [10.0 + 13.0j, 8.0 + 11.0j], dtype=torch.complex64, device=device
        )
        self.assertEqual(res, expected)

        # mismatched alpha
        m1 = torch.tensor([1], dtype=torch.int8, device=device)
        m2 = torch.tensor([2], dtype=torch.int8, device=device)
        self.assertRaisesRegex(
            RuntimeError,
            r"Boolean alpha only supported for Boolean results\.",
            lambda: torch.add(m1, m2, alpha=True),
        )
        self.assertRaisesRegex(
            RuntimeError,
            r"For integral input tensors, argument alpha must not be a floating point number\.",
            lambda: torch.add(m1, m2, alpha=1.0),
        )

        # mismatched alpha, float / double tensor and complex alpha
        msg = r"For non-complex input tensors, argument alpha must not be a complex number\."
        m1 = torch.tensor([3.0, 4.0], device=device)
        m2 = torch.tensor([4.0, 3.0], device=device)
        self.assertRaisesRegex(
            RuntimeError, msg, lambda: torch.add(m1, m2, alpha=complex(0.1, 0.2))
        )

        m1 = torch.tensor([3.0, 4.0], dtype=torch.double, device=device)
        m2 = torch.tensor([4.0, 3.0], dtype=torch.double, device=device)
        self.assertRaisesRegex(
            RuntimeError, msg, lambda: torch.add(m1, m2, alpha=complex(0.1, 0.2))
        )

        # complex
        m1 = torch.tensor((4.0000 + 4.0000j), dtype=torch.complex64)
        m2 = torch.tensor(4.0, dtype=torch.float64)
        self.assertRaisesRegex(
            RuntimeError,
            r"result type ComplexFloat can't be cast to the desired output type Double",
            lambda: torch.add(m1, m1, out=m2),
        )

    @onlyCUDA
    def test_addsub_half_tensor(self, device):
        x = torch.tensor([60000.0], dtype=torch.half, device=device)
        for op, y, alpha in (
            (torch.add, torch.tensor([-60000.0], dtype=torch.half, device=device), 2),
            (torch.sub, torch.tensor([60000.0], dtype=torch.half, device=device), 2),
            (torch.add, -70000.0, 1),
            (torch.sub, 70000.0, 1),
        ):
            actual = op(x, y, alpha=alpha)
            self.assertTrue(not (actual.isnan() or actual.isinf()))

    def test_sub_typing(self, device):
        m1 = torch.tensor(
            [True, False, False, True, False, False], dtype=torch.bool, device=device
        )
        m2 = torch.tensor(
            [True, True, False, False, False, True], dtype=torch.bool, device=device
        )
        self.assertRaisesRegex(
            RuntimeError,
            r"Subtraction, the `\-` operator, with two bool tensors is not supported. "
            r"Use the `\^` or `logical_xor\(\)` operator instead.",
            lambda: m1 - m2,
        )
        self.assertRaisesRegex(
            RuntimeError,
            r"Subtraction, the `\-` operator, with a bool tensor is not supported. "
            r"If you are trying to invert a mask, use the `\~` or `logical_not\(\)` operator instead.",
            lambda: 1 - m1,
        )
        self.assertRaisesRegex(
            RuntimeError,
            r"Subtraction, the `\-` operator, with a bool tensor is not supported. "
            r"If you are trying to invert a mask, use the `\~` or `logical_not\(\)` operator instead.",
            lambda: m2 - 1,
        )

        # mismatched alpha
        m1 = torch.tensor([1], dtype=torch.int8, device=device)
        m2 = torch.tensor([2], dtype=torch.int8, device=device)
        self.assertRaisesRegex(
            RuntimeError,
            r"Boolean alpha only supported for Boolean results\.",
            lambda: torch.sub(m1, m2, alpha=True),
        )
        self.assertRaisesRegex(
            RuntimeError,
            r"For integral input tensors, argument alpha must not be a floating point number\.",
            lambda: torch.sub(m1, m2, alpha=1.0),
        )

    def test_mul(self, device):
        m1 = torch.randn(10, 10, device=device)
        res1 = m1.clone()
        res1[:, 3].mul_(2)
        res2 = m1.clone()
        for i in range(res1.size(0)):
            res2[i, 3] = res2[i, 3] * 2
        self.assertEqual(res1, res2)

        a1 = torch.tensor([True, False, False, True], dtype=torch.bool, device=device)
        a2 = torch.tensor([True, False, True, False], dtype=torch.bool, device=device)
        self.assertEqual(
            a1 * a2,
            torch.tensor([True, False, False, False], dtype=torch.bool, device=device),
        )

        if device == "cpu":
            a1 = torch.tensor([0.1, 0.1], dtype=torch.bfloat16, device=device)
            a2 = torch.tensor([1.1, 0.1], dtype=torch.bfloat16, device=device)
            self.assertEqual(
                a1 * a2,
                torch.tensor([0.11, 0.01], dtype=torch.bfloat16, device=device),
                atol=0.01,
                rtol=0,
            )
            self.assertEqual(a1.mul(a2), a1 * a2)

    def test_bool_tensor_comparison_ops(self, device):
        a = torch.tensor(
            [True, False, True, False, True, False], dtype=torch.bool, device=device
        )
        b = torch.tensor(
            [True, False, True, True, True, True], dtype=torch.bool, device=device
        )
        self.assertEqual(
            a == b, torch.tensor([1, 1, 1, 0, 1, 0], dtype=torch.bool, device=device)
        )
        self.assertEqual(
            a != b, torch.tensor([0, 0, 0, 1, 0, 1], dtype=torch.bool, device=device)
        )
        self.assertEqual(
            a < b, torch.tensor([0, 0, 0, 1, 0, 1], dtype=torch.bool, device=device)
        )
        self.assertEqual(
            a > b, torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.bool, device=device)
        )
        self.assertEqual(
            a >= b, torch.tensor([1, 1, 1, 0, 1, 0], dtype=torch.bool, device=device)
        )
        self.assertEqual(
            a <= b, torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.bool, device=device)
        )
        self.assertEqual(
            a > False, torch.tensor([1, 0, 1, 0, 1, 0], dtype=torch.bool, device=device)
        )
        self.assertEqual(
            a == torch.tensor(True, dtype=torch.bool, device=device),
            torch.tensor([1, 0, 1, 0, 1, 0], dtype=torch.bool, device=device),
        )
        self.assertEqual(
            a == torch.tensor(0, dtype=torch.bool, device=device),
            torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.bool, device=device),
        )
        self.assertFalse(a.equal(b))

    @dtypes(*all_types_and(torch.half, torch.bfloat16, torch.bool))
    def test_logical(self, device, dtype):
        if dtype != torch.bool:
            x = torch.tensor([1, 2, 3, 4], device=device, dtype=dtype)
            b = torch.tensor([2], device=device, dtype=dtype)
            self.assertEqual(x.lt(2), torch.tensor([True, False, False, False]))
            self.assertEqual(x.le(2), torch.tensor([True, True, False, False]))
            self.assertEqual(x.ge(2), torch.tensor([False, True, True, True]))
            self.assertEqual(x.gt(2), torch.tensor([False, False, True, True]))
            self.assertEqual(x.eq(2), torch.tensor([False, True, False, False]))
            self.assertEqual(x.ne(2), torch.tensor([True, False, True, True]))

            self.assertEqual(x.lt(b), torch.tensor([True, False, False, False]))
            self.assertEqual(x.le(b), torch.tensor([True, True, False, False]))
            self.assertEqual(x.ge(b), torch.tensor([False, True, True, True]))
            self.assertEqual(x.gt(b), torch.tensor([False, False, True, True]))
            self.assertEqual(x.eq(b), torch.tensor([False, True, False, False]))
            self.assertEqual(x.ne(b), torch.tensor([True, False, True, True]))
        else:
            x = torch.tensor([True, False, True, False], device=device)
            self.assertEqual(x.lt(True), torch.tensor([False, True, False, True]))
            self.assertEqual(x.le(True), torch.tensor([True, True, True, True]))
            self.assertEqual(x.ge(True), torch.tensor([True, False, True, False]))
            self.assertEqual(x.gt(True), torch.tensor([False, False, False, False]))
            self.assertEqual(x.eq(True), torch.tensor([True, False, True, False]))
            self.assertEqual(x.ne(True), torch.tensor([False, True, False, True]))

    def test_atan2(self, device):
        def _test_atan2_with_size(size, device):
            a = torch.rand(size=size, device=device, dtype=torch.double)
            b = torch.rand(size=size, device=device, dtype=torch.double)
            actual = a.atan2(b)
            x = a.view(-1)
            y = b.view(-1)
            expected = torch.tensor(
                [math.atan2(x[i].item(), y[i].item()) for i in range(x.numel())],
                device=device,
                dtype=torch.double,
            )
            self.assertEqual(expected, actual.view(-1), rtol=0, atol=0.02)

            # bfloat16/float16
            for lowp_dtype in [torch.bfloat16, torch.float16]:
                if lowp_dtype == torch.bfloat16:
                    rtol = 0
                    atol = 0.02
                else:
                    rtol = 0
                    atol = 0.001
                a_16 = a.to(dtype=lowp_dtype)
                b_16 = b.to(dtype=lowp_dtype)
                actual_16 = a_16.atan2(b_16)
                self.assertEqual(actual_16, actual.to(dtype=lowp_dtype))
                self.assertEqual(
                    expected,
                    actual_16.view(-1),
                    exact_dtype=False,
                    rtol=rtol,
                    atol=atol,
                )

        _test_atan2_with_size((2, 2), device)
        _test_atan2_with_size((3, 3), device)
        _test_atan2_with_size((5, 5), device)

    def test_atan2_edgecases(self, device):
        def _test_atan2(x, y, expected, device, dtype):
            expected_tensor = torch.tensor([expected], dtype=dtype, device=device)
            x_tensor = torch.tensor([x], dtype=dtype, device=device)
            y_tensor = torch.tensor([y], dtype=dtype, device=device)
            actual = torch.atan2(y_tensor, x_tensor)
            self.assertEqual(expected_tensor, actual, rtol=0, atol=0.02)

        for dtype in [torch.float, torch.double]:
            _test_atan2(0, 0, 0, device, dtype)
            _test_atan2(0, 1, math.pi / 2, device, dtype)
            _test_atan2(0, -1, math.pi / -2, device, dtype)
            _test_atan2(-1, 0, math.pi, device, dtype)
            _test_atan2(1, 0, 0, device, dtype)
            _test_atan2(-1, -1, math.pi * -3 / 4, device, dtype)
            _test_atan2(1, 1, math.pi / 4, device, dtype)
            _test_atan2(1, -1, math.pi / -4, device, dtype)
            _test_atan2(-1, 1, math.pi * 3 / 4, device, dtype)

    def test_trapezoid(self, device):
        def test_dx(sizes, dim, dx, device):
            t = torch.randn(sizes, device=device)
            actual = torch.trapezoid(t, dx=dx, dim=dim)
            expected = np.trapz(t.cpu().numpy(), dx=dx, axis=dim)  # noqa: NPY201
            self.assertEqual(expected.shape, actual.shape)
            self.assertEqual(expected, actual, exact_dtype=False)

        def test_x(sizes, dim, x, device):
            t = torch.randn(sizes, device=device)
            actual = torch.trapezoid(t, x=torch.tensor(x, device=device), dim=dim)
            expected = np.trapz(t.cpu().numpy(), x=x, axis=dim)  # noqa: NPY201
            self.assertEqual(expected.shape, actual.shape)
            self.assertEqual(expected, actual.cpu(), exact_dtype=False)

        test_dx((2, 3, 4), 1, 1, device)
        test_dx((10, 2), 0, 0.1, device)
        test_dx((1, 10), 0, 2.3, device)
        test_dx((0, 2), 0, 1.0, device)
        test_dx((0, 2), 1, 1.0, device)
        test_x((2, 3, 4), 1, [1.0, 2.0, 3.0], device)
        test_x(
            (10, 2), 0, [2.0, 3.0, 4.0, 7.0, 11.0, 14.0, 22.0, 26.0, 26.1, 30.3], device
        )
        test_x((1, 10), 0, [1.0], device)
        test_x((0, 2), 0, [], device)
        test_x((0, 2), 1, [1.0, 2.0], device)
        test_x((2, 3, 4), -1, [1.0, 2.0, 3.0, 4.0], device)
        test_x((2, 3, 4), 0, [1.0, 2.0], device)
        test_x((2, 3, 4), 1, [1.0, 2.0, 3.0], device)
        test_x((2, 3, 4), 2, [1.0, 2.0, 3.0, 4.0], device)
        test_x((2, 2, 4), -1, [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]], device)
        with self.assertRaisesRegex(IndexError, "Dimension out of range"):
            test_x((2, 3), 2, [], device)
            test_dx((2, 3), 2, 1.0, device)
        with self.assertRaisesRegex(
            RuntimeError, "There must be one `x` value for each sample point"
        ):
            test_x((2, 3), 1, [1.0, 2.0], device)
            test_x((2, 3), 1, [1.0, 2.0, 3.0, 4.0], device)

    @skipIf(not TEST_SCIPY, "Scipy required for the test.")
    def test_cumulative_trapezoid(self, device):
        import scipy.integrate

        if hasattr(scipy.integrate, "cumulative_trapezoid"):
            _scipy_cumulative_trapezoid = scipy.integrate.cumulative_trapezoid
        else:  # Older version of SciPy uses a different name
            _scipy_cumulative_trapezoid = scipy.integrate.cumtrapz

        def scipy_cumulative_trapezoid(y, x=None, dx=1.0, axis=-1, initial=None):
            if y.shape[axis] == 0:
                return np.empty_like(y)
            else:
                return _scipy_cumulative_trapezoid(y, x, dx, axis, initial)

        def test_dx(sizes, dim, dx, device):
            t = torch.randn(sizes, device=device)
            y = t.cpu().numpy()
            actual = torch.cumulative_trapezoid(t, dx=dx, dim=dim)
            expected = scipy_cumulative_trapezoid(t.cpu().numpy(), dx=dx, axis=dim)
            self.assertEqual(expected.shape, actual.shape)
            self.assertEqual(expected, actual, exact_dtype=False, atol=1e-4, rtol=1e-4)

        def test_x(sizes, dim, x, device):
            t = torch.randn(sizes, device=device)
            actual = torch.cumulative_trapezoid(
                t, x=torch.tensor(x, device=device), dim=dim
            )
            expected = scipy_cumulative_trapezoid(t.cpu().numpy(), x=x, axis=dim)
            self.assertEqual(expected.shape, actual.shape)
            self.assertEqual(
                expected, actual.cpu(), exact_dtype=False, atol=1e-4, rtol=1e-4
            )

        def test_empty_x(sizes, dim, x, device):
            t = torch.randn(sizes, device=device)
            actual = torch.cumulative_trapezoid(
                t, x=torch.tensor(x, device=device), dim=dim
            )
            self.assertEqual(torch.empty(actual.shape), actual)

        test_dx((2,), -1, 1, device)
        test_dx((3, 3), -1, 1, device)
        test_dx((4, 2), 0, 1, device)
        test_dx((2, 3, 4), 1, 1, device)
        test_dx((10, 2), 0, 0.1, device)
        test_dx((1, 10), 0, 2.3, device)
        test_dx((0, 2), 0, 1.0, device)
        test_dx((0, 2), 1, 1.0, device)
        test_dx((512, 512), 1, 1.0, device)
        test_dx((100, 100, 100), 1, 1.0, device)

        test_x((2,), -1, [100, 50], device)
        test_x((4, 2), 0, [2, 3, 4, 5], device)
        test_x((2, 3, 4), 1, [1.0, 2.0, 3.0], device)
        test_x(
            (10, 2), 0, [2.0, 3.0, 4.0, 7.0, 11.0, 14.0, 22.0, 26.0, 26.1, 30.3], device
        )
        test_x((1, 10), 0, [1.0], device)
        test_x((0, 2), 1, [1, 2], device)
        test_x((2, 3, 4), -1, [1.0, 2.0, 3.0, 4.0], device)
        test_x((2, 3, 4), 0, [1.0, 2.0], device)
        test_x((2, 3, 4), 1, [1.0, 2.0, 3.0], device)
        test_x((2, 3, 4), 2, [1.0, 2.0, 3.0, 4.0], device)

        test_empty_x(
            (0, 2), 0, [], device
        )  # SciPy failing when x == [], but our version returns empty

        with self.assertRaisesRegex(IndexError, "Dimension out of range"):
            test_x((2, 3), 2, [], device)
            test_dx((2, 3), 2, 1.0, device)
        with self.assertRaisesRegex(
            RuntimeError, "There must be one `x` value for each sample point"
        ):
            test_x((2, 3), 1, [1.0, 2.0], device)
            test_x((0, 2), 0, [1.0, 2.0], device)
            test_x((2, 3), 1, [1.0, 2.0, 3.0, 4.0], device)
        with self.assertRaisesRegex(
            RuntimeError, "Currently, we only support dx as a real number"
        ):
            test_dx((2, 2), -1, complex(1, 1), device)
        with self.assertRaisesRegex(
            TypeError, "received an invalid combination of arguments"
        ):
            actual = torch.cumulative_trapezoid(
                torch.randn((3, 3)), x=torch.randn((3, 3)), dx=3
            )

    @skipMeta
    @dtypes(torch.double)
    def test_pow_scalar_overloads_mem_overlap(self, device, dtype):
        sz = 3
        doubles = torch.randn(2 * sz, dtype=dtype, device=device)
        self.check_internal_mem_overlap(lambda t: t.pow_(42), 1, dtype, device)
        self.unary_check_input_output_mem_overlap(
            doubles, sz, lambda input, out: torch.pow(input, 42, out=out)
        )
        self.unary_check_input_output_mem_overlap(
            doubles, sz, lambda input, out: torch.pow(42, input, out=out)
        )

    @dtypes(
        *list(
            product(
                all_types_and_complex_and(torch.half, torch.bfloat16),
                all_types_and_complex_and(torch.half, torch.bfloat16),
            )
        )
    )
    def test_float_power(self, device, dtypes):
        def to_np(value):
            if isinstance(value, torch.Tensor) and value.dtype == torch.bfloat16:
                return value.to(torch.float).cpu().numpy()
            return value.cpu().numpy() if isinstance(value, torch.Tensor) else value

        base_dtype = dtypes[0]
        exp_dtype = dtypes[1]
        out_dtype = (
            torch.complex128
            if base_dtype.is_complex or exp_dtype.is_complex
            else torch.float64
        )

        base = make_tensor((30,), dtype=base_dtype, device=device, low=1, high=100)
        # Complex and real results do not agree between PyTorch and NumPy when computing negative and zero power of 0
        # Related: https://github.com/pytorch/pytorch/issues/48000
        # base[0] = base[3] = base[7] = 0
        exp = make_tensor((30,), dtype=exp_dtype, device=device, low=-2, high=2)
        exp[0] = exp[4] = exp[6] = 0

        expected = torch.from_numpy(np.float_power(to_np(base), to_np(exp)))

        exponents = [-2.8, -2, -1, -0.5, 0.5, 1, 2]
        complex_exponents = exponents + [
            -2.5j,
            -1.0j,
            1.0j,
            2.5j,
            1.0 + 1.0j,
            -1.0 - 1.5j,
            3.3j,
        ]

        for op in (
            torch.float_power,
            torch.Tensor.float_power,
            torch.Tensor.float_power_,
        ):
            # Case of Tensor x Tensor
            if op is torch.Tensor.float_power_ and base_dtype != out_dtype:
                with self.assertRaisesRegex(
                    RuntimeError, "operation's result requires dtype"
                ):
                    op(base.clone(), exp)
            else:
                result = op(base.clone(), exp)
                self.assertEqual(expected, result)

            if op is torch.float_power:
                out = torch.empty_like(base).to(device=device, dtype=out_dtype)
                op(base, exp, out=out)
                self.assertEqual(expected, out)

            # Case of Tensor x Scalar
            for i in complex_exponents if exp_dtype.is_complex else exponents:
                out_dtype_scalar_exp = (
                    torch.complex128
                    if base_dtype.is_complex or type(i) == complex
                    else torch.float64
                )
                expected_scalar_exp = torch.from_numpy(np.float_power(to_np(base), i))

                if (
                    op is torch.Tensor.float_power_
                    and base_dtype != out_dtype_scalar_exp
                ):
                    with self.assertRaisesRegex(
                        RuntimeError, "operation's result requires dtype"
                    ):
                        op(base.clone(), i)
                else:
                    result = op(base.clone(), i)
                    self.assertEqual(expected_scalar_exp, result)

                if op is torch.float_power:
                    out = torch.empty_like(base).to(
                        device=device, dtype=out_dtype_scalar_exp
                    )
                    op(base, i, out=out)
                    self.assertEqual(expected_scalar_exp, out)

        # Case of Scalar x Tensor
        for i in complex_exponents if base_dtype.is_complex else exponents:
            out_dtype_scalar_base = (
                torch.complex128
                if exp_dtype.is_complex or type(i) == complex
                else torch.float64
            )
            expected_scalar_base = torch.from_numpy(np.float_power(i, to_np(exp)))

            result = torch.float_power(i, exp)
            self.assertEqual(expected_scalar_base, result)

            out = torch.empty_like(exp).to(device=device, dtype=out_dtype_scalar_base)
            torch.float_power(i, exp, out=out)
            self.assertEqual(expected_scalar_base, out)

    def test_float_power_exceptions(self, device):
        def _promo_helper(x, y):
            for i in (x, y):
                if type(i) == complex:
                    return torch.complex128
                elif type(i) == torch.Tensor and i.is_complex():
                    return torch.complex128
            return torch.double

        test_cases = (
            (torch.tensor([-2, -1, 0, 1, 2], device=device), -0.25),
            (
                torch.tensor([-1.0j, 0j, 1.0j, 1.0 + 1.0j, -1.0 - 1.5j], device=device),
                2.0,
            ),
        )
        for base, exp in test_cases:
            for out_dtype in (torch.long, torch.float, torch.double, torch.cdouble):
                out = torch.empty(1, device=device, dtype=out_dtype)
                required_dtype = _promo_helper(base, exp)

                if out.dtype == required_dtype:
                    torch.float_power(base, exp, out=out)
                else:
                    with self.assertRaisesRegex(
                        RuntimeError, "operation's result requires dtype"
                    ):
                        torch.float_power(base, exp, out=out)

                if base.dtype == required_dtype:
                    torch.Tensor.float_power_(base.clone(), exp)
                else:
                    with self.assertRaisesRegex(
                        RuntimeError, "operation's result requires dtype"
                    ):
                        torch.Tensor.float_power_(base.clone(), exp)

    @skipIf(not TEST_SCIPY, "Scipy required for the test.")
    @dtypes(
        *product(
            all_types_and(torch.half, torch.bool), all_types_and(torch.half, torch.bool)
        )
    )
    def test_xlogy_xlog1py(self, device, dtypes):
        x_dtype, y_dtype = dtypes

        def out_variant_helper(torch_fn, x, y):
            expected = torch_fn(x, y)
            out = torch.empty_like(expected)
            torch_fn(x, y, out=out)
            self.assertEqual(expected, out)

        def xlogy_inplace_variant_helper(x, y):
            if x.dtype in integral_types_and(torch.bool):
                with self.assertRaisesRegex(
                    RuntimeError, "can't be cast to the desired output type"
                ):
                    x.clone().xlogy_(y)
            else:
                expected = torch.empty_like(x)
                torch.xlogy(x, y, out=expected)
                inplace_out = x.clone().xlogy_(y)
                self.assertEqual(expected, inplace_out)

        def test_helper(torch_fn, reference_fn, inputs, scalar=None):
            x, y, z = inputs
            torch_fn_partial = partial(torch_fn, x)
            reference_fn_partial = partial(reference_fn, x.cpu().numpy())
            self.compare_with_numpy(
                torch_fn_partial, reference_fn_partial, x, exact_dtype=False
            )
            self.compare_with_numpy(
                torch_fn_partial, reference_fn_partial, y, exact_dtype=False
            )
            self.compare_with_numpy(
                torch_fn_partial, reference_fn_partial, z, exact_dtype=False
            )

            val = scalar if scalar is not None else x
            out_variant_helper(torch_fn, val, x)
            out_variant_helper(torch_fn, val, y)
            out_variant_helper(torch_fn, val, z)

        # Tensor-Tensor Test (tensor of same and different shape)
        x = make_tensor((3, 2, 4, 5), dtype=x_dtype, device=device, low=0.5, high=1000)
        y = make_tensor((3, 2, 4, 5), dtype=y_dtype, device=device, low=0.5, high=1000)
        z = make_tensor((4, 5), dtype=y_dtype, device=device, low=0.5, high=1000)

        x_1p = make_tensor(
            (3, 2, 4, 5), dtype=x_dtype, device=device, low=-0.5, high=1000
        )
        y_1p = make_tensor(
            (3, 2, 4, 5), dtype=y_dtype, device=device, low=-0.5, high=1000
        )
        z_1p = make_tensor((4, 5), dtype=y_dtype, device=device, low=-0.5, high=1000)

        xlogy_fns = torch.xlogy, scipy.special.xlogy
        xlog1py_fns = torch.special.xlog1py, scipy.special.xlog1py

        test_helper(*xlogy_fns, (x, y, z))
        xlogy_inplace_variant_helper(x, x)
        xlogy_inplace_variant_helper(x, y)
        xlogy_inplace_variant_helper(x, z)
        test_helper(*xlog1py_fns, (x_1p, y_1p, z_1p))

        # Scalar-Tensor Test
        test_helper(*xlogy_fns, (x, y, z), 3.14)
        test_helper(*xlog1py_fns, (x_1p, y_1p, z_1p), 3.14)

        # Special Values Tensor-Tensor
        t = torch.tensor(
            [-1.0, 0.0, 1.0, 2.0, float("inf"), -float("inf"), float("nan")],
            device=device,
        )
        zeros = torch.zeros(7, dtype=y_dtype, device=device)

        def test_zeros_special_helper(torch_fn, reference_fn, scalar=False):
            zeros_t = 0 if scalar else zeros
            zeros_np = 0 if scalar else zeros.cpu().numpy()
            torch_fn_partial = partial(torch_fn, zeros_t)
            reference_fn_partial = partial(reference_fn, zeros_np)
            self.compare_with_numpy(
                torch_fn_partial, reference_fn_partial, t, exact_dtype=False
            )
            out_variant_helper(torch_fn, zeros_t, t)

        test_zeros_special_helper(*xlogy_fns)
        xlogy_inplace_variant_helper(zeros, t)
        test_zeros_special_helper(*xlog1py_fns)

        # Special Values Scalar-Tensor
        test_zeros_special_helper(*xlogy_fns, scalar=True)
        test_zeros_special_helper(*xlog1py_fns, scalar=True)

    @dtypes(torch.float64)
    def test_xlogy_xlog1py_gradients(self, device, dtype):
        make_arg = partial(torch.tensor, dtype=dtype, device=device, requires_grad=True)

        zeros = torch.zeros((2,), dtype=dtype, device=device)

        x = make_arg([0.0, 0.0])
        y = make_arg([-1.5, 0.0])
        torch.special.xlogy(x, y).sum().backward()
        self.assertEqual(x.grad, zeros)

        x = make_arg([0.0, 0.0])
        y = make_arg([-2.5, -1.0])
        torch.special.xlog1py(x, y).sum().backward()
        self.assertEqual(x.grad, zeros)

    def test_xlogy_xlog1py_scalar_type_promotion(self, device):
        # Test that python numbers don't participate in type promotion at the same
        # priority level as 0-dim tensors
        t = torch.randn((), dtype=torch.float32, device=device)

        self.assertEqual(t.dtype, torch.xlogy(t, 5).dtype)
        self.assertEqual(t.dtype, torch.xlogy(t, 5.0).dtype)
        self.assertEqual(t.dtype, torch.special.xlog1py(t, 5).dtype)
        self.assertEqual(t.dtype, torch.special.xlog1py(t, 5.0).dtype)

        self.assertEqual(t.dtype, torch.xlogy(5, t).dtype)
        self.assertEqual(t.dtype, torch.xlogy(5.0, t).dtype)
        self.assertEqual(t.dtype, torch.special.xlog1py(5, t).dtype)
        self.assertEqual(t.dtype, torch.special.xlog1py(5.0, t).dtype)

    @skipIf(not TEST_SCIPY, "Scipy required for the test.")
    def test_xlogy_xlog1py_bfloat16(self, device):
        def _compare_helper(x, y, torch_fn, reference_fn):
            x_np = x if isinstance(x, float) else x.cpu().to(torch.float).numpy()
            y_np = y if isinstance(y, float) else y.cpu().to(torch.float).numpy()
            expected = torch.from_numpy(reference_fn(x_np, y_np))
            actual = torch_fn(x, y)
            self.assertEqual(expected, actual, exact_dtype=False)

        x_dtype, y_dtype = torch.bfloat16, torch.bfloat16

        # Tensor-Tensor Test (tensor of same and different shape)
        x = make_tensor((3, 2, 4, 5), dtype=x_dtype, device=device, low=0.5, high=1000)
        y = make_tensor((3, 2, 4, 5), dtype=y_dtype, device=device, low=0.5, high=1000)
        z = make_tensor((4, 5), dtype=y_dtype, device=device, low=0.5, high=1000)

        x_1p = make_tensor(
            (3, 2, 4, 5), dtype=x_dtype, device=device, low=-0.8, high=1000
        )
        y_1p = make_tensor(
            (3, 2, 4, 5), dtype=y_dtype, device=device, low=-0.8, high=1000
        )
        z_1p = make_tensor((4, 5), dtype=y_dtype, device=device, low=-0.8, high=1000)

        xlogy_fns = torch.xlogy, scipy.special.xlogy
        xlog1py_fns = torch.special.xlog1py, scipy.special.xlog1py

        _compare_helper(x, x, *xlogy_fns)
        _compare_helper(x, y, *xlogy_fns)
        _compare_helper(x, z, *xlogy_fns)
        _compare_helper(x, 3.14, *xlogy_fns)
        _compare_helper(y, 3.14, *xlogy_fns)
        _compare_helper(z, 3.14, *xlogy_fns)

        _compare_helper(x_1p, x_1p, *xlog1py_fns)
        _compare_helper(x_1p, y_1p, *xlog1py_fns)
        _compare_helper(x_1p, z_1p, *xlog1py_fns)
        _compare_helper(x_1p, 3.14, *xlog1py_fns)
        _compare_helper(y_1p, 3.14, *xlog1py_fns)
        _compare_helper(z_1p, 3.14, *xlog1py_fns)

        # Special Values Tensor-Tensor
        t = torch.tensor(
            [-1.0, 0.0, 1.0, 2.0, float("inf"), -float("inf"), float("nan")],
            device=device,
        )
        zeros = torch.tensor(7, dtype=y_dtype, device=device)

        _compare_helper(t, zeros, *xlogy_fns)
        _compare_helper(t, 0.0, *xlogy_fns)

        _compare_helper(t, zeros, *xlog1py_fns)
        _compare_helper(t, 0.0, *xlog1py_fns)

    @dtypes(*product(all_types_and(torch.bool), all_types_and(torch.bool)))
    @skipIf(not TEST_SCIPY, "Scipy required for the test.")
    @slowTest
    def test_zeta(self, device, dtypes):
        x_dtype, q_dtype = dtypes

        def test_helper(x, q):
            x_np = x if isinstance(x, float) else x.cpu().numpy()
            q_np = q if isinstance(q, float) else q.cpu().numpy()
            expected = torch.from_numpy(scipy.special.zeta(x_np, q_np))
            actual = torch.special.zeta(x, q)

            rtol, atol = None, None
            if self.device_type == "cpu":
                rtol, atol = 1e-6, 1e-6
            self.assertEqual(expected, actual, rtol=rtol, atol=atol, exact_dtype=False)

        # x tensor - q tensor same size
        x = make_tensor((2, 3, 4), dtype=x_dtype, device=device)
        q = make_tensor((2, 3, 4), dtype=q_dtype, device=device)
        test_helper(x, q)

        # x tensor - q tensor broadcast lhs
        x = make_tensor((2, 1, 4), dtype=x_dtype, device=device)
        q = make_tensor((2, 3, 4), dtype=q_dtype, device=device)
        test_helper(x, q)

        # x tensor - q tensor broadcast rhs
        x = make_tensor((2, 3, 4), dtype=x_dtype, device=device)
        q = make_tensor((2, 1, 4), dtype=q_dtype, device=device)
        test_helper(x, q)

        # x tensor - q tensor broadcast all
        x = make_tensor((2, 3, 1), dtype=x_dtype, device=device)
        q = make_tensor((2, 1, 4), dtype=q_dtype, device=device)
        test_helper(x, q)

        # x scalar - q tensor
        for x in np.linspace(-5, 5, num=10).tolist():
            if not q_dtype.is_floating_point:
                q_dtype = torch.get_default_dtype()
            q = make_tensor((2, 3, 4), dtype=q_dtype, device=device)
            test_helper(x, q)

        # x tensor - q scalar
        for q in np.linspace(-5, 5, num=10).tolist():
            if not x_dtype.is_floating_point:
                x_dtype = torch.get_default_dtype()
            x = make_tensor((2, 3, 4), dtype=x_dtype, device=device)
            test_helper(x, q)

    @onlyCUDA
    @dtypes(torch.chalf)
    def test_mul_chalf_tensor_and_cpu_scalar(self, device, dtype):
        # Tests that Tensor and CPU Scalar work for `mul` for chalf.
        # Ideally, this should be covered by `test_complex_half_reference_testing`
        # from test_ops.py by checking reference_samples from the OpInfo.
        # But currently that doesn't work as sample generation requires support of
        # `index_select` which is not implemented for `complex32` at the
        # time of writing this test.
        # TODO: Remove this test once above issue is fixed.
        # Ref: https://github.com/pytorch/pytorch/pull/76364
        x = make_tensor((2, 2), device=device, dtype=dtype)
        self.assertEqual(x * 2.5, x * torch.tensor(2.5, device=device, dtype=dtype))


tensor_binary_ops = [
    "__lt__",
    "__le__",
    "__gt__",
    "__ge__",
    "__eq__",
    "__ne__",
    "__add__",
    "__radd__",
    "__iadd__",
    "__sub__",
    "__rsub__",
    "__isub__",
    "__mul__",
    "__rmul__",
    "__imul__",
    "__matmul__",
    "__rmatmul__",
    "__truediv__",
    "__rtruediv__",
    "__itruediv__",
    "__floordiv__",
    "__rfloordiv__",
    "__ifloordiv__",
    "__mod__",
    "__rmod__",
    "__imod__",
    "__pow__",
    "__rpow__",
    "__ipow__",
    "__lshift__",
    "__rlshift__",
    "__ilshift__",
    "__rshift__",
    "__rrshift__",
    "__irshift__",
    "__and__",
    "__rand__",
    "__iand__",
    "__xor__",
    "__rxor__",
    "__ixor__",
    "__or__",
    "__ror__",
    "__ior__",
    # Unsupported operators
    # '__imatmul__',
    # '__divmod__', '__rdivmod__', '__idivmod__',
]


# Test that binary math operations return NotImplemented for unknown types.
def generate_not_implemented_tests(cls):
    class UnknownType:
        pass

    # TODO: refactor to inline these
    _types = [
        torch.half,
        torch.float,
        torch.double,
        torch.int8,
        torch.short,
        torch.int,
        torch.long,
        torch.uint8,
    ]

    def create_test_func(op):
        @dtypes(*_types)
        def test(self, device, dtype):
            # Generate the inputs
            tensor = torch.empty((), device=device, dtype=dtype)

            # Runs the tensor op on the device
            result = getattr(tensor, op)(UnknownType())
            self.assertEqual(result, NotImplemented)

        return test

    for op in tensor_binary_ops:
        test_name = f"test_{op}_not_implemented"
        assert not hasattr(cls, test_name), f"{test_name} already in {cls.__name__}"

        setattr(cls, test_name, create_test_func(op))


generate_not_implemented_tests(TestBinaryUfuncs)
instantiate_device_type_tests(TestBinaryUfuncs, globals())

if __name__ == "__main__":
    run_tests()
