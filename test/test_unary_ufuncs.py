# Owner(s): ["module: tests"]

import torch
import numpy as np

import math
from numbers import Number
import random
import unittest

from torch import inf, nan
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
    torch_to_numpy_dtype_dict,
    numpy_to_torch_dtype_dict,
    suppress_warnings,
    TEST_SCIPY,
    slowTest,
    skipIfNoSciPy,
    IS_WINDOWS,
    gradcheck,
)
from torch.testing._internal.common_methods_invocations import (
    unary_ufuncs,
    generate_elementwise_unary_tensors,
    generate_elementwise_unary_small_value_tensors,
    generate_elementwise_unary_large_value_tensors,
    generate_elementwise_unary_extremal_value_tensors,
)
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    ops,
    dtypes,
    onlyCPU,
    onlyNativeDeviceTypes,
    onlyCUDA,
    dtypesIfCUDA,
    precisionOverride,
    dtypesIfCPU,
)

from torch.testing import make_tensor
from torch.testing._internal.common_dtype import (
    floating_types_and,
    all_types_and_complex_and,
    integral_types_and,
    get_all_math_dtypes,
    complex_types,
    all_types_and,
    floating_and_complex_types_and,
)

if TEST_SCIPY:
    import scipy

# Refer [scipy reference filter]
# Filter operators for which the reference function
# is available in the current environment (for reference_numerics tests).
reference_filtered_ops = list(filter(lambda op: op.ref is not None, unary_ufuncs))

# Tests for unary "universal functions (ufuncs)" that accept a single
# tensor and have common properties like:
#   - they are elementwise functions
#   - the input shape is the output shape
#   - they typically have method and inplace variants
#   - they typically support the out kwarg
#   - they typically have NumPy or SciPy references

# See NumPy's universal function documentation
# (https://numpy.org/doc/1.18/reference/ufuncs.html) for more details
# about the concept of ufuncs.


# TODO: port test_unary_out_op_mem_overlap
# TODO: add test for inplace variants erroring on broadcasted inputs
class TestUnaryUfuncs(TestCase):
    exact_dtype = True

    @ops(
        [_fn for _fn in unary_ufuncs if _fn.domain != (None, None)],
        allowed_dtypes=floating_types_and(torch.bfloat16, torch.half),
    )
    def test_float_domains(self, device, dtype, op):
        eps = (1e-5, 1e-3, 1e-1, 1, 2, 10, 20, 50, 100)

        low, high = op.domain
        # NOTE: the following two loops are separated for readability
        if low is not None:
            low_tensor = torch.tensor(low, device=device, dtype=dtype)
            for epsilon in eps:
                lower_tensor = low_tensor - epsilon

                # Skips the test if the difference is not representable,
                #   which can occur if, for example, the difference is small
                #   and the dtype is imprecise (like bfloat16 is)
                if lower_tensor.item() == low_tensor.item():
                    continue

                result = op(lower_tensor)
                self.assertEqual(
                    result.item(),
                    float("nan"),
                    msg=(
                        "input of {0} outside lower domain boundary"
                        " {1} produced {2}, not nan!"
                    ).format(lower_tensor.item(), low, result.item()),
                )

        if high is not None:
            high_tensor = torch.tensor(high, device=device, dtype=dtype)
            for epsilon in eps:
                higher_tensor = high_tensor + epsilon

                # See above comment
                if higher_tensor.item() == high_tensor.item():
                    continue

                result = op(higher_tensor)
                self.assertEqual(
                    result.item(),
                    float("nan"),
                    msg=(
                        "input of {0} outside upper domain boundary"
                        " {1} produced {2}, not nan!"
                    ).format(higher_tensor.item(), high, result.item()),
                )

    # Helper for comparing torch tensors and numpy arrays
    # TODO: should this or assertEqual also validate that strides are equal?
    def assertEqualHelper(
        self, actual, expected, msg, *, dtype, exact_dtype=True, **kwargs
    ):
        assert isinstance(actual, torch.Tensor)

        # Some NumPy functions return scalars, not arrays
        if isinstance(expected, Number):
            self.assertEqual(actual.item(), expected, msg, **kwargs)
        elif isinstance(expected, np.ndarray):
            # Handles exact dtype comparisons between arrays and tensors
            if exact_dtype:
                if (
                    actual.dtype is torch.bfloat16
                    or expected.dtype != torch_to_numpy_dtype_dict[actual.dtype]
                ):
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
                    elif expected.dtype == np.float64:
                        assert actual.dtype in (
                            torch.float16,
                            torch.bfloat16,
                            torch.float32,
                            torch.float64,
                        )
                    else:
                        self.fail(
                            "Expected dtype {0} but got {1}!".format(
                                expected.dtype, actual.dtype
                            )
                        )

            self.assertEqual(
                actual,
                torch.from_numpy(expected).to(actual.dtype),
                msg,
                exact_device=False,
                **kwargs
            )
        else:
            self.assertEqual(actual, expected, msg, exact_device=False, **kwargs)

    # Tests that the function and its (array-accepting) reference produce the same
    #   values on given tensors
    def _test_reference_numerics(self, dtype, op, tensors, equal_nan=True):
        def _helper_reference_numerics(
            expected, actual, msg, exact_dtype, equal_nan=True
        ):
            if not torch.can_cast(
                numpy_to_torch_dtype_dict[expected.dtype.type], dtype
            ):
                exact_dtype = False

            if dtype in [torch.uint8, torch.int8, torch.bool]:
                # NOTE: For these dtypes, PyTorch computes in the default scalar type (float)
                # while NumPy computes in float16
                self.assertEqualHelper(
                    actual,
                    expected,
                    msg,
                    dtype=dtype,
                    exact_dtype=exact_dtype,
                    rtol=1e-3,
                    atol=1e-2,
                )
            elif dtype is torch.bfloat16:
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
            elif dtype is torch.half:
                self.assertEqualHelper(
                    actual,
                    expected,
                    msg,
                    dtype=dtype,
                    exact_dtype=exact_dtype,
                    rtol=1.2e-03,
                    atol=1e-03,
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

        for t in tensors:
            t = t.input
            torch_kwargs, numpy_kwargs = op.sample_kwargs(t.device, dtype, t)
            if dtype is torch.bfloat16:
                a = t.cpu().to(torch.float32).numpy()
            elif dtype is torch.complex32:
                a = t.cpu().to(torch.complex64).numpy()
            else:
                a = t.cpu().numpy()

            actual = op(t, **torch_kwargs)
            expected = op.ref(a, **numpy_kwargs)

            # Crafts a custom error message for smaller, printable tensors
            if t.numel() < 10:
                msg = (
                    "Failed to produce expected results! Input tensor was"
                    " {0}, torch result is {1}, and reference result is"
                    " {2}."
                ).format(t, actual, expected)
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

    # Tests that the function and its (array-accepting) reference produce the same
    #   values on a range of tensors, including empty tensors, scalar tensors,
    #   1D tensors and a large 2D tensor with interesting and extremal values
    #   and noncontiguities.
    @suppress_warnings
    @ops(reference_filtered_ops)
    def test_reference_numerics_normal(self, device, dtype, op):
        tensors = generate_elementwise_unary_tensors(
            op, device=device, dtype=dtype, requires_grad=False
        )
        self._test_reference_numerics(dtype, op, tensors)

    @suppress_warnings
    @ops(reference_filtered_ops)
    def test_reference_numerics_small(self, device, dtype, op):
        if dtype in (torch.bool,):
            raise self.skipTest("bool has no small values")

        tensors = generate_elementwise_unary_small_value_tensors(
            op, device=device, dtype=dtype, requires_grad=False
        )
        self._test_reference_numerics(dtype, op, tensors)

    @suppress_warnings
    @ops(reference_filtered_ops)
    def test_reference_numerics_large(self, device, dtype, op):
        if dtype in (torch.bool, torch.uint8, torch.int8):
            raise self.skipTest("bool, uint8, and int8 dtypes have no large values")

        tensors = generate_elementwise_unary_large_value_tensors(
            op, device=device, dtype=dtype, requires_grad=False
        )
        self._test_reference_numerics(dtype, op, tensors)

    @suppress_warnings
    @ops(
        reference_filtered_ops,
        allowed_dtypes=floating_and_complex_types_and(torch.bfloat16, torch.half),
    )
    def test_reference_numerics_extremal(self, device, dtype, op):
        tensors = generate_elementwise_unary_extremal_value_tensors(
            op, device=device, dtype=dtype, requires_grad=False
        )
        self._test_reference_numerics(dtype, op, tensors)

    # Tests for testing (non)contiguity consistency
    @ops(unary_ufuncs)
    def test_contig_vs_every_other(self, device, dtype, op):
        contig = make_tensor(
            (1026,), device=device, dtype=dtype, low=op.domain[0], high=op.domain[1]
        )
        non_contig = contig[::2]

        self.assertTrue(contig.is_contiguous())
        self.assertFalse(non_contig.is_contiguous())

        torch_kwargs, _ = op.sample_kwargs(device, dtype, non_contig)
        self.assertEqual(
            op(contig, **torch_kwargs)[::2], op(non_contig, **torch_kwargs)
        )

    @ops(unary_ufuncs)
    def test_contig_vs_transposed(self, device, dtype, op):
        contig = make_tensor(
            (789, 357), device=device, dtype=dtype, low=op.domain[0], high=op.domain[1]
        )
        non_contig = contig.T

        self.assertTrue(contig.is_contiguous())
        self.assertFalse(non_contig.is_contiguous())

        torch_kwargs, _ = op.sample_kwargs(device, dtype, contig)
        self.assertEqual(op(contig, **torch_kwargs).T, op(non_contig, **torch_kwargs))

    @ops(unary_ufuncs)
    def test_non_contig(self, device, dtype, op):
        shapes = [(5, 7), (1024,)]
        for shape in shapes:
            contig = make_tensor(
                shape, dtype=dtype, device=device, low=op.domain[0], high=op.domain[1]
            )
            non_contig = torch.empty(shape + (2,), device=device, dtype=dtype)[..., 0]
            non_contig.copy_(contig)

            self.assertTrue(contig.is_contiguous())
            self.assertFalse(non_contig.is_contiguous())

            torch_kwargs, _ = op.sample_kwargs(device, dtype, contig)
            self.assertEqual(op(contig, **torch_kwargs), op(non_contig, **torch_kwargs))

    @ops(unary_ufuncs)
    def test_non_contig_index(self, device, dtype, op):
        contig = make_tensor(
            (2, 2, 1, 2),
            dtype=dtype,
            device=device,
            low=op.domain[0],
            high=op.domain[1],
        )
        non_contig = contig[:, 1, ...]
        contig = non_contig.contiguous()

        self.assertTrue(contig.is_contiguous())
        self.assertFalse(non_contig.is_contiguous())

        torch_kwargs, _ = op.sample_kwargs(device, dtype, contig)
        self.assertEqual(op(contig, **torch_kwargs), op(non_contig, **torch_kwargs))

    @ops(unary_ufuncs)
    def test_non_contig_expand(self, device, dtype, op):
        shapes = [(1, 3), (1, 7), (5, 7)]
        for shape in shapes:
            contig = make_tensor(
                shape, dtype=dtype, device=device, low=op.domain[0], high=op.domain[1]
            )
            non_contig = contig.clone().expand(3, -1, -1)

            self.assertTrue(contig.is_contiguous())
            self.assertFalse(non_contig.is_contiguous())

            torch_kwargs, _ = op.sample_kwargs(device, dtype, contig)
            contig = op(contig, **torch_kwargs)
            non_contig = op(non_contig, **torch_kwargs)
            for i in range(3):
                self.assertEqual(
                    contig, non_contig[i], msg="non-contiguous expand[" + str(i) + "]"
                )

    @ops(unary_ufuncs)
    def test_contig_size1(self, device, dtype, op):
        contig = make_tensor(
            (5, 100), dtype=dtype, device=device, low=op.domain[0], high=op.domain[1]
        )
        contig = contig[:1, :50]
        contig2 = torch.empty(contig.size(), device=device, dtype=dtype)
        contig2.copy_(contig)

        self.assertTrue(contig.is_contiguous())
        self.assertTrue(contig2.is_contiguous())

        torch_kwargs, _ = op.sample_kwargs(device, dtype, contig)
        self.assertEqual(op(contig, **torch_kwargs), op(contig2, **torch_kwargs))

    @ops(unary_ufuncs)
    def test_contig_size1_large_dim(self, device, dtype, op):
        contig = make_tensor(
            (5, 2, 3, 1, 4, 5, 3, 2, 1, 2, 3, 4),
            dtype=dtype,
            device=device,
            low=op.domain[0],
            high=op.domain[1],
        )
        contig = contig[:1, :, :, :, :, :, :, :, :, :, :, :]
        contig2 = torch.empty(contig.size(), device=device, dtype=dtype)
        contig2.copy_(contig)

        self.assertTrue(contig.is_contiguous())
        self.assertTrue(contig2.is_contiguous())

        torch_kwargs, _ = op.sample_kwargs(device, dtype, contig)
        self.assertEqual(op(contig, **torch_kwargs), op(contig2, **torch_kwargs))

    # Tests that computation on a multiple batches is the same as
    # per-batch computation.
    @ops(unary_ufuncs)
    def test_batch_vs_slicing(self, device, dtype, op):
        input = make_tensor(
            (1024, 512), dtype=dtype, device=device, low=op.domain[0], high=op.domain[1]
        )

        torch_kwargs, _ = op.sample_kwargs(device, dtype, input)
        actual = op(input, **torch_kwargs)
        expected = torch.stack([op(slice, **torch_kwargs) for slice in input])

        self.assertEqual(actual, expected)

    @dtypes(*all_types_and(torch.bool, torch.half))
    def test_nan_to_num(self, device, dtype):
        for contiguous in [False, True]:
            x = make_tensor((64, 64), low=0.0, high=100.0, dtype=dtype, device=device)

            if dtype.is_floating_point:
                # Add extremal values.
                extremals = [float("nan"), float("inf"), -float("inf")]
                for idx, extremal in zip(torch.randint(0, 63, (3,)), extremals):
                    x[idx, :] = extremal

            if not contiguous:
                x = x.T

            # With args
            nan = random.random()
            posinf = random.random() * 5
            neginf = random.random() * 10

            self.compare_with_numpy(
                lambda x: x.nan_to_num(nan=nan, posinf=posinf),
                lambda x: np.nan_to_num(x, nan=nan, posinf=posinf),
                x,
            )
            self.compare_with_numpy(
                lambda x: x.nan_to_num(posinf=posinf, neginf=neginf),
                lambda x: np.nan_to_num(x, posinf=posinf, neginf=neginf),
                x,
            )

            # Out Variant
            out = torch.empty_like(x)
            result = torch.nan_to_num(x)
            torch.nan_to_num(x, out=out)
            self.assertEqual(result, out)

            result = torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)
            torch.nan_to_num(x, out=out, nan=nan, posinf=posinf, neginf=neginf)
            self.assertEqual(result, out)

    @onlyCPU
    def test_nan_to_num_bfloat16(self, device):
        def test_dtype(fn, input, dtype):
            input = input.detach().clone().to(dtype=dtype).requires_grad_(True)
            input2 = input.detach().clone().float().requires_grad_(True)
            out = fn(input)
            out.sum().backward()
            out2 = fn(input2)
            out2.sum().backward()
            self.assertEqual(out.dtype, dtype)
            self.assertEqual(input.grad.dtype, dtype)
            self.assertEqual(out, out2, exact_dtype=False)
            self.assertEqual(input.grad, input2.grad, exact_dtype=False)

        def func():
            return torch.nan_to_num

        shapes = [[1, 3, 6, 6], [1, 3, 6, 128], [1, 3, 256, 256]]
        for shape in shapes:
            x = torch.randn(shape, device=device)
            extremals = [float('nan'), float('inf'), -float('inf')]
            for id1, id2, extremal in zip(torch.randint(0, 2, (3,)), torch.randint(0, 5, (3,)), extremals):
                x[0, id1, id2, :] = extremal
            test_dtype(func(), x, torch.bfloat16)

    @dtypes(torch.cdouble)
    def test_complex_edge_values(self, device, dtype):
        # sqrt Test Reference: https://github.com/pytorch/pytorch/pull/47424
        x = torch.tensor(0.0 - 1.0e20j, dtype=dtype, device=device)
        self.compare_with_numpy(torch.sqrt, np.sqrt, x)
        # acos test reference: https://github.com/pytorch/pytorch/issue/42952
        # Skip on Windows, as CUDA acos  returns conjugate value
        # see https://github.com/pytorch/pytorch/issues/52299
        if not (IS_WINDOWS and dtype == torch.cdouble and "cuda" in device):
            self.compare_with_numpy(torch.acos, np.arccos, x)

        x = torch.tensor(
            (-1.0e60 if dtype == torch.cdouble else -1.0e20) - 4988429.2j,
            dtype=dtype,
            device=device,
        )
        self.compare_with_numpy(torch.sqrt, np.sqrt, x)

    @unittest.skipIf(not TEST_SCIPY, "Requires SciPy")
    @dtypes(torch.float, torch.double)
    def test_digamma_special(self, device, dtype):
        # Based on SciPy test for the following special values.
        # Reference:
        # https://github.com/scipy/scipy/blob/3a8a3a1d4657254a6611e77e9c28feafa26e6645/scipy/special/tests/test_digamma.py#L22
        euler = 0.57721566490153286
        dataset = [
            (0.0, -0.0),
            (1, -euler),
            (0.5, -2 * math.log(2) - euler),
            (1 / 3, -math.pi / (2 * math.sqrt(3)) - 3 * math.log(3) / 2 - euler),
            (1 / 4, -math.pi / 2 - 3 * math.log(2) - euler),
            (
                1 / 6,
                -math.pi * math.sqrt(3) / 2
                - 2 * math.log(2)
                - 3 * math.log(3) / 2
                - euler,
            ),
            (
                1 / 8,
                -math.pi / 2
                - 4 * math.log(2)
                - (math.pi + math.log(2 + math.sqrt(2)) - math.log(2 - math.sqrt(2)))
                / math.sqrt(2)
                - euler,
            ),
        ]
        x = torch.tensor(dataset, device=device, dtype=dtype)
        self.compare_with_numpy(torch.digamma, scipy.special.digamma, x)

    @unittest.skipIf(not TEST_SCIPY, "Requires SciPy")
    @dtypes(torch.float, torch.double)
    def test_digamma(self, device, dtype):
        # Tests pole behavior
        tensor = torch.tensor(
            [
                -0.999999994,
                -1.999999994,
                -2.0000000111,
                -100.99999994,
                0.000000111,
                -1931.99999994,
                -0.000000111,
                0,
                -0,
                -1,
                -2,
                -931,
            ],
            dtype=dtype,
            device=device,
        )
        self.compare_with_numpy(torch.digamma, scipy.special.digamma, tensor)

    @dtypes(*floating_types_and(torch.half))
    def test_frexp(self, device, dtype):
        input = make_tensor((50, 50), dtype=dtype, device=device)
        mantissa, exponent = torch.frexp(input)
        np_mantissa, np_exponent = np.frexp(input.cpu().numpy())

        self.assertEqual(mantissa, np_mantissa)
        self.assertEqual(exponent, np_exponent)

        # torch.frexp returns exponent in int32 to be compatible with np.frexp
        self.assertTrue(exponent.dtype == torch.int32)
        self.assertTrue(torch_to_numpy_dtype_dict[exponent.dtype] == np_exponent.dtype)

    def test_frexp_assert_raises(self, device):
        invalid_input_dtypes = integral_types_and(torch.bool) + complex_types()
        for dtype in invalid_input_dtypes:
            input = make_tensor((50, 50), dtype=dtype, device=device)
            with self.assertRaisesRegex(
                RuntimeError, r"torch\.frexp\(\) only supports floating-point dtypes"
            ):
                torch.frexp(input)

        for dtype in floating_types_and(torch.half):
            input = make_tensor((50, 50), dtype=dtype, device=device)

            dtypes = list(
                all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16)
            )
            dtypes.remove(dtype)
            for mantissa_dtype in dtypes:
                mantissa = torch.empty_like(input, dtype=mantissa_dtype)
                exponent = torch.empty_like(input, dtype=torch.int)
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"torch\.frexp\(\) expects mantissa to have dtype .+ but got .+",
                ):
                    torch.frexp(input, out=(mantissa, exponent))

            dtypes.append(dtype)
            dtypes.remove(torch.int)
            for exponent_dtype in dtypes:
                mantissa = torch.empty_like(input)
                exponent = torch.empty_like(input, dtype=exponent_dtype)
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"torch\.frexp\(\) expects exponent to have int dtype but got .+",
                ):
                    torch.frexp(input, out=(mantissa, exponent))

    def test_polygamma_neg(self, device):
        with self.assertRaisesRegex(
            RuntimeError, r"polygamma\(n, x\) does not support negative n\."
        ):
            torch.polygamma(-1, torch.tensor([1.0, 2.0], device=device))

    # TODO resolve with opinfos
    @onlyCPU
    def test_op_invert(self, device):
        res = 0xFFFF - torch.arange(127, dtype=torch.int8)
        for dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            a = torch.arange(127, dtype=dtype)
            self.assertEqual(res.to(dtype), ~a)

        self.assertEqual(torch.tensor([True, False]), ~torch.tensor([False, True]))

        # test exceptions
        for dtype in (torch.half, torch.float, torch.double):
            a = torch.zeros(10, dtype=dtype)
            with self.assertRaises(TypeError):
                b = ~a

    @dtypes(torch.complex64, torch.complex128)
    def test_abs_angle_complex_to_float(self, device, dtype):
        # Constructs random complex values
        from random import random

        random_vals = []
        for multiplier in (-1, 1, -10, 10, -100, 100):
            for _ in range(10):
                random_vals.append(
                    complex(random() * multiplier, random() * multiplier)
                )

        for vals in (random_vals, []):
            a = np.array(vals, dtype=torch_to_numpy_dtype_dict[dtype])
            t = torch.tensor(vals, device=device, dtype=dtype)

            for fn_name in ("abs", "angle"):
                torch_fn = getattr(torch, fn_name)
                np_fn = getattr(np, fn_name)

                # Tests function
                np_result = torch.from_numpy(np_fn(a))
                torch_result = torch_fn(t).cpu()
                self.assertEqual(np_result, torch_result, exact_dtype=True)

                # Tests float out
                float_dtype = (
                    torch.float32 if dtype is torch.complex64 else torch.float64
                )
                np_float_out = np_fn(a).astype(torch_to_numpy_dtype_dict[float_dtype])
                float_out = torch.empty_like(t, dtype=float_dtype)
                torch_fn(t, out=float_out)
                self.assertEqual(torch.from_numpy(np_float_out), float_out.cpu())

                # Tests float out (resized out)
                float_out = torch.empty(1, device=device, dtype=float_dtype)
                torch_fn(t, out=float_out)
                self.assertEqual(torch.from_numpy(np_float_out), float_out.cpu())

                # Tests complex out
                np_complex_out = np_fn(a).astype(torch_to_numpy_dtype_dict[dtype])
                complex_out = torch.empty_like(t)
                torch_fn(t, out=complex_out)
                self.assertEqual(torch.from_numpy(np_complex_out), complex_out.cpu())

                # Tests complex out (resized out)
                complex_out = torch.empty(0, device=device, dtype=dtype)
                torch_fn(t, out=complex_out)
                self.assertEqual(torch.from_numpy(np_complex_out), complex_out.cpu())

                # Tests long out behavior (expected failure)
                long_out = torch.empty(0, device=device, dtype=torch.long)
                with self.assertRaises(RuntimeError):
                    torch_fn(t, out=long_out)

                # Tests inplace
                if fn_name == "abs":
                    torch_inplace_method = getattr(torch.Tensor, fn_name + "_")
                    np_fn(a, out=a)
                    if dtype.is_complex:
                        with self.assertRaisesRegex(
                            RuntimeError,
                            "In-place abs is not supported for complex tensors.",
                        ):
                            torch_inplace_method(t)
                        return
                    torch_inplace_method(t)
                    self.assertEqual(torch.from_numpy(a), t.cpu())

                # Note: angle does not have an in-place variant
                if fn_name == "angle":
                    with self.assertRaises(AttributeError):
                        torch_inplace_method = getattr(torch.Tensor, fn_name + "_")

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

    # TODO: run on non-native device types
    @dtypes(torch.double)
    def test_unary_out_op_mem_overlap(self, device, dtype):
        sz = 3
        doubles = torch.randn(2 * sz, dtype=dtype, device=device)
        positives = torch.randint(1, 100, (2 * sz,), device=device).double()
        ints = torch.randint(-100, 100, (2 * sz,), device=device)
        unary_mem_overlap_cases = [
            ("abs", doubles, True, True, "cpu"),
            ("abs", doubles, True, True, "cuda"),
            ("acos", doubles, True, True, "cpu"),
            ("acos", doubles, True, True, "cuda"),
            ("asin", doubles, True, True, "cpu"),
            ("asin", doubles, True, True, "cuda"),
            ("atan", doubles, True, True, "cpu"),
            ("atan", doubles, True, True, "cuda"),
            ("acosh", doubles, True, True, "cpu"),
            ("acosh", doubles, True, True, "cuda"),
            ("asinh", doubles, True, True, "cpu"),
            ("asinh", doubles, True, True, "cuda"),
            ("atanh", doubles, True, True, "cpu"),
            ("atanh", doubles, True, True, "cuda"),
            ("bitwise_not", ints, True, True, "cpu"),
            ("bitwise_not", ints, True, True, "cuda"),
            ("ceil", doubles, True, True, "cpu"),
            ("ceil", doubles, True, True, "cuda"),
            ("cos", doubles, True, True, "cpu"),
            ("cos", doubles, True, True, "cuda"),
            ("cosh", doubles, True, True, "cpu"),
            ("cosh", doubles, True, True, "cuda"),
            ("digamma", doubles, True, True, "cpu"),
            ("erf", doubles, True, True, "cpu"),
            ("erf", doubles, True, True, "cuda"),
            ("erfc", doubles, True, True, "cpu"),
            ("erfc", doubles, True, True, "cuda"),
            ("erfinv", doubles, True, True, "cpu"),
            ("erfinv", doubles, True, True, "cuda"),
            ("exp", doubles, True, True, "cpu"),
            ("exp", doubles, True, True, "cuda"),
            ("exp2", doubles, True, True, "cpu"),
            ("exp2", doubles, True, True, "cuda"),
            ("expm1", doubles, True, True, "cpu"),
            ("expm1", doubles, True, True, "cuda"),
            ("floor", doubles, True, True, "cpu"),
            ("floor", doubles, True, True, "cuda"),
            ("frac", doubles, True, True, "cpu"),
            ("frac", doubles, True, True, "cuda"),
            ("i0", doubles, True, True, "cpu"),
            ("i0", doubles, True, True, "cuda"),
            ("log", positives, True, True, "cpu"),
            ("log", positives, True, True, "cuda"),
            ("log10", positives, True, True, "cpu"),
            ("log10", positives, True, True, "cuda"),
            ("log1p", positives, True, True, "cpu"),
            ("log1p", positives, True, True, "cuda"),
            ("log2", positives, True, True, "cpu"),
            ("log2", positives, True, True, "cuda"),
            ("neg", doubles, True, True, "cpu"),
            ("neg", doubles, True, True, "cuda"),
            ("reciprocal", doubles, True, True, "cpu"),
            ("reciprocal", doubles, True, True, "cuda"),
            ("round", doubles, True, True, "cpu"),
            ("round", doubles, True, True, "cuda"),
            ("rsqrt", positives, True, True, "cpu"),
            ("rsqrt", positives, True, True, "cuda"),
            ("sin", doubles, True, True, "cpu"),
            ("sin", doubles, True, True, "cuda"),
            ("sinh", doubles, True, True, "cpu"),
            ("sinh", doubles, False, True, "cuda"),
            ("sigmoid", doubles, True, True, "cpu"),
            ("sigmoid", doubles, True, True, "cuda"),
            ("logit", doubles, True, True, "cpu"),
            ("logit", doubles, True, True, "cuda"),
            ("sqrt", doubles, True, True, "cpu"),
            ("sqrt", doubles, False, True, "cuda"),
            ("tan", doubles, True, True, "cpu"),
            ("tan", doubles, True, True, "cuda"),
            ("tanh", doubles, True, True, "cpu"),
            ("tanh", doubles, True, True, "cuda"),
            ("trunc", doubles, True, True, "cpu"),
            ("trunc", doubles, True, True, "cuda"),
        ]

        for (
            fn,
            inputs,
            has_input_output_mem_overlap_check,
            has_internal_mem_overlap_check,
            dev,
        ) in unary_mem_overlap_cases:
            if dev != device:
                continue
            out_fn = getattr(torch, fn)
            in_fn = getattr(torch.Tensor, fn + "_")

            self.unary_check_input_output_mem_overlap(
                inputs,
                sz,
                out_fn,
                expected_failure=not has_input_output_mem_overlap_check,
            )

            self.check_internal_mem_overlap(
                in_fn,
                1,
                dtype,
                dev,
                expected_failure=not has_internal_mem_overlap_check,
            )

    # TODO: opinfo hardshrink
    @onlyCPU
    @dtypes(torch.float, torch.double, torch.bfloat16)
    def test_hardshrink(self, device, dtype):
        data = torch.tensor([1, 0.5, 0.3, 0.6], dtype=dtype, device=device).view(2, 2)
        self.assertEqual(
            torch.tensor([1, 0.5, 0, 0.6], dtype=dtype, device=device).view(2, 2),
            data.hardshrink(0.3),
        )
        self.assertEqual(
            torch.tensor([1, 0, 0, 0.6], dtype=dtype, device=device).view(2, 2),
            data.hardshrink(0.5),
        )

        # test default lambd=0.5
        self.assertEqual(data.hardshrink(), data.hardshrink(0.5))

        # test non-contiguous case
        self.assertEqual(
            torch.tensor([1, 0, 0.5, 0.6], dtype=dtype, device=device).view(2, 2),
            data.t().hardshrink(0.3),
        )

    @onlyCPU
    @dtypes(torch.float, torch.double, torch.bfloat16)
    def test_hardshrink_edge_cases(self, device, dtype) -> None:
        def h(values, l_expected):
            for l, expected in l_expected.items():
                values_tensor = torch.tensor(
                    [float(v) for v in values], dtype=dtype, device=device
                )
                expected_tensor = torch.tensor(
                    [float(v) for v in expected], dtype=dtype, device=device
                )
                self.assertEqual(
                    expected_tensor == values_tensor.hardshrink(l),
                    torch.ones_like(values_tensor, dtype=torch.bool),
                )

        def test_helper(min, max):
            h(
                [0.0, min, -min, 0.1, -0.1, 1.0, -1.0, max, -max, inf, -inf],
                {
                    0.0: [0.0, min, -min, 0.1, -0.1, 1.0, -1.0, max, -max, inf, -inf],
                    min: [0.0, 0.0, 0.0, 0.1, -0.1, 1.0, -1.0, max, -max, inf, -inf],
                    0.1: [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, max, -max, inf, -inf],
                    1.0: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, max, -max, inf, -inf],
                    max: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, inf, -inf],
                    inf: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                },
            )

        test_helper(torch.finfo(dtype).tiny, torch.finfo(dtype).max)

    @onlyCPU
    @slowTest
    @dtypes(torch.float)
    @unittest.skipIf(True, "Insufficient memory on linux.(2|4)xlarge")
    def test_exp_slow(self, device, dtype):
        # Test for https://github.com/pytorch/pytorch/issues/17271
        # This is pretty slow on my Macbook but it only takes a few
        # seconds on a beefy Xeon server
        a = torch.exp(torch.ones(2**31, dtype=dtype, device=device))
        b = torch.exp(torch.ones(1, dtype=dtype, device=device))
        self.assertEqual(a, b.expand(2**31))

    @precisionOverride(
        {torch.bfloat16: 1e-2, torch.float: 0.0002, torch.double: 0.0002}
    )
    @dtypes(torch.float, torch.double, torch.bfloat16)
    def test_hardswish(self, device, dtype):
        inputValues = [-1000, -4, -3, -2, 0, 2, 3, 4, 1000]
        expectedOutput = np.multiply(
            inputValues, np.minimum(np.maximum((np.add(inputValues, 3)), 0), 6) / 6.0
        )

        inputTensor = torch.tensor(inputValues, dtype=dtype, device=device)
        expectedOutputTensor = torch.tensor(expectedOutput, dtype=dtype, device=device)

        # normal
        self.assertEqual(
            torch.nn.functional.hardswish(inputTensor), expectedOutputTensor
        )

        # inplace
        inputTensorCpy = inputTensor.clone().detach()
        torch.nn.functional.hardswish(inputTensorCpy, inplace=True)
        self.assertEqual(inputTensorCpy, expectedOutputTensor)

    @precisionOverride(
        {torch.bfloat16: 1e-2, torch.float: 0.0002, torch.double: 0.0002}
    )
    @dtypes(torch.float, torch.double, torch.bfloat16)
    def test_hardsigmoid(self, device, dtype):
        inputValues = [-1000, -4, -3, -2, 0, 2, 3, 4, 1000]
        expectedOutput = np.minimum(np.maximum((np.add(inputValues, 3)), 0), 6) / 6.0

        inputTensor = torch.tensor(inputValues, dtype=dtype, device=device)

        # normal
        self.assertEqual(
            torch.nn.functional.hardsigmoid(inputTensor),
            torch.tensor(expectedOutput, dtype=dtype, device=device),
        )

        # inplace
        inputTensorCpy = inputTensor.clone().detach()
        self.assertEqual(
            torch.nn.functional.hardsigmoid(inputTensorCpy, inplace=True),
            torch.tensor(expectedOutput, dtype=dtype, device=device),
        )

    @precisionOverride(
        {torch.bfloat16: 1e-2, torch.float: 0.0002, torch.double: 0.0002}
    )
    @dtypes(torch.float, torch.double, torch.bfloat16)
    def test_hardsigmoid_backward(self, device, dtype):
        inputValues = [-3.0, 3.0, -2.0, 2.0, -6.0, 6.0]
        expectedValues = [0.0, 0.0, 1.0 / 6.0, 1.0 / 6.0, 0.0, 0.0]
        inputTensor = torch.tensor(
            inputValues, dtype=dtype, device=device
        ).requires_grad_()
        expetedTensor = torch.tensor(expectedValues, dtype=dtype, device=device)
        out = torch.nn.functional.hardsigmoid(inputTensor)
        out.backward(torch.ones_like(inputTensor))
        self.assertEqual(inputTensor.grad, expetedTensor)

    @skipIfNoSciPy
    @dtypes(torch.float, torch.double)
    def test_silu(self, device, dtype):
        input_np = np.random.randn(5, 8)
        special_input = [[-1000, -1, -0.1, 0, 0.5, 1, 2, 1000]]
        input_np = np.concatenate((input_np, special_input), axis=0).astype(
            torch_to_numpy_dtype_dict[dtype]
        )
        expected_output_np = input_np * scipy.special.expit(input_np)

        expected_output = torch.from_numpy(expected_output_np).to(device)
        expected_output_noncontig = expected_output.transpose(0, 1)

        atol = 1e-6
        rtol = 1e-6

        input = torch.from_numpy(input_np).clone().contiguous().to(device)
        self.assertEqual(
            torch.nn.functional.silu(input), expected_output, atol=atol, rtol=rtol
        )
        self.assertEqual(
            torch.nn.functional.silu(input, inplace=True),
            expected_output,
            atol=atol,
            rtol=rtol,
        )

        input = torch.from_numpy(input_np).clone().to(device)
        input_noncontig = input.transpose(0, 1)
        self.assertEqual(
            torch.nn.functional.silu(input_noncontig),
            expected_output_noncontig,
            atol=atol,
            rtol=rtol,
        )
        self.assertEqual(
            torch.nn.functional.silu(input_noncontig, inplace=True),
            expected_output_noncontig,
            atol=atol,
            rtol=rtol,
        )

    # It is not obvious how to merge this into OpInfo becuase these inputs
    # succeed for gradcheck but are expected to fail for gradgradcheck
    @dtypes(torch.double)
    def test_sinc(self, device, dtype):
        # The derivative of sinc(x) at x=0 has to be special cased.
        # A naive computation will result in 0/0 -> NaN.
        # We also need to be careful when we are very close to 0, as the
        # derivative's denominator is squared, and there are some floats
        # that are positive and whose squares are zero.
        a = torch.tensor(
            [0.0, torch.finfo(torch.double).tiny, 1.0],
            dtype=dtype,
            requires_grad=True,
            device=device,
        )
        gradcheck(torch.sinc, a)

    @skipIfNoSciPy
    @dtypes(torch.float, torch.double)
    def test_mish(self, device, dtype):
        input_np = np.random.randn(5, 8)
        special_input = [[-1000, -1, -0.1, 0, 0.5, 1, 2, 1000]]
        input_np = np.concatenate((input_np, special_input), axis=0).astype(
            torch_to_numpy_dtype_dict[dtype]
        )
        expected_output_np = input_np * np.tanh(np.log1p(np.exp(input_np)))

        expected_output = torch.from_numpy(expected_output_np).to(device)
        expected_output_noncontig = expected_output.transpose(0, 1)

        atol = 1e-6
        rtol = 1e-6

        input = torch.from_numpy(input_np).clone().contiguous().to(device)
        self.assertEqual(
            torch.nn.functional.mish(input), expected_output, atol=atol, rtol=rtol
        )
        self.assertEqual(
            torch.nn.functional.mish(input, inplace=True),
            expected_output,
            atol=atol,
            rtol=rtol,
        )

        input = torch.from_numpy(input_np).clone().to(device)
        input_noncontig = input.transpose(0, 1)
        self.assertEqual(
            torch.nn.functional.mish(input_noncontig),
            expected_output_noncontig,
            atol=atol,
            rtol=rtol,
        )
        self.assertEqual(
            torch.nn.functional.mish(input_noncontig, inplace=True),
            expected_output_noncontig,
            atol=atol,
            rtol=rtol,
        )

    @dtypes(torch.complex64, torch.complex128)
    def test_log1p_complex(self, device, dtype):
        # The output values here were obtained using arbitrary precision math (mpmath)
        # and double checked with WolframAlpha.
        # Not using numpy's log1p here because by the time of writing this,
        # np.log1p has precision problems for small complex input values, see here:
        # https://github.com/numpy/numpy/issues/22609
        inouts = [
            (0.2 + 0.3j, 0.21263386770217202 + 0.24497866312686414j),
            (1e-19 + 1e-18j, 1e-19 + 1e-18j),
            (1e-18 + 0.1j, 0.00497517 + 0.0996687j),
            (0.1 + 1e-18j, 0.0953102 + 9.090909090909090909e-19j),
            (0.5 + 0j, 0.40546510810816 + 0j),
            (0.0 + 0.5j, 0.111571776 + 0.463647609j),
            (2.0 + 1.0j, 1.151292546497023 + 0.3217505543966422j),
            (-1.0 + 2.0j, 0.6931471805599453 + 1.570796326794897j),
            (2.0j, 0.80471895621705014 + 1.1071487177940904j),
            (-2.0j, 0.80471895621705014 - 1.1071487177940904j),
        ]
        # test the extreme values
        if dtype == torch.complex128:
            inouts += [
                (-1 + 1e250j, 575.6462732485114 + 1.5707963267948966j),
                (1e250 + 1j, 575.6462732485114 + 1e-250j),
                (1e250 + 1e250j, 575.9928468387914 + 0.7853981633974483j),
                (1e-250 + 1e250j, 575.6462732485114 + 1.5707963267948966j),
                (1e-250 + 2e-250j, 1e-250 + 2e-250j),
                (1e250 + 1e-250j, 575.6462732485114 + 0.0j),
            ]
        elif dtype == torch.complex64:
            inouts += [
                (-1 + 1e30j, 69.07755278982137 + 1.5707963267948966j),
                (1e30 + 1j, 69.07755278982137 + 1e-30j),
                (1e30 + 1e30j, 69.42412638010134 + 0.7853981633974483j),
                (1e-30 + 1e30j, 69.07755278982137 + 1.5707963267948966j),
                (1e-30 + 2e-30j, 1e-30 + 2e-30j),
                (1e30 + 1e-30j, 69.07755278982137 + 0.0j),
            ]

        # test the log1p individually
        for inp, out in inouts:
            res = torch.log1p(torch.tensor(inp, dtype=dtype, device=device))
            self.assertFalse(torch.any(torch.isnan(res)))
            # setting up atol == 0.0 because some part has very small values
            self.assertEqual(res.real, out.real, atol=0.0, rtol=1e-6)
            self.assertEqual(res.imag, out.imag, atol=0.0, rtol=1e-6)

        # test the log1p in tensor
        inp_lst, out_lst = [list(elmt) for elmt in zip(*inouts)]
        inp_tens = torch.tensor(inp_lst, dtype=dtype, device=device)
        out_tens = torch.tensor(out_lst, dtype=dtype, device=device)
        res_tens = torch.log1p(inp_tens)
        self.assertEqual(res_tens.real, out_tens.real, atol=0.0, rtol=1e-6)
        self.assertEqual(res_tens.imag, out_tens.imag, atol=0.0, rtol=1e-6)

    # do ops like threshold need a test_unary(_nonufunc) test suite?
    @onlyCPU
    @dtypes(*get_all_math_dtypes("cpu"))
    def test_threshold(self, device, dtype):
        if dtype != torch.uint8 and dtype != torch.float16 and not dtype.is_complex:
            # 100 is wide enough to use AVX2 instructions for all types
            x = (
                torch.randn(100, dtype=torch.float, device=device)
                .sign()
                .to(dtype=dtype)
            )
            y = torch.threshold(x, 0, 0)
            self.assertTrue(y.le(0).any())

    def _helper_test_igamma(self, loglo, loghi, device, dtype, torch_fcn, scipy_fcn):
        exp1 = 2.71828182846
        vec1 = torch.logspace(
            loglo, loghi, steps=500, base=exp1, dtype=torch.float64, device=device
        ).unsqueeze(-1)
        vec1 = vec1.to(dtype)
        inputs = [
            (vec1, vec1.transpose(0, 1)),
            (vec1, vec1),  # for large number, it should approach 0.5
            (vec1, 0.5 * vec1),  # test for considerable ratio
            (vec1, 2.0 * vec1),
            (vec1[::2, :], vec1[::2, :]),  # contiguous/noncontiguous tests
            (vec1[::2, :], vec1[: vec1.shape[0] // 2, :]),
            (vec1[: vec1.shape[0] // 2, :], vec1[::2, :]),
        ]
        half_prec = dtype in [torch.bfloat16, torch.float16]
        for input0, input1 in inputs:
            actual = torch_fcn(input0, input1)
            if half_prec:
                input0 = input0.to(torch.float)
                input1 = input1.to(torch.float)
            expected = scipy_fcn(input0.cpu().numpy(), input1.cpu().numpy())
            expected = torch.from_numpy(expected).to(dtype)
            self.assertEqual(actual, expected)

    @dtypesIfCPU(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @dtypes(torch.float32, torch.float64)
    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    @onlyNativeDeviceTypes
    def test_igamma_common(self, device, dtype):
        # test igamma for reasonable range of values
        loglo = -4  # approx 0.018
        loghi = 4  # approx 54.6
        self._helper_test_igamma(
            loglo, loghi, device, dtype, torch.igamma, scipy.special.gammainc
        )

    @dtypesIfCPU(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @dtypes(torch.float32, torch.float64)
    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    @onlyNativeDeviceTypes
    def test_igammac_common(self, device, dtype):
        # test igammac for reasonable range of values
        loglo = -4  # approx 0.018
        loghi = 4  # approx 54.6
        self._helper_test_igamma(
            loglo, loghi, device, dtype, torch.igammac, scipy.special.gammaincc
        )

    @dtypesIfCPU(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @dtypes(torch.float32, torch.float64)
    @onlyNativeDeviceTypes
    def test_igamma_edge_cases(self, device, dtype):
        tkwargs = {"dtype": dtype, "device": device}
        infs = torch.zeros((3,), **tkwargs) + float("inf")
        zeros = torch.zeros((3,), **tkwargs)
        ones = torch.ones((3,), **tkwargs)
        zero_to_large = torch.tensor([0.0, 1.0, 1e3], **tkwargs)
        small_to_inf = torch.tensor([1e-3, 1.0, float("inf")], **tkwargs)
        nans = torch.zeros((3,), **tkwargs) + float("nan")
        inpouts = [
            # (a    ,    x),       out
            ((zeros, small_to_inf), ones),
            ((small_to_inf, zeros), zeros),
            ((infs, zero_to_large), zeros),
            ((zero_to_large, infs), ones),
            ((zeros, zeros), nans),
            ((infs, infs), nans),
            ((-small_to_inf, small_to_inf), nans),
        ]
        for inputs, output in inpouts:
            input0, input1 = inputs
            calc = torch.igamma(input0, input1)
            if torch.all(torch.isnan(output)):
                self.assertTrue(torch.all(torch.isnan(calc)))
            else:
                self.assertEqual(calc, output)

    @dtypesIfCPU(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @dtypes(torch.float32, torch.float64)
    @onlyNativeDeviceTypes
    def test_igammac_edge_cases(self, device, dtype):
        tkwargs = {"dtype": dtype, "device": device}
        infs = torch.zeros((3,), **tkwargs) + float("inf")
        zeros = torch.zeros((3,), **tkwargs)
        ones = torch.ones((3,), **tkwargs)
        zero_to_large = torch.tensor([0.0, 1.0, 1e3], **tkwargs)
        small_to_inf = torch.tensor([1e-3, 1.0, float("inf")], **tkwargs)
        nans = torch.zeros((3,), **tkwargs) + float("nan")
        inpouts = [
            # (a    ,    x),       out
            ((zeros, small_to_inf), zeros),
            ((small_to_inf, zeros), ones),
            ((infs, zero_to_large), ones),
            ((zero_to_large, infs), zeros),
            ((zeros, zeros), nans),
            ((infs, infs), nans),
            ((-small_to_inf, small_to_inf), nans),
        ]
        for inputs, output in inpouts:
            input0, input1 = inputs
            calc = torch.igammac(input0, input1)
            if torch.all(torch.isnan(output)):
                self.assertTrue(torch.all(torch.isnan(calc)))
            else:
                self.assertEqual(calc, output)

    def _i0_helper(self, t):
        # Test by comparing to scipy
        dtype = t.dtype
        actual = torch.i0(t)
        if dtype is torch.bfloat16:
            t = t.to(torch.float32)
        expected = scipy.special.i0(t.cpu().numpy())
        # Casting down for dtype float16 is required since scipy upcasts to float32
        if dtype is torch.bfloat16 or dtype is torch.float16:
            expected = torch.from_numpy(expected).to(dtype)
        self.assertEqual(actual, expected)

    def _i0_range_helper(self, range, device, dtype):
        # i0 tests are broken up by the domain for which the function does not overflow for each dtype
        # This is done to ensure that the function performs well across all possible input values, without worrying
        # about inf or nan possibilities
        for r in (range, -range):
            t = torch.rand(1000, device=device).to(dtype) * r
            self._i0_helper(t)

    @dtypesIfCUDA(*floating_types_and(torch.half, torch.bfloat16))
    @dtypes(torch.bfloat16, torch.float32, torch.float64)
    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    def test_i0_range1(self, device, dtype):
        # This tests the domain for i0 for which float16 does not overflow
        # The domain is (-13.25, 13.25)
        self._i0_range_helper(13.25, device, dtype)

    @dtypesIfCUDA(*floating_types_and(torch.half, torch.bfloat16))
    @dtypes(torch.bfloat16, torch.float32, torch.float64)
    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    def test_i0_range2(self, device, dtype):
        # This tests the domain for i0 for which float32 and bfloat16 does not overflow
        # The domain is (-88.5, 88.5)
        self._i0_range_helper(88.5, device, dtype)

    @dtypes(torch.float64)
    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    def test_i0_range3(self, device, dtype):
        # This tests the domain for i0 for which float64 does not overflow
        # The domain is (-709.75, 709.75)
        self._i0_range_helper(709.75, device, dtype)

    @dtypesIfCUDA(*floating_types_and(torch.half, torch.bfloat16))
    @dtypes(torch.bfloat16, torch.float32, torch.float64)
    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    def test_i0_special(self, device, dtype):
        t = torch.tensor([], device=device, dtype=dtype)
        self._i0_helper(t)

        t = torch.tensor([inf, -inf, nan], device=device, dtype=dtype)
        self.assertTrue(torch.i0(t).isnan().all())

    @dtypesIfCUDA(*floating_types_and(torch.half, torch.bfloat16))
    @dtypes(torch.bfloat16, torch.float32, torch.float64)
    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    def test_special_i0_i1_vs_scipy(self, device, dtype):
        def check_equal(t, torch_fn, scipy_fn):
            # Test by comparing to scipy
            actual = torch_fn(t)
            if dtype is torch.bfloat16:
                t = t.to(torch.float32)
            expected = scipy_fn(t.cpu().numpy())

            # Casting down for dtype float16 is required since scipy upcasts to float32
            if dtype is torch.bfloat16 or dtype is torch.float16:
                expected = torch.from_numpy(expected).to(dtype)
            self.assertEqual(actual, expected)

        t = torch.tensor([], device=device, dtype=dtype)
        check_equal(t, torch.i0, scipy.special.i0)
        check_equal(t, torch.special.i0e, scipy.special.i0e)
        if dtype not in [torch.half, torch.bfloat16]:
            check_equal(t, torch.special.i1, scipy.special.i1)
            check_equal(t, torch.special.i1e, scipy.special.i1e)

        range = (-1e7, 1e7)
        if dtype == torch.half:
            range = (-65000, 65000)

        t = torch.linspace(*range, int(1e4), device=device, dtype=dtype)
        check_equal(t, torch.i0, scipy.special.i0)
        check_equal(t, torch.special.i0e, scipy.special.i0e)
        if dtype not in [torch.half, torch.bfloat16]:
            check_equal(t, torch.special.i1, scipy.special.i1)
            check_equal(t, torch.special.i1e, scipy.special.i1e)

        # NaN, inf, -inf are tested in reference_numerics tests.
        info = torch.finfo(dtype)
        min, max, eps, tiny = info.min, info.max, info.eps, info.tiny
        t = torch.tensor([min, max, eps, tiny], dtype=dtype, device=device)
        check_equal(t, torch.i0, scipy.special.i0)
        check_equal(t, torch.special.i0e, scipy.special.i0e)
        if dtype not in [torch.half, torch.bfloat16]:
            check_equal(t, torch.special.i1, scipy.special.i1)
            check_equal(t, torch.special.i1e, scipy.special.i1e)

    @dtypes(torch.float32, torch.float64)
    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    def test_special_ndtr_vs_scipy(self, device, dtype):
        def check_equal(t):
            # Test by comparing to scipy
            actual = torch.special.ndtr(t)
            expected = scipy.special.ndtr(t.cpu().numpy())
            self.assertEqual(actual, expected)

        range = (-10, 10)
        t = torch.linspace(*range, 1, device=device, dtype=dtype)
        check_equal(t)

        # Skip testing NaN, inf, -inf since they are tested in reference_numerics tests.
        info = torch.finfo(dtype)
        min, max, eps, tiny = info.min, info.max, info.eps, info.tiny
        t = torch.tensor([min, max, eps, tiny], dtype=dtype, device=device)
        check_equal(t)

    @dtypes(torch.float32, torch.float64)
    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    def test_special_log_ndtr_vs_scipy(self, device, dtype):
        def check_equal(t):
            # Test by comparing with scipy
            actual = torch.special.log_ndtr(t)
            expected = scipy.special.log_ndtr(t.cpu().numpy())
            self.assertEqual(actual, expected)

        # Skip testing NaN, inf, -inf since they are tested in reference_numerics tests.
        info = torch.finfo(dtype)
        min, max, eps, tiny = info.min, info.max, info.eps, info.tiny
        t = torch.tensor([min, max, eps, tiny], dtype=dtype, device=device)
        check_equal(t)

    # TODO: allow large opinfo values to be opted-into via metadata
    @dtypes(torch.long)
    def test_abs_big_number(self, device, dtype):
        bignumber = 2**31 + 1
        res = torch.tensor([bignumber], device=device, dtype=dtype)
        self.assertGreater(res.abs()[0], 0)

    # TODO: add signed zero testing to opinfos
    @dtypes(torch.float, torch.double)
    def test_abs_signed_zero(self, device, dtype):
        # Both abs(0.0) and abs(-0.0) should result in 0.0
        size = 128 + 1  # pick a large enough number with remainder so that
        # both vectorized and nonvectorized op is tested
        inp = torch.zeros(size, device=device, dtype=dtype)
        inp[::2] = -0.0
        inp = inp.abs()
        for v in inp:
            self.assertGreater(math.copysign(1.0, v), 0.0)

    # TODO: update to compare against NumPy by rationalizing with OpInfo
    @onlyCUDA
    @dtypes(torch.float, torch.double)
    def test_abs_zero(self, device, dtype):
        # Both abs(0.0) and abs(-0.0) should result in 0.0
        abs_zeros = torch.tensor([0.0, -0.0], device=device, dtype=dtype).abs().tolist()
        for num in abs_zeros:
            self.assertGreater(math.copysign(1.0, num), 0.0)

    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16))
    def test_isposinf_isneginf_non_boolean_output(self, device, dtype):
        # test non-boolean tensors as the `out=` parameters
        # boolean outputs are tested in the above testcases
        vals = (float("inf"), -float("inf"), 1.2)
        t = torch.tensor(vals, device=device)
        for torch_op in (torch.isposinf, torch.isneginf):
            out = torch.empty_like(t, dtype=dtype)
            with self.assertRaisesRegex(
                RuntimeError, "does not support non-boolean outputs"
            ):
                torch_op(t, out=out)

    def test_nonzero_empty(self, device):
        def assert_tuple_empty(tup, dim):
            self.assertEqual(dim, len(tup))
            for t in tup:
                self.assertEqual(torch.Size([0]), t.shape)

        x = torch.randn(0, 2, 0, 5, 0, device=device)
        y = torch.nonzero(x)
        z = torch.nonzero(x, as_tuple=True)

        self.assertEqual(0, y.numel())
        self.assertEqual(torch.Size([0, 5]), y.shape)
        assert_tuple_empty(z, 5)

        x = torch.tensor(0.5, device=device)
        y = torch.nonzero(x)
        # nonzero with as_tuple returns a
        # tuple of len 1 for a zero-dim tensor.
        # This is done to match Numpy behavior.
        z = torch.nonzero(x, as_tuple=True)
        self.assertEqual(1, len(z))
        self.assertEqual(torch.zeros(1, dtype=torch.long), z[0])

        x = torch.zeros((), device=device)
        y = torch.nonzero(x)
        z = torch.nonzero(x, as_tuple=True)
        self.assertEqual(torch.Size([0, 0]), y.shape)
        self.assertEqual(1, len(z))
        self.assertEqual(torch.empty(0, dtype=torch.long), z[0])

    # TODO: rationalize with exp OpInfo
    @dtypes(*floating_and_complex_types_and(torch.bfloat16))
    @dtypesIfCUDA(*floating_and_complex_types_and(torch.half, torch.bfloat16))
    def test_exp(self, device, dtype):
        for v in (2, -2) + ((1j, 1 + 1j) if dtype.is_complex else ()):
            a = (
                torch.tensor(v, dtype=dtype, device=device)
                * torch.arange(18, device=device)
                / 3
                * math.pi
            )
            a = a.to(dtype)
            # bfloat16 overflows
            if dtype == torch.bfloat16:
                return
            self.compare_with_numpy(torch.exp, np.exp, a)

            if dtype.is_complex:
                inf_real_zero_imag_in = torch.tensor(
                    complex(float("inf"), 0), device=device, dtype=dtype
                )
                inf_real_zero_imag_out = torch.exp(inf_real_zero_imag_in).item()
                self.assertTrue(math.isinf(inf_real_zero_imag_out.real))
                if self.device_type == "cpu":
                    pass
                    # These are commented out because it cannot be consistently reproduced.
                    # This is incorrect. It should be zero. Need fix!
                    # https://github.com/pytorch/pytorch/issues/40590
                    # self.assertNotEqual(inf_real_zero_imag_out.imag, 0)
                    # This is incorrect. They should equal. Need fix!
                    # https://github.com/pytorch/pytorch/issues/40590
                    # with self.assertRaises(AssertionError):
                    #     self.compare_with_numpy(torch.exp, np.exp, inf_real_zero_imag_in)
                else:
                    self.assertEqual(inf_real_zero_imag_out.imag, 0, atol=0, rtol=0)
                    self.compare_with_numpy(torch.exp, np.exp, inf_real_zero_imag_in)

                zero_real_inf_imag_in = torch.tensor(
                    complex(0, float("inf")), device=device, dtype=dtype
                )
                zero_real_inf_imag_out = torch.exp(zero_real_inf_imag_in).item()
                self.assertTrue(math.isnan(zero_real_inf_imag_out.real))
                self.assertTrue(math.isnan(zero_real_inf_imag_out.imag))
                # Ensure we are notified when NumPy changes its behavior
                self.compare_with_numpy(torch.exp, np.exp, zero_real_inf_imag_in)

                inf_real_imag_in = torch.tensor(
                    complex(float("inf"), float("inf")), device=device, dtype=dtype
                )
                inf_real_imag_out = torch.exp(inf_real_imag_in).item()
                if self.device_type == "cpu":
                    pass
                    # This is incorrect. Need fix! https://github.com/pytorch/pytorch/issues/40590
                    # This is commented out because it cannot be consistently reproduced.
                    # with self.assertRaises(AssertionError):
                    #     self.compare_with_numpy(torch.exp, np.exp, inf_real_imag_in)
                else:
                    self.assertTrue(math.isinf(inf_real_imag_out.real))
                    self.assertTrue(math.isnan(inf_real_imag_out.imag))
                    self.compare_with_numpy(torch.exp, np.exp, inf_real_imag_in)

                inf_real_nan_imag_in = torch.tensor(
                    complex(float("inf"), float("nan")), device=device, dtype=dtype
                )
                inf_real_nan_imag_out = torch.exp(inf_real_nan_imag_in).item()
                if self.device_type == "cpu":
                    pass
                    # This is incorrect. It should be inf. Need fix! https://github.com/pytorch/pytorch/issues/40590
                    # This is commented out because it cannot be consistently reproduced.
                    # with self.assertRaises(AssertionError):
                    #     self.compare_with_numpy(torch.exp, np.exp, inf_real_nan_imag_in)
                else:
                    self.assertTrue(math.isinf(inf_real_nan_imag_out.real))
                    self.assertTrue(math.isnan(inf_real_nan_imag_out.imag))
                    self.compare_with_numpy(torch.exp, np.exp, inf_real_nan_imag_in)

                nan_real_inf_imag_in = torch.tensor(
                    complex(float("nan"), float("inf")), device=device, dtype=dtype
                )
                nan_real_inf_imag_out = torch.exp(nan_real_inf_imag_in).item()
                self.assertTrue(math.isnan(nan_real_inf_imag_out.real))
                self.assertTrue(math.isnan(nan_real_inf_imag_out.imag))
                # Ensure we are notified when NumPy changes its behavior
                self.compare_with_numpy(torch.exp, np.exp, nan_real_inf_imag_in)


instantiate_device_type_tests(TestUnaryUfuncs, globals())

if __name__ == "__main__":
    run_tests()
