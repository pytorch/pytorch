# Owner(s): ["module: complex"]
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist


# Support both when imported from elsewhere or directly as a file
try:
    from .utils import (
        COMPLEX_DTYPES,
        decomposed_op_db,
        Descriptor,
        force_test_op_db,
        implemented_op_db,
        TestCase,
        Variant,
    )
except ImportError:
    from utils import (
        COMPLEX_DTYPES,
        decomposed_op_db,
        Descriptor,
        force_test_op_db,
        implemented_op_db,
        TestCase,
        Variant,
    )

from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    OpDTypes,
    ops,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    unMarkDynamoStrictTest,
)


if TYPE_CHECKING:
    from torch.testing._internal.opinfo.core import OpInfo

aten = torch.ops.aten

SKIPS = {
    Descriptor(op=aten.empty_like, variant=None): "Non-deterministic output",
    Descriptor(op=aten.randn_like, variant=None): "Non-deterministic output",
    Descriptor(op=aten.empty_strided, variant=None): "Non-deterministic output",
    Descriptor(op=aten.new_empty, variant=None): "Non-deterministic output",
    Descriptor(
        op=aten.any, variant=Variant.Distributed
    ): "does not have a sharding strategy registered",
    Descriptor(
        op=aten.all, variant=Variant.Distributed
    ): "does not have a sharding strategy registered",
    Descriptor(
        op=aten.allclose, variant=Variant.Distributed
    ): "does not have a sharding strategy registered",
    Descriptor(
        op=aten.conj_physical, variant=Variant.Distributed
    ): "does not have a sharding strategy registered",
    Descriptor(
        op=aten._conj_physical, variant=Variant.Distributed
    ): "does not have a sharding strategy registered",
    Descriptor(
        op=aten.cumprod, variant=Variant.Distributed
    ): "does not have a sharding strategy registered",
    Descriptor(
        op=aten.index_add, variant=Variant.Distributed
    ): "does not have a sharding strategy registered",
    Descriptor(
        op=aten.diagonal_scatter, variant=Variant.Distributed
    ): "does not have a sharding strategy registered",
    Descriptor(
        op=aten.flip, variant=Variant.Distributed
    ): "does not have a sharding strategy registered",
    Descriptor(
        op=aten.masked_fill, variant=Variant.Distributed
    ): "does not have a sharding strategy registered",
    Descriptor(
        op=aten.masked_scatter, variant=Variant.Distributed
    ): "does not have a sharding strategy registered",
    Descriptor(
        op=aten.rsub, variant=Variant.Distributed
    ): "does not have a sharding strategy registered",
    Descriptor(
        op=aten.ne, variant=Variant.Distributed
    ): "does not have a sharding strategy registered",
    Descriptor(
        op=aten.squeeze, variant=Variant.Distributed
    ): "does not have a sharding strategy registered",
    Descriptor(
        op=aten.index_select, variant=Variant.Distributed
    ): "Sharding propagation failed",
    Descriptor(op=aten.real, variant=Variant.Distributed): "No scalar support",
    Descriptor(op=aten.imag, variant=Variant.Distributed): "No scalar support",
    Descriptor(op=aten.isfinite, variant=Variant.Distributed): "No scalar support",
    Descriptor(op=aten.transpose, variant=Variant.Distributed): "No scalar support",
    Descriptor(op=aten.view_as_real, variant=Variant.Distributed): "No scalar support",
    Descriptor(
        op=aten.mm, variant=Variant.OpInterleaved | Variant.GradCheck
    ): "Conjugated inputs",
    Descriptor(
        op=aten.dot, variant=Variant.OpInterleaved | Variant.GradCheck
    ): "Conjugated inputs",
    Descriptor(
        op=aten.addmm, variant=Variant.OpInterleaved | Variant.GradCheck
    ): "Conjugated inputs",
    Descriptor(op=aten.vdot, variant=Variant.OpInterleaved): "Conjugated inputs",
    Descriptor(
        op=aten.index_fill, variant=Variant.OpInterleaved
    ): "prims.copy_strided not implemented",
}

EXTRA_KWARGS = {
    Descriptor(
        op=aten.asinh, dtype=torch.complex64, variant=Variant.Op | Variant.OpInterleaved
    ): {
        "rtol": 2e-5,
        "atol": 5e-5,
    },
    Descriptor(
        op=aten.tanh, dtype=torch.complex64, variant=Variant.Op | Variant.OpInterleaved
    ): {
        "rtol": 1e-4,
        "atol": 1e-5,
    },
    Descriptor(
        op=aten.pow, dtype=torch.complex64, variant=Variant.Op | Variant.OpInterleaved
    ): {
        "rtol": 2e-2,
        "atol": 2e-6,
    },
    Descriptor(op=aten.asinh, dtype=torch.complex64, variant=Variant.Distributed): {
        "rtol": 2e-5,
        "atol": 5e-5,
    },
    Descriptor(op=aten.tanh, dtype=torch.complex64, variant=Variant.Distributed): {
        "rtol": 1e-4,
        "atol": 1e-5,
    },
    Descriptor(op=aten.pow, dtype=torch.complex64, variant=Variant.Distributed): {
        "rtol": 2e-2,
        "atol": 2e-6,
    },
    Descriptor(op=aten.tan, dtype=torch.complex64, variant=Variant.Distributed): {
        "rtol": 2e-6,
        "atol": 1e-2,
    },
    Descriptor(op=aten.tan, dtype=torch.complex64, variant=Variant.OpInterleaved): {
        "rtol": 2e-6,
        "atol": 5e-5,
    },
}


class TestComplexTensor(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR
    _default_dtype_check_enabled = True

    @ops(
        implemented_op_db,
        dtypes=OpDTypes.supported,
        allowed_dtypes=list(COMPLEX_DTYPES),
    )
    def test_consistency(self, device, dtype, op: OpInfo):
        self.check_consistency(device, dtype, op, Variant.Op)

    @ops(
        decomposed_op_db,
        dtypes=OpDTypes.supported,
        allowed_dtypes=list(COMPLEX_DTYPES),
    )
    def test_decompositions(self, device, dtype, op: OpInfo):
        self.check_consistency(device, dtype, op, Variant.Op)

    @ops(force_test_op_db, allowed_dtypes=list(COMPLEX_DTYPES))
    def test_maybe_error(self, device, dtype, op: OpInfo):
        self.check_consistency(device, dtype, op, Variant.Op)

    @ops(
        implemented_op_db,
        dtypes=OpDTypes.supported,
        allowed_dtypes=list(COMPLEX_DTYPES),
    )
    def test_consistency_interleaved(self, device, dtype, op: OpInfo):
        self.check_consistency(device, dtype, op, Variant.OpInterleaved)

    @ops(force_test_op_db, allowed_dtypes=list(COMPLEX_DTYPES))
    def test_maybe_error_interleaved(self, device, dtype, op: OpInfo):
        self.check_consistency(device, dtype, op, Variant.OpInterleaved)

    @ops(
        decomposed_op_db,
        dtypes=OpDTypes.supported,
        allowed_dtypes=list(COMPLEX_DTYPES),
    )
    def test_decompositions_interleaved(self, device, dtype, op: OpInfo):
        self.check_consistency(device, dtype, op, Variant.OpInterleaved)


class TestComplexTensorGeneric(TestCase):
    hw_classification = HardwareClassification.GENERIC
    _default_dtype_check_enabled = True

    def test_get_set_components(self):
        from torch._subclasses.complex_tensor import ComplexTensor

        c = ComplexTensor.from_interleaved(
            torch.tensor([1 + 2j, 3 + 4j], dtype=torch.complex64)
        )
        self.assertEqual(c.real, torch.tensor([1, 3], dtype=torch.float32))
        self.assertEqual(c.imag, torch.tensor([2, 4], dtype=torch.float32))

        c.real = torch.tensor([5, 6])
        c.imag = torch.tensor([7, 8])
        self.assertEqual(c.real, torch.tensor([5, 6], dtype=torch.float32))
        self.assertEqual(c.imag, torch.tensor([7, 8], dtype=torch.float32))
        self.assertEqual(c, torch.tensor([5 + 7j, 6 + 8j], dtype=torch.complex64))

    def test_mul_inplace_complex(self):
        from torch._subclasses.complex_tensor import ComplexTensor

        a = torch.tensor([1 + 2j, 3 + 4j], dtype=torch.complex64)
        b = torch.tensor([7 - 8j, -9 + 1j], dtype=torch.complex64)
        expected = a * b

        xa = ComplexTensor.from_interleaved(a)
        result = xa.mul_(ComplexTensor.from_interleaved(b))

        self.assertIs(result, xa)
        self.assertEqual(xa.as_interleaved(), expected)

    def test_ne_real_operand(self):
        from torch._subclasses.complex_tensor import ComplexTensor

        r = torch.tensor([1.0, 3.0, 5.0])
        c = torch.tensor([1 + 2j, 3 + 0j, 0 + 4j], dtype=torch.complex64)
        xc = ComplexTensor.from_interleaved(c)

        self.assertEqual(torch.ne(r, xc), torch.ne(r, c))

    def test_as_strided_inplace_beyond_wrapper_storage(self):
        from torch._subclasses.complex_tensor import ComplexTensor

        xc = ComplexTensor(torch.zeros(10)[:2], torch.zeros(10)[:2])
        result = xc.as_strided_((10,), (1,))

        self.assertIs(result, xc)
        self.assertEqual(xc.shape, (10,))
        self.assertEqual(xc.re.shape, (10,))
        self.assertEqual(xc.im.shape, (10,))

    def test_as_strided_inplace_error_leaves_components_untouched(self):
        from torch._subclasses.complex_tensor import ComplexTensor

        xc = ComplexTensor(torch.zeros(10)[:2], torch.zeros(2))
        with self.assertRaises(RuntimeError):
            xc.as_strided_((10,), (1,))

        self.assertEqual(xc.shape, (2,))
        self.assertEqual(xc.re.shape, (2,))
        self.assertEqual(xc.im.shape, (2,))

    def test_out_overload_writes_destination(self):
        from torch._subclasses.complex_tensor import ComplexTensor

        t = torch.randn(4, dtype=torch.complex64)
        out = ComplexTensor.from_interleaved(torch.zeros(4, dtype=torch.complex64))
        result = torch.fft.fft(ComplexTensor.from_interleaved(t), out=out)

        self.assertIs(result, out)
        self.assertEqual(out.as_interleaved(), torch.fft.fft(t))

    def test_resize_inplace(self):
        from torch._subclasses.complex_tensor import ComplexTensor

        xc = ComplexTensor(torch.randn(2), torch.randn(2))
        result = xc.resize_(3, 4)

        self.assertIs(result, xc)
        self.assertEqual(xc.shape, (3, 4))
        self.assertEqual(xc.stride(), (4, 1))
        self.assertEqual(xc.re.shape, (3, 4))
        self.assertEqual(xc.im.shape, (3, 4))

    def test_out_overload_resizes_destination(self):
        from torch._subclasses.complex_tensor import ComplexTensor

        t = torch.randn(4, dtype=torch.complex64)
        out = ComplexTensor.from_interleaved(torch.empty(0, dtype=torch.complex64))
        result = torch.fft.fft(ComplexTensor.from_interleaved(t), out=out)

        self.assertIs(result, out)
        self.assertEqual(out.as_interleaved(), torch.fft.fft(t))

    def test_out_overload_resizes_nonempty_destination(self):
        from torch._subclasses.complex_tensor import ComplexTensor

        t = torch.randn(4, dtype=torch.complex64)
        for shape in [(2,), (2, 3)]:
            with self.subTest(shape=shape):
                out = ComplexTensor.from_interleaved(
                    torch.randn(shape, dtype=torch.complex64)
                )
                result = torch.fft.fft(ComplexTensor.from_interleaved(t), out=out)

                self.assertIs(result, out)
                self.assertEqual(out.shape, (4,))
                self.assertEqual(out.as_interleaved(), torch.fft.fft(t))

    def test_copy(self):
        from torch._subclasses.complex_tensor import ComplexTensor

        src = torch.randn(3, dtype=torch.complex64)
        xsrc = ComplexTensor.from_interleaved(src)

        result = aten.copy(ComplexTensor.from_interleaved(torch.zeros_like(src)), xsrc)
        self.assertEqual(result.as_interleaved(), src)

        with self.assertWarnsRegex(UserWarning, "discards the imaginary part"):
            result = aten.copy(torch.zeros(3), xsrc)
        self.assertEqual(result, src.real)

    def test_index_put_matches_native(self):
        from torch._subclasses.complex_tensor import ComplexTensor

        indices = (torch.tensor([0, 2]),)
        dtypes = (torch.float32, torch.complex64)
        for self_dtype in dtypes:
            for values_dtype in dtypes:
                if not (self_dtype.is_complex or values_dtype.is_complex):
                    continue
                for accumulate in (False, True):
                    with self.subTest(
                        self_dtype=self_dtype,
                        values_dtype=values_dtype,
                        accumulate=accumulate,
                    ):
                        t = torch.randn(4, dtype=self_dtype)
                        v = torch.randn(2, dtype=values_dtype)
                        xt, xv = (
                            ComplexTensor.from_interleaved(a)
                            if a.dtype.is_complex
                            else a
                            for a in (t, v)
                        )
                        if self_dtype != values_dtype:
                            with self.assertRaisesRegex(RuntimeError, "dtypes match"):
                                aten.index_put(t, indices, v, accumulate)
                            with self.assertRaisesRegex(RuntimeError, "dtypes match"):
                                aten.index_put(xt, indices, xv, accumulate)
                            continue
                        expected = aten.index_put(t, indices, v, accumulate)
                        result = aten.index_put(xt, indices, xv, accumulate)
                        self.assertEqual(result.as_interleaved(), expected)

    def test_prims_rev_and_fft(self):
        from torch._subclasses.complex_tensor import ComplexTensor

        prims = torch.ops.prims
        t = torch.randn(4, 6, dtype=torch.complex128)
        xt = ComplexTensor.from_interleaved(t)

        self.assertEqual(prims.rev(xt, [1]).as_interleaved(), prims.rev(t, [1]))
        self.assertEqual(
            prims.fft_c2c(xt, dim=[1], forward=True).as_interleaved(),
            prims.fft_c2c(t, dim=[1], forward=True),
        )
        self.assertEqual(
            prims.fft_c2r(xt, dim=[1], last_dim_size=10),
            prims.fft_c2r(t, dim=[1], last_dim_size=10),
        )

    def test_out_overload_writes_multiple_destinations(self):
        from torch._subclasses.complex_tensor import ComplexTensor

        a = torch.randn(3, 3, dtype=torch.complex128)
        lu = ComplexTensor.from_interleaved(torch.zeros(3, 3, dtype=torch.complex128))
        pivots = torch.zeros(3, dtype=torch.int32)
        info = torch.ones((), dtype=torch.int32)
        result = torch.linalg.lu_factor_ex(
            ComplexTensor.from_interleaved(a), out=(lu, pivots, info)
        )

        self.assertIs(result[0], lu)
        self.assertIs(result[1], pivots)
        self.assertIs(result[2], info)
        expected = torch.linalg.lu_factor_ex(a)
        self.assertEqual(lu.as_interleaved(), expected.LU)
        self.assertEqual(pivots, expected.pivots)
        self.assertEqual(info, expected.info)

    def test_div_inplace(self):
        from torch._subclasses.complex_tensor import ComplexTensor

        a = torch.randn(3, dtype=torch.complex128)
        b = torch.randn(3, dtype=torch.complex128)
        xa = ComplexTensor.from_interleaved(a)
        result = xa.div_(ComplexTensor.from_interleaved(b))

        self.assertIs(result, xa)
        self.assertEqual(xa.as_interleaved(), a / b)

    def test_tril_inplace(self):
        from torch._subclasses.complex_tensor import ComplexTensor

        a = torch.randn(3, 3, dtype=torch.complex128)
        xa = ComplexTensor.from_interleaved(a)
        result = xa.tril_(-1)

        self.assertIs(result, xa)
        self.assertEqual(xa.as_interleaved(), a.tril(-1))

    def test_interleaved_views_input(self):
        from torch._subclasses.complex_tensor import InterleavedComplexTensor

        t = torch.randn(4, 6, dtype=torch.complex128)[:, ::2]
        x = InterleavedComplexTensor(t)

        self.assertEqual(x.stride(), t.stride())
        self.assertEqual(x.re.data_ptr(), t.data_ptr())
        self.assertEqual(x.as_interleaved(), t)

        y = x * x
        self.assertIsInstance(y, InterleavedComplexTensor)
        self.assertEqual(y.as_interleaved(), t * t)

        x.real = torch.zeros(4, 3)
        self.assertEqual(t.real, torch.zeros(4, 3, dtype=torch.float64))

    def test_interleaved_result_views_alias(self):
        from torch._subclasses.complex_tensor import InterleavedComplexTensor

        x = InterleavedComplexTensor(torch.randn(3, 3, dtype=torch.complex128))
        y = x + x.mH
        torch.real(y.diagonal()).mul_(0)
        y.select(-1, 0).fill_(1)

        self.assertEqual(y.re.diagonal()[1:], torch.zeros(2, dtype=torch.float64))
        self.assertEqual(
            y.as_interleaved()[:, 0], torch.ones(3, dtype=torch.complex128)
        )

    def test_interleaved_inplace_writes_through(self):
        from torch._subclasses.complex_tensor import InterleavedComplexTensor

        a = torch.tensor([1 + 2j, 3 + 4j], dtype=torch.complex64)
        b = torch.tensor([7 - 8j, -9 + 1j], dtype=torch.complex64)
        expected = a * b

        xa = InterleavedComplexTensor(a)
        result = xa.mul_(b)

        self.assertIs(result, xa)
        self.assertEqual(a, expected)

        xa.copy_(b)
        self.assertEqual(a, b)

    def test_interleaved_resize_inplace(self):
        from torch._subclasses.complex_tensor import InterleavedComplexTensor

        t = torch.randn(2, dtype=torch.complex64)
        x = InterleavedComplexTensor(t.clone())
        result = x.resize_(3, 4)

        self.assertIs(result, x)
        self.assertEqual(x.shape, (3, 4))
        self.assertEqual(x.as_interleaved().shape, (3, 4))
        self.assertEqual(x.as_interleaved().flatten()[:2], t)

    def test_interleaved_rejects_invalid_input(self):
        from torch._subclasses.complex_tensor import InterleavedComplexTensor

        with self.assertRaises(TypeError):
            InterleavedComplexTensor(torch.randn(3))


@unMarkDynamoStrictTest
class TestComplexBwdGradients(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR
    _default_dtype_check_enabled = True

    @ops(
        implemented_op_db,
        dtypes=OpDTypes.supported_backward,
        allowed_dtypes=[torch.complex128],
    )
    def test_fn_grad(self, device: str, dtype: torch.dtype, op: OpInfo) -> None:
        self.check_grad(device, dtype, op, Variant.Op)

    @ops(
        implemented_op_db,
        dtypes=OpDTypes.supported_backward,
        allowed_dtypes=[torch.complex128],
    )
    def test_fn_grad_interleaved(
        self, device: str, dtype: torch.dtype, op: OpInfo
    ) -> None:
        self.check_grad(device, dtype, op, Variant.OpInterleaved)


instantiate_device_type_tests(TestComplexTensor, globals())
instantiate_device_type_tests(TestComplexBwdGradients, globals())


if dist.is_available():
    from torch.testing._internal.common_distributed import MultiProcessTestCase

    @unMarkDynamoStrictTest
    class TestComplexDistributed(TestCase, MultiProcessTestCase):
        hw_classification = HardwareClassification.ACCELERATOR

        @ops(implemented_op_db, allowed_dtypes=list(COMPLEX_DTYPES))
        def test_distributed(self, device, dtype, op: OpInfo):
            self.check_consistency(device, dtype, op, Variant.Distributed)

    instantiate_device_type_tests(TestComplexDistributed, globals())

if __name__ == "__main__":
    run_tests()
