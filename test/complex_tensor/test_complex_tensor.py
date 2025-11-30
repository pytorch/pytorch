# Owner(s): ["module: complex"]
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist


# Support both when imported from elsewhere or directly as a file
try:
    from .utils import (
        COMPLEX_DTYPES,
        Descriptor,
        force_test_op_db,
        get_overload_packet_from_name,
        implemented_op_db,
        TestCase,
        Variant,
    )
except ImportError:
    from utils import (
        COMPLEX_DTYPES,
        Descriptor,
        force_test_op_db,
        get_overload_packet_from_name,
        implemented_op_db,
        TestCase,
        Variant,
    )

from torch._subclasses.complex_tensor._ops.common import ComplexTensorMode
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    OpDTypes,
    ops,
)
from torch.testing._internal.common_utils import (
    run_tests,
    TestGradients,
    unMarkDynamoStrictTest,
)


if TYPE_CHECKING:
    from torch.testing._internal.opinfo.core import OpInfo

aten = torch.ops.aten

SKIPS = {
    Descriptor(op=aten.empty_like, variant=None): "Non-deterministic output",
    Descriptor(op=aten.randn_like, variant=None): "Non-deterministic output",
    Descriptor(op=aten.angle, variant=Variant.GradCheck): "Numerical inconsistency",
    Descriptor(op=aten.asinh, variant=Variant.GradCheck): "Numerical inconsistency",
    Descriptor(op=aten.atanh, variant=Variant.GradCheck): "Numerical inconsistency",
    Descriptor(
        op=aten.reciprocal, variant=Variant.GradCheck
    ): "Numerical inconsistency",
    Descriptor(op=aten.rsqrt, variant=Variant.GradCheck): "Numerical inconsistency",
    Descriptor(op=aten.select, variant=Variant.GradCheck): "Numerical inconsistency",
    Descriptor(op=aten.asin, variant=Variant.GradCheck): "Numerical inconsistency",
    Descriptor(op=aten.log, variant=Variant.GradCheck): "Numerical inconsistency",
    Descriptor(op=aten.sgn, variant=Variant.GradCheck): "Numerical inconsistency",
    Descriptor(op=aten.cumprod, variant=Variant.GradCheck): "Numerical inconsistency",
    Descriptor(op=aten.slice, variant=Variant.GradCheck): "Numerical inconsistency",
    Descriptor(op=aten.sqrt, variant=Variant.GradCheck): "Numerical inconsistency",
    Descriptor(op=aten.tan, variant=Variant.GradCheck): "Numerical inconsistency",
    Descriptor(
        op=aten.true_divide, variant=Variant.GradCheck
    ): "Numerical inconsistency",
    Descriptor(op=aten.prod, variant=Variant.GradCheck): "Numerical inconsistency",
    Descriptor(op=aten.div, variant=Variant.GradCheck): "Numerical inconsistency",
    Descriptor(op=aten.expm1, variant=Variant.GradCheck): "Numerical inconsistency",
    Descriptor(op=aten.var, variant=Variant.GradCheck): "Numerical inconsistency",
    Descriptor(op=aten.bmm, variant=Variant.GradCheck): "Numerical inconsistency",
    Descriptor(op=aten.diagonal, variant=Variant.GradCheck): "Numerical inconsistency",
    Descriptor(op=aten.sinh, variant=Variant.GradCheck): "Numerical inconsistency",
    Descriptor(op=aten.abs, variant=Variant.GradCheck): "Numerical inconsistency",
    Descriptor(op=aten.sin, variant=Variant.GradCheck): "Numerical inconsistency",
    Descriptor(op=aten.atan, variant=Variant.GradCheck): "Numerical inconsistency",
    Descriptor(op=aten.acos, variant=Variant.GradCheck): "Numerical inconsistency",
    Descriptor(op=aten.acosh, variant=Variant.GradCheck): "Numerical inconsistency",
    Descriptor(op=aten.cos, variant=Variant.GradCheck): "Numerical inconsistency",
    Descriptor(op=aten.cosh, variant=Variant.GradCheck): "Numerical inconsistency",
    Descriptor(op=aten.addmm, variant=Variant.GradCheck): "Numerical inconsistency",
    Descriptor(op=aten.pow, variant=Variant.GradCheck): "Numerical inconsistency",
    Descriptor(op=aten.log1p, variant=Variant.GradCheck): "Numerical inconsistency",
    Descriptor(op=aten.tanh, variant=Variant.GradCheck): "Numerical inconsistency",
    Descriptor(op=aten.mm, variant=Variant.GradCheck): "Numerical inconsistency",
    Descriptor(op=aten.dot, variant=Variant.GradCheck): "Numerical inconsistency",
    Descriptor(op=aten.mul, variant=Variant.GradCheck): "Numerical inconsistency",
    Descriptor(op=aten.exp, variant=Variant.GradCheck): "Numerical inconsistency",
    Descriptor(op=aten.to, variant=Variant.GradCheck): "Numerical inconsistency",
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
}

EXTRA_KWARGS = {
    Descriptor(op=aten.asinh, dtype=torch.complex64, variant=Variant.Op): {
        "rtol": 2e-5,
        "atol": 5e-5,
    },
    Descriptor(op=aten.tanh, dtype=torch.complex64, variant=Variant.Op): {
        "rtol": 1e-4,
        "atol": 1e-5,
    },
    Descriptor(op=aten.pow, dtype=torch.complex64, variant=Variant.Op): {
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
}


class TestComplexTensor(TestCase):
    _default_dtype_check_enabled = True

    @ops(
        implemented_op_db,
        dtypes=OpDTypes.supported,
        allowed_dtypes=list(COMPLEX_DTYPES),
    )
    def test_consistency(self, device, dtype, op: OpInfo):
        self.check_consistency(device, dtype, op, Variant.Op)

    @ops(force_test_op_db, allowed_dtypes=list(COMPLEX_DTYPES))
    def test_maybe_error(self, device, dtype, op: OpInfo):
        self.check_consistency(device, dtype, op, Variant.Op)


@unMarkDynamoStrictTest
class TestComplexBwdGradients(TestGradients):
    _default_dtype_check_enabled = True

    @ops(
        implemented_op_db,
        dtypes=OpDTypes.supported_backward,
        allowed_dtypes=[torch.complex128],
    )
    def test_fn_grad(self, device: str, dtype: torch.dtype, op: OpInfo) -> None:
        test_info = Descriptor(
            op=get_overload_packet_from_name(op.name),
            device_type=torch.device(device).type,
            dtype=dtype,
            variant=Variant.GradCheck,
        )
        for xfail_info, reason in SKIPS.items():
            if xfail_info.matches(test_info):
                self.skipTest(reason)

        if dtype not in op.supported_backward_dtypes(torch.device(device).type):
            self.skipTest(f"Skipped! {dtype=} is not in supported backward dtypes!")

        with ComplexTensorMode():
            op.gradcheck_fast_mode = False
            self._grad_test_helper(device, dtype, op, op.get_op())


instantiate_device_type_tests(TestComplexTensor, globals())
instantiate_device_type_tests(TestComplexBwdGradients, globals())


if dist.is_available():
    from torch.testing._internal.common_distributed import MultiProcessTestCase

    @unMarkDynamoStrictTest
    class TestComplexDistributed(TestCase, MultiProcessTestCase):
        @ops(implemented_op_db, allowed_dtypes=list(COMPLEX_DTYPES))
        def test_distributed(self, device, dtype, op: OpInfo):
            self.check_consistency(device, dtype, op, Variant.Distributed)

    instantiate_device_type_tests(TestComplexDistributed, globals())

if __name__ == "__main__":
    run_tests()
