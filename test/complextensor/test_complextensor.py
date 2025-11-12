# Owner(s): ["module: complex"]
from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch
import torch.distributed as dist
from torch.complex.complextensor.ops.common import (
    _as_complex_tensor,
    _get_op_name,
    COMPLEX_OPS_TABLE,
    ComplexTensorMode,
    FORCE_TEST_LIST,
    OpOverloadPacket,
)
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    OpDTypes,
    ops,
)
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_utils import (
    parametrize,
    run_tests,
    TestGradients,
    unMarkDynamoStrictTest,
)

from .utils import COMPLEX_DTYPES, Descriptor, TestCase, Variant


if TYPE_CHECKING:
    from torch.testing._internal.opinfo.core import OpInfo


torch._dynamo.config.recompile_limit = float("inf")
torch._dynamo.config.accumulated_recompile_limit = float("inf")

aten = torch.ops.aten

complex_op_db = tuple(
    filter(lambda op: any(op.supports_dtype(ct, "cpu") for ct in COMPLEX_DTYPES), op_db)
)


def get_overload_packet_from_name(name: str) -> OpOverloadPacket:
    for domain_name in torch.ops:
        op_namespace = getattr(torch.ops, domain_name)
        op: OpOverloadPacket | None = getattr(op_namespace, name, None)
        if op is not None:
            return op

    raise RuntimeError(f"No op with {name=} found.")


force_test_names = set(map(_get_op_name, FORCE_TEST_LIST))
implemented_op_names = (
    set(map(_get_op_name, COMPLEX_OPS_TABLE.keys())) - force_test_names
)
implemented_op_db = tuple(
    filter(lambda op: op.name in implemented_op_names, complex_op_db)
)
force_test_op_db = tuple(filter(lambda op: op.name in force_test_names, op_db))

tested_op_names = {op.name for op in implemented_op_db} | {
    op.name for op in force_test_op_db
}
non_tested_ops = {
    op for op in COMPLEX_OPS_TABLE if _get_op_name(op) not in tested_op_names
}

if len(non_tested_ops) != 0:
    import textwrap
    import warnings

    list_missing_ops = "\n".join(sorted([str(op) for op in non_tested_ops]))
    warnings.warn(
        "Not all implemented ops are tested. List of ops missing tests:"
        f"\n{textwrap.indent(list_missing_ops, '    ')}",
        UserWarning,
        stacklevel=2,
    )


SKIPS = {
    Descriptor(op=aten.empty_like, variant=None): "Non-deterministic output",
    # This passes with `PYTORCH_OPINFO_SAMPLE_INPUT_INDEX=35 ...
    # but when the whole test is run, it fails with this exact
    # sample.
    Descriptor(op=aten.repeat, compile=True, variant=None): "Heisenbug",
    Descriptor(
        op=aten.allclose, compile=True, variant=None
    ): "`aten.allclose` requires data-dependent control-flow",
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

STORE = dist.HashStore()
dist.init_process_group(store=STORE, rank=0, world_size=1)
DEVICE_MESH = dist.init_device_mesh("cpu", mesh_shape=(1,))


def _as_complex_dtensor(arg: torch.Tensor | Any) -> torch.Tensor | Any:
    if not isinstance(arg, torch.Tensor):
        return arg

    return dist.tensor.DTensor.from_local(
        _as_complex_tensor(arg), device_mesh=DEVICE_MESH
    )


TRANSFORM_FUNCS = {
    Variant.Op: _as_complex_tensor,
    Variant.Distributed: _as_complex_dtensor,
}


class TestComplexTensor(TestCase):
    _default_dtype_check_enabled = True

    @parametrize("compile", [False, True])
    @ops(
        implemented_op_db,
        dtypes=OpDTypes.supported,
        allowed_dtypes=list(COMPLEX_DTYPES),
    )
    def test_consistency(self, device, dtype, op: OpInfo, compile: bool):
        self.check_consistency(device, dtype, op, compile, Variant.Op)

    @parametrize("compile", [False, True])
    @ops(force_test_op_db, allowed_dtypes=list(COMPLEX_DTYPES))
    def test_maybe_error(self, device, dtype, op: OpInfo, compile: bool):
        self.check_consistency(device, dtype, op, compile, Variant.Op)

    @ops(implemented_op_db, allowed_dtypes=list(COMPLEX_DTYPES))
    def test_distributed(self, device, dtype, op: OpInfo):
        self.check_consistency(device, dtype, op, False, Variant.Distributed)

    def check_consistency(
        self, device: torch.device, dtype, op: OpInfo, compile: bool, variant: Variant
    ) -> None:
        test_info = Descriptor(
            op=get_overload_packet_from_name(op.name),
            device=device,
            dtype=dtype,
            compile=compile,
            variant=variant,
        )
        for xfail_info, reason in SKIPS.items():
            if xfail_info.matches(test_info):
                self.skipTest(reason)

        kwargs = {}
        for extra_info, extra_kw in EXTRA_KWARGS.items():
            if extra_info.matches(test_info):
                kwargs = extra_kw
                break

        sample_inputs = op.sample_inputs(device, dtype)
        op_eager = op
        if compile:
            op = torch.compile(op, fullgraph=True)

        transform_fn = TRANSFORM_FUNCS[variant]

        for sample_input in sample_inputs:

            def expected(sample_input=sample_input):
                return op_eager(
                    sample_input.input, *sample_input.args, **sample_input.kwargs
                )

            subclass_sample = sample_input.transform(transform_fn)

            def actual(subclass_sample=subclass_sample):
                return op(
                    subclass_sample.input,
                    *subclass_sample.args,
                    **subclass_sample.kwargs,
                )

            self.assertSameResult(expected, actual, ignore_exc_types=compile, **kwargs)


@unMarkDynamoStrictTest
class TestComplexBwdGradients(TestGradients):
    @ops(
        implemented_op_db,
        dtypes=OpDTypes.supported_backward,
        allowed_dtypes=[torch.complex128],
    )
    def test_fn_grad(
        self, device: torch.device, dtype: torch.dtype, op: OpInfo
    ) -> None:
        test_info = Descriptor(
            op=get_overload_packet_from_name(op.name),
            device=device,
            dtype=dtype,
            compile=False,
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

if __name__ == "__main__":
    run_tests()
