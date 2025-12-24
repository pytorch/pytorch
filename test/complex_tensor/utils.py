from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import auto, Enum
from typing import Any, TYPE_CHECKING

import torch
import torch.distributed as dist
from torch._subclasses.complex_tensor._ops.common import (
    _as_complex_tensor,
    _as_interleaved,
    _get_op_name,
    COMPLEX_OPS_TABLE,
    COMPLEX_TO_REAL,
    FORCE_TEST_LIST,
    OpOverloadPacket,
)
from torch.autograd.gradcheck import gradcheck
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_utils import TestCase as PytorchTestCase
from torch.utils._pytree import tree_flatten, tree_map


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.distributed.tensor import DTensor
    from torch.testing._internal.opinfo.core import OpInfo

COMPLEX_DTYPES = set(COMPLEX_TO_REAL)


class Variant(Enum):
    Op = auto()
    GradCheck = auto()
    Distributed = auto()


def _as_local(arg: DTensor | Any) -> torch.Tensor | Any:
    if not (dist.is_available() and isinstance(arg, dist.tensor.DTensor)):
        return arg

    return arg.full_tensor()


def _as_complex_dtensor(arg: torch.Tensor | Any) -> torch.Tensor | Any:
    if not isinstance(arg, torch.Tensor):
        return arg

    return dist.tensor.DTensor.from_local(_as_complex_tensor(arg))


TRANSFORM_FUNCS = {
    Variant.Op: _as_complex_tensor,
    Variant.Distributed: _as_complex_dtensor,
}


@dataclass(frozen=True, kw_only=True)
class Descriptor:
    op: OpOverloadPacket
    variant: Variant | None
    device_type: str | None = field(default=None)
    dtype: torch.dtype | None = field(default=None)

    def matches(self, other: Descriptor) -> bool:
        fields1 = fields(self)
        fields2 = fields(other)
        if fields1 != fields2:
            return False

        for f in fields1:
            f1 = getattr(self, f.name)
            f2 = getattr(other, f.name)
            if f1 is not None and f2 is not None and f1 != f2:
                return False

        return True


class TestCase(PytorchTestCase):
    def assertSameResult(
        self,
        expected: Callable[[], Any],
        actual: Callable[[], Any],
        *args,
        **kwargs,
    ) -> None:
        try:
            result_e = expected()
            exception_e = None
        except Exception as e:  # noqa: BLE001
            result_e = None
            exception_e = e

        try:
            result_a = actual()
            exception_a = None
        except Exception as e:  # noqa: BLE001
            result_a = None
            exception_a = e

        if (exception_e is None) != (exception_a is None):
            if exception_a is not None and exception_e is None:
                raise exception_a
            self.assertIs(
                type(exception_e),
                type(exception_a),
                f"\n{exception_e=}\n{exception_a=}",
            )

        if exception_e is None:
            flattened_e, spec_e = tree_flatten(result_e)
            flattened_a, spec_a = tree_flatten(result_a)

            self.assertEqual(
                spec_e,
                spec_a,
                "Both functions must return a result with the same tree structure.",
            )
            for value_e, value_a in zip(flattened_e, flattened_a, strict=True):
                value_e = _as_interleaved(_as_local(value_e))
                value_a = _as_interleaved(_as_local(value_a))

                self.assertEqual(value_e, value_a, *args, **kwargs)

    def conditional_skip(self, desc: Descriptor) -> None:
        try:
            from .test_complex_tensor import SKIPS
        except ImportError:
            from test_complex_tensor import SKIPS
        for xfail_info, reason in SKIPS.items():
            if xfail_info.matches(desc):
                self.skipTest(reason)

    def get_extra_kwargs(self, desc: Descriptor) -> dict[str, Any]:
        try:
            from .test_complex_tensor import EXTRA_KWARGS
        except ImportError:
            from test_complex_tensor import EXTRA_KWARGS

        for extra_info, extra_kw in EXTRA_KWARGS.items():
            if extra_info.matches(desc):
                return extra_kw

        return {}

    def check_consistency(
        self, device: str, dtype: torch.dtype, op: OpInfo, variant: Variant
    ) -> None:
        assert variant in {Variant.Op, Variant.Distributed}, (
            "`check_consistency` called with the wrong `variant`."
        )
        desc = Descriptor(
            op=get_overload_packet_from_name(op.name),
            device_type=torch.device(device).type,
            dtype=dtype,
            variant=variant,
        )
        self.conditional_skip(desc)
        kwargs = self.get_extra_kwargs(desc)

        transform_fn = TRANSFORM_FUNCS[variant]

        sample_inputs = op.sample_inputs(device, dtype)

        for sample_input in sample_inputs:

            def expected(sample_input=sample_input):
                return op(
                    sample_input.input,
                    *sample_input.args,
                    **sample_input.kwargs,
                )

            subclass_sample = sample_input.transform(transform_fn)

            def actual(subclass_sample=subclass_sample):
                return op(
                    subclass_sample.input,
                    *subclass_sample.args,
                    **subclass_sample.kwargs,
                )

            self.assertSameResult(expected, actual, **kwargs)

    def check_grad(self, device: str, dtype: torch.dtype, op: OpInfo) -> None:
        desc = Descriptor(
            op=get_overload_packet_from_name(op.name),
            device_type=torch.device(device).type,
            dtype=dtype,
            variant=Variant.GradCheck,
        )
        self.conditional_skip(desc)

        sample_inputs = op.sample_inputs(device, dtype, requires_grad=True)

        for sample_input in sample_inputs:
            # We don't use `sample_input.transform` here as it
            # uses a `no_grad` context which makes one lose the
            # `requires_grad` flag.
            args, kwargs = (sample_input.input, *sample_input.args), sample_input.kwargs
            args, kwargs = tree_map(
                _as_complex_tensor,
                (args, kwargs),
                is_leaf=lambda x: isinstance(x, torch.Tensor) and x.dtype.is_complex,
            )

            if isinstance(args[0], torch.Tensor):
                input_list = list(args)

                def func(*t: torch.Tensor):
                    return op(*t, **kwargs)
            else:
                # For ops like `stack` and `cat`, we need to specify
                # the list of inputs a bit differently
                input_list = list(args[0])

                def func(*t: torch.Tensor):
                    return op(list(t), *args[1:], **kwargs)

            # Do the actual gradcheck
            self.assertTrue(
                gradcheck(
                    func,
                    input_list,
                    raise_exception=True,
                    fast_mode=True,
                    check_batched_grad=True,
                    check_forward_ad=True,
                )
            )


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


# TODO (hameerabbasi): There are a number of ops that don't have any associated
# OpInfos. We still need to write tests for those ops.
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
