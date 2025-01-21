# mypy: allow-untyped-defs
from typing import Optional
import torch
from torch._ops import OpOverload, HigherOrderOperator
from torch._export.error import InternalError
from torch._export.pass_base import _ExportPassBaseDeprecatedDoNotUse


__all__ = ["ReplaceViewOpsWithViewCopyOpsPass"]


_NON_FUNCTIONAL_OPS_TO_FUNCTIONAL_OPS: dict[OpOverload, OpOverload] = {
    torch.ops.aten._unsafe_view.default: torch.ops.aten.view_copy.default,
}


def is_view_op(schema: torch._C.FunctionSchema) -> bool:
    if len(schema.arguments) == 0:
        return False
    alias_info = schema.arguments[0].alias_info
    return (alias_info is not None) and (not alias_info.is_write)


def get_view_copy_of_view_op(schema: torch._C.FunctionSchema) -> Optional[OpOverload]:
    if is_view_op(schema) and schema.name.startswith("aten::"):
        view_op_name = schema.name.split("::")[1]
        view_op_overload = (
            schema.overload_name
            if schema.overload_name != ""
            else "default"
        )
        view_copy_op_name = view_op_name + "_copy"
        if not hasattr(torch.ops.aten, view_copy_op_name):
            raise InternalError(f"{schema.name} is missing a view_copy variant")

        view_copy_op_overload_packet = getattr(torch.ops.aten, view_copy_op_name)

        if not hasattr(view_copy_op_overload_packet, view_op_overload):
            raise InternalError(f"{schema.name} is missing a view_copy variant")

        return getattr(view_copy_op_overload_packet, view_op_overload)

    return None


class ReplaceViewOpsWithViewCopyOpsPass(_ExportPassBaseDeprecatedDoNotUse):
    """
    Our backend expects pure functional operators. For efficiency
    purposes, we keep view ops around while functionalizing the exported
    program. This pass replaces view ops with view copy ops for backends that
    need AOT memory planning.
    """
    def call_operator(self, op, args, kwargs, meta):
        if op in _NON_FUNCTIONAL_OPS_TO_FUNCTIONAL_OPS:
            return super().call_operator(
                (_NON_FUNCTIONAL_OPS_TO_FUNCTIONAL_OPS[op]), args, kwargs, meta
            )

        if isinstance(op, HigherOrderOperator):
            return super().call_operator(op, args, kwargs, meta)

        if view_copy_op := get_view_copy_of_view_op(op._schema):
            return super().call_operator(view_copy_op, args, kwargs, meta)

        return super().call_operator(op, args, kwargs, meta)
