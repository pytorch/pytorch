from typing import Dict, List, Set, Type

import torch
from torch._ops import OpOverload, OpOverloadPacket
from torch._export.pass_base import ExportPassBase


__all__ = ["ReplaceViewOpsWithViewCopyOpsPass"]


_NON_FUNCTIONAL_OPS_TO_FUNCTIONAL_OPS: Dict[OpOverload, OpOverload] = {
    torch.ops.aten._unsafe_view.default: torch.ops.aten.view_copy.default,
}

# TODO (tmanlaibaatar) remove this after https://github.com/pytorch/pytorch/pull/100749
_BLACK_LISTED_OPS: Set[OpOverloadPacket] = {
    torch.ops.aten.sym_size,
    torch.ops.aten.sym_stride,
    torch.ops.aten.sym_numel,
}

def is_view_op(schema: torch._C.FunctionSchema) -> bool:
    if len(schema.arguments) == 0:
        return False
    return schema.arguments[0].alias_info is not None


class ReplaceViewOpsWithViewCopyOpsPass(ExportPassBase):
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

        if op in _BLACK_LISTED_OPS:
            return super().call_operator(op, args, kwargs, meta)

        if op._schema.name.startswith("aten::") and is_view_op(
            op._schema
        ):
            view_op_name = op._schema.name.split("::")[1]
            view_op_overload = (
                op._schema.overload_name
                if op._schema.overload_name != ""
                else "default"
            )
            view_copy_op_name = view_op_name + "_copy"
            view_copy_op = getattr(
                getattr(torch.ops.aten, view_copy_op_name), view_op_overload
            )
            return super().call_operator(view_copy_op, args, kwargs, meta)

        return super().call_operator(op, args, kwargs, meta)
