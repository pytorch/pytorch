from typing import Dict, List, Type

import torch
from torch._ops import OpOverload
from torch._export.pass_base import ExportPassBase


__all__ = ["ReplaceBrokenOpsWithFunctionalOpsPass"]


_NON_FUNCTIONAL_OPS_TO_FUNCTIONAL_OPS: Dict[OpOverload, OpOverload] = {
    torch.ops.aten._unsafe_view.default: torch.ops.aten.view_copy.default,
    torch.ops.aten.t.default: torch.ops.aten.t_copy.default,
    torch.ops.aten.view.default: torch.ops.aten.view_copy.default,
    torch.ops.aten.expand.default: torch.ops.aten.expand_copy.default,
    torch.ops.aten.permute.default: torch.ops.aten.permute_copy.default,
}


class ReplaceBrokenOpsWithFunctionalOpsPass(ExportPassBase):
    """
    Our backend expects pure functions. However, some operators
    are not functionalized properly. This pass intends to replace
    non-functionalized operators with their functionalized variant.
    """
    def get_valid_dialects(self) -> List[Type]:
        return []

    def call_operator(self, op, args, kwargs, meta):
        if op in _NON_FUNCTIONAL_OPS_TO_FUNCTIONAL_OPS:
            return super().call_operator(
                (_NON_FUNCTIONAL_OPS_TO_FUNCTIONAL_OPS[op]), args, kwargs, meta
            )
        return super().call_operator(op, args, kwargs, meta)
