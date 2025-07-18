# mypy: allow-untyped-defs
import torch
from torch._ops import OpOverload, OpOverloadPacket


def _register_decomposition(op: OpOverload, graph: torch._C.Graph):
    assert not isinstance(op, OpOverloadPacket), (
        f"Must pass specific op overload, not overload packet, found {op}"
    )
    assert isinstance(op, OpOverload)

    torch._C._jit_register_decomposition_for_schema(op._schema, graph)
