# mypy: allow-untyped-defs
import torch
from torch._ops import OpOverload, OpOverloadPacket


def _register_decomposition(op: OpOverload, graph: torch._C.Graph) -> None:
    if isinstance(op, OpOverloadPacket):
        raise AssertionError(
            f"Must pass specific op overload, not overload packet, found {op}"
        )
    if not isinstance(op, OpOverload):
        raise AssertionError(f"Expected OpOverload, got {type(op)}")

    torch._C._jit_register_decomposition_for_schema(op._schema, graph)
