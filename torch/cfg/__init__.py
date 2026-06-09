"""
`torch.cfg` is a strict control-flow graph API.

Unlike `torch.fx`, control flow is explicit through basic blocks and terminators,
and value facts live in structured `ValueInfo` objects instead of a generic
metadata bag. Example values are optional and separated from abstract tensor
descriptions so callers do not have to guess whether `node.meta["val"]` or
`node.meta["example_value"]` is authoritative.
"""

from .graph import (
    BasicBlock,
    BlockParameterSpec,
    BranchTerminator,
    ControlFlowGraph,
    Graph,
    Instruction,
    InstructionKind,
    JumpTerminator,
    RaiseTerminator,
    ReturnTerminator,
    SourceLocation,
    TensorSpec,
    Value,
    ValueInfo,
)


__all__ = [
    "BasicBlock",
    "BlockParameterSpec",
    "BranchTerminator",
    "ControlFlowGraph",
    "Graph",
    "Instruction",
    "InstructionKind",
    "JumpTerminator",
    "RaiseTerminator",
    "ReturnTerminator",
    "SourceLocation",
    "TensorSpec",
    "Value",
    "ValueInfo",
]
