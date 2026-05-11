r"""
Typed control-flow graphs for PyTorch.

``torch.cfg`` is a small, principled IR for analyses and transforms that need
explicit basic blocks and typed values. Unlike FX nodes, semantic metadata lives
in a single ``Value.spec`` field instead of an open-ended ``meta`` dictionary.
Value names are globally unique across a graph, even across blocks, so textual
rendering and validation can use them as stable identifiers.

Example
-------

.. code-block:: python

    import torch
    import torch.cfg as cfg

    x = cfg.Value("x", cfg.TensorSpec.from_tensor(torch.randn(2, 3)))
    pred = cfg.Value("pred", cfg.ScalarSpec(bool))
    negative_input = cfg.Value("negative_input", x.spec)
    result = cfg.Value("result", x.spec)
    neg = cfg.Value("neg", x.spec)

    graph = cfg.Graph(
        name="branchy",
        entry="entry",
        blocks=(
            cfg.Block(
                "entry",
                parameters=(x,),
                instructions=(
                    cfg.Instruction(
                        name="lt_zero",
                        opcode="call_function",
                        target=torch.ops.aten.lt.Scalar,
                        inputs=(x, 0),
                        outputs=(pred,),
                    ),
                ),
                terminator=cfg.Branch(
                    pred,
                    cfg.Successor("negative", (x,)),
                    cfg.Successor("done", (x,)),
                ),
            ),
            cfg.Block(
                "negative",
                parameters=(negative_input,),
                instructions=(
                    cfg.Instruction(
                        name="negate",
                        opcode="call_function",
                        target=torch.neg,
                        inputs=(negative_input,),
                        outputs=(neg,),
                    ),
                ),
                terminator=cfg.Jump(cfg.Successor("done", (neg,))),
            ),
            cfg.Block(
                "done",
                parameters=(result,),
                terminator=cfg.Return(result),
            ),
        ),
    )
"""

from .fx import from_fx
from .ir import (
    Block,
    Branch,
    DictSpec,
    Graph,
    Instruction,
    Jump,
    ListSpec,
    Literal,
    literal,
    Location,
    ObjectSpec,
    OptionalSpec,
    Return,
    ScalarSpec,
    Spec,
    Successor,
    TensorSpec,
    TupleSpec,
    ValidationError,
    Value,
)


__all__ = [
    "Block",
    "Branch",
    "DictSpec",
    "Graph",
    "Instruction",
    "Jump",
    "ListSpec",
    "Literal",
    "Location",
    "ObjectSpec",
    "OptionalSpec",
    "Return",
    "ScalarSpec",
    "Spec",
    "Successor",
    "TensorSpec",
    "TupleSpec",
    "ValidationError",
    "Value",
    "literal",
    "from_fx",
]
