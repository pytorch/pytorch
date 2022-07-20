"""This file exports ONNX ops for opset 15.

Note [ONNX operators that are added/updated in opset 15]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
https://github.com/onnx/onnx/blob/master/docs/Changelog.md#version-15-of-the-default-onnx-operator-set
New operators:
    Bernoulli
    CastLike
    Optional
    OptionalGetElement
    OptionalHasElement

Updated operators:
    BatchNormalization https://github.com/onnx/onnx/pull/3545
                        Backwards compatible
                        TODO: test coverage for mixed types inputs.
    Pow                https://github.com/onnx/onnx/pull/3412
                        Backwards compatible
                        TODO: bfloat16 support.
    Shape              https://github.com/onnx/onnx/pull/3580
                        Backwards compatible
                        TODO: optional start/end attribute.
"""

# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in symbolic_helper.py

import torch
from torch import _C
from torch.onnx import symbolic_helper
from torch.onnx import symbolic_opset9 as opset9


def __is_(g, self, other):
    if symbolic_helper._is_none(other):
        if isinstance(self.type(), _C.OptionalType):
            none = g.op("OptionalHasElement", self)
            return g.op("Not", none)
        else:
            return g.op("Constant", value_t=torch.BoolTensor([0]))
    return opset9.eq(g, self, other)


@opset9.wrap_logical_op_with_negation
def __isnot_(g, self, other):
    return __is_(g, self, other)


class Prim:
    domain = "prim"

    @staticmethod
    def unchecked_cast(g, self):
        # exists to refine the type of the Value
        # if x is Optional[Tensor], unchecked_cast will cast
        # x to Tensor, so the rest of the graph knows that x is a Tensor.
        if isinstance(self.type(), _C.OptionalType):
            return g.op("OptionalGetElement", self)

        return self
