# exists to refine the type of the Value
# if x is an optional Tensor, unchecked_cast will cast
# x to Tensor, so the rest of the graph knows that x is a Tensor
# this doesn't do anything in runtime and is a noop in ONNX
import torch
from torch.onnx.symbolic_opset9 import eq, wrap_logical_op_with_negation
from torch.onnx.symbolic_helper import _is_none


def __is_(g, self, other):
    if _is_none(other):
        if self.type().kind() == 'OptionalType' or self.type().kind() == 'NoneType':
            none = g.op("OptionalHasElement", self)
            return g.op("Not", none)
        else:
            return g.op("Constant", value_t=torch.BoolTensor([0]))
    return eq(g, self, other)


@wrap_logical_op_with_negation
def __isnot_(g, self, other):
    return __is_(g, self, other)


def prim_unchecked_cast(g, self):
    if self.type().kind() == 'OptionalType' or self.type().kind() == 'NoneType':
        return g.op("OptionalGetElement", self)
    else:
        return self
