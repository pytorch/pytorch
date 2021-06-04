# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in symbolic_helper.py

# This file exports ONNX ops for opset 14
from torch.onnx.symbolic_helper import parse_args
from torch.onnx.symbolic_helper import _block_list_in_opset

# Note [ONNX operators that are added/updated in opset 14]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# New operators:
#   HardSwish, Trilu
#
# Updated operators:
#   Reshape
#   Add, Sub, Mul, Div
#   GRU, LSTM, RNN
#   BatchNorm, Cumsum, Relu

block_listed_operators = [
]

for block_listed_op in block_listed_operators:
    vars()[block_listed_op] = _block_list_in_opset(block_listed_op)

@parse_args("v")
def hardswish(g, self):
    return g.op("HardSwish", self)
