from __future__ import absolute_import, division, print_function, unicode_literals

from torch.onnx.symbolic_helper import parse_args


# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in symbolic_helper.py

# This file exports ONNX ops for opset 12

@parse_args('s', 'v')
def einsum(g, equation, tensor_list):
    return g.op("Einsum", tensor_list, equation_s=equation)
