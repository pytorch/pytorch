from torch.onnx.symbolic_helper import parse_args, _unimplemented
from torch.onnx.symbolic_helper import _black_list_in_opset


# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in symbolic_helper.py

# This file exports ONNX ops for opset 11

black_listed_operators = [
    "cumsum", "eq", "ne", "scatter", "clamp", "clamp_min", "clamp_max", "sort", "topk", "hardtanh"
]

for black_listed_op in black_listed_operators:
    vars()[black_listed_op] = _black_list_in_opset(black_listed_op)

@parse_args('v', 'i')
def pixel_shuffle(g, self, upscale_factor):
    dims = self.type().sizes()
    if len(dims) != 4:
        return _unimplemented("pixel_shuffle", "only support 4d input")
    return g.op("DepthToSpace", self, blocksize_i=upscale_factor, mode_s="CRD")
