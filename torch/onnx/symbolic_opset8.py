import torch.onnx.symbolic_helper as sym_help
from torch.onnx.symbolic_helper import parse_args, _unimplemented, _black_list_in_opset


black_listed_operators = ["nonzero", "where", "scatter", "erf", "sign", "isnan", "zeros",
                          "zeros_like", "ones", "ones_like", "full", "full_like", "gather"]

for black_listed_op in black_listed_operators:
    vars()[black_listed_op] = _black_list_in_opset(black_listed_op)


@parse_args('v', 'v')
def upsample_nearest2d(g, input, output_size):
    output_size = sym_help._maybe_get_const(output_size, 'is')
    if sym_help._is_value(output_size):
        return _unimplemented("upsample_nearest2d", "torch._C.Value (oputut_size) indexing")
    else:
        height_scale = float(output_size[-2]) / input.type().sizes()[-2]
        width_scale = float(output_size[-1]) / input.type().sizes()[-1]
        scales = [1., 1., height_scale, width_scale]
        return g.op("Upsample", input, mode_s="nearest",
                    scales_f=scales)

@parse_args('v', 'v', 'i')
def upsample_bilinear2d(g, input, output_size, align_corners):
    if align_corners:
        return _unimplemented("upsample_bilinear2d", "align_corners == True")

    output_size = sym_help._maybe_get_const(output_size, 'is')
    if sym_help._is_value(output_size):
        return _unimplemented("upsample_bilinear2d", "torch._C.Value (oputut_size) indexing")
    else:
        height_scale = float(output_size[-2]) / input.type().sizes()[-2]
        width_scale = float(output_size[-1]) / input.type().sizes()[-1]
        return g.op("Upsample", input, mode_s="linear",
                    scales_f=[1., 1., height_scale, width_scale])
