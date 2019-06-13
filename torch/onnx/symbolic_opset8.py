import torch
import torch.onnx
import torch.onnx.utils

from torch.onnx.symbolic_helper import parse_args

@parse_args('v', 'v')
def upsample_nearest2d(g, input, output_size):
    height_scale = float(output_size[-2]) / input.type().sizes()[-2]
    width_scale = float(output_size[-1]) / input.type().sizes()[-1]

    print(height_scale, width_scale)
    return g.op("Upsample", input, mode_s="nearest",
                scales_f=[1., 1., height_scale, width_scale])


@parse_args('v', 'is', 'i')
def upsample_bilinear2d(g, input, output_size, align_corners):
    if align_corners:
        return _unimplemented("upsample_bilinear2d", "align_corners == True")
    height_scale = float(output_size[-2]) / input.type().sizes()[-2]
    width_scale = float(output_size[-1]) / input.type().sizes()[-1]

    return g.op("Upsample", input, mode_s="linear",
                scales_f=[1., 1., height_scale, width_scale])
