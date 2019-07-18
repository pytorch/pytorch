from __future__ import absolute_import, division, print_function, unicode_literals

from .modules.conv_relu import ConvReLU2d
from .modules.linear_relu import LinearReLU

from torch._ops import ops

fq_per_tensor_affine_forward = ops.quantized.fake_quantize_per_tensor_affine_forward
fq_per_tensor_affine_backward = ops.quantized.fake_quantize_per_tensor_affine_backward

# Modules
__all__ = [
    'ConvReLU2d',
    'LinearReLU',
]

# Other stuff
__all__ += [
    'fq_per_tensor_affine_forward',
    'fq_per_tensor_affine_backward'
]
