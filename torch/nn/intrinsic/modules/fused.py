import torch
from torch.nn import Conv1d, Conv2d, Conv3d, ReLU, Linear, BatchNorm1d, BatchNorm2d, BatchNorm3d

# Used for identifying intrinsic modules used in quantization
class _FusedModule(torch.nn.Sequential):
    pass

class ConvReLU1d(_FusedModule):
    r"""This is a sequential container which calls the Conv1d and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, conv, relu):
        assert type(conv) == Conv1d and type(relu) == ReLU, \
            'Incorrect types for input modules{}{}'.format(
                type(conv), type(relu))
        super().__init__(conv, relu)

class ConvReLU2d(_FusedModule):
    r"""This is a sequential container which calls the Conv2d and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, conv, relu):
        assert type(conv) == Conv2d and type(relu) == ReLU, \
            'Incorrect types for input modules{}{}'.format(
                type(conv), type(relu))
        super().__init__(conv, relu)

class ConvReLU3d(_FusedModule):
    r"""This is a sequential container which calls the Conv3d and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, conv, relu):
        assert type(conv) == Conv3d and type(relu) == ReLU, \
            'Incorrect types for input modules{}{}'.format(
                type(conv), type(relu))
        super().__init__(conv, relu)

class LinearReLU(_FusedModule):
    r"""This is a sequential container which calls the Linear and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, linear, relu):
        assert type(linear) == Linear and type(relu) == ReLU, \
            'Incorrect types for input modules{}{}'.format(
                type(linear), type(relu))
        super().__init__(linear, relu)

class ConvBn1d(_FusedModule):
    r"""This is a sequential container which calls the Conv 1d and Batch Norm 1d modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, conv, bn):
        assert type(conv) == Conv1d and type(bn) == BatchNorm1d, \
            'Incorrect types for input modules{}{}'.format(
                type(conv), type(bn))
        super().__init__(conv, bn)

class ConvBn2d(_FusedModule):
    r"""This is a sequential container which calls the Conv 2d and Batch Norm 2d modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, conv, bn):
        assert type(conv) == Conv2d and type(bn) == BatchNorm2d, \
            'Incorrect types for input modules{}{}'.format(
                type(conv), type(bn))
        super(ConvBn2d, self).__init__(conv, bn)

class ConvBnReLU1d(_FusedModule):
    r"""This is a sequential container which calls the Conv 1d, Batch Norm 1d, and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, conv, bn, relu):
        assert type(conv) == Conv1d and type(bn) == BatchNorm1d and \
            type(relu) == ReLU, 'Incorrect types for input modules{}{}{}' \
            .format(type(conv), type(bn), type(relu))
        super().__init__(conv, bn, relu)

class ConvBnReLU2d(_FusedModule):
    r"""This is a sequential container which calls the Conv 2d, Batch Norm 2d, and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, conv, bn, relu):
        assert type(conv) == Conv2d and type(bn) == BatchNorm2d and \
            type(relu) == ReLU, 'Incorrect types for input modules{}{}{}' \
            .format(type(conv), type(bn), type(relu))
        super().__init__(conv, bn, relu)

class ConvBn3d(_FusedModule):
    r"""This is a sequential container which calls the Conv 3d and Batch Norm 3d modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, conv, bn):
        assert type(conv) == Conv3d and type(bn) == BatchNorm3d, \
            'Incorrect types for input modules{}{}'.format(
                type(conv), type(bn))
        super().__init__(conv, bn)

class ConvBnReLU3d(_FusedModule):
    r"""This is a sequential container which calls the Conv 3d, Batch Norm 3d, and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, conv, bn, relu):
        assert type(conv) == Conv3d and type(bn) == BatchNorm3d and \
            type(relu) == ReLU, 'Incorrect types for input modules{}{}{}' \
            .format(type(conv), type(bn), type(relu))
        super().__init__(conv, bn, relu)


class BNReLU2d(_FusedModule):
    r"""This is a sequential container which calls the BatchNorm 2d and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, batch_norm, relu):
        assert type(batch_norm) == BatchNorm2d and type(relu) == ReLU, \
            'Incorrect types for input modules{}{}'.format(
                type(batch_norm), type(relu))
        super().__init__(batch_norm, relu)

class BNReLU3d(_FusedModule):
    r"""This is a sequential container which calls the BatchNorm 3d and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, batch_norm, relu):
        assert type(batch_norm) == BatchNorm3d and type(relu) == ReLU, \
            'Incorrect types for input modules{}{}'.format(
                type(batch_norm), type(relu))
        super().__init__(batch_norm, relu)
