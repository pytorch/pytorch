
import torch
import torch.nn.intrinsic
import torch.nn.intrinsic.qat
import torch.nn.quantized as nnq


class BNReLU2d(nnq.BatchNorm2d):
    r"""
    A BNReLU2d module is a fused module of BatchNorm2d and ReLU

    We adopt the same interface as :class:`torch.nn.quantized.BatchNorm2d`.

    Attributes:
        Same as torch.nn.quantized.BatchNorm2d

    """
    # TODO: Add qat support for BNReLU2d
    _NAME = 'QuantizedBNReLU2d'

    def forward(self, input):
        self._check_input_dim(input)
        return torch.ops.quantized.batch_norm2d_relu(
            input, self.weight, self.bias, self.running_mean,
            self.running_var, self.eps, self.scale, self.zero_point)


class BNReLU3d(nnq.BatchNorm3d):
    r"""
    A BNReLU3d module is a fused module of BatchNorm3d and ReLU

    We adopt the same interface as :class:`torch.nn.quantized.BatchNorm3d`.

    .. note::
    Attributes: Same as torch.nn.quantized.BatchNorm3d

    """
    # TODO: Add qat support for BNReLU3d
    _NAME = 'QuantizedBNReLU3d'

    def forward(self, input):
        self._check_input_dim(input)
        return torch.ops.quantized.batch_norm3d_relu(
            input, self.weight, self.bias, self.running_mean,
            self.running_var, self.eps, self.scale, self.zero_point)
