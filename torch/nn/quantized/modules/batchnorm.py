import torch
import torch.nn.quantized.functional
import torch.nn.intrinsic as nni
from torch import Tensor

class _BatchNorm(torch.nn.modules.batchnorm._BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(num_features, eps, momentum, True, True, **factory_kwargs)
        self.register_buffer('scale', torch.tensor(1.0, **factory_kwargs))
        self.register_buffer('zero_point', torch.tensor(0, **factory_kwargs))

    @staticmethod
    def from_float(cls, mod):
        activation_post_process = mod.activation_post_process
        if type(mod) == cls._NNI_BN_RELU_MODULE:
            mod = mod[0]
        scale, zero_point = activation_post_process.calculate_qparams()
        new_mod = cls(mod.num_features, mod.eps)
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        new_mod.running_mean = mod.running_mean
        new_mod.running_var = mod.running_var
        new_mod.scale = scale
        new_mod.zero_point = zero_point
        return new_mod

    @classmethod
    def from_reference(cls, bn, output_scale, output_zero_point):
        qbn = cls(
            bn.num_features,
            bn.eps,
            bn.momentum,
            device=bn.weight.device,
            dtype=bn.weight.dtype
        )
        qbn.weight = bn.weight
        qbn.bias = bn.bias
        qbn.running_mean = bn.running_mean
        qbn.running_var = bn.running_var
        qbn.scale = output_scale
        qbn.zero_point = output_zero_point
        return qbn

class BatchNorm2d(_BatchNorm):
    r"""This is the quantized version of :class:`~torch.nn.BatchNorm2d`.
    """

    _NNI_BN_RELU_MODULE = nni.BNReLU2d

    def __init__(self, num_features, eps=1e-5, momentum=0.1, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(num_features, eps, momentum, **factory_kwargs)

    def _get_name(self):
        return 'QuantizedBatchNorm2d'

    def _check_input_dim(self, input):
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 4:
            raise ValueError("Input shape must be `(N, C, H, W)`!")

    def forward(self, input: Tensor) -> Tensor:
        # disabling this since this is not symbolically traceable
        # self._check_input_dim(input)
        return torch.ops.quantized.batch_norm2d(
            input, self.weight, self.bias, self.running_mean,
            self.running_var, self.eps, self.scale, self.zero_point)

    @classmethod
    def from_float(cls, mod):
        return _BatchNorm.from_float(cls, mod)

class BatchNorm3d(_BatchNorm):
    r"""This is the quantized version of :class:`~torch.nn.BatchNorm3d`.
    """

    _NNI_BN_RELU_MODULE = nni.BNReLU3d

    def __init__(self, num_features, eps=1e-5, momentum=0.1, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(num_features, eps, momentum, **factory_kwargs)

    def _get_name(self):
        return 'QuantizedBatchNorm3d'

    def _check_input_dim(self, input):
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 5:
            raise ValueError("Input shape must be `(N, C, H, W)`!")

    def forward(self, input: Tensor) -> Tensor:
        # disabling this since this is not symbolically traceable
        # self._check_input_dim(input)
        return torch.ops.quantized.batch_norm3d(
            input, self.weight, self.bias, self.running_mean,
            self.running_var, self.eps, self.scale, self.zero_point)

    @classmethod
    def from_float(cls, mod):
        return _BatchNorm.from_float(cls, mod)
