import torch
import torch.nn.quantized.functional


class _BatchNormBase(torch.nn.modules.batchnorm._BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        r"""Base class for the batch norm"""
        super().__init__(num_features, eps, momentum)
        self.scale = 1.0
        self.zero_point = 0

    def _get_name(self):
        return self._NAME

    def forward(self, x):
        raise NotImplementedError("Cannot use instance of `_BatchNormBase`")

    @classmethod
    def from_float(cls, mod):
        activation_post_process = mod.activation_post_process
        if hasattr(mod, '__getitem__') and type(mod[0]) == cls._FLOAT_MODULE:
            # Intrinsic (fused) modules are treated as a sequential set.
            # However, we need to treat them as fused, and will only take the
            # batch norm layer.
            mod = mod[0]
        scale, zero_point = activation_post_process.calculate_qparams()
        new_mod = cls(mod.num_features, mod.eps)
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        new_mod.running_mean = mod.running_mean
        new_mod.running_var = mod.running_var
        new_mod.scale = float(scale)
        new_mod.zero_point = int(zero_point)
        return new_mod

    def _check_input_dim(self, x):
        if self._DIM == 1 and x.dim() != 2 and x.dim() != 3:
            raise ValueError(f'Expected 2D or 3D input (got {x.dim()}D)')
        elif x.dim() != self._DIM + 2:
            raise ValueError(f'Expected {self._DIM}D input (got {x.dim()}D)')


class BatchNorm1d(_BatchNormBase):
    r"""This is the quantized version of :class:`~torch.nn.BatchNorm1d`.
    """
    _FLOAT_MODULE = torch.nn.BatchNorm1d
    _DIM = 1
    _NAME = 'QuantizedBatchNorm1d'

    def forward(self, input):
        return torch.ops.quantized.batch_norm1d(input, self.weight, self.bias, self.running_mean,
                                                self.running_var, self.eps, self.scale, self.zero_point)


class BatchNorm2d(_BatchNormBase):
    r"""This is the quantized version of :class:`~torch.nn.BatchNorm2d`.
    """
    _FLOAT_MODULE = torch.nn.BatchNorm2d
    _DIM = 2
    _NAME = 'QuantizedBatchNorm2d'

    def forward(self, input):
        return torch.ops.quantized.batch_norm2d(input, self.weight, self.bias, self.running_mean,
                                                self.running_var, self.eps, self.scale, self.zero_point)


class BatchNorm3d(_BatchNormBase):
    r"""This is the quantized version of :class:`~torch.nn.BatchNorm3d`.
    """
    _FLOAT_MODULE = torch.nn.BatchNorm3d
    _DIM = 3
    _NAME = 'QuantizedBatchNorm3d'

    def forward(self, input):
        return torch.ops.quantized.batch_norm3d(input, self.weight, self.bias, self.running_mean,
                                                self.running_var, self.eps, self.scale, self.zero_point)
