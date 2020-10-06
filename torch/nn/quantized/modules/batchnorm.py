import torch
import torch.nn.quantized.functional
import torch.nn.intrinsic as nni

class BatchNorm2d(torch.nn.BatchNorm2d):
    r"""This is the quantized version of :class:`~torch.nn.BatchNorm2d`.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(BatchNorm2d, self).__init__(num_features)
        self.eps = eps
        self.scale = 1.0
        self.zero_point = 0

    def forward(self, input):
        return torch.ops.quantized.batch_norm2d(input, self.weight, self.bias, self.running_mean,
                                                self.running_var, self.eps, self.scale, self.zero_point)

    def _get_name(self):
        return 'QuantizedBatchNorm2d'

    @classmethod
    def from_float(cls, mod):
        if type(mod) == nni.BNReLU2d:
            activation_post_process = mod[1].activation_post_process
            mod = mod[0]
        else:
            activation_post_process = mod.activation_post_process
        scale, zero_point = activation_post_process.calculate_qparams()
        new_mod = cls(mod.num_features, mod.eps)
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        new_mod.running_mean = mod.running_mean
        new_mod.running_var = mod.running_var
        new_mod.scale = float(scale)
        new_mod.zero_point = int(zero_point)
        return new_mod

class BatchNorm3d(torch.nn.BatchNorm3d):
    r"""This is the quantized version of :class:`~torch.nn.BatchNorm3d`.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(BatchNorm3d, self).__init__(num_features)
        self.eps = eps
        self.scale = 1.0
        self.zero_point = 0

    def forward(self, input):
        return torch.ops.quantized.batch_norm3d(input, self.weight, self.bias, self.running_mean,
                                                self.running_var, self.eps, self.scale, self.zero_point)

    def _get_name(self):
        return 'QuantizedBatchNorm3d'

    @classmethod
    def from_float(cls, mod):
        if type(mod) == nni.BNReLU3d:
            activation_post_process = mod[1].activation_post_process
            mod = mod[0]
        else:
            activation_post_process = mod.activation_post_process

        scale, zero_point = activation_post_process.calculate_qparams()
        new_mod = cls(mod.num_features, mod.eps)
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        new_mod.running_mean = mod.running_mean
        new_mod.running_var = mod.running_var
        new_mod.scale = float(scale)
        new_mod.zero_point = int(zero_point)
        return new_mod
