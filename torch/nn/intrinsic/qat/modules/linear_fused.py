import torch
import torch.nn as nn
import torch.ao.nn.intrinsic as nni
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.utils.fusion import fuse_linear_bn_weights


class LinearBn1d(nn.modules.linear.Linear, nni._FusedModule):
    r"""
    A LinearBn1d module is a module fused from Linear and BatchNorm1d, attached
    with FakeQuantize modules for weight, used in quantization aware training.

    We combined the interface of :class:`torch.nn.Linear` and
    :class:torch.nn.BatchNorm1d`.

    Similar to :class:`torch.nn.Linear`, with FakeQuantize modules initialized
    to default.

    Attributes:
        freeze_bn:
        weight_fake_quant: fake quant module for weight

    """
    def __init__(self,
                 # Linear args
                 in_features, out_features, bias=True,
                 # BatchNorm1d args
                 # num_features: out_features
                 eps=1e-05, momentum=0.1,
                 # affine: True
                 # track_running_stats: True
                 # Args for this module
                 freeze_bn=False,
                 qconfig=None):
        nn.modules.linear.Linear.__init__(self, in_features, out_features, bias)
        assert qconfig, 'qconfig must be provded for QAT module'
        self.qconfig = qconfig
        self.freeze_bn = freeze_bn if self.training else True
        self.bn = nn.BatchNorm1d(out_features, eps, momentum, True, True)
        self.weight_fake_quant = self.qconfig.weight()
        if bias:
            self.bias = Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_bn_parameters()

        # this needs to be called after reset_bn_parameters,
        # as they modify the same state
        if self.training:
            if freeze_bn:
                self.freeze_bn_stats()
            else:
                self.update_bn_stats()
        else:
            self.freeze_bn_stats()

    def reset_running_stats(self):
        self.bn.reset_running_stats()

    def reset_bn_parameters(self):
        self.bn.reset_running_stats()
        init.uniform_(self.bn.weight)
        init.zeros_(self.bn.bias)

    def reset_parameters(self):
        super(LinearBn1d, self).reset_parameters()

    def update_bn_stats(self):
        self.freeze_bn = False
        self.bn.training = True
        return self

    def freeze_bn_stats(self):
        self.freeze_bn = True
        self.bn.training = False
        return self

    def forward(self, input):
        assert self.bn.running_var is not None

        # Scale the linear weights by BN's running statistics to reduce
        # weight jitter, see https://arxiv.org/pdf/1806.08342.pdf, page 18
        # for motivation.
        #
        # Instead of
        #
        #   x1 = F.linear(x0, fq(w), b)
        #   x2 = self.bn(x1)
        #
        # We have
        #
        #   # scale the weight by previous batch's running statistics
        #   scale_factor = bn.w / bn.running_std_from_prev_batch
        #   # do the linear transformation without bias
        #   x1_scaled = F.linear(x0, fq(w * scale_factor), 0)
        #   # reverse the scaling and add original bias
        #   x1_orig = x1_scaled / scale_factor + b
        #   x2 = self.bn(x1_orig)

        running_std = torch.sqrt(self.bn.running_var + self.bn.eps)
        scale_factor = self.bn.weight / running_std
        weight_shape = [1] * len(self.weight.shape)
        weight_shape[0] = -1
        bias_shape = [1] * len(self.weight.shape)
        bias_shape[1] = -1
        scaled_weight = self.weight_fake_quant(self.weight * scale_factor.reshape(weight_shape))
        if self.bias is not None:
            zero_bias = torch.zeros_like(self.bias)
        else:
            zero_bias = torch.zeros(self.out_features, device=scaled_weight.device)
        linear_out = F.linear(input, scaled_weight, zero_bias)
        linear_out_orig = linear_out / scale_factor.reshape(bias_shape)
        if self.bias is not None:
            linear_out_orig = linear_out_orig + self.bias.reshape(bias_shape)
        bn_out = self.bn(linear_out_orig)
        return bn_out

    def train(self, mode=True):
        """
        Batchnorm's training behavior is using the self.training flag. Prevent
        changing it if BN is frozen. This makes sure that calling `model.train()`
        on a model with a frozen BN will behave properly.
        """
        self.training = mode
        if not self.freeze_bn:
            for module in self.children():
                module.train(mode)
        return self

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict

            Args: `mod' a float module, either produced by torch.ao.quantization
            utilities or directly from user
        """
        assert type(mod) == nni.LinearBn1d, 'qat.' + cls.__name__ + \
            '.from_float only works for ' + nni.LinearBn1d.__name__
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
        assert mod.qconfig, 'Input float module must have a valid config'
        qconfig = mod.qconfig
        linear, bn = mod[0], mod[1]
        qat_linearbn = cls(linear.in_features, linear.out_features, linear.bias is not None,
                           bn.eps, bn.momentum,
                           False, qconfig)
        qat_linearbn.weight = linear.weight
        qat_linearbn.bias = linear.bias
        qat_linearbn.bn.weight = bn.weight
        qat_linearbn.bn.bias = bn.bias
        qat_linearbn.bn.running_mean = bn.running_mean
        qat_linearbn.bn.running_var = bn.running_var
        qat_linearbn.bn.num_batches_tracked = bn.num_batches_tracked
        return qat_linearbn

    def to_float(self):
        linear = torch.nn.Linear(self.in_features, self.out_features)
        linear.weight, linear.bias = fuse_linear_bn_weights(
            self.weight,
            self.bias,
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.eps,
            self.bn.weight,
            self.bn.bias)
        return linear
