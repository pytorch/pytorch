import math
import torch
import torch.nn as nn
import torch.nn.intrinsic as nni
import torch.nn.qat as nnqat
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter


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
    _FLOAT_BN_MODULE = nn.BatchNorm1d
    _FLOAT_RELU_MODULE = None
    _FLOAT_MODULE = nni.LinearBn1d
    _FLOAT_LINEAR_MODULE = nn.Linear

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
        # note: below is actully for conv, not BN
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

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

    def _forward(self, input):
        assert self.bn.running_var is not None
        linear = F.linear(input, self.weight, self.bias)
        output = self.bn(linear)
        return output

    def extra_repr(self):
        return super(LinearBn1d, self).extra_repr()

    def forward(self, input):
        return self._forward(input)

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
        # The ignore is because _FLOAT_MODULE is a TypeVar here where the bound
        # has no __name__ (code is fine though)
        assert type(mod) == cls._FLOAT_MODULE, 'qat.' + cls.__name__ + \
            '.from_float only works for ' + cls._FLOAT_MODULE.__name
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
        assert mod.qconfig, 'Input float module must have a valid config'
        qconfig = mod.qconfig
        linear, bn = mod[0], mod[1]
        qat_linearbn = cls(linear.in_features, linear.out_features, linear.bias,
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
        modules = []
        cls = type(self)
        lienar = cls._FLOAT_LINEAR_MODULE(
            self.in_features,
            self.out_features,
            self.bias)
        linear.weight = Parameter(self.weight.detach())
        if self.bias is not None:
            lienar.bias = Parameter(self.bias.detach())
        modules.append(linear)

        if cls._FLOAT_BN_MODULE:
            bn = cls._FLOAT_BN_MODULE(
                self.bn.num_features,
                self.bn.eps,
                self.bn.momentum,
                self.bn.affine,
                self.bn.track_running_stats)
            bn.weight = Paramter(self.bn.weight.detach())
            if self.bn.affine:
                bn.bias = Parameter(self.bn.bias.detach())
            modueles.append(bn)

        if cls._FLOAT_RELU_MODULE:
            relu = cls._FLOAT_RELU_MODULE()
            modules.append(relu)

        result = cls._FLOAT_MODULE(*modules)
        result.train(self.training)
        return result
