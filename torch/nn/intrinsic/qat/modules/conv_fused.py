import math
import torch
import torch.nn as nn
import torch.nn.intrinsic as nni
import torch.ao.nn.qat as nnqat
import torch.nn.functional as F
from torch.nn import init
from torch.nn.utils import fuse_conv_bn_weights
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn.parameter import Parameter
from typing import TypeVar

__all__ = ['ConvBn1d', 'ConvBnReLU1d', 'ConvReLU1d', 'ConvBn2d', 'ConvBnReLU2d', 'ConvReLU2d', 'ConvBn3d',
           'ConvBnReLU3d', 'ConvReLU3d', 'update_bn_stats', 'freeze_bn_stats']
_BN_CLASS_MAP = {
    1: nn.BatchNorm1d,
    2: nn.BatchNorm2d,
    3: nn.BatchNorm3d,
}


MOD = TypeVar('MOD', bound=nn.modules.conv._ConvNd)


class _ConvBnNd(nn.modules.conv._ConvNd, nni._FusedModule):

    _version = 2
    _FLOAT_MODULE = MOD

    def __init__(self,
                 # ConvNd args
                 in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups,
                 bias,
                 padding_mode,
                 # BatchNormNd args
                 # num_features: out_channels
                 eps=1e-05, momentum=0.1,
                 # affine: True
                 # track_running_stats: True
                 # Args for this module
                 freeze_bn=False,
                 qconfig=None,
                 dim=2):
        nn.modules.conv._ConvNd.__init__(self, in_channels, out_channels, kernel_size,
                                         stride, padding, dilation, transposed,
                                         output_padding, groups, False, padding_mode)
        assert qconfig, 'qconfig must be provided for QAT module'
        self.qconfig = qconfig
        self.freeze_bn = freeze_bn if self.training else True
        self.bn = _BN_CLASS_MAP[dim](out_channels, eps, momentum, True, True)
        self.weight_fake_quant = self.qconfig.weight()
        if bias:
            self.bias = Parameter(torch.empty(out_channels))
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
        super(_ConvBnNd, self).reset_parameters()

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
        running_std = torch.sqrt(self.bn.running_var + self.bn.eps)
        scale_factor = self.bn.weight / running_std
        weight_shape = [1] * len(self.weight.shape)
        weight_shape[0] = -1
        bias_shape = [1] * len(self.weight.shape)
        bias_shape[1] = -1
        scaled_weight = self.weight_fake_quant(self.weight * scale_factor.reshape(weight_shape))
        # using zero bias here since the bias for original conv
        # will be added later
        if self.bias is not None:
            zero_bias = torch.zeros_like(self.bias)
        else:
            zero_bias = torch.zeros(self.out_channels, device=scaled_weight.device)
        conv = self._conv_forward(input, scaled_weight, zero_bias)
        conv_orig = conv / scale_factor.reshape(bias_shape)
        if self.bias is not None:
            conv_orig = conv_orig + self.bias.reshape(bias_shape)
        conv = self.bn(conv_orig)
        return conv

    def extra_repr(self):
        # TODO(jerryzh): extend
        return super(_ConvBnNd, self).extra_repr()

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

    # ===== Serialization version history =====
    #
    # Version 1/None
    #   self
    #   |--- weight : Tensor
    #   |--- bias : Tensor
    #   |--- gamma : Tensor
    #   |--- beta : Tensor
    #   |--- running_mean : Tensor
    #   |--- running_var : Tensor
    #   |--- num_batches_tracked : Tensor
    #
    # Version 2
    #   self
    #   |--- weight : Tensor
    #   |--- bias : Tensor
    #   |--- bn : Module
    #        |--- weight : Tensor (moved from v1.self.gamma)
    #        |--- bias : Tensor (moved from v1.self.beta)
    #        |--- running_mean : Tensor (moved from v1.self.running_mean)
    #        |--- running_var : Tensor (moved from v1.self.running_var)
    #        |--- num_batches_tracked : Tensor (moved from v1.self.num_batches_tracked)
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)
        if version is None or version == 1:
            # BN related parameters and buffers were moved into the BN module for v2
            v2_to_v1_names = {
                'bn.weight': 'gamma',
                'bn.bias': 'beta',
                'bn.running_mean': 'running_mean',
                'bn.running_var': 'running_var',
                'bn.num_batches_tracked': 'num_batches_tracked',
            }
            for v2_name, v1_name in v2_to_v1_names.items():
                if prefix + v1_name in state_dict:
                    state_dict[prefix + v2_name] = state_dict[prefix + v1_name]
                    state_dict.pop(prefix + v1_name)
                elif prefix + v2_name in state_dict:
                    # there was a brief period where forward compatibility
                    # for this module was broken (between
                    # https://github.com/pytorch/pytorch/pull/38478
                    # and https://github.com/pytorch/pytorch/pull/38820)
                    # and modules emitted the v2 state_dict format while
                    # specifying that version == 1. This patches the forward
                    # compatibility issue by allowing the v2 style entries to
                    # be used.
                    pass
                elif strict:
                    missing_keys.append(prefix + v2_name)

        super(_ConvBnNd, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict

            Args: `mod` a float module, either produced by torch.ao.quantization utilities
            or directly from user
        """
        # The ignore is because _FLOAT_MODULE is a TypeVar here where the bound
        # has no __name__ (code is fine though)
        assert type(mod) == cls._FLOAT_MODULE, 'qat.' + cls.__name__ + '.from_float only works for ' + \
            cls._FLOAT_MODULE.__name__  # type: ignore[attr-defined]
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
        assert mod.qconfig, 'Input float module must have a valid qconfig'
        qconfig = mod.qconfig
        conv, bn = mod[0], mod[1]
        qat_convbn = cls(conv.in_channels, conv.out_channels, conv.kernel_size,
                         conv.stride, conv.padding, conv.dilation,
                         conv.groups, conv.bias is not None,
                         conv.padding_mode,
                         bn.eps, bn.momentum,
                         False,
                         qconfig)
        qat_convbn.weight = conv.weight
        qat_convbn.bias = conv.bias
        qat_convbn.bn.weight = bn.weight
        qat_convbn.bn.bias = bn.bias
        qat_convbn.bn.running_mean = bn.running_mean
        qat_convbn.bn.running_var = bn.running_var
        # mypy error: Cannot determine type of 'num_batches_tracked'
        qat_convbn.bn.num_batches_tracked = bn.num_batches_tracked  # type: ignore[has-type]
        return qat_convbn

    def to_float(self):
        cls = type(self)
        conv = cls._FLOAT_CONV_MODULE(  # type: ignore[attr-defined]
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.bias is not None,
            self.padding_mode)
        conv.weight = torch.nn.Parameter(self.weight.detach())
        if self.bias is not None:
            conv.bias = torch.nn.Parameter(self.bias.detach())

        if cls._FLOAT_BN_MODULE:  # type: ignore[attr-defined]
            # fuse bn into conv
            conv.weight, conv.bias = fuse_conv_bn_weights(
                conv.weight,
                conv.bias,
                self.bn.running_mean,
                self.bn.running_var,
                self.bn.eps,
                self.bn.weight,
                self.bn.bias
            )

        if cls._FLOAT_RELU_MODULE:  # type: ignore[attr-defined]
            modules = []
            modules.append(conv)
            relu = cls._FLOAT_RELU_MODULE()  # type: ignore[attr-defined]
            modules.append(relu)
            conv_relu = cls._FUSED_FLOAT_MODULE(*modules)  # type: ignore[attr-defined]
            conv_relu.train(self.training)
            return conv_relu
        else:
            conv.train(self.training)
            return conv

class ConvBn1d(_ConvBnNd, nn.Conv1d):
    r"""
    A ConvBn1d module is a module fused from Conv1d and BatchNorm1d,
    attached with FakeQuantize modules for weight,
    used in quantization aware training.

    We combined the interface of :class:`torch.nn.Conv1d` and
    :class:`torch.nn.BatchNorm1d`.

    Similar to :class:`torch.nn.Conv1d`, with FakeQuantize modules initialized
    to default.

    Attributes:
        freeze_bn:
        weight_fake_quant: fake quant module for weight

    """
    _FLOAT_BN_MODULE = nn.BatchNorm1d
    _FLOAT_RELU_MODULE = None
    _FLOAT_MODULE = nni.ConvBn1d
    _FLOAT_CONV_MODULE = nn.Conv1d

    def __init__(self,
                 # Conv1d args
                 in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=None,
                 padding_mode='zeros',
                 # BatchNorm1d args
                 # num_features: out_channels
                 eps=1e-05, momentum=0.1,
                 # affine: True
                 # track_running_stats: True
                 # Args for this module
                 freeze_bn=False,
                 qconfig=None):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        _ConvBnNd.__init__(self, in_channels, out_channels, kernel_size, stride,
                           padding, dilation, False, _single(0), groups, bias, padding_mode,
                           eps, momentum, freeze_bn, qconfig, dim=1)

class ConvBnReLU1d(ConvBn1d):
    r"""
    A ConvBnReLU1d module is a module fused from Conv1d, BatchNorm1d and ReLU,
    attached with FakeQuantize modules for weight,
    used in quantization aware training.

    We combined the interface of :class:`torch.nn.Conv1d` and
    :class:`torch.nn.BatchNorm1d` and :class:`torch.nn.ReLU`.

    Similar to `torch.nn.Conv1d`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight_fake_quant: fake quant module for weight

    """
    # base class defines _FLOAT_MODULE as "ConvBn1d"
    _FLOAT_MODULE = nni.ConvBnReLU1d  # type: ignore[assignment]
    _FLOAT_CONV_MODULE = nn.Conv1d
    _FLOAT_BN_MODULE = nn.BatchNorm1d
    _FLOAT_RELU_MODULE = nn.ReLU  # type: ignore[assignment]
    # module class after fusing bn into conv
    _FUSED_FLOAT_MODULE = nni.ConvReLU1d

    def __init__(self,
                 # Conv1d args
                 in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=None,
                 padding_mode='zeros',
                 # BatchNorm1d args
                 # num_features: out_channels
                 eps=1e-05, momentum=0.1,
                 # affine: True
                 # track_running_stats: True
                 # Args for this module
                 freeze_bn=False,
                 qconfig=None):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias,
                         padding_mode, eps, momentum,
                         freeze_bn,
                         qconfig)

    def forward(self, input):
        return F.relu(ConvBn1d._forward(self, input))

    @classmethod
    def from_float(cls, mod):
        return super(ConvBnReLU1d, cls).from_float(mod)

class ConvReLU1d(nnqat.Conv1d, nni._FusedModule):
    r"""A ConvReLU1d module is a fused module of Conv1d and ReLU, attached with
    FakeQuantize modules for weight for
    quantization aware training.

    We combined the interface of :class:`~torch.nn.Conv1d` and
    :class:`~torch.nn.BatchNorm1d`.

    Attributes:
        weight_fake_quant: fake quant module for weight

    """
    _FLOAT_MODULE = nni.ConvReLU1d
    _FLOAT_CONV_MODULE = nn.Conv1d
    _FLOAT_BN_MODULE = None
    _FLOAT_RELU_MODULE = nn.ReLU

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros',
                 qconfig=None):
        super(ConvReLU1d, self).__init__(in_channels, out_channels, kernel_size,
                                         stride=stride, padding=padding, dilation=dilation,
                                         groups=groups, bias=bias, padding_mode=padding_mode,
                                         qconfig=qconfig)
        assert qconfig, 'qconfig must be provided for QAT module'
        self.qconfig = qconfig
        self.weight_fake_quant = self.qconfig.weight()

    def forward(self, input):
        return F.relu(
            self._conv_forward(input, self.weight_fake_quant(self.weight), self.bias))

    @classmethod
    def from_float(cls, mod):
        return super(ConvReLU1d, cls).from_float(mod)

class ConvBn2d(_ConvBnNd, nn.Conv2d):
    r"""
    A ConvBn2d module is a module fused from Conv2d and BatchNorm2d,
    attached with FakeQuantize modules for weight,
    used in quantization aware training.

    We combined the interface of :class:`torch.nn.Conv2d` and
    :class:`torch.nn.BatchNorm2d`.

    Similar to :class:`torch.nn.Conv2d`, with FakeQuantize modules initialized
    to default.

    Attributes:
        freeze_bn:
        weight_fake_quant: fake quant module for weight

    """
    _FLOAT_MODULE = nni.ConvBn2d
    _FLOAT_CONV_MODULE = nn.Conv2d
    _FLOAT_BN_MODULE = nn.BatchNorm2d
    _FLOAT_RELU_MODULE = None

    def __init__(self,
                 # ConvNd args
                 in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=None,
                 padding_mode='zeros',
                 # BatchNorm2d args
                 # num_features: out_channels
                 eps=1e-05, momentum=0.1,
                 # affine: True
                 # track_running_stats: True
                 # Args for this module
                 freeze_bn=False,
                 qconfig=None):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        _ConvBnNd.__init__(self, in_channels, out_channels, kernel_size, stride,
                           padding, dilation, False, _pair(0), groups, bias, padding_mode,
                           eps, momentum, freeze_bn, qconfig, dim=2)

class ConvBnReLU2d(ConvBn2d):
    r"""
    A ConvBnReLU2d module is a module fused from Conv2d, BatchNorm2d and ReLU,
    attached with FakeQuantize modules for weight,
    used in quantization aware training.

    We combined the interface of :class:`torch.nn.Conv2d` and
    :class:`torch.nn.BatchNorm2d` and :class:`torch.nn.ReLU`.

    Similar to `torch.nn.Conv2d`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight_fake_quant: fake quant module for weight

    """
    # base class defines _FLOAT_MODULE as "ConvBn2d"
    _FLOAT_MODULE = nni.ConvBnReLU2d  # type: ignore[assignment]
    _FLOAT_CONV_MODULE = nn.Conv2d
    _FLOAT_BN_MODULE = nn.BatchNorm2d
    _FLOAT_RELU_MODULE = nn.ReLU  # type: ignore[assignment]
    # module class after fusing bn into conv
    _FUSED_FLOAT_MODULE = nni.ConvReLU2d

    def __init__(self,
                 # Conv2d args
                 in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=None,
                 padding_mode='zeros',
                 # BatchNorm2d args
                 # num_features: out_channels
                 eps=1e-05, momentum=0.1,
                 # affine: True
                 # track_running_stats: True
                 # Args for this module
                 freeze_bn=False,
                 qconfig=None):
        super(ConvBnReLU2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                           padding, dilation, groups, bias,
                                           padding_mode, eps, momentum,
                                           freeze_bn,
                                           qconfig)

    def forward(self, input):
        return F.relu(ConvBn2d._forward(self, input))

    @classmethod
    def from_float(cls, mod):
        return super(ConvBnReLU2d, cls).from_float(mod)

class ConvReLU2d(nnqat.Conv2d, nni._FusedModule):
    r"""A ConvReLU2d module is a fused module of Conv2d and ReLU, attached with
    FakeQuantize modules for weight for
    quantization aware training.

    We combined the interface of :class:`~torch.nn.Conv2d` and
    :class:`~torch.nn.BatchNorm2d`.

    Attributes:
        weight_fake_quant: fake quant module for weight

    """
    _FLOAT_MODULE = nni.ConvReLU2d
    _FLOAT_CONV_MODULE = nn.Conv2d
    _FLOAT_BN_MODULE = None
    _FLOAT_RELU_MODULE = nn.ReLU

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros',
                 qconfig=None):
        super(ConvReLU2d, self).__init__(in_channels, out_channels, kernel_size,
                                         stride=stride, padding=padding, dilation=dilation,
                                         groups=groups, bias=bias, padding_mode=padding_mode,
                                         qconfig=qconfig)
        assert qconfig, 'qconfig must be provided for QAT module'
        self.qconfig = qconfig
        self.weight_fake_quant = self.qconfig.weight()

    def forward(self, input):
        return F.relu(
            self._conv_forward(input, self.weight_fake_quant(self.weight), self.bias))

    @classmethod
    def from_float(cls, mod):
        return super(ConvReLU2d, cls).from_float(mod)

class ConvBn3d(_ConvBnNd, nn.Conv3d):
    r"""
    A ConvBn3d module is a module fused from Conv3d and BatchNorm3d,
    attached with FakeQuantize modules for weight,
    used in quantization aware training.

    We combined the interface of :class:`torch.nn.Conv3d` and
    :class:`torch.nn.BatchNorm3d`.

    Similar to :class:`torch.nn.Conv3d`, with FakeQuantize modules initialized
    to default.

    Attributes:
        freeze_bn:
        weight_fake_quant: fake quant module for weight

    """
    _FLOAT_MODULE = nni.ConvBn3d
    _FLOAT_CONV_MODULE = nn.Conv3d
    _FLOAT_BN_MODULE = nn.BatchNorm3d
    _FLOAT_RELU_MODULE = None

    def __init__(
        self,
        # ConvNd args
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=None,
        padding_mode="zeros",
        # BatchNorm3d args
        # num_features: out_channels
        eps=1e-05,
        momentum=0.1,
        # affine: True
        # track_running_stats: True
        # Args for this module
        freeze_bn=False,
        qconfig=None,
    ):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        _ConvBnNd.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            False,
            _triple(0),
            groups,
            bias,
            padding_mode,
            eps,
            momentum,
            freeze_bn,
            qconfig,
            dim=3,
        )

class ConvBnReLU3d(ConvBn3d):
    r"""
    A ConvBnReLU3d module is a module fused from Conv3d, BatchNorm3d and ReLU,
    attached with FakeQuantize modules for weight,
    used in quantization aware training.

    We combined the interface of :class:`torch.nn.Conv3d` and
    :class:`torch.nn.BatchNorm3d` and :class:`torch.nn.ReLU`.

    Similar to `torch.nn.Conv3d`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight_fake_quant: fake quant module for weight

    """
    _FLOAT_MODULE = nni.ConvBnReLU3d  # type: ignore[assignment]
    _FLOAT_CONV_MODULE = nn.Conv3d
    _FLOAT_BN_MODULE = nn.BatchNorm3d
    _FLOAT_RELU_MODULE = nn.ReLU  # type: ignore[assignment]
    # module class after fusing bn into conv
    _FUSED_FLOAT_MODULE = nni.ConvReLU3d

    def __init__(
        self,
        # Conv3d args
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=None,
        padding_mode="zeros",
        # BatchNorm3d args
        # num_features: out_channels
        eps=1e-05,
        momentum=0.1,
        # affine: True
        # track_running_stats: True
        # Args for this module
        freeze_bn=False,
        qconfig=None,
    ):
        super(ConvBnReLU3d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            eps,
            momentum,
            freeze_bn,
            qconfig,
        )

    def forward(self, input):
        return F.relu(ConvBn3d._forward(self, input))

    @classmethod
    def from_float(cls, mod):
        return super(ConvBnReLU3d, cls).from_float(mod)

class ConvReLU3d(nnqat.Conv3d, nni._FusedModule):
    r"""A ConvReLU3d module is a fused module of Conv3d and ReLU, attached with
    FakeQuantize modules for weight for
    quantization aware training.

    We combined the interface of :class:`~torch.nn.Conv3d` and
    :class:`~torch.nn.BatchNorm3d`.

    Attributes:
        weight_fake_quant: fake quant module for weight

    """
    _FLOAT_MODULE = nni.ConvReLU3d
    _FLOAT_CONV_MODULE = nn.Conv3d
    _FLOAT_BN_MODULE = None
    _FLOAT_RELU_MODULE = nn.ReLU

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        qconfig=None,
    ):
        super(ConvReLU3d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            qconfig=qconfig,
        )
        assert qconfig, "qconfig must be provided for QAT module"
        self.qconfig = qconfig
        self.weight_fake_quant = self.qconfig.weight()

    def forward(self, input):
        return F.relu(
            self._conv_forward(input, self.weight_fake_quant(self.weight), self.bias)
        )

    @classmethod
    def from_float(cls, mod):
        return super(ConvReLU3d, cls).from_float(mod)

def update_bn_stats(mod):
    if type(mod) in set(
        [ConvBnReLU1d, ConvBnReLU2d, ConvBnReLU3d, ConvBn1d, ConvBn2d, ConvBn3d]
    ):
        mod.update_bn_stats()

def freeze_bn_stats(mod):
    if type(mod) in set(
        [ConvBnReLU1d, ConvBnReLU2d, ConvBnReLU3d, ConvBn1d, ConvBn2d, ConvBn3d]
    ):
        mod.freeze_bn_stats()
