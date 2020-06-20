from __future__ import absolute_import, division, print_function, unicode_literals
import torch.nn as nn

class GroupNorm(nn.GroupNorm):
    r"""
    A GroupNorm module attached with FakeQuantize modules for output
    activation, used for quantization aware training.

    Similar to `torch.nn.GroupNorm`, with FakeQuantize modules initialized to
    default.

    Attributes:
        activation_post_process: fake quant module for output activation
        weight: fake quant module for weight
    """
    _FLOAT_MODULE = nn.GroupNorm

    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True,
                 qconfig=None):
        super(GroupNorm, self).__init__(num_groups, num_channels, eps, affine)
        assert qconfig, 'qconfig must be provided for QAT module'
        self.qconfig = qconfig
        self.activation_post_process = qconfig.activation()

    def forward(self, input):
        return self.activation_post_process(
            super(GroupNorm, self).forward(input))

    @classmethod
    def from_float(cls, mod, qconfig=None):
        r"""Create a qat module from a float module or qparams_dict

            Args: `mod` a float module, either produced by
            torch.quantization utilities or directly from user
        """
        assert type(mod) == cls._FLOAT_MODULE, ' qat.' + cls.__name__ + '.from_float only works for ' + \
            cls._FLOAT_MODULE.__name__
        if not qconfig:
            assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
            assert mod.qconfig, 'Input float module must have a valid qconfig'

        qconfig = mod.qconfig
        qat_groupnorm = cls(
            mod.num_groups, mod.num_channels, mod.eps, mod.affine,
            qconfig=qconfig)
        return qat_groupnorm

class InstanceNorm1d(nn.InstanceNorm1d):
    r"""
    A InstanceNorm1d module attached with FakeQuantize modules for output
    activation, used for quantization aware training.

    Similar to `torch.nn.InstanceNorm1d`, with FakeQuantize modules initialized to
    default.

    Attributes:
        activation_post_process: fake quant module for output activation
        weight: fake quant module for weight
    """
    _FLOAT_MODULE = nn.InstanceNorm1d

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=False,
                 track_running_stats=False, qconfig=None):
        super(InstanceNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        assert qconfig, 'qconfig must be provided for QAT module'
        self.qconfig = qconfig
        self.activation_post_process = qconfig.activation()

    def forward(self, input):
        return self.activation_post_process(
            super(InstanceNorm1d, self).forward(input))

    @classmethod
    def from_float(cls, mod, qconfig=None):
        r"""Create a qat module from a float module or qparams_dict

            Args: `mod` a float module, either produced by
            torch.quantization utilities or directly from user
        """
        assert type(mod) == cls._FLOAT_MODULE, ' qat.' + cls.__name__ + '.from_float only works for ' + \
            cls._FLOAT_MODULE.__name__
        if not qconfig:
            assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
            assert mod.qconfig, 'Input float module must have a valid qconfig'

        qconfig = mod.qconfig
        qat_instancenorm = cls(
            mod.num_features, mod.eps, mod.momentum, mod.affine,
            mod.track_running_stats, qconfig=qconfig)
        return qat_instancenorm

class InstanceNorm2d(nn.InstanceNorm2d):
    r"""
    A InstanceNorm2d module attached with FakeQuantize modules for output
    activation, used for quantization aware training.

    Similar to `torch.nn.InstanceNorm2d`, with FakeQuantize modules initialized to
    default.

    Attributes:
        activation_post_process: fake quant module for output activation
        weight: fake quant module for weight
    """
    _FLOAT_MODULE = nn.InstanceNorm2d

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=False,
                 track_running_stats=False, qconfig=None):
        super(InstanceNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        assert qconfig, 'qconfig must be provided for QAT module'
        self.qconfig = qconfig
        self.activation_post_process = qconfig.activation()

    def forward(self, input):
        return self.activation_post_process(
            super(InstanceNorm2d, self).forward(input))

    @classmethod
    def from_float(cls, mod, qconfig=None):
        r"""Create a qat module from a float module or qparams_dict

            Args: `mod` a float module, either produced by
            torch.quantization utilities or directly from user
        """
        assert type(mod) == cls._FLOAT_MODULE, ' qat.' + cls.__name__ + '.from_float only works for ' + \
            cls._FLOAT_MODULE.__name__
        if not qconfig:
            assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
            assert mod.qconfig, 'Input float module must have a valid qconfig'

        qconfig = mod.qconfig
        qat_instancenorm = cls(
            mod.num_features, mod.eps, mod.momentum, mod.affine,
            mod.track_running_stats, qconfig=qconfig)
        return qat_instancenorm

class InstanceNorm3d(nn.InstanceNorm3d):
    r"""
    A InstanceNorm3d module attached with FakeQuantize modules for output
    activation, used for quantization aware training.

    Similar to `torch.nn.InstanceNorm3d`, with FakeQuantize modules initialized to
    default.

    Attributes:
        activation_post_process: fake quant module for output activation
        weight: fake quant module for weight
    """
    _FLOAT_MODULE = nn.InstanceNorm3d

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=False,
                 track_running_stats=False, qconfig=None):
        super(InstanceNorm3d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        assert qconfig, 'qconfig must be provided for QAT module'
        self.qconfig = qconfig
        self.activation_post_process = qconfig.activation()

    def forward(self, input):
        return self.activation_post_process(
            super(InstanceNorm3d, self).forward(input))

    @classmethod
    def from_float(cls, mod, qconfig=None):
        r"""Create a qat module from a float module or qparams_dict

            Args: `mod` a float module, either produced by
            torch.quantization utilities or directly from user
        """
        assert type(mod) == cls._FLOAT_MODULE, ' qat.' + cls.__name__ + '.from_float only works for ' + \
            cls._FLOAT_MODULE.__name__
        if not qconfig:
            assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
            assert mod.qconfig, 'Input float module must have a valid qconfig'

        qconfig = mod.qconfig
        qat_instancenorm = cls(
            mod.num_features, mod.eps, mod.momentum, mod.affine,
            mod.track_running_stats, qconfig=qconfig)
        return qat_instancenorm

class LayerNorm(nn.LayerNorm):
    r"""
    A LayerNorm module attached with FakeQuantize modules for output
    activation, used for quantization aware training.

    Similar to :class:`torch.nn.LayerNorm`, with FakeQuantize modules initialized to
    default.

    Attributes:
        activation_post_process: fake quant module for output activation
        weight: fake quant module for weight
    """
    _FLOAT_MODULE = nn.LayerNorm

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True,
                 qconfig=None):
        super(LayerNorm, self).__init__(normalized_shape, eps, elementwise_affine)
        assert qconfig, 'qconfig must be provided for QAT module'
        self.qconfig = qconfig
        self.activation_post_process = qconfig.activation()

    def forward(self, input):
        return self.activation_post_process(
            super(LayerNorm, self).forward(input))

    @classmethod
    def from_float(cls, mod, qconfig=None):
        r"""Create a qat module from a float module or qparams_dict

            Args: `mod` a float module, either produced by
            torch.quantization utilities or directly from user
        """
        assert type(mod) == cls._FLOAT_MODULE, ' qat.' + cls.__name__ + '.from_float only works for ' + \
            cls._FLOAT_MODULE.__name__
        if not qconfig:
            assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
            assert mod.qconfig, 'Input float module must have a valid qconfig'

        qconfig = mod.qconfig
        qat_layernorm = cls(
            mod.normalized_shape, mod.eps, mod.elementwise_affine,
            qconfig=qconfig)
        return qat_layernorm
