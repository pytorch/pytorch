from __future__ import absolute_import, division, print_function, unicode_literals
from torch.nn import Conv2d as NNConv2d
from torch.nn._intrinsic import ConvBn2d as NNConvBn2d
from torch.nn._intrinsic import ConvBnReLU2d as NNConvBnReLU2d
from torch.quantization.QConfig import default_qat_qconfig
import torch.nn.functional as F

class ConvBn2d(NNConv2d):
    r"""
    A ConvBn2d module is a module fused from Conv2d and BatchNorm2d,
    attached with FakeQuantize modules for both output activation and weight,
    used in quantization aware training.

    We combined the interface of :class:`torch.nn.Conv2d` and
    :class:`torch.nn.BatchNorm2d`.

    Implementation details: https://arxiv.org/pdf/1806.08342.pdf

    Similar to :class:`torch.nn.Conv2d`, with FakeQuantize modules initialized
    to default.

    Attributes:
        freeze_bn:
        observer: fake quant module for output activation, it's called observer
            to align with post training flow
        weight_fake_quant: fake quant module for weight

    """
    __FLOAT_MODULE__ = NNConvBn2d

    def __init__(self,
                 # conv2d args
                 in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros',
                 # bn args
                 # num_features: enforce this matches out_channels before fusion
                 eps=1e-05, momentum=0.1,
                 # affine: enforce this is True before fusion?
                 # tracking_running_stats: enforce this is True before fusion
                 # args for this module
                 freeze_bn=False,
                 activation_fake_quant=default_qat_qconfig.activation,
                 weight_fake_quant=default_qat_qconfig.weight):
        super(ConvBn2d, self).__init__(in_channels, out_channels, kernel_size,
                                       stride=stride, padding=padding, dilation=dilation,
                                       groups=groups, bias=bias, padding_mode=padding_mode)
        self.eps = eps
        self.momentum = momentum
        self.freeze_bn = freeze_bn
        self.num_features = out_channels
        self.gamma = Parameter(torch.Tensor(out_channels))
        self.beta = Parameter(torch.Tensor(out_channels))
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.register_buffer('running_var', torch.ones(out_channels))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        self.observer = activation_fake_quant()
        self.weight_fake_quant = weight_fake_quant()

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.uniform_(self.gamma)
            init.zeros_(self.beta)

    def _forward(self, input):
        # exponential_average_factor is self.momentum set to
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        scaled_weight = self.gamma * self.weight / torch.sqrt(self.running_var)
        conv = conv2d_forward(input, self.padding_mode, self.padding,
                              self.weight_fake_quant(scaled_weight), self.bias,
                              self.stride, self.dilation, self.groups)
        batch_mean = torch.mean(conv, dim=[0, 2, 3])
        n = input.numel() / self.num_features
        batch_var = torch.var(conv, dim=[0, 2, 3]) * (n / (n - 1))
        self.running_mean = exponential_average_factor * batch_mean + (1 - exponential_average_factor) * self.running_mean
        self.running_var = exponential_average_factor * batch_var + (1 - exponential_average_factor) * self.running_var

        if not self.freeze_bn:
            conv *= torch.sqrt(self.running_var) / torch.sqrt(batch_var)
            conv += self.beta - self.gamma * (batch_mean / torch.sqrt(batch_var))
        else:
            conv += self.beta - self.gamma * self.running_mean / torch.sqrt(self.running_var)

        return conv

    def forward(self, input):
        return self.observer(self._forward(input))

    @classmethod
    def from_float(cls, mod, qconfig):
        r"""Create a qat module from a float module or qparams_dict

            Args: `mod` a float module, either produced by torch.quantization utilities
            or directly from user
        """
        assert type(mod) == cls.__FLOAT_MODULE__, ' qat.' + cls.__name__ + '.from_float only works for ' + \
            cls.__FLOAT_MODULE__.__name__
        if not qconfig:
            assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
            assert hasattr(mod, 'observer'), 'Input float module must have observer attached'
            qconfig = mod.qconfig
        qat_convbn = cls(mod.in_channels, mod.out_channels, mod.kernel_size,
                         stride=mod.stride, padding=mod.padding, dilation=mod.dilation,
                         groups=mod.groups, bias=mod.bias is not None,
                         padding_mode=mod.padding_mode,
                         eps=mod.eps, momentum=mod.momentum,
                         freeze_bn=False,
                         activation_fake_quant=qconfig.activation,
                         weight_fake_quant=qconfig.weight)

        qat_convbn.weight = mod.weight
        qat_convbn.bias = mod.bias
        qat_convbn.gamma = mod.gamma
        qat_convbn.beta = mod.beta
        qat_convbn.running_mean = mod.running_mean
        qat_convbn.running_var = mod.running_var
        qat_convbn.num_batches_tracked = mod.num_batches_tracked
        return qat_convbn

class ConvBnReLU2d(ConvBn2d):
        r"""
        A ConvBn2d module is a module fused from Conv2d, BatchNorm2d and ReLU,
        attached with FakeQuantize modules for both output activation and weight,
        used in quantization aware training.

        We combined the interface of :class:`torch.nn.Conv2d` and
        :class:`torch.nn.BatchNorm2d` and :class:`torch.nn.ReLU`.

        Implementation details: https://arxiv.org/pdf/1806.08342.pdf

        Similar to `torch.nn.Conv2d`, with FakeQuantize modules initialized to
        default.

        Attributes:
            observer: fake quant module for output activation, it's called observer
                to align with post training flow
            weight_fake_quant: fake quant module for weight

        """
        __FLOAT_MODULE__ = NNConvBnReLU2d

        def __init__(self,
                     # conv2d args
                     in_channels, out_channels, kernel_size, stride=1,
                     padding=0, dilation=1, groups=1,
                     bias=True, padding_mode='zeros',
                     # bn args
                     # num_features: enforce this matches out_channels before fusion
                     eps=1e-05, momentum=0.1,
                     # affine: enforce this is True before fusion?
                     # tracking_running_stats: enforce this is True before fusion
                     # args for this module
                     freeze_bn=False,
                     activation_fake_quant=default_qat_qconfig.activation,
                     weight_fake_quant=default_qat_qconfig.weight):
            super(ConvBnReLU2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                               padding=padding, dilation=dilation, groups=groups, bias=bias,
                                               padding_mode=padding_mode, eps=eps, momentum=momentum,
                                               freeze_bn=freeze_bn,
                                               activation_fake_quant=activation_fake_quant,
                                               weight_fake_quant=weight_fake_quant)

        def forward(self, input):
            return self.observer(F.relu(super(ConvBnReLU2d, self)._forward(input)))
