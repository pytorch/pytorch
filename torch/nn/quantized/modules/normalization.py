import torch
import torch.nn.quantized.functional

class LayerNorm(torch.nn.LayerNorm):
    r"""This is the quantized version of :class:`~torch.nn.LayerNorm`.

    Additional args:
        * **scale** - quantization scale of the output, type: double.
        * **zero_point** - quantization zero point of the output, type: long.

    """

    def __init__(self, normalized_shape, weight, bias, scale, zero_point, eps=1e-5,
                 elementwise_affine=True):
        super(LayerNorm, self).__init__(
            normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
        self.weight = weight
        self.bias = bias
        self.scale = scale
        self.zero_point = zero_point

    def forward(self, input):
        return torch.ops.quantized.layer_norm(
            input, self.normalized_shape, weight=self.weight, bias=self.bias,
            eps=self.eps, output_scale=self.scale, output_zero_point=self.zero_point)

    def _get_name(self):
        return 'QuantizedLayerNorm'

    @classmethod
    def from_float(cls, mod):
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        new_mod = cls(
            mod.normalized_shape, mod.weight, mod.bias, float(scale),
            int(zero_point), mod.eps, mod.elementwise_affine)
        return new_mod

class GroupNorm(torch.nn.GroupNorm):
    r"""This is the quantized version of :class:`~torch.nn.GroupNorm`.

    Additional args:
        * **scale** - quantization scale of the output, type: double.
        * **zero_point** - quantization zero point of the output, type: long.

    """
    __constants__ = ['num_groups', 'num_channels', 'eps', 'affine']

    def __init__(self, num_groups, num_channels, weight, bias, scale, zero_point, eps=1e-5, affine=True):
        super(GroupNorm, self).__init__(num_groups, num_channels, eps, affine)
        self.weight = weight
        self.bias = bias
        self.scale = scale
        self.zero_point = zero_point

    def forward(self, input):
        return torch.ops.quantized.group_norm(
            input, self.num_groups, self.weight, self.bias, self.eps, self.scale,
            self.zero_point)

    def _get_name(self):
        return 'QuantizedGroupNorm'

    @classmethod
    def from_float(cls, mod):
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        new_mod = cls(
            mod.num_groups, mod.num_channels, mod.weight, mod.bias, float(scale), int(zero_point),
            mod.eps, mod.affine)
        return new_mod

class InstanceNorm1d(torch.nn.InstanceNorm1d):
    r"""This is the quantized version of :class:`~torch.nn.InstanceNorm1d`.

    Additional args:
        * **scale** - quantization scale of the output, type: double.
        * **zero_point** - quantization zero point of the output, type: long.

    """
    def __init__(self, num_features, weight, bias, scale, zero_point,
                 eps=1e-5, momentum=0.1, affine=False,
                 track_running_stats=False):
        super(InstanceNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.weight = weight
        self.bias = bias
        self.scale = scale
        self.zero_point = zero_point

    def forward(self, input):
        return torch.ops.quantized.instance_norm(
            input, self.weight, self.bias, self.eps, self.scale,
            self.zero_point)

    def _get_name(self):
        return 'QuantizedInstanceNorm1d'

    @classmethod
    def from_float(cls, mod):
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        new_mod = cls(
            mod.num_features, mod.weight, mod.bias, float(scale), int(zero_point),
            mod.eps, mod.affine)
        return new_mod

class InstanceNorm2d(torch.nn.InstanceNorm2d):
    r"""This is the quantized version of :class:`~torch.nn.InstanceNorm2d`.

    Additional args:
        * **scale** - quantization scale of the output, type: double.
        * **zero_point** - quantization zero point of the output, type: long.

    """
    def __init__(self, num_features, weight, bias, scale, zero_point,
                 eps=1e-5, momentum=0.1, affine=False,
                 track_running_stats=False):
        super(InstanceNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.weight = weight
        self.bias = bias
        self.scale = scale
        self.zero_point = zero_point

    def forward(self, input):
        return torch.ops.quantized.instance_norm(
            input, self.weight, self.bias, self.eps, self.scale,
            self.zero_point)

    def _get_name(self):
        return 'QuantizedInstanceNorm2d'

    @classmethod
    def from_float(cls, mod):
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        new_mod = cls(
            mod.num_features, mod.weight, mod.bias, float(scale), int(zero_point),
            mod.eps, mod.affine)
        return new_mod

class InstanceNorm3d(torch.nn.InstanceNorm3d):
    r"""This is the quantized version of :class:`~torch.nn.InstanceNorm3d`.

    Additional args:
        * **scale** - quantization scale of the output, type: double.
        * **zero_point** - quantization zero point of the output, type: long.

    """
    def __init__(self, num_features, weight, bias, scale, zero_point,
                 eps=1e-5, momentum=0.1, affine=False,
                 track_running_stats=False):
        super(InstanceNorm3d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.weight = weight
        self.bias = bias
        self.scale = scale
        self.zero_point = zero_point

    def forward(self, input):
        return torch.ops.quantized.instance_norm(
            input, self.weight, self.bias, self.eps, self.scale,
            self.zero_point)

    def _get_name(self):
        return 'QuantizedInstanceNorm3d'

    @classmethod
    def from_float(cls, mod):
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        new_mod = cls(
            mod.num_features, mod.weight, mod.bias, float(scale), int(zero_point),
            mod.eps, mod.affine)
        return new_mod
