from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn.quantized.functional

class ReLU(torch.nn.ReLU):
    r"""Applies quantized rectified linear unit function element-wise:

    :math:`\text{ReLU}(x)= \max(x_0, x)`, where :math:`x_0` is the zero point.

    Please see https://pytorch.org/docs/stable/nn.html#torch.nn.ReLU
    for more documentation on ReLU.

    Args:
        inplace: (Currently not supported) can optionally do the operation in-place.

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    Examples::

        >>> m = nn.quantized.ReLU()
        >>> input = torch.randn(2)
        >>> input = torch.quantize_per_tensor(input, 1.0, 0, dtype=torch.qint32)
        >>> output = m(input)
    """
    def __init__(self, inplace=False):
        super(ReLU, self).__init__(inplace)
        self.inplace = inplace

    def forward(self, input):
        return torch.nn.quantized.functional.relu(input, inplace=self.inplace)

    def _get_name(self):
        return 'QuantizedReLU'

    @staticmethod
    def from_float(mod):
        return ReLU(mod.inplace)


class ReLU6(torch.nn.ReLU):
    r"""Applies the element-wise function:

    :math:`\text{ReLU6}(x) = \min(\max(x_0, x), q(6))`, where :math:`x_0` is the
    zero_point, and :math:`q(6)` is the quantized representation of number 6.

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    .. image:: scripts/activation_images/ReLU6.png

    Examples::

        >>> m = nn.quantized.ReLU6()
        >>> input = torch.randn(2)
        >>> input = torch.quantize_per_tensor(input, 1.0, 0, dtype=torch.qint32)
        >>> output = m(input)
    """
    def __init__(self, inplace=False):
        super(ReLU6, self).__init__(inplace)
        self.inplace = inplace

    def forward(self, input):
        return torch.ops.quantized.relu6(input, self.inplace)

    def _get_name(self):
        return 'QuantizedReLU6'

    @staticmethod
    def from_float(mod):
        return ReLU6(mod.inplace)

class GELU(torch.nn.ReLU):
    r"""Applies quantized Gaussian Error Linear Units function:

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    .. image:: scripts/activation_images/GELU.png


    Examples::

        >>> m = nn.quantized.GELU()
        >>> input = torch.randn(2)
        >>> input = torch.quantize_per_tensor(input, 1.0, 0, dtype=torch.qint32)
        >>> output = m(input)
    """
    def __init__(self, inplace=False):
        super(GELU, self).__init__(inplace)
        self.scale = 1.0
        self.zero_point = 0 
        self.register_buffer("gelu_lut", torch.zeros([256]))
        self.lut_validate = False
    def forward(self, input):
        if not self.lut_validate:
           act_scale = input.q_scale()
           act_zp = input.q_zero_point()
           for i in range(0, 256):
             self.gelu_lut[i] = torch.nn.functional.gelu(torch.tensor((i - act_zp)) * act_scale)
           #self.int2gelu = torch.quantize_per_tensor(self.int2gelu, self.scale, self.zero_point, dtype=torch.quint8)
           self.lut_validate = True
        return torch.quantized_elmentwise_helper(input, self.gelu_lut, self.scale, self.zero_point, True) 

    # ===== Serialization methods =====
    # The special consideration here is that we have to unpack the weights into their
    # regular QTensor form for serialization. Packed weights should not live
    # outside the process in which they were created, rather they should be derived
    # from the QTensor weight.
    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super(GELU, self)._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'scale'] = torch.tensor(self.scale)
        destination[prefix + 'zero_point'] = torch.tensor(self.zero_point)

    # ===== Deserialization methods =====
    # Counterpart to the serialization methods, we must pack the serialized QTensor
    # weight into its packed format for use by the FBGEMM ops.
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        self.scale = float(state_dict[prefix + 'scale'])
        state_dict.pop(prefix + 'scale')

        self.zero_point = int(state_dict[prefix + 'zero_point'])
        state_dict.pop(prefix + 'zero_point')

        super(GELU, self)._load_from_state_dict(state_dict, prefix, local_metadata, False,
                                                  missing_keys, unexpected_keys, error_msgs)

    

    def _get_name(self):
        return 'QuantizedGELU'
    
    def extra_repr(self):
        return 'scale={}, zero_point={}'.format(
                self.scale, self.zero_point)

    @staticmethod
    def from_float(mod):
        r"""Create a quantized module from a float module or qparams_dict

        Args:
            mod (Module): a float module, either produced by torch.quantization
                          utilities or provided by the user
        """
        activation_post_process = mod.activation_post_process
        act_scale, act_zp = activation_post_process.calculate_qparams()
        qgelu = GELU()
        qgelu.scale = float(act_scale)
        qgelu.zero_point = int(act_zp)
        return qgelu
