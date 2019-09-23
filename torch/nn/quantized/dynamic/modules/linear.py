from __future__ import absolute_import, division, print_function, unicode_literals
import torch
from ....modules.linear import Linear as NNLinear
import torch.nn.quantized as nnq

class Linear(nnq.Linear):
    r"""
    A dynamic quantized linear module with quantized tensor as inputs and outputs.
    We adopt the same interface as `torch.nn.Linear`, please see
    https://pytorch.org/docs/stable/nn.html#torch.nn.Linear for documentation.

    Similar to :class:`torch.nn.Linear`, attributes will be randomly
    initialized at module creation time and will be overwritten later

    Attributes:
        weight (Tensor): the non-learnable quantized weights of the module which are of
                         shape :math:`(\text{out\_features}, \text{in\_features})`.
        bias (Tensor): the non-learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized to zero.
        scale: `scale` parameter of weight Quantized Tensor, type: double
        zero_point: `zero_point` parameter for weight Quantized Tensor, type: long

    Examples::

        >>> m = nn.quantized.dynamic.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    def __init__(self, in_features, out_features, bias_=True):
        super(Linear, self).__init__(in_features, out_features, bias_)
        # We don't muck around with buffers or attributes or anything here
        # to keep the module simple. *everything* is simply a Python attribute.
        # Serialization logic is explicitly handled in the below serialization and
        # deserialization modules

    def forward(self, x):
        # Note that we can handle self.bias == None case.
        Y = torch.ops.quantized.linear_dynamic(
            x, self._packed_params)
        return Y.to(x.dtype)

    @classmethod
    def from_float(cls, mod):
        r"""Create a dynamic quantized module from a float module or qparams_dict

        Args:
            mod (Module): a float module, either produced by torch.quantization
                          utilities or provided by the user
        """
        assert type(mod) == NNLinear, 'nn.quantized.dynamic.Linear.from_float only works for nn.Linear'
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
        if mod.qconfig is not None and mod.qconfig.weight() is not None:
            weight_observer = mod.qconfig.weight()
        else:
            # We have the circular import issues if we import the qconfig in the beginning of this file:
            # https://github.com/pytorch/pytorch/pull/24231. The current workaround is to postpone the
            # import until we need it.
            from torch.quantization.QConfig import default_dynamic_qconfig
            weight_observer = default_dynamic_qconfig.weight()
        assert weight_observer.dtype == torch.qint8, 'Weight observer must have dtype torch.qint8'
        weight_observer(mod.weight)
        wt_scale, wt_zp = weight_observer.calculate_qparams()
        qweight = torch.quantize_per_tensor(mod.weight.float(), float(wt_scale), int(wt_zp), torch.qint8)
        qlinear = Linear(mod.in_features, mod.out_features)
        qlinear.set_weight_bias(qweight, mod.bias)
        return qlinear
