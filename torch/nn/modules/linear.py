import math
import operator
from collections import Iterable
from functools import reduce

import torch
from torch.nn.parameter import Parameter
from .. import functional as F
from .. import init
from .module import Module
from ..._jit_internal import weak_module, weak_script_method


@weak_module
class Identity(Module):
    r"""A placeholder identity operator that is argument-insensitive.

    Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)

    Examples::

        >>> m = nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 20])

    """
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    @weak_script_method
    def forward(self, input):
        return input


@weak_module
class Linear(Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['bias']

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        sizes = []
        sizes.append(out_features) \
            if isinstance(out_features, int) else sizes.extend(out_features)
        sizes.append(in_features) \
            if isinstance(in_features, int) else sizes.extend(in_features)

        self.in_prod = self._sizes_prod(in_features)
        self.out_prod = self._sizes_prod(out_features)

        self.weight = Parameter(torch.empty(sizes))
        if bias:
            self.bias = Parameter(torch.empty(self.out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def _sizes_prod(self, features):
        assert isinstance(features, (int, Iterable)), (
            "nn.Linear in_features and out_features must be either an int or "
            "an Iterable, but got {}."
        ).format(type(features))

        return features if isinstance(features, int) else \
            reduce(operator.mul, features, 1) if features else 1

    def reset_parameters(self):
        weight, bias = self._reshape_params()
        init.kaiming_uniform_(weight, a=math.sqrt(5))
        if bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(bias, -bound, bound)

    def _reshape_params(self):
        if self.in_prod != self.in_features or self.out_prod != self.out_features:
            weight_view = self.weight.view(self.out_prod, self.in_prod)
            bias_view = self.bias.view(self.out_prod)
            return weight_view, bias_view
        else:
            return self.weight, self.bias

    @weak_script_method
    def forward(self, input):
        # In order to keep parameters() show correct dimensions, weight
        # and bias need to be reshaped in every iteration.
        weight, bias = self._reshape_params()

        if self.in_prod != self.in_features:
            # reshape input
            view_sizes = input.size()[:-len(self.in_features)] \
                if self.in_features else input.size()
            view_sizes = view_sizes.__add__(torch.Size([self.in_prod]))
            input = input.view(view_sizes)

        output = F.linear(input, weight, bias)

        if self.out_prod != self.out_features:
            # reshape output
            view_sizes = output.size()[:-1] \
                               .__add__(torch.Size(self.out_features))
            output = output.view(view_sizes)

        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


@weak_module
class Bilinear(Module):
    r"""Applies a bilinear transformation to the incoming data:
    :math:`y = x_1 A x_2 + b`

    Args:
        in1_features: size of each first input sample
        in2_features: size of each second input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input1: :math:`(N, *, H_{in1})` where :math:`H_{in1}=\text{in1\_features}` and
          :math:`*` means any number of additional dimensions. All but the last dimension
          of the inputs should be the same.
        - Input2: :math:`(N, *, H_{in2})` where :math:`H_{in2}=\text{in2\_features}`.
        - Output: :math:`(N, *, H_{out})` where :math:`H_{out}=\text{out\_features}`
          and all but the last dimension are the same shape as the input.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in1\_features}, \text{in2\_features})`.
            The values are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in1\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
                :math:`k = \frac{1}{\text{in1\_features}}`

    Examples::

        >>> m = nn.Bilinear(20, 30, 40)
        >>> input1 = torch.randn(128, 20)
        >>> input2 = torch.randn(128, 30)
        >>> output = m(input1, input2)
        >>> print(output.size())
        torch.Size([128, 40])
    """
    __constants__ = ['in1_features', 'in2_features', 'out_features', 'bias']

    def __init__(self, in1_features, in2_features, out_features, bias=True):
        super(Bilinear, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in1_features, in2_features))

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.weight.size(1))
        init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            init.uniform_(self.bias, -bound, bound)

    @weak_script_method
    def forward(self, input1, input2):
        return F.bilinear(input1, input2, self.weight, self.bias)

    def extra_repr(self):
        return 'in1_features={}, in2_features={}, out_features={}, bias={}'.format(
            self.in1_features, self.in2_features, self.out_features, self.bias is not None
        )

# TODO: PartialLinear - maybe in sparse?
