from __future__ import absolute_import, division, print_function, unicode_literals
import torch
from ...modules.module import Module
from ...._jit_internal import weak_module

@weak_module
class Linear(Module):
    r"""
    A module that wraps the quantized fbgemm linear operator function
    We adopt the same interface as `torch.nn.Linear`, please see https://pytorch.org/docs/stable/nn.html#torch.nn.Linear
    for documentation.

    Similar to `torch.nn.Linear`, attributes will be randomly initialized at
        module creation time and will be overwritten later

    Attributes:
        _packed_weight: the non-learnable packed weights of the
            module which are of shape :math:`(\text{out\_features}, \text{in\_features})`.
        bias:   the non-learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized to zero.
        output_scale: `scale` parameter of output Quantized Tensor
        output_zero_point: `zero_point` parameter for output Quantized Tensor

    Examples::

        >>> m = nn.quantized.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        weight = torch.randn(out_features, in_features, dtype=torch.float32)
        weight = torch.quantize_linear(weight, 1.0, 0, torch.qint8)
        _packed_weight = torch.ops.quantized.fbgemm_linear_prepack(weight)

        output_scale = 1.0
        self.register_buffer('output_scale', torch.Tensor([output_scale]))
        output_zero_point = 0
        self.register_buffer('output_zero_point', torch.Tensor([output_zero_point]))
        self.register_buffer('_packed_weight', _packed_weight)
        _bias = torch.quantize_linear(torch.zeros(out_features).float(), output_scale,
                                      output_zero_point, torch.qint32)
        self.register_buffer('bias', _bias)


    def forward(self, x):
        Y_q = torch.ops.quantized.fbgemm_linear(x, self._packed_weight, self.bias, self.output_scale, self.output_zero_point)
        return Y_q

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        r"""Adaptation of state_dict method to handle packing of weights
        Returns a dictionary containing a whole state of the module.

        Both parameters and persistent buffers (e.g. running averages) are
        included. Keys are corresponding parameter and buffer names.

        Returns:
            dict:
                a dictionary containing a whole state of the module

        Example::

            >>> module.state_dict().keys()
            ['bias', 'weight']

        """
        raw_dict = super().state_dict(destination, prefix, keep_vars)
        weight = torch.ops.quantized.fbgemm_linear_unpack(raw_dict[prefix + '_packed_weight'])
        raw_dict[prefix + 'weight'] = weight
        raw_dict.pop(prefix + '_packed_weight')
        return raw_dict

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        r"""
        Adaptation of default _load_from_state_dict method
        to handle packing and unpacking of weights
        Copies parameters and buffers from :attr:`state_dict` into only
        this module, but not its descendants. This is called on every submodule
        in :meth:`~torch.nn.Module.load_state_dict`. Metadata saved for this
        module in input :attr:`state_dict` is provided as :attr:`local_metadata`.
        For state dicts without metadata, :attr:`local_metadata` is empty.
        Subclasses can achieve class-specific backward compatible loading using
        the version number at `local_metadata.get("version", None)`.

        .. note::
            :attr:`state_dict` is not the same object as the input
            :attr:`state_dict` to :meth:`~torch.nn.Module.load_state_dict`. So
            it can be modified.

        Arguments:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            prefix (str): the prefix for parameters and buffers used in this
                module
            local_metadata (dict): a dict containing the metadata for this module.
                See
            strict (bool): whether to strictly enforce that the keys in
                :attr:`state_dict` with :attr:`prefix` match the names of
                parameters and buffers in this module
            missing_keys (list of str): if ``strict=True``, add missing keys to
                this list
            unexpected_keys (list of str): if ``strict=True``, add unexpected
                keys to this list
            error_msgs (list of str): error messages should be added to this
                list, and will be reported together in
                :meth:`~torch.nn.Module.load_state_dict`
        """
        # Modify state_dict first and then use default load function
        self._packed_weight = torch.ops.quantized.fbgemm_linear_prepack(state_dict[prefix + 'weight'])
        self.bias = state_dict[prefix + 'bias']
        # TODO: Remove once  https://github.com/pytorch/pytorch/pull/21852 lands

        super()._load_from_state_dict(state_dict, prefix, local_metadata, False,
                                      missing_keys, unexpected_keys, error_msgs)
        return
