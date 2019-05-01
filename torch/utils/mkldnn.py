from __future__ import absolute_import, division, print_function, unicode_literals
import functools

import torch


def to_mkldnn(module):
    def t_fn(t):
        if t.is_floating_point():
            return t.to_mkldnn()

    def m_fn(m):
        # TODO: This is a temporary hack to work around the fact that
        # nn.Linear is decomposed into addmm/matmul. Later we will
        # change nn.Linear to directly call aten linear and we can
        # remove this patch
        if isinstance(m, torch.nn.Linear):
            m.forward = functools.partial(
                torch._C._nn.linear,
                weight=m.weight,
                bias=m.bias)

        for param in m._parameters.values():
            if param is not None:
                # Tensors stored in modules are graph leaves, and we don't
                # want to create copy nodes, so we have to unpack the data.
                param.data = t_fn(param.data)
                if param._grad is not None:
                    param._grad.data = t_fn(param._grad.data)

        for key, buf in m._buffers.items():
            if buf is not None:
                m._buffers[key] = t_fn(buf)

        if isinstance(m, torch.nn.Conv2d):
            m.weight.data = torch._C._nn.mkldnn_reorder_conv2d_weight(
                m.weight.data,
                m.padding,
                m.stride,
                m.dilation,
                m.groups)

    return module.apply(m_fn)
