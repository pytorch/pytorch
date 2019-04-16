from __future__ import absolute_import, division, print_function, unicode_literals

import torch


def to_mkldnn(module):
    def t_fn(t):
        if t.is_floating_point():
            return t.to_mkldnn()

    def m_fn(m):
        for param in m._parameters.values():
            if param is not None:
                # Tensors stored in modules are graph leaves, and we don't
                # want to create copy nodes, so we have to unpack the data.
                param.data = t_fn(param.data)
                if param._grad is not None:
                    param._grad.data = t_fn(param._grad.data)

        for key, buf in module._buffers.items():
            if buf is not None:
                module._buffers[key] = t_fn(buf)

        if isinstance(module, torch.nn.Conv2d):
            module.weight.data = torch._C._nn.mkldnn_reorder_conv2d_weight(
                module.weight.data,
                module.padding,
                module.stride,
                module.dilation,
                module.groups)

    return module.apply(m_fn)
