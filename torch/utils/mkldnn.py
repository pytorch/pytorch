from __future__ import absolute_import, division, print_function, unicode_literals

import torch


def to_mkldnn(module):
    for child in module.children():
        to_mkldnn(child)

    def fn(t):
        if t.is_floating_point():
            return t.to_mkldnn()

    for param in module._parameters.values():
        if param is not None:
            # Tensors stored in modules are graph leaves, and we don't
            # want to create copy nodes, so we have to unpack the data.
            param.data = fn(param.data)
            if param._grad is not None:
                param._grad.data = fn(param._grad.data)

    for key, buf in module._buffers.items():
        if buf is not None:
            module._buffers[key] = fn(buf)

    if isinstance(module, torch.nn.Conv2d):
        module.weight.data = module.weight.data.mkldnn_reorder_conv2d_weight(
            module.padding,
            module.stride,
            module.dilation,
            module.groups)

    return module
