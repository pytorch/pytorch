from __future__ import absolute_import, division, print_function, unicode_literals
import functools

import torch


def to_mkldnn(module):
    def t_fn(t):
        if t.is_floating_point():
            return t.to_mkldnn()

    def m_fn(m):
        for key, param in m._parameters.items():
            if param is not None:
                # yf225 TODO: explain the change here!
                m._parameters[key] = t_fn(param)
                if param._grad is not None:
                    m._parameters[key]._grad = t_fn(param._grad)

        for key, buf in m._buffers.items():
            if buf is not None:
                m._buffers[key] = t_fn(buf)

        # TODO: This is a temporary hack to work around the fact that
        # nn.Linear is decomposed into addmm/matmul. Later we will
        # change nn.Linear to directly call aten linear and we can
        # remove this patch
        # yf225 TODO: add comment: this needs to happen after the conversion for param and buf, because
        # the conversion can potentially change the values that `m.weight` and `m.bias` point to, which invalidates
        # the original pointers
        if isinstance(m, torch.nn.Linear):
            m.forward = functools.partial(
                torch._C._nn.linear,
                weight=m.weight,
                bias=m.bias)

        if isinstance(m, torch.nn.Conv2d):
            m.weight.data = torch._C._nn.mkldnn_reorder_conv2d_weight(
                m.weight.data,
                m.padding,
                m.stride,
                m.dilation,
                m.groups)

    return module.apply(m_fn)
