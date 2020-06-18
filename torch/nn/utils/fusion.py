
from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import torch

def fuse_conv_bn_eval(conv, bn):
    assert(not (conv.training or bn.training)), "Fusion only for eval!"
    fused_conv = copy.deepcopy(conv)

    fused_conv.weight, fused_conv.bias = \
        fuse_conv_bn_weights(fused_conv.weight, fused_conv.bias,
                             bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)

    return fused_conv

def fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    if conv_b is None:
        conv_b = bn_rm.new_zeros(bn_rm.shape)
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

    conv_w = conv_w * (bn_w * bn_var_rsqrt).reshape([-1] + [1] * (len(conv_w.shape) - 1))
    conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b

    return torch.nn.Parameter(conv_w), torch.nn.Parameter(conv_b)
