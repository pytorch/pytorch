
from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import torch

def fuse_conv_bn_eval(conv, bn):
    assert(not (conv.training or bn.training)), "Fusion only for eval!"
    fused_conv = copy.deepcopy(conv)

    w_conv = fused_conv.weight
    b_conv = fused_conv.bias

    bn_mean = bn.running_mean
    bn_var_sqrt = torch.rsqrt(bn.running_var + bn.eps)

    bn_weight = bn.weight
    bn_bias = bn.bias

    if b_conv is None:
        b_conv = bn_mean.new_zeros(bn_mean.shape)

    w_conv = w_conv * (bn_weight * bn_var_sqrt).reshape([-1, 1, 1, 1])
    b_conv = (b_conv - bn_mean) * bn_var_sqrt * bn_weight + bn_bias

    fused_conv.weight = torch.nn.Parameter(w_conv)
    fused_conv.bias = torch.nn.Parameter(b_conv)

    return fused_conv
