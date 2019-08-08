
from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import torch

def fuse_conv_bn_eval(conv, bn):
    assert(not (conv.training or bn.training)), "Fusion only for eval!"
    fused_conv = copy.deepcopy(conv)

    conv_w = fused_conv.weight
    conv_b = fused_conv.bias

    bn_mean = bn.running_mean
    bn_var_sqrt = torch.sqrt(bn.running_var + bn.eps)

    bn_weight = bn.weight
    bn_bias = bn.bias

    if conv_b is None:
        conv_b = bn_mean.new_zeros(bn_mean.shape)

    conv_w = conv_w * (bn_weight / bn_var_sqrt).reshape([-1, 1, 1, 1])
    conv_b = (conv_b - bn_mean) / bn_var_sqrt * bn_weight + bn_bias

    fused_conv.weight = torch.nn.Parameter(conv_w)
    fused_conv.bias = torch.nn.Parameter(conv_b)

    return fused_conv

def fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    bn_var_sqrt = torch.sqrt(bn_rv + bn_eps)

    if conv_b is None:
        conv_b = bn_rm.new_zeros(bn_rm.shape)

    conv_w = conv_w * (bn_w / bn_var_sqrt).reshape([-1, 1, 1, 1])
    conv_b = (conv_b - bn_rm) / bn_var_sqrt * bn_w + bn_b

    return conv_w, conv_b
