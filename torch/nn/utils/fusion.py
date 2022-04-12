

import copy
import torch

def fuse_conv_bn_eval(conv, bn, transpose=False):
    assert(not (conv.training or bn.training)), "Fusion only for eval!"
    fused_conv = copy.deepcopy(conv)

    fused_conv.weight, fused_conv.bias = \
        fuse_conv_bn_weights(fused_conv.weight, fused_conv.bias,
                             bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias, transpose)

    return fused_conv

def fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b, transpose=False):
    if conv_b is None:
        conv_b = torch.zeros_like(bn_rm)
    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)
    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

    if transpose:
        shape = [1, -1] + [1] * (len(conv_w.shape) - 2)
    else:
        shape = [-1, 1] + [1] * (len(conv_w.shape) - 2)

    conv_w = conv_w * (bn_w * bn_var_rsqrt).reshape(shape)
    conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b

    return torch.nn.Parameter(conv_w), torch.nn.Parameter(conv_b)

def fuse_linear_bn_eval(linear, bn):
    assert(not (linear.training or bn.training)), "Fusion only for eval!"
    fused_linear = copy.deepcopy(linear)

    fused_linear.weight, fused_linear.bias = fuse_linear_bn_weights(
        fused_linear.weight, fused_linear.bias,
        bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)

    return fused_linear

def fuse_linear_bn_weights(linear_w, linear_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    if linear_b is None:
        linear_b = torch.zeros_like(bn_rm)
    bn_scale = bn_w * torch.rsqrt(bn_rv + bn_eps)

    fused_w = linear_w * bn_scale.unsqueeze(-1)
    fused_b = (linear_b - bn_rm) * bn_scale + bn_b

    return torch.nn.Parameter(fused_w), torch.nn.Parameter(fused_b)
