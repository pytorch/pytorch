from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import copy
from collections import OrderedDict

def fuse_conv_bn(conv, bn):
    r"""Given the conv and bn modules, fuses them and returns the fused module

    Args:
        conv: Module instance of type conv2d
        bn: Spatial BN instance that needs to be fused with the conv

    Examples::

        >>> m1 = nn.Conv2d(10, 20, 3)
        >>> b1 = nn.BatchNorm2d(20)
        >>> m2 = fuse_conv_bn(m1, b1, eval=True)
    """
    assert(conv.training == bn.training),\
        "Conv and BN both must be in the same mode (train or eval)."

    if conv.training:
        fused_conv = copy.deepcopy(conv)

        w_conv = fused_conv.weight
        b_conv = fused_conv.bias

        bn_mean = bn.running_mean
        bn_var_sqrt = torch.sqrt(bn.running_var + bn.eps)

        bn_weight = bn.weight
        bn_bias = bn.bias

        if b_conv is None:
            b_conv = bn_mean.new_zeros(bn_mean.shape)

        w_conv = w_conv * (bn_weight / bn_var_sqrt).reshape([fused_conv.out_channels, 1, 1, 1])
        b_conv = (b_conv - bn_mean) / bn_var_sqrt * bn_weight + bn_bias

        fused_conv.weight = torch.nn.Parameter(w_conv)
        fused_conv.bias = torch.nn.Parameter(b_conv)
        return fused_conv
    else:
        fused_conv = torch.nn.Sequential(OrderedDict([('conv', conv), ('bn', bn)]))
        return fused_conv


def fuse_conv_bn_relu(conv, bn, relu):
    r"""Given the conv and bn modules, fuses them and returns the fused module

    Args:
        conv: Module instance of type conv2d
        bn: Spatial BN instance that needs to be fused with the conv
        eval: Boolean specifying if batch norm folding occurs for train or eval.

    Examples::

        >>> m1 = nn.Conv2d(10, 20, 3)
        >>> b1 = nn.BatchNorm2d(20)
        >>> m2 = fuse_conv_bn(m1, b1, eval=True)
    """
    assert(conv.training == bn.training),\
        "Conv and BN both must be in the same mode (train or eval)."

    if eval:
        return torch.nn._intrinsic.Conv2dReLU.from_modules(fuse_conv_bn(conv, bn), relu)
    else:
        # TODO(zaf): Use fusion ConvBnRelu here or throw NotImplementedError
        return torch.nn.Sequential(OrderedDict([('conv', conv), ('bn', bn), ('relu', relu)]))

def _fuse_modules(model, named_module_dict, modules_to_fuse, eval):
    assert(len(modules_to_fuse) == 2 or len(modules_to_fuse) == 3),\
        "Can fuse only 2 or 3 modules."

    mod = []
    parent_mod = []
    for i in range(len(modules_to_fuse)):
        parent_module_name = '.'.join(modules_to_fuse[i].split('.')[:-1])
        mod.append(named_module_dict[modules_to_fuse[i]])
        parent_mod.append(named_module_dict.get(parent_module_name, model))

    new_mod = mod[0]
    types = [type(m) for m in mod]
    if types == [torch.nn.Conv2d, torch.nn.BatchNorm2d]:
        new_mod = fuse_conv_bn(mod[0], mod[1], eval)
    elif types == [torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.ReLU]:
        new_mod = fuse_conv_bn_relu(mod[0], mod[1], mod[2], eval)
    else:
        raise NotImplementedError("Cannot fuse modules: {}".format(types))

    # Assign new_mod to module and set remaining modules to identity
    if new_mod is not mod[0]:
        setattr(parent_mod[0], modules_to_fuse[0].split('.')[-1], new_mod)
        for i in range(1, len(modules_to_fuse)):
            setattr(parent_mod[i], modules_to_fuse[i].split('.')[-1], torch.nn.Identity())


def fuse_modules(model, modules_to_fuse, eval=True):
    r"""Fuses a list of modules into a single module

    Fuses only the following sequence of modules:
    conv, bn
    conv, bn, relu
    All other sequences are left unchanged.
    For these sequences, replaces the first item in the list
    with the fused module, replacing the rest of the modules
    with identity.

    Arguments:
        model: Model containing the modules to be fused
        modules_to_fuse: list of list of module names to fuse.
        eval: Boolean specifying if fusion is done for training or eval

    Returns:
        Modifies the model in place.

    Examples::

            >>> m = myModel()
            >>> # m is a module containing  the sub-modules below
            >>> modules_to_fuse = [ ['conv1', 'bn1', 'relu1'], ['submodule.conv', 'submodule.relu']]
            >>> torch.quantization.fuse_module(m,modules_to_fuse)
            >>> output = m(input)

    """
    named_module_dict = {name: mod for name, mod in model.named_modules()}
    for module_list in modules_to_fuse:
        _fuse_modules(model, named_module_dict, module_list, eval)
