from __future__ import absolute_import, division, print_function, unicode_literals

import torch

import torch.nn._intrinsic.modules.fused as torch_fused
def fuse_conv_relu(conv, relu):
    r"""Given the conv and relu modules, fuses them and returns the fused module
    """
    return torch.nn._intrinsic.ConvReLU2d(conv, relu)

def fuse_linear_relu(linear, relu):
    r"""Given the linear and relu modules, fuses them and returns the fused module
    """
    return torch.nn._intrinsic.LinearReLU(linear, relu)

def fuse_conv_bn(conv, bn):
    r"""Given the conv and bn modules, fuses them and returns the fused module

    Args:
        conv: Module instance of type conv2d
        bn: Spatial BN instance that needs to be fused with the conv

    Examples::

        >>> m1 = nn.Conv2d(10, 20, 3)
        >>> b1 = nn.BatchNorm2d(20)
        >>> m2 = fuse_conv_bn(m1, b1)
    """
    assert(conv.training == bn.training),\
        "Conv and BN both must be in the same mode (train or eval)."

    if conv.training:
        assert conv.bias is None, 'Only support fusing Conv2d that does not have bias'
        assert bn.num_features == conv.out_channels, 'Output channel of Conv2d must match num_features of BatchNorm2d'
        assert bn.affine, 'Only support fusing BatchNorm2d with affine set to True'
        assert bn.track_running_stats, 'Only support fusing BatchNorm2d with tracking_running_stats set to True'
        return torch.nn._intrinsic.ConvBn2d(conv, bn)
    else:
        return torch.nn.utils.fuse_conv_bn_eval(conv, bn)

def fuse_conv_bn_relu(conv, bn, relu):
    r"""Given the conv and bn modules, fuses them and returns the fused module

    Args:
        conv: Module instance of type conv2d
        bn: Spatial BN instance that needs to be fused with the conv

    Examples::

        >>> m1 = nn.Conv2d(10, 20, 3)
        >>> b1 = nn.BatchNorm2d(20)
        >>> m2 = fuse_conv_bn(m1, b1)
    """
    assert(conv.training == bn.training == relu.training),\
        "Conv and BN both must be in the same mode (train or eval)."

    if conv.training:
        assert not relu.inplace, 'We only support fusion of non-inplace ReLU.'
        return torch_fused.ConvBnReLU2d(conv, bn, relu)
    else:
        return torch_fused.ConvReLU2d(
            torch.nn.utils.fusion.fuse_conv_bn_eval(conv, bn), relu)

def _get_module(model, submodule_key):
    submodule = ''
    tokens = submodule_key.split('.')
    cur_mod = model
    for s in tokens:
        if s.isdigit():
            cur_mod = cur_mod[int(s)]
        else:
            cur_mod = getattr(cur_mod, s)
    return cur_mod

# Generalization of setattr
def _set_module(model, submodule_key, module):
    submodule = ''
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    prev_mod = model
    for s in sub_tokens:
        if s.isdigit():
            cur_mod = cur_mod[int(s)]
        else:
            cur_mod = getattr( cur_mod, s)

    if tokens[-1].isdigit():
        cur_mod[int(tokens[-1])] = module
    else:
        setattr(cur_mod, tokens[-1], module)

def _fuse_modules(model, modules_to_fuse, fuser_func=None):

    OP_LIST_TO_FUSER_FUNC = {
        (torch.nn.Conv2d, torch.nn.BatchNorm2d): fuse_conv_bn,
        (torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.ReLU): fuse_conv_bn_relu,
        (torch.nn.Conv2d, torch.nn.ReLU): fuse_conv_relu,
        (torch.nn.Linear, torch.nn.ReLU): fuse_linear_relu
    }

    mod = []
    for item in modules_to_fuse:
        mod.append(_get_module(model, item))

    if fuser_func is None:
        types = tuple(type(m) for m in mod)
        fuser_func = OP_LIST_TO_FUSER_FUNC.get(types, None)
        if fuser_func is None:
            raise NotImplementedError("Cannot fuse modules: {}".format(types))
    new_mod = fuser_func(*mod)

    _set_module(model, modules_to_fuse[0], new_mod)

    for item in modules_to_fuse[1:]:
        _set_module(model, item, torch.nn.Identity())

def fuse_modules(model, modules_to_fuse):
    r"""Fuses a list of modules into a single module

    Fuses only the following sequence of modules:
    conv, bn
    conv, bn, relu
    conv, relu
    linear, relu
    All other sequences are left unchanged.
    For these sequences, replaces the first item in the list
    with the fused module, replacing the rest of the modules
    with identity.

    Arguments:
        model: Model containing the modules to be fused
        modules_to_fuse: list of list of module names to fuse.

    Returns:
        Modifies the model in place.

    Examples::

            >>> m = myModel()
            >>> # m is a module containing  the sub-modules below
            >>> modules_to_fuse = [ ['conv1', 'bn1', 'relu1'], ['submodule.conv', 'submodule.relu']]
            >>> nn.quantization.fuse_module.fuse_module(m, modules_to_fuse)
            >>> output = m(input)

    """
    for module_list in modules_to_fuse:
        _fuse_modules(model, module_list)
