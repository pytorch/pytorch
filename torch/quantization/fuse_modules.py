from __future__ import absolute_import, division, print_function, unicode_literals

import torch

import torch.nn._intrinsic.modules.fused as torch_fused

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
        return torch_fused.ConvBnReLU2d(conv, bn, relu)
    else:
        return torch_fused.ConvReLU2d(
            torch.nn.utils.fusion.fuse_conv_bn_eval(conv, bn), relu)


def _fuse_modules(model, named_module_dict, modules_to_fuse, fuser_func=None):
    assert(len(modules_to_fuse) == 2 or len(modules_to_fuse) == 3),\
        "Can fuse only 2 or 3 modules."

    OP_LIST_TO_FUSER_FUNC = {
        (torch.nn.Conv2d, torch.nn.BatchNorm2d): fuse_conv_bn,
        (torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.ReLU): fuse_conv_bn_relu
    }

    mod = []
    parent_mod = []
    for i in range(len(modules_to_fuse)):
        parent_module_name = '.'.join(modules_to_fuse[i].split('.')[:-1])
        mod.append(named_module_dict[modules_to_fuse[i]])
        parent_mod.append(named_module_dict.get(parent_module_name, model))

    new_mod = mod[0]
    if fuser_func is None:
        types = tuple(type(m) for m in mod)
        fuser_func = OP_LIST_TO_FUSER_FUNC.get(types, None)
        if fuser_func is None:
            raise NotImplementedError("Cannot fuse modules: {}".format(types))
    new_mod = fuser_func(*mod)

    # Assign new_mod to module and set remaining modules to identity
    if new_mod is not mod[0]:
        setattr(parent_mod[0], modules_to_fuse[0].split('.')[-1], new_mod)
        for i in range(1, len(modules_to_fuse)):
            setattr(parent_mod[i], modules_to_fuse[i].split('.')[-1], torch.nn.Identity())


def fuse_modules(model, modules_to_fuse):
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

    Returns:
        Modifies the model in place.

    Examples::

            >>> m = myModel()
            >>> # m is a module containing  the sub-modules below
            >>> modules_to_fuse = [ ['conv1', 'bn1', 'relu1'], ['submodule.conv', 'submodule.relu']]
            >>> nn.quantization.fuse_module.fuse_module(m, modules_to_fuse)
            >>> output = m(input)

    """
    named_module_dict = {name: mod for name, mod in model.named_modules()}
    for module_list in modules_to_fuse:
        _fuse_modules(model, named_module_dict, module_list)
