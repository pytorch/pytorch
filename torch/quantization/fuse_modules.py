from __future__ import absolute_import, division, print_function, unicode_literals

import copy

import torch.nn.intrinsic.modules.fused as torch_fused
import torch.nn as nn
import torch.nn.intrinsic as nni

from typing import Type, List, Optional, Union, Callable, Tuple, Dict

def fuse_conv_bn(conv, bn):
    r"""Given the conv and bn modules, fuses them and returns the fused module

    Args:
        conv: Module instance of type conv2d/conv3d
        bn: Spatial BN instance that needs to be fused with the conv

    Examples::

        >>> m1 = nn.Conv2d(10, 20, 3)
        >>> b1 = nn.BatchNorm2d(20)
        >>> m2 = fuse_conv_bn(m1, b1)
    """
    assert(conv.training == bn.training),\
        "Conv and BN both must be in the same mode (train or eval)."

    is_3d = isinstance(conv, nn.Conv3d)

    if conv.training:
        assert bn.num_features == conv.out_channels, 'Output channel of Conv2d must match num_features of BatchNorm2d'
        assert bn.affine, 'Only support fusing BatchNorm2d with affine set to True'
        assert bn.track_running_stats, 'Only support fusing BatchNorm2d with tracking_running_stats set to True'
        return nni.ConvBn3d(conv, bn) if is_3d \
            else nni.ConvBn2d(conv, bn)
    else:
        return nn.utils.fuse_conv_bn_eval(conv, bn)

def fuse_conv_bn_relu(conv, bn, relu):
    r"""Given the conv and bn modules, fuses them and returns the fused module

    Args:
        conv: Module instance of type conv2d/conv3d
        bn: Spatial BN instance that needs to be fused with the conv

    Examples::

        >>> m1 = nn.Conv2d(10, 20, 3)
        >>> b1 = nn.BatchNorm2d(20)
        >>> m2 = fuse_conv_bn(m1, b1)
    """
    assert(conv.training == bn.training == relu.training),\
        "Conv and BN both must be in the same mode (train or eval)."
    fused_module : Optional[Type[nn.Sequential]] = None
    if conv.training:
        map_to_fused_module_train = {
            nn.Conv2d: torch_fused.ConvBnReLU2d,
            nn.Conv3d: torch_fused.ConvBnReLU3d,
        }
        assert bn.num_features == conv.out_channels, 'Output channel of Conv must match num_features of BatchNorm'
        assert bn.affine, 'Only support fusing BatchNorm with affine set to True'
        assert bn.track_running_stats, 'Only support fusing BatchNorm with tracking_running_stats set to True'
        fused_module = map_to_fused_module_train.get(type(conv))
        if fused_module is not None:
            return fused_module(conv, bn, relu)
        else:
            raise NotImplementedError("Cannot fuse train modules: {}".format((conv, bn, relu)))
    else:
        map_to_fused_module_eval = {
            nn.Conv1d: torch_fused.ConvReLU1d,
            nn.Conv2d: torch_fused.ConvReLU2d,
            nn.Conv3d: torch_fused.ConvReLU3d,
        }
        fused_module = map_to_fused_module_eval[type(conv)]
        if fused_module is not None:
            fused_conv = nn.utils.fusion.fuse_conv_bn_eval(conv, bn)
            return fused_module(fused_conv, relu)
        else:
            raise NotImplementedError("Cannot fuse eval modules: {}".format((conv, bn, relu)))

OP_LIST_TO_FUSER_METHOD : Dict[Tuple, Union[nn.Sequential, Callable]] = {
    (nn.Conv1d, nn.BatchNorm1d): fuse_conv_bn,
    (nn.Conv1d, nn.BatchNorm1d, nn.ReLU): fuse_conv_bn_relu,
    (nn.Conv2d, nn.BatchNorm2d): fuse_conv_bn,
    (nn.Conv2d, nn.BatchNorm2d, nn.ReLU): fuse_conv_bn_relu,
    (nn.Conv3d, nn.BatchNorm3d): fuse_conv_bn,
    (nn.Conv3d, nn.BatchNorm3d, nn.ReLU): fuse_conv_bn_relu,
    (nn.Conv1d, nn.ReLU): nni.ConvReLU1d,
    (nn.Conv2d, nn.ReLU): nni.ConvReLU2d,
    (nn.Conv3d, nn.ReLU): nni.ConvReLU3d,
    (nn.Linear, nn.ReLU): nni.LinearReLU,
    (nn.BatchNorm2d, nn.ReLU): nni.BNReLU2d,
    (nn.BatchNorm3d, nn.ReLU): nni.BNReLU3d,
}

# Generalization of getattr
def _get_module(model, submodule_key):
    tokens = submodule_key.split('.')
    cur_mod = model
    for s in tokens:
        cur_mod = getattr(cur_mod, s)
    return cur_mod

# Generalization of setattr
def _set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)

    setattr(cur_mod, tokens[-1], module)

def fuse_known_modules(mod_list):
    r"""Returns a list of modules that fuses the operations specified
     in the input module list.

    Fuses only the following sequence of modules:
    conv, bn
    conv, bn, relu
    conv, relu
    linear, relu
    For these sequences, the first element in the output module list performs
    the fused operation. The rest of the elements are set to nn.Identity()
    """
    types = tuple(type(m) for m in mod_list)
    fuser_method = OP_LIST_TO_FUSER_METHOD.get(types)
    if fuser_method is None:
        raise NotImplementedError("Cannot fuse modules: {}".format(types))
    new_mod : List[Optional[nn.Module]] = [None] * len(mod_list)
    fused = fuser_method(*mod_list)
    # NOTE: forward hooks not processed in the two following for loops will be lost after the fusion
    # Move pre forward hooks of the base module to resulting fused module
    for handle_id, pre_hook_fn in mod_list[0]._forward_pre_hooks.items():
        fused.register_forward_pre_hook(pre_hook_fn)
        del mod_list[0]._forward_pre_hooks[handle_id]
    # Move post forward hooks of the last module to resulting fused module
    for handle_id, hook_fn in mod_list[-1]._forward_hooks.items():
        fused.register_forward_hook(hook_fn)
        del mod_list[-1]._forward_hooks[handle_id]
    new_mod[0] = fused

    for i in range(1, len(mod_list)):
        identity = nn.Identity()
        identity.training = mod_list[0].training
        new_mod[i] = identity

    return new_mod

def _fuse_modules(model, modules_to_fuse, fuser_func=fuse_known_modules):

    mod_list = []
    for item in modules_to_fuse:
        mod_list.append(_get_module(model, item))

    # Fuse list of modules
    new_mod_list = fuser_func(mod_list)

    # Replace original module list with fused module list
    for i, item in enumerate(modules_to_fuse):
        _set_module(model, item, new_mod_list[i])

def fuse_modules(model, modules_to_fuse, inplace=False, fuser_func=fuse_known_modules):
    r"""Fuses a list of modules into a single module

    Fuses only the following sequence of modules:
    conv, bn
    conv, bn, relu
    conv, relu
    linear, relu
    bn, relu
    All other sequences are left unchanged.
    For these sequences, replaces the first item in the list
    with the fused module, replacing the rest of the modules
    with identity.

    Arguments:
        model: Model containing the modules to be fused
        modules_to_fuse: list of list of module names to fuse. Can also be a list
                         of strings if there is only a single list of modules to fuse.
        inplace: bool specifying if fusion happens in place on the model, by default
                 a new model is returned
        fuser_func: Function that takes in a list of modules and outputs a list of fused modules
                    of the same length. For example,
                    fuser_func([convModule, BNModule]) returns the list [ConvBNModule, nn.Identity()]
                    Defaults to torch.quantization.fuse_known_modules
    Returns:
        model with fused modules. A new copy is created if inplace=True.

    Examples::

            >>> m = myModel()
            >>> # m is a module containing  the sub-modules below
            >>> modules_to_fuse = [ ['conv1', 'bn1', 'relu1'], ['submodule.conv', 'submodule.relu']]
            >>> fused_m = torch.quantization.fuse_modules(m, modules_to_fuse)
            >>> output = fused_m(input)

            >>> m = myModel()
            >>> # Alternately provide a single list of modules to fuse
            >>> modules_to_fuse = ['conv1', 'bn1', 'relu1']
            >>> fused_m = torch.quantization.fuse_modules(m, modules_to_fuse)
            >>> output = fused_m(input)

    """
    if not inplace:
        model = copy.deepcopy(model)

    if all(isinstance(module_element, str) for module_element in modules_to_fuse):
        # Handle case of modules_to_fuse being a list
        _fuse_modules(model, modules_to_fuse, fuser_func)
    else:
        # Handle case of modules_to_fuse being a list of lists
        for module_list in modules_to_fuse:
            _fuse_modules(model, module_list, fuser_func)
    return model
