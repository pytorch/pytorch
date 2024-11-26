#!/usr/bin/env python3
# Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>

# mypy: allow-untyped-defs
import copy

import torch

__all__ = ["pack_linear_weights"]


def _get_unique_devices_(module):
    return {p.device for p in module.parameters()} | {
        p.device for p in module.buffers()
    }


def _observer_forward_hook(self, input, output):
    r"""Forward hook that calls observer on the output"""
    return self.activation_post_process(output)


def pack_linear_weights(model, inplace=False):
    r"""
    Packs linear weights
    """

    torch._C._log_api_usage_once("utils_api.pack_linear.pack_linear_weights")
    if not inplace:
        model_ = copy.deepcopy(model)
    model_.eval()

    new_module = convert(model_)
    return new_module


def convert(
    module,
    inplace=False,
):
    r"""
    Packs weights in all submodules in a graph_module
    """
    torch._C._log_api_usage_once("utils_api.pack_linear.convert")

    if not inplace:
        mod_ = copy.deepcopy(module)
    else:
        mod_ = module

    for name, mod in mod_.named_children():
        if not isinstance(mod, torch.ao.nn.intrinsic._FusedModule):
            mod_._modules[name] = convert(mod, inplace)

        if isinstance(mod, torch.nn.Linear):
            mod_._modules[name] = pack_module(
                mod,
            )

    return mod_


def pack_module(
    mod,
):
    r"""
    Packs the weights of a linear module
    """
    # respect device affinity when swapping modules
    devices = _get_unique_devices_(mod)
    assert (
        len(devices) <= 1
    ), f"swap_module only works with cpu or single-device CUDA modules, but got devices {devices}"
    device = next(iter(devices)) if len(devices) > 0 else None

    if torch._C._has_mkldnn:
        packed_weight = torch._C._nn.mkldnn_reorder_linear_weight(torch.Tensor(mod.weight))
    else:
        raise Exception()

    if packed_weight is not None:
        mod.weight = torch.nn.Parameter(packed_weight)

    # Preserve module's pre forward hooks. They'll be called on quantized input
    for pre_hook_fn in mod._forward_pre_hooks.values():
        mod.register_forward_pre_hook(pre_hook_fn)
    # Preserve module's post forward hooks except _observer_forward_hook
    # After convert they'll work with quantized output
    for hook_fn in mod._forward_hooks.values():
        if hook_fn is not _observer_forward_hook:
            mod.register_forward_hook(hook_fn)

    if device:
        mod.to(device)

    return mod
