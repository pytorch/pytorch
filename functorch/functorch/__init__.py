# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import functools
import textwrap
from . import _C

# Top-level APIs. Please think carefully before adding something to the
# top-level namespace:
# - private helper functions should go into functorch._src
# - very experimental things should go into functorch.experimental
# - compilation related things should go into functorch.compile

# functorch transforms
from ._src.vmap import vmap
from ._src.eager_transforms import grad, grad_and_value, vjp, jacrev
from ._src.python_key import make_fx, pythonkey_decompose, register_decomposition

# utilities. Maybe these should go in their own namespace in the future?
from ._src.make_functional import (
    make_functional_with_buffers,
    make_functional,
    combine_state_for_ensemble,
    FunctionalModule,
)

# Monkeypatching lol
_old_cross_entropy = torch.nn.functional.cross_entropy


# **kwargs to handle the new label_smoothing arg
def cross_entropy(input, target, weight=None, size_average=None,
                  ignore_index=-100, reduce=None, reduction='mean', **kwargs):
    if input.dim() == 1 and target.dim() == 0:
        input = input.unsqueeze(0)
        target = target.unsqueeze(0)

    result = _old_cross_entropy(
            input, target, weight, size_average,
            ignore_index, reduce, reduction, **kwargs)
    if reduction == 'none':
        return result.squeeze(0)
    return result


torch.nn.functional.cross_entropy = cross_entropy

# Monkeypatch .backward() to error out if any transforms are active.
# TODO: remove the monkeypatching and add an extension point into PyTorch core
_old_backward = torch.Tensor.backward


@functools.wraps(_old_backward)
def _backward(*args, **kwargs):
    if _C.are_transforms_active():
        raise RuntimeError(
            "backward() called inside a functorch transform. This is not "
            "supported, please use functorch.grad or functorch.vjp instead "
            "or call backward() outside of functorch transforms.")
    return _old_backward(*args, **kwargs)


setattr(torch.Tensor, 'backward', _backward)


# Monkeypatch tensor printing in pytorch
_old_str = torch._tensor_str._str


def prep_value(text, indent=4):
    first_line_txt = ''
    lines = text.split('\n')
    lines[0] = lines[0]
    lines[0] = ' ' * indent + first_line_txt + lines[0]
    for i in range(1, len(lines)):
        lines[i] = ' ' * (indent + len(first_line_txt)) + lines[i]
    return '\n'.join(lines)


@functools.wraps(_old_str)
def _functorch_str(tensor):
    level = _C.maybe_get_level(tensor)
    if level == -1:
        return _old_str(tensor)

    value = _C.get_unwrapped(tensor)
    value_repr = repr(value)
    if _C.is_batchedtensor(tensor):
        bdim = _C.maybe_get_bdim(tensor)
        assert bdim != -1
        return (
            f'BatchedTensor(lvl={level}, bdim={bdim}, value=\n'
            f'{prep_value(value_repr)}\n'
            f')'
        )
    if _C.is_gradtrackingtensor(tensor):
        return (
            f'GradTrackingTensor(lvl={level}, value=\n'
            f'{prep_value(value_repr)}\n'
            f')'
        )

    raise ValueError("We don't know how to print this, please file us an issue")


torch._tensor_str._str = _functorch_str
