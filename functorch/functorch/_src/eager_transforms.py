# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from functools import partial, wraps
import collections
import torch.nn as nn
import torch.nn.functional as F
from torch.utils._pytree import tree_flatten, tree_unflatten, tree_map
from .pytree_hacks import tree_map_, treespec_pprint
from collections import namedtuple
import torch.autograd.forward_ad as fwAD
import gc

from .vmap import vmap
from .make_functional import make_functional_deprecated_v1, make_functional_with_buffers_deprecated_v1

from functorch._C import (
    _wrap_for_grad,
    _unwrap_for_grad,
    _grad_increment_nesting,
    _grad_decrement_nesting,
)

def _create_differentiable(inps, level=None):
    def create_differentiable(x):
        if isinstance(x, torch.Tensor):
            return x.requires_grad_()
        raise ValueError(f'Thing passed to transform API must be Tensor,'
                         f'got {type(x)}')
    return tree_map(create_differentiable, inps)

def _undo_create_differentiable(inps, level=None):
    def unwrap_tensors(x):
        if isinstance(x, torch.Tensor):
            return _unwrap_for_grad(x, level)
        # TODO: Remove the following hack for namedtuples
        if isinstance(x, tuple):
            return tree_map(unwrap_tensors, tuple(x))

        raise RuntimeError("Expected tensors, got unsupported type {type(x)}")

    return tree_map(unwrap_tensors, inps)

def _is_differentiable(maybe_tensor):
    if not isinstance(maybe_tensor, torch.Tensor):
        return False
    return maybe_tensor.requires_grad

def _any_differentiable(tensor_or_tuple_of_tensors):
    flat_args, _ = tree_unflatten(tensor_or_tuple_of_tensors)
    return any(tuple(map(_is_differentiable, flat_args)))

def _wrap_tensor_for_grad(maybe_tensor, level):
    if not isinstance(maybe_tensor, torch.Tensor):
        return maybe_tensor
    return _wrap_for_grad(maybe_tensor, level)

def _wrap_all_tensors(tensor_pytree, level):
    return tree_map(partial(_wrap_tensor_for_grad, level=level), tensor_pytree)

def _as_tuple(val):
    if isinstance(val, tuple):
        return val
    return (val,)

# Version of autograd.grad that handles outputs that don't depend on inputs
def _autograd_grad(outputs, inputs, grad_outputs=None, retain_graph=False, create_graph=True):
    if grad_outputs is None:
        diff_outputs = tuple(out for out in outputs if out.requires_grad)
    else:
        result = tuple((out, go) for out, go in zip(outputs, grad_outputs) if out.requires_grad)
        if len(result) == 0:
            diff_outputs, grad_outputs = (), ()
        else:
            diff_outputs, grad_outputs = zip(*result)
    if len(diff_outputs) == 0:
        return tuple(torch.zeros_like(inp) for inp in inputs)
    grad_inputs = torch.autograd.grad(diff_outputs, inputs, grad_outputs,
                                      retain_graph=retain_graph,
                                      create_graph=create_graph,
                                      allow_unused=True)
    grad_inputs = tuple(torch.zeros_like(inp) if gi is None else gi
                        for gi, inp in zip(grad_inputs, inputs))
    return grad_inputs

# How do we increment and decrement the nesting? I don't think we can.
def vjp(f, *primals):
    level = _grad_increment_nesting()
    try:
        primals = _wrap_all_tensors(primals, level)
        diff_primals = _create_differentiable(primals, level)
        primals_out = f(*diff_primals)

        results = _undo_create_differentiable(primals_out, level)
        flat_diff_primals, primals_spec = tree_flatten(diff_primals)
        flat_primals_out, primals_out_spec = tree_flatten(primals_out)

        for primal_out in flat_primals_out:
            assert isinstance(primal_out, torch.Tensor)
            if primal_out.is_floating_point() or primal_out.is_complex():
                continue
            raise RuntimeError("vjp(f, ...): All outputs of f must be "
                               "floating-point or complex Tensors, got Tensor "
                               f"with dtype {primal_out.dtype}")

        def wrapper(cotangents, retain_graph=True, create_graph=True):
            flat_cotangents, cotangents_spec = tree_flatten(cotangents)
            if primals_out_spec != cotangents_spec:
                raise RuntimeError(
                    f'Expected pytree structure of cotangents to be the same '
                    f'as pytree structure of outputs to the function. '
                    f'cotangents: {treespec_pprint(cotangents_spec)}, '
                    f'primal output: {treespec_pprint(primals_out_spec)}')
            result = _autograd_grad(flat_primals_out, flat_diff_primals, flat_cotangents,
                                    retain_graph=retain_graph, create_graph=create_graph)
            return tree_unflatten(result, primals_spec)

    finally:
        _grad_decrement_nesting()

    return results, wrapper

def jacrev(f):
    def wrapper_fn(primal):
        output, vjp_fn = vjp(f, primal)
        assert isinstance(output, torch.Tensor)
        # TODO: does jacrev compose with vmap...? the eye call should make it so that it doesn't
        basis = torch.eye(output.numel(), dtype=output.dtype, device=output.device) \
                     .view(output.numel(), *output.shape)
        result, = vmap(vjp_fn)(basis)
        result = result.view(*output.shape, *primal.shape)
        return result
    return wrapper_fn

def _safe_index(args, argnum):
    if not isinstance(argnum, int):
        raise RuntimeError(f'argnum must be int, got: {type(argnum)}')
    if argnum >= 0 and argnum < len(args):
        return args[argnum]
    raise RuntimeError(f'Got argnum={argnum}, but only {len(args)} positional inputs')

def _slice_argnums(args, argnums):
    if isinstance(argnums, int):
        return _safe_index(args, argnums)
    if isinstance(argnums, tuple):
        return tuple(_safe_index(args, i) for i in argnums)
    raise RuntimeError(f'argnums must be int or Tuple[int, ...], got: {type(argnums)}')

def jvp(f, primals, tangents):
    level = _grad_increment_nesting()
    try:
        # Some interesting notes:
        # 1. Can't nested jvp of jvp due to forwardAD restrictions
        # 2. Seems like we can indeed vmap over this, given some more batch rules
        # 3. PyTorch doesn't have a lot of jvp rules implemented right now.
        with fwAD.dual_level():
            # TODO: extend this to any number of primals
            assert len(primals) == 1 and len(tangents) == 1
            duals = tuple(fwAD.make_dual(p, t) for p, t in zip(primals, tangents))
            result_duals = f(*duals)
            result_duals, _ = tree_flatten(result_duals)
            assert len(result_duals) == 1
            primals_out, tangents_out = fwAD.unpack_dual(result_duals[0])
            primals_out = _undo_create_differentiable(primals_out, level)
            tangents_out = _undo_create_differentiable(tangents_out, level)
            return primals_out, tangents_out
    finally:
        _grad_decrement_nesting()

def jacfwd(f):
    # TODO: This should take more than just a single primal...
    def wrapper_fn(primal):
        basis = torch.eye(primal.numel(), dtype=primal.dtype, device=primal.device) \
                     .view(primal.numel(), *primal.shape)

        def push_jvp(basis):
            _, jvp_out = jvp(f, (primal,), (basis,))
            return jvp_out

        result = vmap(push_jvp)(basis)
        result = result.view(*primal.shape, *primal.shape)
        return result
    return wrapper_fn

def grad_and_value(f, argnums=0, has_aux=False):
    @wraps(f)
    def wrapper(*args, **kwargs):
        level = _grad_increment_nesting()
        output, aux, grad_input = None, None, None
        try:
            args = _wrap_all_tensors(args, level)
            kwargs = _wrap_all_tensors(kwargs, level)
            diff_args = _slice_argnums(args, argnums)
            tree_map_(partial(_create_differentiable, level=level), diff_args)

            output = f(*args, **kwargs)
            if has_aux:
                output, aux = output

            if not isinstance(output, torch.Tensor):
                raise RuntimeError('grad_and_value(f)(*args): Expected f(*args)'
                                   f'to return a Tensor, got {type(output)}')
            if output.dim() != 0:
                raise RuntimeError('grad_and_value(f)(*args): Expected f(*args)'
                                   'to return a scalar Tensor, got tensor with '
                                   f'{output.dim()} dims. Maybe you wanted to'
                                   'use the vjp or jacrev APIs instead?')

            flat_diff_args, spec = tree_flatten(diff_args)

            # NB: need create_graph so that backward pass isn't run in no_grad mode
            flat_outputs = _as_tuple(output)
            flat_grad_input = _autograd_grad(flat_outputs, flat_diff_args, create_graph=True)
            grad_input = tree_unflatten(flat_grad_input, spec)

        finally:
            if grad_input is not None:
                grad_input = _undo_create_differentiable(grad_input, level)
            if output is not None:
                output = _undo_create_differentiable(output, level)
            if aux is not None:
                aux = _undo_create_differentiable(aux, level)
            _grad_decrement_nesting()
        if has_aux:
            return grad_input, (output, aux)
        return grad_input, output
    return wrapper

def grad(f, argnums=0, has_aux=False):
    @wraps(f)
    def wrapper(*args, **kwargs):
        results = grad_and_value(f, argnums, has_aux=has_aux)(*args, **kwargs)
        if has_aux:
            grad, (value, aux) = results
            return grad, aux
        grad, value = results
        return grad
    return wrapper

def vjpfull(f, primals, tangents):
    out, vjpfn = vjp(f, *primals)
    return out, vjpfn(*tangents)
