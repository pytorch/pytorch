# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from functools import partial, wraps
import contextlib
import collections
import torch.nn as nn
import torch.nn.functional as F
from torch.utils._pytree import tree_flatten, tree_unflatten, tree_map
from typing import Callable, Tuple, Union
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
        raise ValueError(f'Thing passed to transform API must be Tensor, '
                         f'got {type(x)}')
    return tree_map(create_differentiable, inps)

def _undo_create_differentiable(inps, level=None):
    def unwrap_tensors(x):
        if isinstance(x, torch.Tensor):
            return _unwrap_for_grad(x, level)
        # TODO: Remove the following hack for namedtuples
        if isinstance(x, tuple):
            return tree_map(unwrap_tensors, tuple(x))

        raise RuntimeError(f"Expected tensors, got unsupported type {type(x)}")

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

# NOTE [grad and vjp interaction with no_grad]
#
# def f(x):
#   with torch.no_grad():
#     c = x ** 2
#   return x - c
#
# The thing to consider is if enable_grad is on/off before grad gets called.
#
# Case 1: enable_grad is on.
# grad(f)(x)
# In this case, `grad` should respect the inner torch.no_grad.
#
# Case 2: enable_grad is off
# with torch.no_grad():
#   grad(f)(x)
# In this case, `grad` should respect the inner torch.no_grad, but not the
# outer one. This is because `grad` is a "function transform": its result
# should not depend on the result of a context manager outside of `f`.
#
# This gives us the following desired behavior:
# - (nested) grad transforms must obey torch.no_grad inside them
# - (nested) grad transforms should not obey torch.no_grad outside them
#
# To achieve this behavior, upon entering grad/vjp:
# - we save the current ("previous") is_grad_enabled (*)
# - we unconditionally enable grad.
#
# Inside DynamicLayerBackFallback, when we're temporarily popping `grad` layer
# off the stack:
# - if grad_mode is disabled, then we do nothing. (there is a torch.no_grad
#   active, all subsequent grad transforms must obey it).
# - if grad_mode is enabled, and the previous is_grad_enabled (*) is False,
#   then we temporarily restore the previous `is_grad_enabled`. This is
#   because we're crossing the boundary from a `grad` outside the
#   no_grad to a `grad` inside the no_grad.
#
# NB: vjp has some interesting behavior because the vjp's callable can be called
# under a different grad_mode than the forward computation...
#
# TODO: forward-mode AD: does it also respect no_grad? What does that mean
# for our jvp transform?


# How do we increment and decrement the nesting? I don't think we can.
def vjp(f: Callable, *primals):
    """
    Standing for the vector-Jacobian product, returns a tuple containing the
    results of :attr:`f` applied to :attr:`primals` and a function that, when
    given ``cotangents``, computes the reverse-mode Jacobian of :attr:`f` with
    respect to :attr:`primals` times ``cotangents``.

    Args:
        f (Callable): A Python function that takes one or more arguments. Must
            return one or more Tensors.
        primals: Positional arguments to :attr:`f` that must all be Tensors.
            The returned function will also be computing the derivative with
            respect to these arguments

    Returns:
        Returns a tuple containing the output of :attr:`f` applied to
        :attr:`primals` and a function that computes the vjp of :attr:`f` with
        respect to all :attr:`primals` using the cotangents passed to the
        returned function. The returned function will return a tuple of each
        VJP
    
    When used in simple cases, :func:`vjp` behaves the same as :func:`grad`
        >>> x = torch.randn([5])
        >>> f = lambda x: x.sin().sum()
        >>> (_, vjpfunc) = functorch.vjp(f, x)
        >>> grad = vjpfunc(torch.tensor(1.))[0]
        >>> assert torch.allclose(grad, functorch.grad(f)(x))

    However, :func:`vjp` can support functions with multiple outputs by
    passing in the cotangents for each of the outputs
        >>> x = torch.randn([5])
        >>> f = lambda x: (x.sin(), x.cos())
        >>> (_, vjpfunc) = functorch.vjp(f, x)
        >>> vjps = vjpfunc((torch.ones([5]), torch.ones([5])))
        >>> assert torch.allclose(vjps[0], x.cos() + -x.sin())

    :func:`vjp` can even support outputs being Python structs
        >>> x = torch.randn([5])
        >>> f = lambda x: {'first': x.sin(), 'second': x.cos()}
        >>> (_, vjpfunc) = functorch.vjp(f, x)
        >>> cotangents = {'first': torch.ones([5]), 'second': torch.ones([5])}
        >>> vjps = vjpfunc((cotangents,))
        >>> assert torch.allclose(vjps[0], x.cos() + -x.sin())

    The function returned by :func:`vjp` will compute the partials with
    respect to each of the :attr:`primals`
        >>> x, y = torch.randn([5, 4]), torch.randn([4, 5])
        >>> (_, vjpfunc) = functorch.vjp(torch.matmul, x, y)
        >>> cotangents = torch.randn([5, 5])
        >>> vjps = vjpfunc(cotangents)
        >>> assert len(vjps) == 2
        >>> assert torch.allclose(vjps[0], torch.matmul(cotangents, y.transpose(0, 1)))
        >>> assert torch.allclose(vjps[1], torch.matmul(x.transpose(0, 1), cotangents))

    :attr:`primals` are the positional arguments for :attr:`f`. All kwargs use their
    default value
        >>> x = torch.randn([5])
        >>> def f(x, scale=4.):
        >>>   return x * 4.
        >>>
        >>> (_, vjpfunc) = functorch.vjp(f, x)
        >>> vjps = vjpfunc(torch.ones_like(x))
        >>> assert torch.allclose(vjps[0], torch.full(x.shape, 4.))

    .. note:
        Using PyTorch ``torch.no_grad`` together with ``jacrev``.
        Case 1: Using ``torch.no_grad`` inside a function:
            >>> def f(x):
            >>>     with torch.no_grad():
            >>>         c = x ** 2
            >>>     return x - c
        In this case, ``jacrev(f)(x)`` will respect the inner ``torch.no_grad``.
        Case 2: Using ``jacrev`` inside ``torch.no_grad`` context manager:
            >>> with torch.no_grad():
            >>>     jacrev(f)(x)
        In this case, ``jacrev`` will respect the inner ``torch.no_grad``, but not the
        outer one. This is because ``jacrev`` is a "function transform": its result
        should not depend on the result of a context manager outside of ``f``.
    """
    level = _grad_increment_nesting()
    try:
        # See NOTE [grad and vjp interaction with no_grad]
        with torch.enable_grad():
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

        def wrapper(cotangents, retain_graph=True, create_graph=None):
            if create_graph is None:
                create_graph = torch.is_grad_enabled()
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


def jacrev(f: Callable, argnums: Union[int, Tuple[int]] = 0):
    """
    Computes the Jacobian of :attr:`f` with respect to the arg(s) at index :attr:`argnum` using reverse
    mode autodiff

    Args:
        func (function): A Python function that takes one or more arguments, one of which must be a
            Tensor, and returns one or more Tensors
        argnums: Optional, integer or tuple of integers, saying which arguments to get the Jacobian with
            respect to. Default: 0.

    Returns:
        Returns a function that takes in the same inputs as :attr:`f` and returns the Jacobian of :attr:`f` with 
        respect to the arg(s) at :attr:`argnums`

    A basic usage with a pointwise, unary operation will give a diagonal array as the Jacobian
        >>> from functorch import jacrev
        >>> x = torch.randn(5)
        >>> jacobian = jacrev(torch.sin)(x)
        >>> expected = torch.diag(torch.cos(x))
        >>> assert torch.allclose(jacobian, expected)

    :func:`jacrev` can be composed with vmap to produce batched Jacobians:
        >>> from functorch import jacrev
        >>> x = torch.randn(64, 5)
        >>> jacobian = vmap(jacrev(torch.sin))(x)
        >>> assert jacobian.shape == (64, 5, 5)
    
    Additionally, :func:`jacrev` can be composed with itself to produce Hessians
        >>> def f(x):
        >>>   return x.sin().sum()
        >>>
        >>> x = torch.randn(5)
        >>> hessian = jacrev(jacrev(f))(x)
        >>> assert torch.allclose(hessian, torch.diag(-x.sin()))
    
    By default, :func:`jacrev` computes the Jacobian with respect to the first input. However, it can compute the
    Jacboian with respect to a different argument by using :attr:`argnums`:
        >>> from functorch import jacrev
        >>> def f(x, y):
        >>>   return x + y ** 2
        >>>
        >>> x, y = torch.randn(5), torch.randn(5)
        >>> jacobian = jacrev(f, argnums=1)(x, y)
        >>> expected = torch.diag(2 * y)
        >>> assert torch.allclose(jacobian, expected)
    
    Additionally, passing a tuple to :attr:`argnums` will compute the Jacobian with respect to multiple arguments
        >>> from functorch import jacrev
        >>> def f(x, y):
        >>>   return x + y ** 2
        >>>
        >>> x, y = torch.randn(5), torch.randn(5)
        >>> jacobian = jacrev(f, argnums=(0,1))(x, y)
        >>> expectedX = torch.diag(torch.ones_like(x))
        >>> expectedY = torch.diag(2 * y)
        >>> assert torch.allclose(jacobian[0], expectedX)
        >>> assert torch.allclose(jacobian[1], expectedY)

    .. note:
        Using PyTorch ``torch.no_grad`` together with ``jacrev``.
        Case 1: Using ``torch.no_grad`` inside a function:
            >>> def f(x):
            >>>     with torch.no_grad():
            >>>         c = x ** 2
            >>>     return x - c
        In this case, ``jacrev(f)(x)`` will respect the inner ``torch.no_grad``.
        Case 2: Using ``jacrev`` inside ``torch.no_grad`` context manager:
            >>> with torch.no_grad():
            >>>     jacrev(f)(x)
        In this case, ``jacrev`` will respect the inner ``torch.no_grad``, but not the
        outer one. This is because ``jacrev`` is a "function transform": its result
        should not depend on the result of a context manager outside of ``f``.
    """
    @wraps(f)
    def wrapper_fn(*args):
        f_wrapper, primals = _argnums_partial(f, args, argnums)
        output, vjp_fn = vjp(f_wrapper, *primals)
        assert isinstance(output, torch.Tensor)
        # TODO: does jacrev compose with vmap...? the eye call should make it so that it doesn't
        basis = torch.eye(output.numel(), dtype=output.dtype, device=output.device) \
                     .view(output.numel(), *output.shape)
        results = vmap(vjp_fn)(basis)
        results = tuple(r.view(*output.shape, *p.shape) for (r, p) in zip(results, primals))
        return results if len(results) > 1 else results[0]
    return wrapper_fn

def _check_unique_non_empty(argnums):
    if isinstance(argnums, tuple):
        if len(argnums) == 0:
            raise RuntimeError("argnums must be non-empty")
        if len(set(argnums)) != len(argnums):
            raise RuntimeError(f"argnums elements must be unique, got {argnums}")

def _replace_args(old_args, new_args, argnums):
    if isinstance(argnums, int):
        if len(new_args) == 1:
            return tuple(new_args[0] if i == argnums else old_args[i] for i in range(len(old_args)))
        else:
            raise RuntimeError(f'new_args should be of size 1, was of size {len(new_args)}')
    if isinstance(argnums, tuple):
        if len(new_args) == len(argnums):
            get_right_elem = lambda i : new_args[argnums.index(i)] if i in argnums else old_args[i]
            return tuple(get_right_elem(i) for i in range(len(old_args)))
        else:
            raise RuntimeError("new_args should have the same size as argnums. "
            f"Argnums size {len(argnums)}, new_args size {len(new_args)}")
    raise RuntimeError(f'argnums must be int or Tuple[int, ...], got: {type(argnums)}')

def _safe_index(args, argnum):
    if not isinstance(argnum, int):
        raise RuntimeError(f'argnum must be int, got: {type(argnum)}')
    if argnum >= 0 and argnum < len(args):
        return args[argnum]
    raise RuntimeError(f'Got argnum={argnum}, but only {len(args)} positional inputs')

def _slice_argnums(args, argnums):
    _check_unique_non_empty(argnums)
    if isinstance(argnums, int):
        return _safe_index(args, argnums)
    if isinstance(argnums, tuple):
        return tuple(_safe_index(args, i) for i in argnums)
    raise RuntimeError(f'argnums must be int or Tuple[int, ...], got: {type(argnums)}')

def _argnums_partial(f, args, argnums):
    def f_wrapper(*wrapper_args):
        replaced_args = _replace_args(args, wrapper_args, argnums)
        return f(*replaced_args)
    wrapper_args = _slice_argnums(args, argnums)
    wrapper_args = wrapper_args if isinstance(wrapper_args, tuple) else (wrapper_args, )
    return (f_wrapper, wrapper_args)

JVP_NESTING = 0

@contextlib.contextmanager
def noop():
    yield

def jvp(f, primals, tangents):
    level = _grad_increment_nesting()
    try:
        # Some interesting notes:
        # 1. Can't nested jvp of jvp due to forwardAD restrictions
        # 2. Seems like we can indeed vmap over this, given some more batch rules
        # 3. PyTorch doesn't have a lot of jvp rules implemented right now.
        global JVP_NESTING
        JVP_NESTING += 1
        ctx = fwAD.dual_level if JVP_NESTING == 1 else noop
        with ctx():
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
        JVP_NESTING -= 1

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
            # See NOTE [grad and vjp interaction with no_grad]
            with torch.enable_grad():
                args = _wrap_all_tensors(args, level)
                kwargs = _wrap_all_tensors(kwargs, level)
                diff_args = _slice_argnums(args, argnums)
                tree_map_(partial(_create_differentiable, level=level), diff_args)

                output = f(*args, **kwargs)
                if has_aux:
                    output, aux = output

                if not isinstance(output, torch.Tensor):
                    raise RuntimeError('grad_and_value(f)(*args): Expected f(*args) '
                                       f'to return a Tensor, got {type(output)}')
                if output.dim() != 0:
                    raise RuntimeError('grad_and_value(f)(*args): Expected f(*args) '
                                       'to return a scalar Tensor, got tensor with '
                                       f'{output.dim()} dims. Maybe you wanted to '
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
