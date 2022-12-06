# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Union, Tuple, List, Any, Optional
import torch
from functools import partial, wraps
import contextlib
from torch.utils._pytree import tree_flatten, tree_unflatten, tree_map
from .pytree_hacks import tree_map_, treespec_pprint
import torch.autograd.forward_ad as fwAD

from .vmap import vmap, doesnt_support_saved_tensors_hooks
from torch._decomp import decomposition_table

from torch._C._functorch import (
    _wrap_for_grad,
    _unwrap_for_grad,
    _grad_increment_nesting,
    _grad_decrement_nesting,
    _jvp_increment_nesting,
    _jvp_decrement_nesting,
    set_fwd_grad_enabled,
    get_fwd_grad_enabled,
    _wrap_functional_tensor,
    _unwrap_functional_tensor,
    _func_decrement_nesting,
    _func_increment_nesting,
    _assert_wrapped_functional,
    _propagate_functional_input_mutation,
    set_inplace_requires_grad_allowed,
    get_inplace_requires_grad_allowed
)


argnums_t = Union[int, Tuple[int, ...]]


@contextlib.contextmanager
def enable_inplace_requires_grad(enabled=True):
    prev_state = get_inplace_requires_grad_allowed()
    set_inplace_requires_grad_allowed(enabled)
    try:
        yield
    finally:
        set_inplace_requires_grad_allowed(prev_state)


def _create_differentiable(inps, level=None):
    def create_differentiable(x):
        if isinstance(x, torch.Tensor):
            with enable_inplace_requires_grad():
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
# NB: forward-mode AD: forward-mode AD doesn't respect torch.no_grad, but
# it respects c10::AutoFwGradMode. We've implemented the same logic for
# our jvp transform (it will have special handling if FwGradMode is disabled).


# How do we increment and decrement the nesting? I don't think we can.
def vjp(func: Callable, *primals, has_aux: bool = False):
    return _vjp_with_argnums(func, *primals, has_aux=has_aux)


@doesnt_support_saved_tensors_hooks
def _vjp_with_argnums(func: Callable, *primals, argnums: Optional[argnums_t] = None, has_aux: bool = False):
    # This is the same function as vjp but also accepts an argnums argument
    # All args are the same as vjp except for the added argument
    # argnums (Optional[int or tuple[int]]): Optional, specifies the argument(s) to compute gradients with respect to.
    #         If None, computes the gradients with respect to all inputs (used for vjp). Default: None
    #
    # WARN: Users should NOT call this function directly and should just be calling vjp.
    # It is only separated so that inputs passed to jacrev but not differentiated get the correct wrappers.
    #
    # NOTE: All error messages are produced as if vjp was being called, even if this was called by jacrev
    #
    # Returns the same two elements as :func:`vjp` but the function returned, vjp_fn, returns a tuple of VJPs
    # for only the primal elements given by argnums.
    level = _grad_increment_nesting()
    try:
        # See NOTE [grad and vjp interaction with no_grad]
        with torch.enable_grad():
            primals = _wrap_all_tensors(primals, level)
            if argnums is None:
                diff_primals = _create_differentiable(primals, level)
            else:
                diff_primals = _slice_argnums(primals, argnums, as_tuple=False)
                tree_map_(partial(_create_differentiable, level=level), diff_primals)
            primals_out = func(*primals)

            if has_aux:
                if not (isinstance(primals_out, tuple) and len(primals_out) == 2):
                    raise RuntimeError(
                        "vjp(f, *primals): output of function f should be a tuple: (output, aux) "
                        "if has_aux is True"
                    )
                primals_out, aux = primals_out
                aux = _undo_create_differentiable(aux, level)

            flat_primals_out, primals_out_spec = tree_flatten(primals_out)
            assert_non_empty_tensor_output(flat_primals_out, 'vjp(f, *primals)')
            flat_diff_primals, primals_spec = tree_flatten(diff_primals)
            results = _undo_create_differentiable(primals_out, level)

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

    if has_aux:
        return results, wrapper, aux
    else:
        return results, wrapper


def _safe_zero_index(x):
    assert len(x) == 1
    return x[0]


def jacrev(func: Callable, argnums: Union[int, Tuple[int]] = 0, *, has_aux=False):
    @wraps(func)
    def wrapper_fn(*args):
        vjp_out = _vjp_with_argnums(func, *args, argnums=argnums, has_aux=has_aux)
        if has_aux:
            output, vjp_fn, aux = vjp_out
        else:
            output, vjp_fn = vjp_out

        # See NOTE: [Computing jacobian with vmap and vjp for multiple outputs]
        flat_output, output_spec = tree_flatten(output)

        # NB: vjp already checks that all outputs are tensors
        # Step 1: Construct grad_outputs by splitting the standard basis
        flat_output_numels = tuple(out.numel() for out in flat_output)
        flat_basis = _construct_standard_basis_for(flat_output, flat_output_numels)
        basis = tree_unflatten(flat_basis, output_spec)

        results = vmap(vjp_fn)(basis)

        primals = _slice_argnums(args, argnums)
        flat_primals, primals_spec = tree_flatten(primals)
        flat_results, results_spec = tree_flatten(results)

        # Step 2: The returned jacobian is one big tensor per input. In this step,
        # we split each Tensor by output.
        flat_results = [result.split(flat_output_numels, dim=0) for result in flat_results]
        flat_input_flat_output = [
            tuple(split.view(out.shape + primal.shape)
                  for split, out in zip(splits, flat_output))
            for splits, primal in zip(flat_results, flat_primals)
        ]

        # Step 3: Right now, `jacobian` is a List[List[Tensor]].
        # The outer List corresponds to the number of primals,
        # the inner List corresponds to the number of outputs.
        # We need to:
        # a. Exchange the order of the outer List and inner List
        # b. tree_unflatten the inner Lists (which correspond to the primals)
        # c. handle the argnums=int case
        # d. tree_unflatten the outer List (which corresponds to the outputs)
        flat_output_flat_input = tuple(zip(*flat_input_flat_output))

        flat_output_input = tuple(tree_unflatten(flat_input, primals_spec)
                                  for flat_input in flat_output_flat_input)

        if isinstance(argnums, int):
            flat_output_input = tuple(_safe_zero_index(flat_input)
                                      for flat_input in flat_output_input)
        output_input = tree_unflatten(flat_output_input, output_spec)
        if has_aux:
            return output_input, aux
        return output_input
    return wrapper_fn

# NOTE: [Computing jacobian with vmap and vjp for multiple outputs]
#
# Let's consider f(x) = (x**2, x.sum()) and let x = torch.randn(3).
# It turns out we can compute the jacobian of this function with a single
# call to autograd.grad by using vmap over the correct grad_outputs.
#
# Firstly, one way to compute the jacobian is to stack x**2 and x.sum()
# into a 4D vector. E.g., use g(x) = torch.stack([x**2, x.sum()])
#
# To get the first row of the jacobian, we call
# >>> autograd.grad(g(x), x, grad_outputs=torch.tensor([1, 0, 0, 0]))
# To get the 2nd row of the jacobian, we call
# >>> autograd.grad(g(x), x, grad_outputs=torch.tensor([0, 1, 0, 0]))
# and so on.
#
# Using vmap, we can vectorize all 4 of these computations into one by
# passing the standard basis for R^4 as the grad_output.
# vmap(partial(autograd.grad, g(x), x))(torch.eye(4)).
#
# Now, how do we compute the jacobian *without stacking the output*?
# We can just split the standard basis across the outputs. So to
# compute the jacobian of f(x), we'd use
# >>> autograd.grad(f(x), x, grad_outputs=_construct_standard_basis_for(...))
# The grad_outputs looks like the following:
# ( torch.tensor([[1, 0, 0],
#                 [0, 1, 0],
#                 [0, 0, 1],
#                 [0, 0, 0]]),
#   torch.tensor([[0],
#                 [0],
#                 [0],
#                 [1]]) )
#
# But we're not done yet!
# >>> vmap(partial(autograd.grad(f(x), x, grad_outputs=...)))
# returns a Tensor of shape [4, 3]. We have to remember to split the
# jacobian of shape [4, 3] into two:
# - one of shape [3, 3] for the first output
# - one of shape [   3] for the second output


def _construct_standard_basis_for(tensors, tensor_numels):
    # This function:
    # - constructs a N=sum(tensor_numels) standard basis. i.e. an NxN identity matrix.
    # - Splits the identity matrix into chunks with each chunk size determined by `tensor_numels`.
    # - Each chunk corresponds to one tensor. The chunk has the same dtype and
    #   device as the tensor
    #
    # For example, with tensor_numels = [1, 2, 1], this function returns:
    # ( tensor([[1],     tensor([[0, 0],      tensor([[0],
    #           [0],             [1, 0],              [0],
    #           [0],             [0, 1],              [0],
    #           [0]])  ,         [0, 0]])  ,          [1]])  )
    #
    # Precondition: tensor_numels == tuple(tensor.numel() for tensor in tensors)
    # Precondition: tensors always has at least one element.
    #
    # See NOTE: [Computing jacobian with vmap and grad for multiple tensors]
    # for context behind this function.
    assert len(tensors) == len(tensor_numels)
    assert len(tensors) > 0
    total_numel = sum(tensor_numels)
    diag_start_indices = (0, *torch.tensor(tensor_numels).cumsum(dim=0)[:-1].neg().unbind())
    chunks = tuple(tensor.new_zeros(total_numel, tensor_numel)
                   for tensor, tensor_numel in zip(tensors, tensor_numels))
    for chunk, diag_start_idx in zip(chunks, diag_start_indices):
        chunk.diagonal(diag_start_idx).fill_(1)
    chunks = tuple(chunk.view(total_numel, *tensor.shape)
                   for chunk, tensor in zip(chunks, tensors))
    return chunks


def _validate_and_wrap_argnum(argnum, num_args):
    if not isinstance(argnum, int):
        raise RuntimeError(f'argnum must be int, got: {type(argnum)}')
    if argnum >= 0 and argnum < num_args:
        return argnum
    if argnum < 0 and argnum >= -num_args:
        return argnum + num_args
    raise RuntimeError(f'Got argnum={argnum}, but only {num_args} positional inputs')


def _check_unique_non_empty(argnums):
    if isinstance(argnums, tuple):
        if len(argnums) == 0:
            raise RuntimeError("argnums must be non-empty")
        if len(set(argnums)) != len(argnums):
            raise RuntimeError(f"argnums elements must be unique, got {argnums}")


def _replace_args(old_args, new_args, argnums):
    if isinstance(argnums, int):
        if len(new_args) != 1:
            raise RuntimeError(f'new_args should be of size 1, was of size {len(new_args)}')
        return tuple(new_args[0] if i == argnums else old_args[i] for i in range(len(old_args)))
    if isinstance(argnums, tuple):
        if len(new_args) != len(argnums):
            raise RuntimeError(
                "new_args should have the same size as argnums. "
                f"Argnums size {len(argnums)}, new_args size {len(new_args)}")

        def get_right_elem(i):
            return new_args[argnums.index(i)] if i in argnums else old_args[i]

        return tuple(get_right_elem(i) for i in range(len(old_args)))
    raise RuntimeError(f'argnums must be int or Tuple[int, ...], got: {type(argnums)}')


def _validate_and_wrap_argnums(argnums, num_args):
    if isinstance(argnums, int):
        return _validate_and_wrap_argnum(argnums, num_args)
    if isinstance(argnums, tuple):
        return tuple(_validate_and_wrap_argnum(argnum, num_args) for argnum in argnums)
    raise AssertionError("Should never get here")


def _slice_argnums(args, argnums, as_tuple=True):
    if not isinstance(argnums, int) and not isinstance(argnums, tuple):
        raise RuntimeError(f'argnums must be int or Tuple[int, ...], got: {type(argnums)}')
    argnums = _validate_and_wrap_argnums(argnums, len(args))
    _check_unique_non_empty(argnums)
    if isinstance(argnums, int):
        if as_tuple:
            return (args[argnums],)
        else:
            return args[argnums]
    return tuple(args[i] for i in argnums)


JVP_NESTING = 0


@contextlib.contextmanager
def noop():
    yield


@contextlib.contextmanager
def enable_fwd_grad(enabled=True):
    prev_state = get_fwd_grad_enabled()
    set_fwd_grad_enabled(enabled)
    try:
        yield
    finally:
        set_fwd_grad_enabled(prev_state)


def assert_flat_tuple_of_tensors(elts: Any, api: str, argname: str) -> None:
    if not isinstance(elts, tuple):
        raise RuntimeError(
            f'{api}: Expected {argname} to be a tuple of Tensors, got {type(elts)}')
    for elt in elts:
        if isinstance(elt, torch.Tensor):
            continue
        raise RuntimeError(
            f'{api}: Expected {argname} to be a tuple of Tensors, got '
            f'a tuple with an element of type {type(elt)}')
    if len(elts) == 0:
        raise RuntimeError(
            f'{api}: Expected {argname} to be a non-empty tuple of Tensors.')


def assert_non_empty_tensor_output(output: List[Any], api: str) -> None:
    if output == [None] or len(output) < 1:
        raise RuntimeError(
            f'{api}: Expected f to be a function that has non-empty output (got output = {output})'
        )
    for o in output:
        if not isinstance(o, torch.Tensor):
            raise RuntimeError(
                f'{api}: expected f(*primals) to return only tensors'
                f', got unsupported type {type(o)}'
            )


def assert_output_is_tensor_or_tensors(output: Any, api: str) -> None:
    if isinstance(output, torch.Tensor):
        return
    if not isinstance(output, tuple):
        raise RuntimeError(
            f'{api}: Expected output of f to be a Tensor or Tensors, got '
            f'{type(output)}')
    if len(output) == 0:
        raise RuntimeError(
            f'{api}: Expected output of f to be a non-empty tuple of Tensors.')
    for out in output:
        if isinstance(out, torch.Tensor):
            continue
        raise RuntimeError(
            f'{api}: Expected output of f to be a Tensor or Tensors, got '
            f'{type(out)} as an output')


def assert_non_empty_list_of_tensors(output: List[torch.Tensor], api: str, argname: str) -> None:
    if len(output) == 0:
        raise RuntimeError(
            f'{api}: Expected {argname} to contain at least one Tensor.')
    for out in output:
        if isinstance(out, torch.Tensor):
            continue
        raise RuntimeError(
            f'{api}: Expected {argname} to only contain Tensors, got '
            f'{type(out)}')


jvp_str = 'jvp(f, primals, tangents)'


def safe_unpack_dual(dual, strict):
    if not isinstance(dual, torch.Tensor):
        raise RuntimeError(
            f'{jvp_str}: expected f(*args) to return only tensors'
            f', got unsupported type {type(dual)}'
        )

    primal, tangent = fwAD.unpack_dual(dual)
    if tangent is None:
        if strict:
            raise RuntimeError(
                'jvp(f, primals, tangents, strict=True): '
                'The output of f is independent of '
                'the inputs. This is not allowed with strict=True.')
        tangent = torch.zeros_like(primal)
    return primal, tangent


def jvp(func: Callable, primals: Any, tangents: Any, *, strict: bool = False, has_aux: bool = False):
    return _jvp_with_argnums(func, primals, tangents, argnums=None, strict=strict, has_aux=has_aux)


@doesnt_support_saved_tensors_hooks
def _jvp_with_argnums(func: Callable, primals: Any, tangents: Any, argnums: Optional[argnums_t], *,
                      strict: bool = False, has_aux: bool):
    # This is the same function as jvp but also accepts an argnums argument
    # Most args are the same as jvp except for the added argument
    # argnums (Optional[int or tuple[int]]): Optional, specifies the argument(s) to compute gradients with respect to.
    #         If None, computes the gradients with respect to all inputs (used for jvp). Default: None
    # Because of this, tangents must be of length argnums and matches up to the corresponding primal whose index is
    # given by argnums
    #
    # WARN: Users should NOT call this function directly and should just be calling jvp.
    # It is only separated so that inputs passed to jacfwd but not differentiated get the correct wrappers.
    #
    # NOTE: All error messages are produced as if jvp was being called, even if this was called by jacfwd
    #
    # Returns the same two elements as :func:`jvp` but the returned tuple, ``jvp_out``, only has JVPs with respect to
    # the primals given by argnums
    if not isinstance(primals, tuple):
        raise RuntimeError(
            f'{jvp_str}: Expected primals to be a tuple. '
            f'E.g. it should be valid to call f(*primals).')
    diff_args = primals if argnums is None else _slice_argnums(primals, argnums)
    flat_primals, primals_spec = tree_flatten(diff_args)
    flat_tangents, tangents_spec = tree_flatten(tangents)
    if primals_spec != tangents_spec:
        raise RuntimeError(
            f'{jvp_str}: Expected primals and tangents to have the same python '
            f'structure. For example, if primals is a tuple of 3 tensors, '
            f'tangents also must be. Got primals with structure {primals_spec} '
            f'and tangents with structure {tangents_spec}')
    assert_non_empty_list_of_tensors(flat_primals, jvp_str, 'primals')
    assert_non_empty_list_of_tensors(flat_tangents, jvp_str, 'tangents')

    level = _jvp_increment_nesting()
    try:
        global JVP_NESTING
        JVP_NESTING += 1
        with enable_fwd_grad():
            ctx = fwAD.dual_level if JVP_NESTING == 1 else noop
            with ctx():
                flat_duals = tuple(fwAD.make_dual(p, t)
                                   for p, t in zip(flat_primals, flat_tangents))
                duals = tree_unflatten(flat_duals, primals_spec)
                if argnums is not None:
                    primals = _wrap_all_tensors(primals, level)
                    duals = _replace_args(primals, duals, argnums)
                result_duals = func(*duals)
                if has_aux:
                    if not (isinstance(result_duals, tuple) and len(result_duals) == 2):
                        raise RuntimeError(
                            f"{jvp_str}: output of function f should be a tuple: (output, aux) "
                            "if has_aux is True"
                        )
                    result_duals, aux = result_duals
                    aux = _undo_create_differentiable(aux, level)

                result_duals, spec = tree_flatten(result_duals)
                assert_non_empty_tensor_output(result_duals, jvp_str)

                primals_out, tangents_out = \
                    zip(*[safe_unpack_dual(dual, strict) for dual in result_duals])
                primals_out = tree_map(
                    partial(_undo_create_differentiable, level=level), primals_out)
                tangents_out = tree_map(
                    partial(_undo_create_differentiable, level=level), tangents_out)

                primals_out_unflatten = tree_unflatten(primals_out, spec)
                tangents_out_unflatten = tree_unflatten(tangents_out, spec)
                if has_aux:
                    return primals_out_unflatten, tangents_out_unflatten, aux

                return primals_out_unflatten, tangents_out_unflatten
    finally:
        _jvp_decrement_nesting()
        JVP_NESTING -= 1


def safe_unflatten(tensor, dim, shape):
    if len(shape) == 0:
        assert tensor.shape[dim] == 1
        return tensor.squeeze(dim)
    return tensor.unflatten(dim, shape)


def jacfwd(func: Callable, argnums: argnums_t = 0, has_aux: bool = False, *, randomness: str = "error"):
    @wraps(func)
    def wrapper_fn(*args):
        primals = args if argnums is None else _slice_argnums(args, argnums)
        flat_primals, primals_spec = tree_flatten(primals)
        flat_primals_numels = tuple(p.numel() for p in flat_primals)
        flat_basis = _construct_standard_basis_for(flat_primals, flat_primals_numels)
        basis = tree_unflatten(flat_basis, primals_spec)

        def push_jvp(basis):
            output = _jvp_with_argnums(func, args, basis, argnums=argnums, has_aux=has_aux)
            if has_aux:
                _, jvp_out, aux = output
                return jvp_out, aux
            _, jvp_out = output
            return jvp_out

        results = vmap(push_jvp, randomness=randomness)(basis)
        if has_aux:
            results, aux = results
            # aux is in the standard basis format, e.g. NxN matrix
            # We need to fetch the first element as original `func` output
            flat_aux, aux_spec = tree_flatten(aux)
            flat_aux = [value[0] for value in flat_aux]
            aux = tree_unflatten(flat_aux, aux_spec)

        jac_outs, spec = tree_flatten(results)
        # Most probably below output check can never raise an error
        # as jvp should test the output before
        # assert_non_empty_output(jac_outs, 'jacfwd(f, ...)(*args)')

        jac_outs_ins = tuple(
            tuple(
                safe_unflatten(jac_out_in, -1, primal.shape)
                for primal, jac_out_in in
                zip(flat_primals, jac_out.movedim(0, -1).split(flat_primals_numels, dim=-1))
            )
            for jac_out in jac_outs
        )
        jac_outs_ins = tuple(tree_unflatten(jac_ins, primals_spec) for jac_ins in jac_outs_ins)

        if isinstance(argnums, int):
            jac_outs_ins = tuple(jac_ins[0] for jac_ins in jac_outs_ins)
        if has_aux:
            return tree_unflatten(jac_outs_ins, spec), aux
        return tree_unflatten(jac_outs_ins, spec)
    return wrapper_fn


def grad_and_value(func: Callable, argnums: argnums_t = 0, has_aux: bool = False) -> Callable:
    @doesnt_support_saved_tensors_hooks
    @wraps(func)
    def wrapper(*args, **kwargs):
        level = _grad_increment_nesting()
        try:
            output, aux, grad_input = None, None, None
            # See NOTE [grad and vjp interaction with no_grad]
            with torch.enable_grad():
                args = _wrap_all_tensors(args, level)
                kwargs = _wrap_all_tensors(kwargs, level)
                diff_args = _slice_argnums(args, argnums, as_tuple=False)
                tree_map_(partial(_create_differentiable, level=level), diff_args)

                output = func(*args, **kwargs)
                if has_aux:
                    if not (isinstance(output, tuple) and len(output) == 2):
                        raise RuntimeError(
                            "grad_and_value(f)(*args): output of function f should be a tuple: (output, aux) "
                            "if has_aux is True"
                        )
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

                grad_input = _undo_create_differentiable(grad_input, level)
                output = _undo_create_differentiable(output, level)
                if aux is not None:
                    aux = _undo_create_differentiable(aux, level)

            if has_aux:
                return grad_input, (output, aux)
            return grad_input, output
        finally:
            _grad_decrement_nesting()
    return wrapper


def grad(func: Callable, argnums: argnums_t = 0, has_aux: bool = False) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        results = grad_and_value(func, argnums, has_aux=has_aux)(*args, **kwargs)
        if has_aux:
            grad, (_, aux) = results
            return grad, aux
        grad, _ = results
        return grad
    return wrapper


def _maybe_wrap_functional_tensor(maybe_tensor, level):
    if not isinstance(maybe_tensor, torch.Tensor):
        return maybe_tensor
    wrapped = _wrap_functional_tensor(maybe_tensor, level)
    _assert_wrapped_functional(maybe_tensor, wrapped)
    return wrapped


def _wrap_all_tensors_to_functional(tensor_pytree, level):
    return tree_map(partial(_maybe_wrap_functional_tensor, level=level), tensor_pytree)


def _maybe_unwrap_functional_tensor(maybe_tensor, *, reapply_views: bool):
    if not isinstance(maybe_tensor, torch.Tensor):
        return maybe_tensor
    if not torch._is_functional_tensor(maybe_tensor):
        # If it's not a functional tensor, just return it.
        # This can happen if we functionalize a fn that returns a global,
        # which was never wrapped properly.
        return maybe_tensor
    return _unwrap_functional_tensor(maybe_tensor, reapply_views)


def _unwrap_all_tensors_from_functional(tensor_pytree, *, reapply_views: bool):
    return tree_map(lambda t: _maybe_unwrap_functional_tensor(t, reapply_views=reapply_views), tensor_pytree)


def functionalize(func: Callable, *, remove: str = 'mutations') -> Callable:
    if remove == 'mutations':
        reapply_views = True
    elif remove == 'mutations_and_views':
        reapply_views = False
    else:
        raise RuntimeError(
            f"functionalize(f, remove='mutations'): received invalid argument for remove={remove}."
            " Valid options are:\n"
            "     remove='mutations': all inplace and out= operators will be removed from the program, and replaced"
            " with their out-of-place equivalents.\n"
            "     remove='mutations_and_views': In addition to the above, all aliasing operators {view} will be"
            " replaced with their non-aliasing counterparts, {view}_copy.\n"
        )

    @doesnt_support_saved_tensors_hooks
    @wraps(func)
    def wrapped(*args, **kwargs):
        try:
            func_level = _func_increment_nesting(reapply_views)
            func_args = _wrap_all_tensors_to_functional(args, func_level)
            func_kwargs = _wrap_all_tensors_to_functional(kwargs, func_level)

            flattened_unwrapped_args, _ = tree_flatten(args)
            flattened_wrapped_args, _ = tree_flatten(func_args)
            flattened_unwrapped_kwargs, _ = tree_flatten(kwargs)
            flattened_wrapped_kwargs, _ = tree_flatten(func_kwargs)

            func_outputs = func(*func_args, **func_kwargs)
            outputs = _unwrap_all_tensors_from_functional(func_outputs, reapply_views=reapply_views)
            flat_outputs, func_out_spec = tree_flatten(outputs)

            for a in flattened_wrapped_args + flattened_wrapped_kwargs:
                if isinstance(a, torch.Tensor):
                    # Call sync_() on the inputs, to ensure that any pending mutations have been applied.
                    torch._sync(a)

            # And if any mutations were applied to the inputs, we need to propagate them back to the user.
            for unwrapped, wrapped in zip(flattened_unwrapped_args, flattened_wrapped_args):
                if isinstance(unwrapped, torch.Tensor) and isinstance(wrapped, torch.Tensor):
                    _propagate_functional_input_mutation(unwrapped, wrapped)
            for unwrapped, wrapped in zip(flattened_unwrapped_kwargs, flattened_wrapped_kwargs):
                if isinstance(unwrapped, torch.Tensor) and isinstance(wrapped, torch.Tensor):
                    _propagate_functional_input_mutation(unwrapped, wrapped)

            return outputs
        finally:
            _func_decrement_nesting()
    return wrapped

# use an alternate way to register an operator into the decomposition table
# _register_jit_decomposition doesn't work for some operators, e.g. addr,
#  because the Tensor types generated cannot be unioned by torchscript
# decomp should be type OpOverload
vmap_decompositions_lib = torch.library.Library("aten", "IMPL", "FuncTorchBatched")


def _register_python_decomposition_vmap(decomp):
    if decomp in decomposition_table:
        vmap_decompositions_lib.impl(decomp, decomposition_table[decomp])
    else:
        raise RuntimeError(f"could not find decomposition for {decomp}")


_register_python_decomposition_vmap(torch.ops.aten.mse_loss_backward.default)
_register_python_decomposition_vmap(torch.ops.aten.addr.default)
