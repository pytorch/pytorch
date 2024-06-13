# mypy: ignore-errors

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
from functools import partial, wraps
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
import torch.autograd.forward_ad as fwAD

from torch._C._functorch import (
    _assert_wrapped_functional,
    _func_decrement_nesting,
    _func_increment_nesting,
    _grad_decrement_nesting,
    _grad_increment_nesting,
    _jvp_decrement_nesting,
    _jvp_increment_nesting,
    _propagate_functional_input_mutation,
    _unwrap_for_grad,
    _unwrap_functional_tensor,
    _wrap_for_grad,
    _wrap_functional_tensor,
    get_inplace_requires_grad_allowed,
    set_inplace_requires_grad_allowed,
)
from torch._functorch.utils import argnums_t, exposed_in
from torch._subclasses.functional_tensor import FunctionalTensor
from torch.fx.experimental import const_fold
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils import _pytree as pytree
from torch.utils._pytree import (
    tree_flatten,
    tree_map,
    tree_map_,
    tree_map_only,
    tree_unflatten,
    treespec_pprint,
)
from .apis import vmap

from .vmap import doesnt_support_saved_tensors_hooks, get_chunk_sizes


def lazy_dynamo_disallow(func):
    import torch._dynamo

    return torch._dynamo.disallow_in_graph(func)


@contextlib.contextmanager
def enable_inplace_requires_grad(enabled):
    prev_state = get_inplace_requires_grad_allowed()
    set_inplace_requires_grad_allowed(enabled)
    try:
        yield
    finally:
        set_inplace_requires_grad_allowed(prev_state)


def _vjp_treespec_compare(primals_out, cotangents):
    # Revert this once #116264 gets fixed
    _, primals_out_spec = tree_flatten(primals_out)
    _, cotangents_spec = tree_flatten(cotangents)
    # Dynamo fails to trace operator.ne below. To bypass this limitation, this
    # function is not inlined.
    if primals_out_spec != cotangents_spec:
        raise RuntimeError(
            f"Expected pytree structure of cotangents to be the same "
            f"as pytree structure of outputs to the function. "
            f"cotangents: {treespec_pprint(cotangents_spec)}, "
            f"primal output: {treespec_pprint(primals_out_spec)}"
        )


def _jvp_treespec_compare(primals, tangents):
    # Revert this once #116264 gets fixed
    _, primals_spec = tree_flatten(primals)
    _, tangents_spec = tree_flatten(tangents)
    if primals_spec != tangents_spec:
        raise RuntimeError(
            f"{jvp_str}: Expected primals and tangents to have the same python "
            f"structure. For example, if primals is a tuple of 3 tensors, "
            f"tangents also must be. Got primals with structure {primals_spec} "
            f"and tangents with structure {tangents_spec}"
        )


def _linearize_treespec_compare(primals, tangents):
    # Revert this once #116264 gets fixed
    _, primals_argspec = tree_flatten(primals)
    _, tangent_argspec = tree_flatten(tangents)
    if tangent_argspec != primals_argspec:
        raise RuntimeError(
            f"Expected the tangents {tangent_argspec} to have "
            f"the same argspec as the primals {primals_argspec}"
        )


def _set_tensor_requires_grad(x):
    # avoid graph-break on x.requires_grad_()
    # https://github.com/pytorch/pytorch/pull/110053
    return x.requires_grad_()


def _create_differentiable(inps, level=None):
    def create_differentiable(x):
        if isinstance(x, torch.Tensor):
            with enable_inplace_requires_grad(True):
                return _set_tensor_requires_grad(x)
        raise ValueError(
            f"Thing passed to transform API must be Tensor, " f"got {type(x)}"
        )

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


def _autograd_grad(
    outputs, inputs, grad_outputs=None, retain_graph=False, create_graph=True
):
    if grad_outputs is None:
        diff_outputs = tuple(out for out in outputs if out.requires_grad)
    else:
        result = tuple(
            (out, go) for out, go in zip(outputs, grad_outputs) if out.requires_grad
        )
        if len(result) == 0:
            diff_outputs, grad_outputs = (), ()
        else:
            diff_outputs, grad_outputs = zip(*result)
    if len(diff_outputs) == 0:
        return tuple(torch.zeros_like(inp) for inp in inputs)
    grad_inputs = torch.autograd.grad(
        diff_outputs,
        inputs,
        grad_outputs,
        retain_graph=retain_graph,
        create_graph=create_graph,
        allow_unused=True,
    )
    grad_inputs = tuple(
        torch.zeros_like(inp) if gi is None else gi
        for gi, inp in zip(grad_inputs, inputs)
    )
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
@exposed_in("torch.func")
def vjp(func: Callable, *primals, has_aux: bool = False):
    """
    Standing for the vector-Jacobian product, returns a tuple containing the
    results of ``func`` applied to ``primals`` and a function that, when
    given ``cotangents``, computes the reverse-mode Jacobian of ``func`` with
    respect to ``primals`` times ``cotangents``.

    Args:
        func (Callable): A Python function that takes one or more arguments. Must
            return one or more Tensors.
        primals (Tensors): Positional arguments to ``func`` that must all be
            Tensors. The returned function will also be computing the
            derivative with respect to these arguments
        has_aux (bool): Flag indicating that ``func`` returns a
            ``(output, aux)`` tuple where the first element is the output of
            the function to be differentiated and the second element is
            other auxiliary objects that will not be differentiated.
            Default: False.

    Returns:
        Returns a ``(output, vjp_fn)`` tuple containing the output of ``func``
        applied to ``primals`` and a function that computes the vjp of
        ``func`` with respect to all ``primals`` using the cotangents passed
        to the returned function. If ``has_aux is True``, then instead returns a
        ``(output, vjp_fn, aux)`` tuple.
        The returned ``vjp_fn`` function will return a tuple of each VJP.

    When used in simple cases, :func:`vjp` behaves the same as :func:`grad`

        >>> x = torch.randn([5])
        >>> f = lambda x: x.sin().sum()
        >>> (_, vjpfunc) = torch.func.vjp(f, x)
        >>> grad = vjpfunc(torch.tensor(1.))[0]
        >>> assert torch.allclose(grad, torch.func.grad(f)(x))

    However, :func:`vjp` can support functions with multiple outputs by
    passing in the cotangents for each of the outputs

        >>> x = torch.randn([5])
        >>> f = lambda x: (x.sin(), x.cos())
        >>> (_, vjpfunc) = torch.func.vjp(f, x)
        >>> vjps = vjpfunc((torch.ones([5]), torch.ones([5])))
        >>> assert torch.allclose(vjps[0], x.cos() + -x.sin())

    :func:`vjp` can even support outputs being Python structs

        >>> x = torch.randn([5])
        >>> f = lambda x: {'first': x.sin(), 'second': x.cos()}
        >>> (_, vjpfunc) = torch.func.vjp(f, x)
        >>> cotangents = {'first': torch.ones([5]), 'second': torch.ones([5])}
        >>> vjps = vjpfunc(cotangents)
        >>> assert torch.allclose(vjps[0], x.cos() + -x.sin())

    The function returned by :func:`vjp` will compute the partials with
    respect to each of the ``primals``

        >>> x, y = torch.randn([5, 4]), torch.randn([4, 5])
        >>> (_, vjpfunc) = torch.func.vjp(torch.matmul, x, y)
        >>> cotangents = torch.randn([5, 5])
        >>> vjps = vjpfunc(cotangents)
        >>> assert len(vjps) == 2
        >>> assert torch.allclose(vjps[0], torch.matmul(cotangents, y.transpose(0, 1)))
        >>> assert torch.allclose(vjps[1], torch.matmul(x.transpose(0, 1), cotangents))

    ``primals`` are the positional arguments for ``f``. All kwargs use their
    default value

        >>> x = torch.randn([5])
        >>> def f(x, scale=4.):
        >>>   return x * scale
        >>>
        >>> (_, vjpfunc) = torch.func.vjp(f, x)
        >>> vjps = vjpfunc(torch.ones_like(x))
        >>> assert torch.allclose(vjps[0], torch.full(x.shape, 4.))

    .. note::
        Using PyTorch ``torch.no_grad`` together with ``vjp``.
        Case 1: Using ``torch.no_grad`` inside a function:

            >>> def f(x):
            >>>     with torch.no_grad():
            >>>         c = x ** 2
            >>>     return x - c

        In this case, ``vjp(f)(x)`` will respect the inner ``torch.no_grad``.

        Case 2: Using ``vjp`` inside ``torch.no_grad`` context manager:

            >>> # xdoctest: +SKIP(failing)
            >>> with torch.no_grad():
            >>>     vjp(f)(x)

        In this case, ``vjp`` will respect the inner ``torch.no_grad``, but not the
        outer one. This is because ``vjp`` is a "function transform": its result
        should not depend on the result of a context manager outside of ``f``.
    """
    return _vjp_with_argnums(func, *primals, has_aux=has_aux)


@contextlib.contextmanager
def grad_increment_nesting():
    try:
        grad_level = _grad_increment_nesting()
        yield grad_level
    finally:
        _grad_decrement_nesting()


def enter_jvp_nesting():
    global JVP_NESTING
    jvp_level = _jvp_increment_nesting()
    JVP_NESTING += 1
    return jvp_level


def exit_jvp_nesting():
    global JVP_NESTING
    _jvp_decrement_nesting()
    JVP_NESTING -= 1


@contextlib.contextmanager
def jvp_increment_nesting():
    try:
        yield enter_jvp_nesting()
    finally:
        exit_jvp_nesting()


@doesnt_support_saved_tensors_hooks
def _vjp_with_argnums(
    func: Callable, *primals, argnums: Optional[argnums_t] = None, has_aux: bool = False
):
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
    with grad_increment_nesting() as level:
        # See NOTE [grad and vjp interaction with no_grad]
        with torch.enable_grad():
            primals = _wrap_all_tensors(primals, level)
            # Note for the reviewer: This is extremely odd but it passes the
            # assertion "len(self.block_stack) == 1" on symbolic_convert.py
            # The equivalent "if argnums is None" fails for some reason
            if not isinstance(argnums, int) and not argnums:
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
            assert_non_empty_tensor_output(flat_primals_out, "vjp(f, *primals)")
            flat_diff_primals, primals_spec = tree_flatten(diff_primals)
            results = _undo_create_differentiable(primals_out, level)

            for primal_out in flat_primals_out:
                assert isinstance(primal_out, torch.Tensor)
                if primal_out.is_floating_point() or primal_out.is_complex():
                    continue
                raise RuntimeError(
                    "vjp(f, ...): All outputs of f must be "
                    "floating-point or complex Tensors, got Tensor "
                    f"with dtype {primal_out.dtype}"
                )

        def wrapper(cotangents, retain_graph=True, create_graph=None):
            if create_graph is None:
                create_graph = torch.is_grad_enabled()
            flat_cotangents, cotangents_spec = tree_flatten(cotangents)
            _vjp_treespec_compare(primals_out, cotangents)
            result = _autograd_grad(
                flat_primals_out,
                flat_diff_primals,
                flat_cotangents,
                retain_graph=retain_graph,
                create_graph=create_graph,
            )
            return tree_unflatten(result, primals_spec)

    if has_aux:
        return results, wrapper, aux
    else:
        return results, wrapper


def _safe_zero_index(x):
    assert len(x) == 1
    return x[0]


# jacrev and jacfwd don't support complex functions
# Helper function to throw appropriate error.
def error_if_complex(func_name, args, is_input):
    flat_args = pytree.tree_leaves(args)
    for idx, arg in enumerate(flat_args):
        if isinstance(arg, torch.Tensor) and arg.dtype.is_complex:
            input_or_output = "inputs" if is_input else "outputs"
            err_msg = (
                f"{func_name}: Expected all {input_or_output} "
                f"to be real but received complex tensor at flattened input idx: {idx}"
            )
            raise RuntimeError(err_msg)


@exposed_in("torch.func")
def jacrev(
    func: Callable,
    argnums: Union[int, Tuple[int]] = 0,
    *,
    has_aux=False,
    chunk_size: Optional[int] = None,
    _preallocate_and_copy=False,
):
    """
    Computes the Jacobian of ``func`` with respect to the arg(s) at index
    ``argnum`` using reverse mode autodiff

    .. note::
        Using :attr:`chunk_size=1` is equivalent to computing the jacobian
        row-by-row with a for-loop i.e. the constraints of :func:`vmap` are
        not applicable.

    Args:
        func (function): A Python function that takes one or more arguments,
            one of which must be a Tensor, and returns one or more Tensors
        argnums (int or Tuple[int]): Optional, integer or tuple of integers,
            saying which arguments to get the Jacobian with respect to.
            Default: 0.
        has_aux (bool): Flag indicating that ``func`` returns a
            ``(output, aux)`` tuple where the first element is the output of
            the function to be differentiated and the second element is
            auxiliary objects that will not be differentiated.
            Default: False.
        chunk_size (None or int): If None (default), use the maximum chunk size
            (equivalent to doing a single vmap over vjp to compute the jacobian).
            If 1, then compute the jacobian row-by-row with a for-loop.
            If not None, then compute the jacobian :attr:`chunk_size` rows at a time
            (equivalent to doing multiple vmap over vjp). If you run into memory issues computing
            the jacobian, please try to specify a non-None chunk_size.

    Returns:
        Returns a function that takes in the same inputs as ``func`` and
        returns the Jacobian of ``func`` with respect to the arg(s) at
        ``argnums``. If ``has_aux is True``, then the returned function
        instead returns a ``(jacobian, aux)`` tuple where ``jacobian``
        is the Jacobian and ``aux`` is auxiliary objects returned by ``func``.

    A basic usage with a pointwise, unary operation will give a diagonal array
    as the Jacobian

        >>> from torch.func import jacrev
        >>> x = torch.randn(5)
        >>> jacobian = jacrev(torch.sin)(x)
        >>> expected = torch.diag(torch.cos(x))
        >>> assert torch.allclose(jacobian, expected)

    If you would like to compute the output of the function as well as the
    jacobian of the function, use the ``has_aux`` flag to return the output
    as an auxiliary object:

        >>> from torch.func import jacrev
        >>> x = torch.randn(5)
        >>>
        >>> def f(x):
        >>>   return x.sin()
        >>>
        >>> def g(x):
        >>>   result = f(x)
        >>>   return result, result
        >>>
        >>> jacobian_f, f_x = jacrev(g, has_aux=True)(x)
        >>> assert torch.allclose(f_x, f(x))

    :func:`jacrev` can be composed with vmap to produce batched
    Jacobians:

        >>> from torch.func import jacrev, vmap
        >>> x = torch.randn(64, 5)
        >>> jacobian = vmap(jacrev(torch.sin))(x)
        >>> assert jacobian.shape == (64, 5, 5)

    Additionally, :func:`jacrev` can be composed with itself to produce
    Hessians

        >>> from torch.func import jacrev
        >>> def f(x):
        >>>   return x.sin().sum()
        >>>
        >>> x = torch.randn(5)
        >>> hessian = jacrev(jacrev(f))(x)
        >>> assert torch.allclose(hessian, torch.diag(-x.sin()))

    By default, :func:`jacrev` computes the Jacobian with respect to the first
    input. However, it can compute the Jacboian with respect to a different
    argument by using ``argnums``:

        >>> from torch.func import jacrev
        >>> def f(x, y):
        >>>   return x + y ** 2
        >>>
        >>> x, y = torch.randn(5), torch.randn(5)
        >>> jacobian = jacrev(f, argnums=1)(x, y)
        >>> expected = torch.diag(2 * y)
        >>> assert torch.allclose(jacobian, expected)

    Additionally, passing a tuple to ``argnums`` will compute the Jacobian
    with respect to multiple arguments

        >>> from torch.func import jacrev
        >>> def f(x, y):
        >>>   return x + y ** 2
        >>>
        >>> x, y = torch.randn(5), torch.randn(5)
        >>> jacobian = jacrev(f, argnums=(0, 1))(x, y)
        >>> expectedX = torch.diag(torch.ones_like(x))
        >>> expectedY = torch.diag(2 * y)
        >>> assert torch.allclose(jacobian[0], expectedX)
        >>> assert torch.allclose(jacobian[1], expectedY)

    .. note::
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
    if not (chunk_size is None or chunk_size > 0):
        raise ValueError("jacrev: `chunk_size` should be greater than 0.")

    def wrapper_fn(*args):
        error_if_complex("jacrev", args, is_input=True)
        vjp_out = _vjp_with_argnums(func, *args, argnums=argnums, has_aux=has_aux)
        if has_aux:
            output, vjp_fn, aux = vjp_out
        else:
            output, vjp_fn = vjp_out

        # See NOTE: [Computing jacobian with vmap and vjp for multiple outputs]
        flat_output, output_spec = tree_flatten(output)

        error_if_complex("jacrev", flat_output, is_input=False)

        # NB: vjp already checks that all outputs are tensors
        # Step 1: Construct grad_outputs by splitting the standard basis
        flat_output_numels = tuple(out.numel() for out in flat_output)

        primals = _slice_argnums(args, argnums)
        flat_primals, primals_spec = tree_flatten(primals)

        def compute_jacobian_stacked():
            # Helper function to compute chunked Jacobian
            # The intermediate chunked calculation are only
            # scoped at this function level.
            chunked_results = []
            for flat_basis_chunk in _chunked_standard_basis_for_(
                flat_output, flat_output_numels, chunk_size=chunk_size
            ):
                if chunk_size == 1:
                    # sanity check.
                    for t in flat_basis_chunk:
                        assert t.size(0) == 1

                    flat_basis_chunk = tree_map(
                        lambda t: torch.squeeze(t, 0), flat_basis_chunk
                    )

                basis = tree_unflatten(flat_basis_chunk, output_spec)

                if chunk_size == 1:
                    # Behaviour with `chunk_size=1` is same as `for-loop`
                    # i.e. user shouldn't deal with the limitations of vmap.
                    chunked_result = vjp_fn(basis)
                else:  # chunk_size is None or chunk_size != 1
                    chunked_result = vmap(vjp_fn)(basis)

                flat_results = pytree.tree_leaves(chunked_result)

                if chunk_size == 1:
                    flat_results = tree_map(
                        lambda t: torch.unsqueeze(t, 0), flat_results
                    )

                chunked_results.append(flat_results)

            if len(chunked_results) == 1:
                # Short-circuit if we used a single chunk
                return chunked_results[0]

            # Concatenate chunks.
            flat_results = []
            # Iterate and concat the jacobians of different
            # inputs.
            for idx in range(len(flat_primals)):
                r = tuple(r_[idx] for r_ in chunked_results)
                flat_results.append(torch.cat(r, 0))

            return flat_results

        def compute_jacobian_preallocate_and_copy():
            # Helper function to compute chunked Jacobian
            # The intermediate chunked calculation are only
            # scoped at this function level.
            out_vec_size = sum(flat_output_numels)

            # Don't pre-allocate if we have a single chunk.
            if not (chunk_size is None or chunk_size >= out_vec_size):
                stacked_results = [
                    primal.new_zeros(out_vec_size, *primal.shape)
                    for primal in flat_primals
                ]

            for idx, flat_basis_chunk in enumerate(
                _chunked_standard_basis_for_(
                    flat_output, flat_output_numels, chunk_size=chunk_size
                )
            ):
                if chunk_size == 1:
                    # sanity check.
                    for t in flat_basis_chunk:
                        assert t.size(0) == 1

                    flat_basis_chunk = [torch.squeeze(t, 0) for t in flat_basis_chunk]

                basis = tree_unflatten(flat_basis_chunk, output_spec)

                if chunk_size == 1:
                    # Behaviour with `chunk_size=1` is same as `for-loop`
                    # i.e. user shouldn't deal with the limitations of vmap.
                    chunked_result = vjp_fn(basis)
                else:  # chunk_size is None or chunk_size != 1
                    chunked_result = vmap(vjp_fn)(basis)

                flat_results = pytree.tree_leaves(chunked_result)

                # Short-circuit if we have a single chunk.
                if chunk_size is None or chunk_size >= out_vec_size:
                    if chunk_size == 1:  # and out_vec_size == 1
                        # Since we squeezed the output dim
                        flat_results = tree_map(
                            lambda t: torch.unsqueeze(t, 0), flat_results
                        )
                    return flat_results

                for r, sr in zip(flat_results, stacked_results):
                    sr[idx * chunk_size : (idx + 1) * chunk_size].copy_(r)

            return stacked_results

        if _preallocate_and_copy:
            flat_jacobians_per_input = compute_jacobian_preallocate_and_copy()
        else:
            flat_jacobians_per_input = compute_jacobian_stacked()

        # Step 2: The returned jacobian is one big tensor per input. In this step,
        # we split each Tensor by output.
        flat_jacobians_per_input = [
            result.split(flat_output_numels, dim=0)
            for result in flat_jacobians_per_input
        ]
        flat_input_flat_output = [
            tuple(
                split.view(out.shape + primal.shape)
                for split, out in zip(splits, flat_output)
            )
            for splits, primal in zip(flat_jacobians_per_input, flat_primals)
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

        flat_output_input = tuple(
            tree_unflatten(flat_input, primals_spec)
            for flat_input in flat_output_flat_input
        )

        if isinstance(argnums, int):
            flat_output_input = tuple(
                _safe_zero_index(flat_input) for flat_input in flat_output_input
            )
        output_input = tree_unflatten(flat_output_input, output_spec)
        if has_aux:
            return output_input, aux
        return output_input

    # Dynamo does not support HOP composition if their inner function is
    # annotated with @functools.wraps(...). We circumvent this issue by applying
    # wraps only if we're not tracing with dynamo.
    if not torch._dynamo.is_compiling():
        wrapper_fn = wraps(func)(wrapper_fn)
    else:
        wrapper_fn = torch._dynamo.disable(wrapper_fn)

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


def _chunked_standard_basis_for_(tensors, tensor_numels, chunk_size=None):
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
    # NOTE: Argument `chunk_size` is used to generate chunked basis instead of
    #       one huge basis matrix. `chunk_size` dictates the maximum size of the
    #       basis matrix along dim=0.
    assert len(tensors) == len(tensor_numels)
    assert len(tensors) > 0
    assert chunk_size is None or chunk_size > 0
    total_numel = sum(tensor_numels)
    if chunk_size and chunk_size < total_numel:
        chunk_numels = get_chunk_sizes(total_numel, chunk_size)
    else:  # chunk_size is None or chunk_size >= total_numel
        chunk_size = total_numel
        chunk_numels = [total_numel]

    diag_start_indices = (
        0,
        *torch.tensor(tensor_numels).cumsum(dim=0)[:-1].neg().unbind(),
    )

    for chunk_idx, total_numel in enumerate(chunk_numels):
        chunks = tuple(
            tensor.new_zeros(total_numel, tensor_numel)
            for tensor, tensor_numel in zip(tensors, tensor_numels)
        )

        for chunk, diag_start_idx in zip(chunks, diag_start_indices):
            chunk.diagonal(diag_start_idx + chunk_idx * chunk_size).fill_(1)
        chunks = tuple(
            chunk.view(total_numel, *tensor.shape)
            for chunk, tensor in zip(chunks, tensors)
        )
        yield chunks


def _construct_standard_basis_for(tensors, tensor_numels):
    for basis in _chunked_standard_basis_for_(tensors, tensor_numels, chunk_size=None):
        return basis


def _validate_and_wrap_argnum(argnum, num_args):
    if not isinstance(argnum, int):
        raise RuntimeError(f"argnum must be int, got: {type(argnum)}")
    if argnum >= 0 and argnum < num_args:
        return argnum
    if argnum < 0 and argnum >= -num_args:
        return argnum + num_args
    raise RuntimeError(f"Got argnum={argnum}, but only {num_args} positional inputs")


def _check_unique_non_empty(argnums):
    if isinstance(argnums, tuple):
        if len(argnums) == 0:
            raise RuntimeError("argnums must be non-empty")
        if len(set(argnums)) != len(argnums):
            raise RuntimeError(f"argnums elements must be unique, got {argnums}")


def _replace_args(old_args, new_args, argnums):
    if isinstance(argnums, int):
        if len(new_args) != 1:
            raise RuntimeError(
                f"new_args should be of size 1, was of size {len(new_args)}"
            )
        return tuple(
            new_args[0] if i == argnums else old_args[i] for i in range(len(old_args))
        )
    if isinstance(argnums, tuple):
        if len(new_args) != len(argnums):
            raise RuntimeError(
                "new_args should have the same size as argnums. "
                f"Argnums size {len(argnums)}, new_args size {len(new_args)}"
            )

        def get_right_elem(i):
            return new_args[argnums.index(i)] if i in argnums else old_args[i]

        return tuple(get_right_elem(i) for i in range(len(old_args)))
    raise RuntimeError(f"argnums must be int or Tuple[int, ...], got: {type(argnums)}")


def _validate_and_wrap_argnums(argnums, num_args):
    if isinstance(argnums, int):
        return _validate_and_wrap_argnum(argnums, num_args)
    if isinstance(argnums, tuple):
        return tuple(_validate_and_wrap_argnum(argnum, num_args) for argnum in argnums)
    raise AssertionError("Should never get here")


def _slice_argnums(args, argnums, as_tuple=True):
    if not isinstance(argnums, int) and not isinstance(argnums, tuple):
        raise RuntimeError(
            f"argnums must be int or Tuple[int, ...], got: {type(argnums)}"
        )
    argnums = _validate_and_wrap_argnums(argnums, len(args))
    _check_unique_non_empty(argnums)
    if isinstance(argnums, int):
        if as_tuple:
            return (args[argnums],)
        else:
            return args[argnums]
    return tuple(args[i] for i in argnums)


JVP_NESTING = 0


def assert_flat_tuple_of_tensors(elts: Any, api: str, argname: str) -> None:
    if not isinstance(elts, tuple):
        raise RuntimeError(
            f"{api}: Expected {argname} to be a tuple of Tensors, got {type(elts)}"
        )
    for elt in elts:
        if isinstance(elt, torch.Tensor):
            continue
        raise RuntimeError(
            f"{api}: Expected {argname} to be a tuple of Tensors, got "
            f"a tuple with an element of type {type(elt)}"
        )
    if len(elts) == 0:
        raise RuntimeError(
            f"{api}: Expected {argname} to be a non-empty tuple of Tensors."
        )


def assert_non_empty_tensor_output(output: List[Any], api: str) -> None:
    if (len(output) == 1 and output[0] is None) or len(output) < 1:
        raise RuntimeError(
            f"{api}: Expected f to be a function that has non-empty output (got output = {output})"
        )
    for o in output:
        if not isinstance(o, torch.Tensor):
            raise RuntimeError(
                f"{api}: expected f(*primals) to return only tensors"
                f", got unsupported type {type(o)}"
            )


def assert_output_is_tensor_or_tensors(output: Any, api: str) -> None:
    if isinstance(output, torch.Tensor):
        return
    if not isinstance(output, tuple):
        raise RuntimeError(
            f"{api}: Expected output of f to be a Tensor or Tensors, got "
            f"{type(output)}"
        )
    if len(output) == 0:
        raise RuntimeError(
            f"{api}: Expected output of f to be a non-empty tuple of Tensors."
        )
    for out in output:
        if isinstance(out, torch.Tensor):
            continue
        raise RuntimeError(
            f"{api}: Expected output of f to be a Tensor or Tensors, got "
            f"{type(out)} as an output"
        )


def assert_non_empty_list_of_tensors(
    output: List[torch.Tensor], api: str, argname: str
) -> None:
    if len(output) == 0:
        raise RuntimeError(f"{api}: Expected {argname} to contain at least one Tensor.")
    for out in output:
        if isinstance(out, torch.Tensor):
            continue
        raise RuntimeError(
            f"{api}: Expected {argname} to only contain Tensors, got " f"{type(out)}"
        )


jvp_str = "jvp(f, primals, tangents)"


def safe_unpack_dual(dual, strict):
    if not isinstance(dual, torch.Tensor):
        raise RuntimeError(
            f"{jvp_str}: expected f(*args) to return only tensors"
            f", got unsupported type {type(dual)}"
        )

    primal, tangent = fwAD.unpack_dual(dual)
    if tangent is None:
        if strict:
            raise RuntimeError(
                "jvp(f, primals, tangents, strict=True): "
                "The output of f is independent of "
                "the inputs. This is not allowed with strict=True."
            )
        tangent = torch.zeros_like(primal)
    return primal, tangent


@exposed_in("torch.func")
def jvp(
    func: Callable,
    primals: Any,
    tangents: Any,
    *,
    strict: bool = False,
    has_aux: bool = False,
):
    """
    Standing for the Jacobian-vector product, returns a tuple containing
    the output of `func(*primals)` and the "Jacobian of ``func`` evaluated at
    ``primals``" times ``tangents``. This is also known as forward-mode autodiff.

    Args:
        func (function): A Python function that takes one or more arguments,
            one of which must be a Tensor, and returns one or more Tensors
        primals (Tensors): Positional arguments to ``func`` that must all be
            Tensors. The returned function will also be computing the
            derivative with respect to these arguments
        tangents (Tensors): The "vector" for which Jacobian-vector-product is
            computed. Must be the same structure and sizes as the inputs to
            ``func``.
        has_aux (bool): Flag indicating that ``func`` returns a
            ``(output, aux)`` tuple where the first element is the output of
            the function to be differentiated and the second element is
            other auxiliary objects that will not be differentiated.
            Default: False.

    Returns:
        Returns a ``(output, jvp_out)`` tuple containing the output of ``func``
        evaluated at ``primals`` and the Jacobian-vector product.
        If ``has_aux is True``, then instead returns a ``(output, jvp_out, aux)`` tuple.

    .. note::
        You may see this API error out with "forward-mode AD not implemented
        for operator X". If so, please file a bug report and we will prioritize it.

    jvp is useful when you wish to compute gradients of a function R^1 -> R^N

        >>> from torch.func import jvp
        >>> x = torch.randn([])
        >>> f = lambda x: x * torch.tensor([1., 2., 3])
        >>> value, grad = jvp(f, (x,), (torch.tensor(1.),))
        >>> assert torch.allclose(value, f(x))
        >>> assert torch.allclose(grad, torch.tensor([1., 2, 3]))

    :func:`jvp` can support functions with multiple inputs by passing in the
    tangents for each of the inputs

         >>> from torch.func import jvp
         >>> x = torch.randn(5)
         >>> y = torch.randn(5)
         >>> f = lambda x, y: (x * y)
         >>> _, output = jvp(f, (x, y), (torch.ones(5), torch.ones(5)))
         >>> assert torch.allclose(output, x + y)

    """

    return _jvp_with_argnums(
        func, primals, tangents, argnums=None, strict=strict, has_aux=has_aux
    )


@doesnt_support_saved_tensors_hooks
def _jvp_with_argnums(
    func: Callable,
    primals: Any,
    tangents: Any,
    argnums: Optional[argnums_t],
    *,
    strict: bool = False,
    has_aux: bool,
):
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
            f"{jvp_str}: Expected primals to be a tuple. "
            f"E.g. it should be valid to call f(*primals)."
        )
    diff_args = primals if argnums is None else _slice_argnums(primals, argnums)
    flat_primals, primals_spec = tree_flatten(diff_args)
    flat_tangents, tangents_spec = tree_flatten(tangents)
    _jvp_treespec_compare(diff_args, tangents)
    assert_non_empty_list_of_tensors(flat_primals, jvp_str, "primals")
    assert_non_empty_list_of_tensors(flat_tangents, jvp_str, "tangents")

    global JVP_NESTING

    with jvp_increment_nesting() as level:
        with fwAD._set_fwd_grad_enabled(True):
            ctx = fwAD.dual_level if JVP_NESTING == 1 else contextlib.nullcontext
            with ctx():
                flat_duals = tuple(
                    fwAD.make_dual(p, t) for p, t in zip(flat_primals, flat_tangents)
                )
                duals = tree_unflatten(flat_duals, primals_spec)
                # Note for the reviewer: This is extremely odd but it passes the
                # assertion "len(self.block_stack) == 1" on symbolic_convert.py
                # The equivalent "if argnums is not None" fails for some reason
                if isinstance(argnums, (int, tuple)):
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

                primals_out, tangents_out = zip(
                    *[safe_unpack_dual(dual, strict) for dual in result_duals]
                )
                primals_out = tree_map(
                    partial(_undo_create_differentiable, level=level), primals_out
                )
                tangents_out = tree_map(
                    partial(_undo_create_differentiable, level=level), tangents_out
                )

                primals_out_unflatten = tree_unflatten(primals_out, spec)
                tangents_out_unflatten = tree_unflatten(tangents_out, spec)
                if has_aux:
                    return primals_out_unflatten, tangents_out_unflatten, aux

                return primals_out_unflatten, tangents_out_unflatten


def safe_unflatten(tensor, dim, shape):
    if len(shape) == 0:
        assert tensor.shape[dim] == 1
        return tensor.squeeze(dim)
    return tensor.unflatten(dim, shape)


@exposed_in("torch.func")
def jacfwd(
    func: Callable,
    argnums: argnums_t = 0,
    has_aux: bool = False,
    *,
    randomness: str = "error",
):
    """
    Computes the Jacobian of ``func`` with respect to the arg(s) at index
    ``argnum`` using forward-mode autodiff

    Args:
        func (function): A Python function that takes one or more arguments,
            one of which must be a Tensor, and returns one or more Tensors
        argnums (int or Tuple[int]): Optional, integer or tuple of integers,
            saying which arguments to get the Jacobian with respect to.
            Default: 0.
        has_aux (bool): Flag indicating that ``func`` returns a
            ``(output, aux)`` tuple where the first element is the output of
            the function to be differentiated and the second element is
            auxiliary objects that will not be differentiated.
            Default: False.
        randomness(str): Flag indicating what type of randomness to use.
            See :func:`vmap` for more detail. Allowed: "different", "same", "error".
            Default: "error"

    Returns:
        Returns a function that takes in the same inputs as ``func`` and
        returns the Jacobian of ``func`` with respect to the arg(s) at
        ``argnums``. If ``has_aux is True``, then the returned function
        instead returns a ``(jacobian, aux)`` tuple where ``jacobian``
        is the Jacobian and ``aux`` is auxiliary objects returned by ``func``.

    .. note::
        You may see this API error out with "forward-mode AD not implemented
        for operator X". If so, please file a bug report and we will prioritize it.
        An alternative is to use :func:`jacrev`, which has better operator coverage.

    A basic usage with a pointwise, unary operation will give a diagonal array
    as the Jacobian

        >>> from torch.func import jacfwd
        >>> x = torch.randn(5)
        >>> jacobian = jacfwd(torch.sin)(x)
        >>> expected = torch.diag(torch.cos(x))
        >>> assert torch.allclose(jacobian, expected)

    :func:`jacfwd` can be composed with vmap to produce batched
    Jacobians:

        >>> from torch.func import jacfwd, vmap
        >>> x = torch.randn(64, 5)
        >>> jacobian = vmap(jacfwd(torch.sin))(x)
        >>> assert jacobian.shape == (64, 5, 5)

    If you would like to compute the output of the function as well as the
    jacobian of the function, use the ``has_aux`` flag to return the output
    as an auxiliary object:

        >>> from torch.func import jacfwd
        >>> x = torch.randn(5)
        >>>
        >>> def f(x):
        >>>   return x.sin()
        >>>
        >>> def g(x):
        >>>   result = f(x)
        >>>   return result, result
        >>>
        >>> jacobian_f, f_x = jacfwd(g, has_aux=True)(x)
        >>> assert torch.allclose(f_x, f(x))

    Additionally, :func:`jacrev` can be composed with itself or :func:`jacrev`
    to produce Hessians

        >>> from torch.func import jacfwd, jacrev
        >>> def f(x):
        >>>   return x.sin().sum()
        >>>
        >>> x = torch.randn(5)
        >>> hessian = jacfwd(jacrev(f))(x)
        >>> assert torch.allclose(hessian, torch.diag(-x.sin()))

    By default, :func:`jacfwd` computes the Jacobian with respect to the first
    input. However, it can compute the Jacboian with respect to a different
    argument by using ``argnums``:

        >>> from torch.func import jacfwd
        >>> def f(x, y):
        >>>   return x + y ** 2
        >>>
        >>> x, y = torch.randn(5), torch.randn(5)
        >>> jacobian = jacfwd(f, argnums=1)(x, y)
        >>> expected = torch.diag(2 * y)
        >>> assert torch.allclose(jacobian, expected)

    Additionally, passing a tuple to ``argnums`` will compute the Jacobian
    with respect to multiple arguments

        >>> from torch.func import jacfwd
        >>> def f(x, y):
        >>>   return x + y ** 2
        >>>
        >>> x, y = torch.randn(5), torch.randn(5)
        >>> jacobian = jacfwd(f, argnums=(0, 1))(x, y)
        >>> expectedX = torch.diag(torch.ones_like(x))
        >>> expectedY = torch.diag(2 * y)
        >>> assert torch.allclose(jacobian[0], expectedX)
        >>> assert torch.allclose(jacobian[1], expectedY)

    """

    def wrapper_fn(*args):
        error_if_complex("jacfwd", args, is_input=True)
        primals = args if argnums is None else _slice_argnums(args, argnums)
        flat_primals, primals_spec = tree_flatten(primals)
        flat_primals_numels = tuple(p.numel() for p in flat_primals)
        flat_basis = _construct_standard_basis_for(flat_primals, flat_primals_numels)
        basis = tree_unflatten(flat_basis, primals_spec)

        def push_jvp(basis):
            output = _jvp_with_argnums(
                func, args, basis, argnums=argnums, has_aux=has_aux
            )
            # output[0] is the output of `func(*args)`
            error_if_complex("jacfwd", output[0], is_input=False)
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
                for primal, jac_out_in in zip(
                    flat_primals,
                    jac_out.movedim(0, -1).split(flat_primals_numels, dim=-1),
                )
            )
            for jac_out in jac_outs
        )
        jac_outs_ins = tuple(
            tree_unflatten(jac_ins, primals_spec) for jac_ins in jac_outs_ins
        )

        if isinstance(argnums, int):
            jac_outs_ins = tuple(jac_ins[0] for jac_ins in jac_outs_ins)
        if has_aux:
            return tree_unflatten(jac_outs_ins, spec), aux
        return tree_unflatten(jac_outs_ins, spec)

    # Dynamo does not support HOP composition if their inner function is
    # annotated with @functools.wraps(...). We circumvent this issue by applying
    # wraps only if we're not tracing with dynamo.
    if not torch._dynamo.is_compiling():
        wrapper_fn = wraps(func)(wrapper_fn)
    else:
        wrapper_fn = torch._dynamo.disable(wrapper_fn)

    return wrapper_fn


@exposed_in("torch.func")
def hessian(func, argnums=0):
    """
    Computes the Hessian of ``func`` with respect to the arg(s) at index
    ``argnum`` via a forward-over-reverse strategy.

    The forward-over-reverse strategy (composing ``jacfwd(jacrev(func))``) is
    a good default for good performance. It is possible to compute Hessians
    through other compositions of :func:`jacfwd` and :func:`jacrev` like
    ``jacfwd(jacfwd(func))`` or ``jacrev(jacrev(func))``.

    Args:
        func (function): A Python function that takes one or more arguments,
            one of which must be a Tensor, and returns one or more Tensors
        argnums (int or Tuple[int]): Optional, integer or tuple of integers,
            saying which arguments to get the Hessian with respect to.
            Default: 0.

    Returns:
        Returns a function that takes in the same inputs as ``func`` and
        returns the Hessian of ``func`` with respect to the arg(s) at
        ``argnums``.

    .. note::
        You may see this API error out with "forward-mode AD not implemented
        for operator X". If so, please file a bug report and we will prioritize it.
        An alternative is to use ``jacrev(jacrev(func))``, which has better
        operator coverage.

    A basic usage with a R^N -> R^1 function gives a N x N Hessian:

        >>> from torch.func import hessian
        >>> def f(x):
        >>>   return x.sin().sum()
        >>>
        >>> x = torch.randn(5)
        >>> hess = hessian(f)(x)  # equivalent to jacfwd(jacrev(f))(x)
        >>> assert torch.allclose(hess, torch.diag(-x.sin()))

    """
    return jacfwd(jacrev(func, argnums), argnums)


@doesnt_support_saved_tensors_hooks
def grad_and_value_impl(func, argnums, has_aux, args, kwargs) -> Callable:
    with grad_increment_nesting() as level:
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
                raise RuntimeError(
                    "grad_and_value(f)(*args): Expected f(*args) "
                    f"to return a Tensor, got {type(output)}"
                )
            if output.dim() != 0:
                raise RuntimeError(
                    "grad_and_value(f)(*args): Expected f(*args) "
                    "to return a scalar Tensor, got tensor with "
                    f"{output.dim()} dims. Maybe you wanted to "
                    "use the vjp or jacrev APIs instead?"
                )

            flat_diff_args, spec = tree_flatten(diff_args)

            # NB: need create_graph so that backward pass isn't run in no_grad mode
            flat_outputs = _as_tuple(output)
            flat_grad_input = _autograd_grad(
                flat_outputs, flat_diff_args, create_graph=True
            )
            grad_input = tree_unflatten(flat_grad_input, spec)

            grad_input = _undo_create_differentiable(grad_input, level)
            output = _undo_create_differentiable(output, level)
            if has_aux:
                aux = _undo_create_differentiable(aux, level)

        if has_aux:
            return grad_input, (output, aux)
        return grad_input, output


def grad_impl(func: Callable, argnums: argnums_t, has_aux: bool, args, kwargs):
    results = grad_and_value_impl(func, argnums, has_aux, args, kwargs)
    if has_aux:
        grad, (_, aux) = results
        return grad, aux
    grad, _ = results
    return grad


def _maybe_wrap_functional_tensor(
    maybe_tensor, level, *, _python_functionalize: bool = False
):
    if not isinstance(maybe_tensor, torch.Tensor):
        return maybe_tensor
    wrapped = _wrap_functional_tensor(maybe_tensor, level)
    _assert_wrapped_functional(maybe_tensor, wrapped)
    if _python_functionalize:
        out = FunctionalTensor(wrapped)
        torch._mirror_autograd_meta_to(maybe_tensor, out)
        return out
    return wrapped


def _wrap_all_tensors_to_functional(
    tensor_pytree, level, *, _python_functionalize: bool = False
):
    return tree_map(
        partial(
            lambda x: _maybe_wrap_functional_tensor(
                x, level, _python_functionalize=_python_functionalize
            )
        ),
        tensor_pytree,
    )


def _maybe_unwrap_functional_tensor(maybe_tensor, *, reapply_views: bool):
    if not isinstance(maybe_tensor, torch.Tensor):
        return maybe_tensor
    if isinstance(maybe_tensor, FunctionalTensor):
        maybe_tensor = maybe_tensor.elem

    if not torch._is_functional_tensor(maybe_tensor):
        # If it's not a functional tensor, just return it.
        # This can happen if we functionalize a fn that returns a global,
        # which was never wrapped properly.
        return maybe_tensor
    # Sync any pending updates on the output tensor
    torch._sync(maybe_tensor)
    return _unwrap_functional_tensor(maybe_tensor, reapply_views)


def _unwrap_all_tensors_from_functional(tensor_pytree, *, reapply_views: bool):
    return tree_map(
        lambda t: _maybe_unwrap_functional_tensor(t, reapply_views=reapply_views),
        tensor_pytree,
    )


@exposed_in("torch.func")
def functionalize(func: Callable, *, remove: str = "mutations") -> Callable:
    """
    functionalize is a transform that can be used to remove (intermediate)
    mutations and aliasing from a function, while preserving the function's
    semantics.

    ``functionalize(func)`` returns a new function with the same semantics
    as ``func``, but with all intermediate mutations removed.
    Every inplace operation performed on an intermediate tensor:
    ``intermediate.foo_()``
    gets replaced by its out-of-place equivalent:
    ``intermediate_updated = intermediate.foo()``.

    functionalize is useful for shipping a pytorch program off to
    backends or compilers that aren't able to easily represent
    mutations or aliasing operators.

    Args:
        func (Callable): A Python function that takes one or more arguments.
        remove (str): An optional string argument, that takes on either
            the value 'mutations' or 'mutations_and_views'.
            If 'mutations' is passed in then all mutating operators
            will be replaced with their non-mutating equivalents.
            If 'mutations_and_views' is passed in, then additionally, all aliasing
            operators will be replaced with their non-aliasing equivalents.
            Default: 'mutations'.

    Returns:
        Returns a new "functionalized" function. It takes the same inputs as
        ``func``, and has the same behavior, but any mutations
        (and optionally aliasing) performed on intermediate tensors
        in the function will be removed.

    functionalize will also remove mutations (and views) that were performed on function inputs.
    However to preserve semantics, functionalize will "fix up" the mutations after
    the transform has finished running, by detecting if any tensor inputs "should have"
    been mutated, and copying the new data back to the inputs if necessary.


    Example::

        >>> # xdoctest: +SKIP
        >>> import torch
        >>> from torch.fx.experimental.proxy_tensor import make_fx
        >>> from torch.func import functionalize
        >>>
        >>> # A function that uses mutations and views, but only on intermediate tensors.
        >>> def f(a):
        ...     b = a + 1
        ...     c = b.view(-1)
        ...     c.add_(1)
        ...     return b
        ...
        >>> inpt = torch.randn(2)
        >>>
        >>> out1 = f(inpt)
        >>> out2 = functionalize(f)(inpt)
        >>>
        >>> # semantics are the same (outputs are equivalent)
        >>> print(torch.allclose(out1, out2))
        True
        >>>
        >>> f_traced = make_fx(f)(inpt)
        >>> f_no_mutations_traced = make_fx(functionalize(f))(inpt)
        >>> f_no_mutations_and_views_traced = make_fx(functionalize(f, remove='mutations_and_views'))(inpt)
        >>>
        >>> print(f_traced.code)



        def forward(self, a_1):
            add = torch.ops.aten.add(a_1, 1);  a_1 = None
            view = torch.ops.aten.view(add, [-1])
            add_ = torch.ops.aten.add_(view, 1);  view = None
            return add

        >>> print(f_no_mutations_traced.code)



        def forward(self, a_1):
            add = torch.ops.aten.add(a_1, 1);  a_1 = None
            view = torch.ops.aten.view(add, [-1]);  add = None
            add_1 = torch.ops.aten.add(view, 1);  view = None
            view_1 = torch.ops.aten.view(add_1, [2]);  add_1 = None
            return view_1

        >>> print(f_no_mutations_and_views_traced.code)



        def forward(self, a_1):
            add = torch.ops.aten.add(a_1, 1);  a_1 = None
            view_copy = torch.ops.aten.view_copy(add, [-1]);  add = None
            add_1 = torch.ops.aten.add(view_copy, 1);  view_copy = None
            view_copy_1 = torch.ops.aten.view_copy(add_1, [2]);  add_1 = None
            return view_copy_1


        >>> # A function that mutates its input tensor
        >>> def f(a):
        ...     b = a.view(-1)
        ...     b.add_(1)
        ...     return a
        ...
        >>> f_no_mutations_and_views_traced = make_fx(functionalize(f, remove='mutations_and_views'))(inpt)
        >>> #
        >>> # All mutations and views have been removed,
        >>> # but there is an extra copy_ in the graph to correctly apply the mutation to the input
        >>> # after the function has completed.
        >>> print(f_no_mutations_and_views_traced.code)



        def forward(self, a_1):
            view_copy = torch.ops.aten.view_copy(a_1, [-1])
            add = torch.ops.aten.add(view_copy, 1);  view_copy = None
            view_copy_1 = torch.ops.aten.view_copy(add, [2]);  add = None
            copy_ = torch.ops.aten.copy_(a_1, view_copy_1);  a_1 = None
            return view_copy_1


    There are a few "failure modes" for functionalize that are worth calling out:
      (1) Like other torch.func transforms, `functionalize()` doesn't work with functions
          that directly use `.backward()`. The same is true for torch.autograd.grad.
          If you want to use autograd, you can compute gradients directly
          with `functionalize(grad(f))`.
      (2) Like other torch.func transforms, `functionalize()` doesn't work with global state.
          If you call `functionalize(f)` on a function that takes views / mutations of
          non-local state, functionalization will simply no-op and pass the view/mutation
          calls directly to the backend.
          One way to work around this is is to ensure that any non-local state creation
          is wrapped into a larger function, which you then call functionalize on.
      (3) `resize_()` has some limitations: functionalize will only work on programs
          that use resize_()` as long as the tensor being resized is not a view.
      (4) `as_strided()` has some limitations: functionalize will not work on
          `as_strided()` calls that result in tensors with overlapping memory.


    Finally, a helpful mental model for understanding functionalization is that
    most user pytorch programs are writing with the public torch API.
    When executed, torch operators are generally decomposed into
    our internal C++ "ATen" API.
    The logic for functionalization happens entirely at the level of ATen.
    Functionalization knows how to take every aliasing operator in ATen,
    and map it to its non-aliasing equivalent
    (e.g. ``tensor.view({-1})`` -> ``at::view_copy(tensor, {-1})``),
    and how to take every mutating operator in ATen,
    and map it to its non-mutating equivalent
    (e.g. ``tensor.add_(1)`` -> ``at::add(tensor, -1)``),
    while tracking aliases and mutations out-of-line to know when to fix things up.
    Information about which ATen operators are aliasing or mutating all comes from
    https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/native_functions.yaml.
    """
    if remove == "mutations":
        reapply_views = True
    elif remove == "mutations_and_views":
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

            flattened_unwrapped_args = pytree.arg_tree_leaves(*args)
            flattened_wrapped_args = pytree.arg_tree_leaves(*func_args)
            flattened_unwrapped_kwargs = pytree.arg_tree_leaves(**kwargs)
            flattened_wrapped_kwargs = pytree.arg_tree_leaves(**func_kwargs)

            func_outputs = func(*func_args, **func_kwargs)
            outputs = _unwrap_all_tensors_from_functional(
                func_outputs, reapply_views=reapply_views
            )
            flat_outputs, func_out_spec = tree_flatten(outputs)

            for a in flattened_wrapped_args + flattened_wrapped_kwargs:
                if isinstance(a, torch.Tensor):
                    # Call sync_() on the inputs, to ensure that any pending mutations have been applied.
                    torch._sync(a)

            # And if any mutations were applied to the inputs, we need to propagate them back to the user.
            for unwrapped, wrapped in zip(
                flattened_unwrapped_args, flattened_wrapped_args
            ):
                if isinstance(unwrapped, torch.Tensor) and isinstance(
                    wrapped, torch.Tensor
                ):
                    _propagate_functional_input_mutation(unwrapped, wrapped)
            for unwrapped, wrapped in zip(
                flattened_unwrapped_kwargs, flattened_wrapped_kwargs
            ):
                if isinstance(unwrapped, torch.Tensor) and isinstance(
                    wrapped, torch.Tensor
                ):
                    _propagate_functional_input_mutation(unwrapped, wrapped)

            return outputs
        finally:
            _func_decrement_nesting()

    return wrapped


@exposed_in("torch.func")
def linearize(func: Callable, *primals) -> Tuple[Any, Callable]:
    """
    Returns the value of ``func`` at ``primals`` and linear approximation
    at ``primals``.

    Args:
        func (Callable): A Python function that takes one or more arguments.
        primals (Tensors): Positional arguments to ``func`` that must all be
            Tensors. These are the values at which the function is linearly approximated.

    Returns:
        Returns a ``(output, jvp_fn)`` tuple containing the output of ``func``
        applied to ``primals`` and a function that computes the jvp of
        ``func`` evaluated at ``primals``.

    linearize is useful if jvp is to be computed multiple times at ``primals``. However,
    to achieve this, linearize saves intermediate computation and has higher memory requirements
    than directly applying `jvp`. So, if all the ``tangents`` are known, it maybe more efficient
    to compute vmap(jvp) instead of using linearize.

    .. note::
        linearize evaluates ``func`` twice. Please file an issue for an implementation
        with a single evaluation.

    Example::
        >>> import torch
        >>> from torch.func import linearize
        >>> def fn(x):
        ...     return x.sin()
        ...
        >>> output, jvp_fn = linearize(fn, torch.zeros(3, 3))
        >>> jvp_fn(torch.ones(3, 3))
        tensor([[1., 1., 1.],
                [1., 1., 1.],
                [1., 1., 1.]])
        >>>

    """
    # Note: We evaluate `fn` twice.
    # Once for returning the output and other while
    # tracing the graph.
    # If this becomes a bottle-neck, we should update
    # make_fx such that it also returns the output.

    output = func(*primals)
    _, output_spec = tree_flatten(output)

    flat_primals, primals_argspec = tree_flatten(primals)

    # tangents for tracing
    flat_tangents = tuple(p.new_empty(()).expand_as(p) for p in flat_primals)

    # function to trace
    def trace_fn(flat_tangents):
        with fwAD.dual_level():
            flat_duals = tuple(
                fwAD.make_dual(p, t) for p, t in zip(flat_primals, flat_tangents)
            )
            duals = tree_unflatten(flat_duals, primals_argspec)
            output = func(*duals)
            tangents = tree_map_only(
                torch.Tensor, lambda t: fwAD.unpack_dual(t)[1], output
            )

        return tangents

    jvp_graph = lazy_dynamo_disallow(make_fx)(trace_fn)(flat_tangents)
    const_folded_jvp_graph = lazy_dynamo_disallow(const_fold.split_const_subgraphs)(
        jvp_graph
    )

    # Hold only the meta-data regarding the primals.
    flat_primals_shape = tuple(p.shape for p in flat_primals)
    flat_primals_device = tuple(p.device for p in flat_primals)
    flat_primals_dtype = tuple(p.dtype for p in flat_primals)

    def forward_ad_checks(flat_tangents):
        for idx, t in enumerate(flat_tangents):
            if t.shape != flat_primals_shape[idx]:
                msg = (
                    f"tangent:{idx} with shape {t.shape} in flattened "
                    f"pytree doesn't match the shape {flat_primals_shape[idx]} "
                    "of the corresponding primal."
                )
                raise RuntimeError(msg)

            if t.device != flat_primals_device[idx]:
                msg = (
                    f"tangent:{idx} with device {t.device} in flattened "
                    f"pytree doesn't match the device {flat_primals_device[idx]} "
                    "of the corresponding primal."
                )
                raise RuntimeError(msg)

            if t.dtype != flat_primals_dtype[idx]:
                msg = (
                    f"tangent:{idx} with dtype {t.dtype} in flattened "
                    f"pytree doesn't match the dtype {flat_primals_dtype[idx]} "
                    "of the corresponding primal."
                )
                raise RuntimeError(msg)

    # jvp_fn : callable to return
    #   It takes care of checking the argspec of tangents,
    #   calling the folded fx graph and unflattening fx graph output
    def jvp_fn(*tangents):
        flat_tangents, tangent_argspec = tree_flatten(tangents)
        _linearize_treespec_compare(primals, tangents)

        forward_ad_checks(flat_tangents)

        flat_output = const_folded_jvp_graph(*flat_tangents)
        # const folded graph can return flat output,
        # so transform output.
        return tree_unflatten(flat_output, output_spec)

    return output, jvp_fn
