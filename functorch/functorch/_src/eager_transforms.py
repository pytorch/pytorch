# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Union, Tuple
import torch
from functools import partial, wraps
import contextlib
from torch.utils._pytree import tree_flatten, tree_unflatten, tree_map
from .pytree_hacks import tree_map_, treespec_pprint
import torch.autograd.forward_ad as fwAD

from .vmap import vmap

from functorch._C import (
    _wrap_for_grad,
    _unwrap_for_grad,
    _grad_increment_nesting,
    _grad_decrement_nesting,
)

argnums_t = Union[int, Tuple[int, ...]]


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
def vjp(func: Callable, *primals, has_aux=False):
    """
    Standing for the vector-Jacobian product, returns a tuple containing the
    results of :attr:`func` applied to :attr:`primals` and a function that, when
    given ``cotangents``, computes the reverse-mode Jacobian of :attr:`func` with
    respect to :attr:`primals` times ``cotangents``.

    Args:
        func (Callable): A Python function that takes one or more arguments. Must
            return one or more Tensors.
        primals (Tensors): Positional arguments to :attr:`func` that must all be
            Tensors. The returned function will also be computing the
            derivative with respect to these arguments
        has_aux (bool): Flag indicating that :attr:`func` returns a
            ``(output, aux)`` tuple where the first element is the output of
            the function to be differentiated and the second element is
            other auxiliary objects that will not be differentiated.
            Default: False.

    Returns:
        Returns a ``(output, vjp_fn)`` tuple containing the output of :attr:`func`
        applied to :attr:`primals` and a function that computes the vjp of
        :attr:`func` with respect to all :attr:`primals` using the cotangents passed
        to the returned function. If ``has_aux is True``, then instead returns a
        ``(output, vjp_fn, aux)`` tuple.
        The returned ``vjp_fn`` function will return a tuple of each VJP.

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
        >>> vjps = vjpfunc(cotangents)
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

    .. note::
        Using PyTorch ``torch.no_grad`` together with ``vjp``.
        Case 1: Using ``torch.no_grad`` inside a function:

            >>> def f(x):
            >>>     with torch.no_grad():
            >>>         c = x ** 2
            >>>     return x - c

        In this case, ``vjp(f)(x)`` will respect the inner ``torch.no_grad``.

        Case 2: Using ``vjp`` inside ``torch.no_grad`` context manager:

            >>> with torch.no_grad():
            >>>     vjp(f)(x)

        In this case, ``vjp`` will respect the inner ``torch.no_grad``, but not the
        outer one. This is because ``vjp`` is a "function transform": its result
        should not depend on the result of a context manager outside of ``f``.
    """
    level = _grad_increment_nesting()
    try:
        # See NOTE [grad and vjp interaction with no_grad]
        with torch.enable_grad():
            primals = _wrap_all_tensors(primals, level)
            diff_primals = _create_differentiable(primals, level)
            primals_out = func(*diff_primals)

            if has_aux:
                primals_out, aux = primals_out
                aux = _undo_create_differentiable(aux, level)

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

    if has_aux:
        return results, wrapper, aux
    else:
        return results, wrapper


def _safe_zero_index(x):
    assert len(x) == 1
    return x[0]


def jacrev(func: Callable, argnums: Union[int, Tuple[int]] = 0, *, has_aux=False):
    """
    Computes the Jacobian of :attr:`func` with respect to the arg(s) at index
    :attr:`argnum` using reverse mode autodiff

    Args:
        func (function): A Python function that takes one or more arguments,
            one of which must be a Tensor, and returns one or more Tensors
        argnums (int or Tuple[int]): Optional, integer or tuple of integers,
            saying which arguments to get the Jacobian with respect to.
            Default: 0.
        has_aux (bool): Flag indicating that :attr:`func` returns a
            ``(output, aux)`` tuple where the first element is the output of
            the function to be differentiated and the second element is
            auxiliary objects that will not be differentiated.
            Default: False.

    Returns:
        Returns a function that takes in the same inputs as :attr:`func` and
        returns the Jacobian of :attr:`func` with respect to the arg(s) at
        :attr:`argnums`. If ``has_aux is True``, then the returned function
        instead returns a ``(jacobian, aux)`` tuple where ``jacobian``
        is the Jacobian and ``aux`` is auxiliary objects returned by ``func``.

    A basic usage with a pointwise, unary operation will give a diagonal array
    as the Jacobian

        >>> from functorch import jacrev
        >>> x = torch.randn(5)
        >>> jacobian = jacrev(torch.sin)(x)
        >>> expected = torch.diag(torch.cos(x))
        >>> assert torch.allclose(jacobian, expected)

    :func:`jacrev` can be composed with vmap to produce batched
    Jacobians:

        >>> from functorch import jacrev, vmap
        >>> x = torch.randn(64, 5)
        >>> jacobian = vmap(jacrev(torch.sin))(x)
        >>> assert jacobian.shape == (64, 5, 5)

    Additionally, :func:`jacrev` can be composed with itself to produce
    Hessians

        >>> from functorch import jacrev
        >>> def f(x):
        >>>   return x.sin().sum()
        >>>
        >>> x = torch.randn(5)
        >>> hessian = jacrev(jacrev(f))(x)
        >>> assert torch.allclose(hessian, torch.diag(-x.sin()))

    By default, :func:`jacrev` computes the Jacobian with respect to the first
    input. However, it can compute the Jacboian with respect to a different
    argument by using :attr:`argnums`:

        >>> from functorch import jacrev
        >>> def f(x, y):
        >>>   return x + y ** 2
        >>>
        >>> x, y = torch.randn(5), torch.randn(5)
        >>> jacobian = jacrev(f, argnums=1)(x, y)
        >>> expected = torch.diag(2 * y)
        >>> assert torch.allclose(jacobian, expected)

    Additionally, passing a tuple to :attr:`argnums` will compute the Jacobian
    with respect to multiple arguments

        >>> from functorch import jacrev
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
    @wraps(func)
    def wrapper_fn(*args):
        f_wrapper, primals = _argnums_partial(func, args, argnums)
        vjp_out = vjp(f_wrapper, *primals, has_aux=has_aux)
        if has_aux:
            output, vjp_fn, aux = vjp_out
        else:
            output, vjp_fn = vjp_out

        # See NOTE: [Computing jacobian with vmap and vjp for multiple outputs]
        flat_output, output_spec = tree_flatten(output)
        if len(flat_output) == 0:
            raise RuntimeError(
                'jacrev(f, ...)(*args): expected f to return at least one Tensor, '
                'got no Tensors.')
        for out in flat_output:
            if isinstance(out, torch.Tensor):
                continue
            raise RuntimeError(
                'jacrev(f, ...)(*args): expected f to only return Tensors as '
                f'outputs, got {type(out)}')
        # NB: vjp already checks that all outputs are tensors
        # Step 1: Construct grad_outputs by splitting the standard basis
        flat_output_numels = tuple(out.numel() for out in flat_output)
        flat_basis = _construct_standard_basis_for(flat_output, flat_output_numels)
        basis = tree_unflatten(flat_basis, output_spec)

        results = vmap(vjp_fn)(basis)

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


def assert_flat_tuple_of_tensors(elts, api, argname):
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


def assert_output_is_tensor_or_tensors(output, api):
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


def assert_non_empty_list_of_tensors(output, api, argname):
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
    primal, tangent = fwAD.unpack_dual(dual)
    if tangent is None:
        if strict:
            raise RuntimeError(
                'jvp(f, primals, tangents, strict=True): '
                'The output of f is independent of '
                'the inputs. This is not allowed with strict=True.')
        tangent = torch.zeros_like(primal)
    return primal, tangent


def jvp(func, primals, tangents, *, strict=False):
    """
    Standing for the Jacobian-vector product, returns a tuple containing
    the output of `func(*primals)` and the "Jacobian of ``func`` evaluated at
    ``primals``" times ``tangents``. This is also known as forward-mode autodiff.

    Args:
        func (function): A Python function that takes one or more arguments,
            one of which must be a Tensor, and returns one or more Tensors
        primals (Tensors): Positional arguments to :attr:`func` that must all be
            Tensors. The returned function will also be computing the
            derivative with respect to these arguments
        tangents (Tensors): The "vector" for which Jacobian-vector-product is
            computed. Must be the same structure and sizes as the inputs to
            ``func``.

    Returns:
        Returns a ``(output, jvp_out)`` tuple containing the output of ``func``
        evaluated at ``primals`` and the Jacobian-vector product.

    .. warning::
        PyTorch's forward-mode AD coverage on operators is not very good at the
        moment. You may see this API error out with "forward-mode AD not
        implemented for operator X". If so, please file us a bug report and we
        will prioritize it.

    jvp is useful when you wish to compute gradients of a function R^1 -> R^N

        >>> from functorch import jvp
        >>> x = torch.randn([])
        >>> f = lambda x: x * torch.tensor([1., 2., 3])
        >>> value, grad = jvp(f, (x,), (torch.tensor(1.),))
        >>> assert torch.allclose(value, f(x))
        >>> assert torch.allclose(grad, torch.tensor([1., 2, 3]))

    :func:`jvp` can support functions with multiple inputs by passing in the
    tangents for each of the inputs

         >>> from functorch import jvp
         >>> x = torch.randn(5)
         >>> y = torch.randn(5)
         >>> f = lambda x, y: (x * y)
         >>> _, output = jvp(f, (x, y), (torch.ones(5), torch.ones(5)))
         >>> assert torch.allclose(output, x + y)

    """
    if not isinstance(primals, tuple):
        raise RuntimeError(
            f'{jvp_str}: Expected primals to be a tuple. '
            f'E.g. it should be valid to call f(*primals).')
    flat_primals, primals_spec = tree_flatten(primals)
    flat_tangents, tangents_spec = tree_flatten(tangents)
    if primals_spec != tangents_spec:
        raise RuntimeError(
            f'{jvp_str}: Expected primals and tangents to have the same python '
            f'structure. For example, if primals is a tuple of 3 tensors, '
            f'tangents also must be. Got primals with structure {primals_spec} '
            f'and tangents with structure {tangents_spec}')
    assert_non_empty_list_of_tensors(flat_primals, jvp_str, 'primals')
    assert_non_empty_list_of_tensors(flat_tangents, jvp_str, 'tangents')

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
            flat_duals = tuple(fwAD.make_dual(p, t)
                               for p, t in zip(flat_primals, flat_tangents))
            duals = tree_unflatten(flat_duals, primals_spec)
            result_duals = func(*duals)
            assert_output_is_tensor_or_tensors(result_duals, jvp_str)
            result_duals, spec = tree_flatten(result_duals)

            primals_out, tangents_out = \
                zip(*[safe_unpack_dual(dual, strict) for dual in result_duals])
            primals_out = tree_map(
                partial(_undo_create_differentiable, level=level), primals_out)
            tangents_out = tree_map(
                partial(_undo_create_differentiable, level=level), tangents_out)
            return tree_unflatten(primals_out, spec), tree_unflatten(tangents_out, spec)
    finally:
        _grad_decrement_nesting()
        JVP_NESTING -= 1


def safe_unflatten(tensor, dim, shape):
    if len(shape) == 0:
        assert tensor.shape[dim] == 1
        return tensor.squeeze(dim)
    return tensor.unflatten(dim, shape)


def jacfwd(func, argnums=0):
    """
    Computes the Jacobian of :attr:`func` with respect to the arg(s) at index
    :attr:`argnum` using forward-mode autodiff

    Args:
        func (function): A Python function that takes one or more arguments,
            one of which must be a Tensor, and returns one or more Tensors
        argnums (int or Tuple[int]): Optional, integer or tuple of integers,
            saying which arguments to get the Jacobian with respect to.
            Default: 0.

    Returns:
        Returns a function that takes in the same inputs as :attr:`func` and
        returns the Jacobian of :attr:`func` with respect to the arg(s) at
        :attr:`argnums`.

    .. warning::
        PyTorch's forward-mode AD coverage on operators is not very good at the
        moment. You may see this API error out with "forward-mode AD not
        implemented for operator X". If so, please file us a bug report and we
        will prioritize it.

    A basic usage with a pointwise, unary operation will give a diagonal array
    as the Jacobian

        >>> from functorch import jacfwd
        >>> x = torch.randn(5)
        >>> jacobian = jacfwd(torch.sin)(x)
        >>> expected = torch.diag(torch.cos(x))
        >>> assert torch.allclose(jacobian, expected)

    :func:`jacfwd` can be composed with vmap to produce batched
    Jacobians:

        >>> from functorch import jacfwd, vmap
        >>> x = torch.randn(64, 5)
        >>> jacobian = vmap(jacfwd(torch.sin))(x)
        >>> assert jacobian.shape == (64, 5, 5)

    Additionally, :func:`jacrev` can be composed with itself or :func:`jacrev`
    to produce Hessians

        >>> from functorch import jacfwd, jacrev
        >>> def f(x):
        >>>   return x.sin().sum()
        >>>
        >>> x = torch.randn(5)
        >>> hessian = jacfwd(jacrev(f))(x)
        >>> assert torch.allclose(hessian, torch.diag(-x.sin()))

    By default, :func:`jacfwd` computes the Jacobian with respect to the first
    input. However, it can compute the Jacboian with respect to a different
    argument by using :attr:`argnums`:

        >>> from functorch import jacfwd
        >>> def f(x, y):
        >>>   return x + y ** 2
        >>>
        >>> x, y = torch.randn(5), torch.randn(5)
        >>> jacobian = jacfwd(f, argnums=1)(x, y)
        >>> expected = torch.diag(2 * y)
        >>> assert torch.allclose(jacobian, expected)

    Additionally, passing a tuple to :attr:`argnums` will compute the Jacobian
    with respect to multiple arguments

        >>> from functorch import jacfwd
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
        f_wrapper, primals = _argnums_partial(func, args, argnums)
        flat_primals, primals_spec = tree_flatten(primals)
        flat_primals_numels = tuple(p.numel() for p in flat_primals)
        flat_basis = _construct_standard_basis_for(flat_primals, flat_primals_numels)
        basis = tree_unflatten(flat_basis, primals_spec)

        def push_jvp(basis):
            _, jvp_out = jvp(f_wrapper, primals, basis)
            return jvp_out

        results = vmap(push_jvp)(basis)
        assert_output_is_tensor_or_tensors(results, 'jacfwd(f, ...)(*args)')

        jac_outs, spec = tree_flatten(results)
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
        return tree_unflatten(jac_outs_ins, spec)
    return wrapper_fn


def hessian(func, argnums=0):
    """
    Computes the Hessian of :attr:`func` with respect to the arg(s) at index
    :attr:`argnum` via a forward-over-reverse strategy.

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
        Returns a function that takes in the same inputs as :attr:`func` and
        returns the Hessian of :attr:`func` with respect to the arg(s) at
        :attr:`argnums`.

    .. warning::
        PyTorch's forward-mode AD coverage on operators is not very good at the
        moment. You may see this API error out with "forward-mode AD not
        implemented for operator X". If so, please file us a bug report and we
        will prioritize it.

    A basic usage with a R^N -> R^1 function gives a N x N Hessian:

        >>> from functorch import hessian
        >>> def f(x):
        >>>   return x.sin().sum()
        >>>
        >>> x = torch.randn(5)
        >>> hess = jacfwd(jacrev(f))(x)
        >>> assert torch.allclose(hess, torch.diag(-x.sin()))

    """
    return jacfwd(jacrev(func, argnums), argnums)


def grad_and_value(func: Callable, argnums: argnums_t = 0, has_aux: bool = False) -> Callable:
    """
    Returns a function to compute a tuple of the gradient and primal, or
    forward, computation.

    Args:
        func (Callable): A Python function that takes one or more arguments.
            Must return a single-element Tensor. If specified :attr:`has_aux`
            equals ``True``, function can return a tuple of single-element
            Tensor and other auxiliary objects: ``(output, aux)``.
        argnums (int or Tuple[int]): Specifies arguments to compute gradients
            with respect to. :attr:`argnums` can be single integer or tuple of
            integers. Default: 0.
        has_aux (bool): Flag indicating that :attr:`func` returns a tensor and
            other auxiliary objects: ``(output, aux)``. Default: False.

    Returns:
        Function to compute a tuple of gradients with respect to its inputs
        and the forward computation. By default, the output of the function is
        a tuple of the gradient tensor(s) with respect to the first argument
        and the primal computation. If specified :attr:`has_aux` equals
        ``True``, tuple of gradients and tuple of the forward computation with
        output auxiliary objects is returned. If :attr:`argnums` is a tuple of
        integers, a tuple of a tuple of the output gradients with respect to
        each :attr:`argnums` value and the forward computation is returned.

    See :func:`grad` for examples
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        level = _grad_increment_nesting()
        output, aux, grad_input = None, None, None
        try:
            # See NOTE [grad and vjp interaction with no_grad]
            with torch.enable_grad():
                args = _wrap_all_tensors(args, level)
                kwargs = _wrap_all_tensors(kwargs, level)
                diff_args = _slice_argnums(args, argnums, as_tuple=False)
                tree_map_(partial(_create_differentiable, level=level), diff_args)

                output = func(*args, **kwargs)
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


def grad(func: Callable, argnums: argnums_t = 0, has_aux: bool = False) -> Callable:
    """``grad`` operator helps computing gradients of :attr:`func` with respect to the
    input(s) specified by :attr:`argnums`. This operator can be nested to
    compute higher-order gradients.

    Args:
        func (Callable): A Python function that takes one or more arguments.
            Must return a single-element Tensor. If specified :attr:`has_aux` equals ``True``,
            function can return a tuple of single-element Tensor and other auxiliary objects:
            ``(output, aux)``.
        argnums (int or Tuple[int]): Specifies arguments to compute gradients with respect to.
            :attr:`argnums` can be single integer or tuple of integers. Default: 0.
        has_aux (bool): Flag indicating that :attr:`func` returns a tensor and other
            auxiliary objects: ``(output, aux)``. Default: False.

    Returns:
        Function to compute gradients with respect to its inputs. By default, the output of
        the function is the gradient tensor(s) with respect to the first argument.
        If specified :attr:`has_aux` equals ``True``, tuple of gradients and output auxiliary objects
        is returned. If :attr:`argnums` is a tuple of integers, a tuple of output gradients with
        respect to each :attr:`argnums` value is returned.

    Example of using ``grad``:

        >>> from functorch import grad
        >>> x = torch.randn([])
        >>> cos_x = grad(lambda x: torch.sin(x))(x)
        >>> assert torch.allclose(cos_x, x.cos())
        >>>
        >>> # Second-order gradients
        >>> neg_sin_x = grad(grad(lambda x: torch.sin(x)))(x)
        >>> assert torch.allclose(neg_sin_x, -x.sin())

    When composed with ``vmap``, ``grad`` can be used to compute per-sample-gradients:

        >>> from functorch import grad
        >>> from functorch import vmap
        >>> batch_size, feature_size = 3, 5
        >>>
        >>> def model(weights, feature_vec):
        >>>     # Very simple linear model with activation
        >>>     assert feature_vec.dim() == 1
        >>>     return feature_vec.dot(weights).relu()
        >>>
        >>> def compute_loss(weights, example, target):
        >>>     y = model(weights, example)
        >>>     return ((y - target) ** 2).mean()  # MSELoss
        >>>
        >>> weights = torch.randn(feature_size, requires_grad=True)
        >>> examples = torch.randn(batch_size, feature_size)
        >>> targets = torch.randn(batch_size)
        >>> inputs = (weights, examples, targets)
        >>> grad_weight_per_example = vmap(grad(compute_loss), in_dims=(None, 0, 0))(*inputs)

    Example of using ``grad`` with :attr:`has_aux` and :attr:`argnums`:

        >>> from functorch import grad
        >>> def my_loss_func(y, y_pred):
        >>>    loss_per_sample = (0.5 * y_pred - y) ** 2
        >>>    loss = loss_per_sample.mean()
        >>>    return loss, (y_pred, loss_per_sample)
        >>>
        >>> fn = grad(my_loss_func, argnums=(0, 1), has_aux=True)
        >>> y_true = torch.rand(4)
        >>> y_preds = torch.rand(4, requires_grad=True)
        >>> out = fn(y_true, y_preds)
        >>> > output is ((grads w.r.t y_true, grads w.r.t y_preds), (y_pred, loss_per_sample))

    .. note::
        Using PyTorch ``torch.no_grad`` together with ``grad``.

        Case 1: Using ``torch.no_grad`` inside a function:

            >>> def f(x):
            >>>     with torch.no_grad():
            >>>         c = x ** 2
            >>>     return x - c

        In this case, ``grad(f)(x)`` will respect the inner ``torch.no_grad``.

        Case 2: Using ``grad`` inside ``torch.no_grad`` context manager:

            >>> with torch.no_grad():
            >>>     grad(f)(x)

        In this case, ``grad`` will respect the inner ``torch.no_grad``, but not the
        outer one. This is because ``grad`` is a "function transform": its result
        should not depend on the result of a context manager outside of ``f``.

    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        results = grad_and_value(func, argnums, has_aux=has_aux)(*args, **kwargs)
        if has_aux:
            grad, (_, aux) = results
            return grad, aux
        grad, _ = results
        return grad
    return wrapper
