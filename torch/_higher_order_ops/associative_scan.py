# mypy: allow-untyped-defs
import functools
import itertools
from typing import Callable, List, Tuple

import torch
import torch._prims_common as utils
import torch._subclasses.functional_tensor
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._C._functorch import _add_batch_dim, get_unwrapped, maybe_get_bdim
from torch._higher_order_ops.utils import (
    autograd_not_implemented,
    reenter_make_fx,
    unique_graph_id,
)

from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)


aten = torch._ops.ops.aten


def wrap_combine_fn_flat(*args, combine_fn, spec, num_leaves):
    if len(args) != 2 * num_leaves:
        raise ValueError(
            "The number of leaves provided to the combine wrapper needs to be twice the number of arguments"
        )
    lhs = pytree.tree_unflatten(args[:num_leaves], spec)
    rhs = pytree.tree_unflatten(args[num_leaves:], spec)
    combined = combine_fn(lhs, rhs)
    combined_leaves = pytree.tree_leaves(combined)
    if num_leaves != len(combined_leaves):
        raise ValueError(
            "The number of levaes of the inputs need to be identical to the number of leaves of the scan output"
        )
    return combined_leaves


def check_args(combine_fn, leaves, tree, dim):
    if not callable(combine_fn):
        raise ValueError("combine_fn must be a callable, but got {combine_fn}")
    if not isinstance(dim, int):
        raise ValueError("dim must be an int, but got {type(dim)}")

    if len(leaves) < 1:
        raise ValueError("expected at least 1 input leaf")

    if not all(isinstance(x, torch.Tensor) for x in leaves):
        raise ValueError("input leaves must be a Tensor")
    shape = leaves[0].shape
    ndim = len(shape)
    dim = utils.canonicalize_dim(ndim, dim)

    for x in leaves[1:]:
        if x.shape != shape:
            raise ValueError("All input tensors must have the same shape")

    out = combine_fn(
        pytree.tree_unflatten(leaves, tree),
        pytree.tree_unflatten(leaves, tree),
    )

    out_leaves, tree_out = pytree.tree_flatten(out)
    if tree.num_nodes != tree_out.num_nodes or any(
        o.shape != i.shape or o.dtype != i.dtype or o.device != i.device
        for o, i in zip(out_leaves, leaves)
    ):
        raise ValueError(
            "The pytree of the output of the operator needs to match the input pytree"
        )


def _interleave(a, b, dim):
    # https://stackoverflow.com/questions/60869537/how-can-i-interleave-5-pytorch-tensors
    if b_trunc := (a.shape[dim] == b.shape[dim] + 1):
        pad = [0, 0] * b.ndim
        pad[
            (b.ndim - dim - 1) * 2 + 1
        ] = 1  # +1=always end of dim, pad-order is reversed so start is at end
        b = torch.nn.functional.pad(b, pad)

    stacked = torch.stack([a, b], dim=dim + 1)
    interleaved = torch.flatten(stacked, start_dim=dim, end_dim=dim + 1)
    if b_trunc:
        # TODO: find torch alternative for slice_along dim for torch.jit.script to work
        interleaved = interleaved[
            slice_along_axis(0, b.shape[dim] + a.shape[dim] - 1, dim=dim)
        ]
    return interleaved


def safe_map(f, *args):
    args = list(map(list, args))
    n = len(args[0])
    for arg in args[1:]:
        if len(arg) != n:
            raise ValueError("length mismatch: {list(map(len, args))}")

    def nf(a):
        return f(*a)

    return list(map(nf, zip(*args)))


def slice_along_axis(start, end, stride=None, dim=0):
    return (slice(None),) * dim + (slice(start, end, stride),)


def associative_scan(
    combine_fn: Callable[[pytree.PyTree, pytree.PyTree], pytree.PyTree],
    input: pytree.PyTree,
    dim: int,
    reverse: bool = False,
    generic_scan: bool = False,
    lifted_args: Tuple = (),
) -> torch.Tensor:
    r"""
    Performs an inclusive scan with an associative pointwise combine function.

    .. warning::
        `torch.associative_scan` is a prototype feature in PyTorch. It currently
        does not support autograd and you may run into miscompiles.
        Read more about feature classification at:
        https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype

    This operator requires runtime code generation and so requires support for
    ``torch.compile``. Further, only CUDA device codegen is supported at the moment.

    Args:
        combine_fn (Callable): A binary callable with type ``(Tensor, Tensor) -> Tensor``,
            or if input is a pytree ``(pytree, pytree) -> pytree``.
            This function must satisfy the associativity property.
        input (Tuple of possibly nested dict/list/tuple of tensors): The input tensor, or nested pytree of tensors.
            All inputs are expected to have the same shape.
        dim (int): The dimension to scan over
        reverse (bool): A boolean stating if the scan should be reversed with respect to the dimension.
        generic_scan (bool): A boolean stating whether a generic scan mode should be used.
            If ``generic_scan=False``, ``combine_op`` must be pure and may only contain pointwise operations.
            Moreover, ``generic_scan=False`` may just be used on CUDA tensors.
            On the other hand, ``generic_scan=False`` should be more efficient than ``generic_scan=True``,
            whenever it can be used.
            Note: This argument is automatically computed internally, but ``generic_scan=True`` can be enforced
            Note: In case the output of `torch.associative_scan` is part of backward(),
            i.e., gradients need to propagate through `torch.associative_scan`,
            then ``generic_scan=True`` is required
        lifted_args (Tuple of tensors): A tuple of lifted parameters from the global scope.
            This parameter will be populated internally.

    Example::

        def add(x: torch.Tensor, y: torch.Tensor):
            return x + y

        cumsum = associative_scan(add, x, dim)

    """
    leaves, spec = pytree.tree_flatten(input)

    check_args(combine_fn, leaves, spec, dim)

    if reverse:
        leaves = [torch.flip(elem, [dim]) for elem in leaves]

    combine_fn = functools.partial(
        wrap_combine_fn_flat, combine_fn=combine_fn, spec=spec, num_leaves=len(leaves)
    )

    if generic_scan:
        result_flat = generic_associative_scan(combine_fn, leaves, dim, lifted_args)
    else:
        result_flat = associative_scan_op(combine_fn, leaves, dim, lifted_args)

    if reverse:
        result_flat = [torch.flip(elem, [dim]) for elem in result_flat]

    return pytree.tree_unflatten(result_flat, spec)


def generic_associative_scan(operator, elems_flat, dim=0, lifted_args=()):
    # TODO: The recursion involved here "unrolls" the scan
    # function for all inputs. Could there be a more efficient
    # way instead of running over the operation in sequence?
    def _scan(elems):
        """Perform scan on `elems`."""
        num_elems = elems[0].shape[dim]

        if num_elems < 2:
            return elems

        reduced_elems = operator(
            *[elem[slice_along_axis(0, -1, stride=2, dim=dim)] for elem in elems],
            *[elem[slice_along_axis(1, None, stride=2, dim=dim)] for elem in elems],
            *lifted_args,
        )

        # Recursively compute scan for partially reduced tensors.
        odd_elems = _scan(reduced_elems)

        if num_elems % 2 == 0:
            even_elems = operator(
                *[e[slice_along_axis(0, -1, dim=dim)] for e in odd_elems],
                *[e[slice_along_axis(2, None, stride=2, dim=dim)] for e in elems],
                *lifted_args,
            )
        else:
            even_elems = operator(
                *odd_elems,
                *[e[slice_along_axis(2, None, stride=2, dim=dim)] for e in elems],
                *lifted_args,
            )

        # The first element of a scan is the same as the first element
        # of the original `elems`.
        even_elems = [
            torch.cat([elem[slice_along_axis(0, 1, dim=dim)], result], dim=dim)
            if result.shape.numel() > 0 and elem.shape[dim] > 0
            else result
            if result.shape.numel() > 0
            else elem[
                slice_along_axis(0, 1, dim=dim)
            ]  # Jax allows/ignores concat with 0-dim, Pytorch does not
            for (elem, result) in zip(elems, even_elems)
        ]

        return list(
            safe_map(functools.partial(_interleave, dim=dim), even_elems, odd_elems)
        )

    scans = _scan(elems_flat)

    return scans


associative_scan_op = HigherOrderOperator("associative_scan")


def trace_associative_scan(
    proxy_mode,
    func_overload,
    combine_fn: Callable,
    input: List[torch.Tensor],
    dim: int,
    lifted_args: Tuple[torch.Tensor],
):
    from torch.fx.experimental.proxy_tensor import maybe_handle_decomp

    with disable_proxy_modes_tracing():
        sample_inputs = [
            torch.empty_like(
                x[slice_along_axis(0, 1, stride=None, dim=dim)],
                dtype=x.dtype,
                device=x.device,
                requires_grad=x.requires_grad,
            )
            for x in itertools.chain(input, input)
        ]
        combine_graph = reenter_make_fx(combine_fn)(*sample_inputs, *lifted_args)

    outputs = None
    for node in combine_graph.graph.nodes:
        if node.op == "output":
            assert outputs is None
            assert len(node.args) == 1
            outputs = node.args[0]

    assert outputs is not None
    assert len(outputs) == len(
        input
    ), f"expected combine_fn to return {len(input)} results but got {len(outputs)}"

    for i, o in zip(input, outputs):
        o_meta = o.meta["tensor_meta"]
        assert o_meta.dtype == i.dtype, (
            f"combine_fn output type mismatch, expected {i.dtype} "
            + f"but got {o_meta.dtype}"
        )

    _, combine_graph_name = unique_graph_id(proxy_mode, prefix="scan_combine_graph")

    proxy_mode.tracer.root.register_module(combine_graph_name, combine_graph)

    args = (combine_graph, input, dim, lifted_args)
    out = maybe_handle_decomp(proxy_mode, associative_scan_op, args, {})
    if out is not NotImplemented:
        return out

    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, args)
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", func_overload, proxy_args, {}, name="associative_scan"
    )

    with disable_proxy_modes_tracing():
        out = [aten.clone(x) for x in input]

    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


@associative_scan_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def associative_scan_op_dense(combine_fn, input, dim, lifted_args):
    return generic_associative_scan(combine_fn, input, dim, lifted_args)


# @associative_scan_op.py_impl(DispatchKey.CompositeExplicitAutograd)
# def associative_scan_op_dense(combine_fn, input, dim):
#     raise NotImplementedError("associative_scan is not implemented for eager")


associative_scan_op.py_impl(DispatchKey.Autograd)(
    autograd_not_implemented(associative_scan_op, deferred_error=True)
)

# @associative_scan_op.py_impl(DispatchKey.Autograd)
# def associative_scan_op_autograd(combine_fn, input, dim):
#     return generic_associative_scan(combine_fn, input, dim)


@associative_scan_op.py_impl(ProxyTorchDispatchMode)
def associative_scan_proxy_mode(mode, combine_fn, input, dim, lifted_args):
    if mode.enable_tracing:
        return trace_associative_scan(
            mode, associative_scan_op, combine_fn, input, dim, lifted_args
        )
    else:
        return associative_scan_op(
            mode, associative_scan_op, combine_fn, input, dim, lifted_args
        )


@associative_scan_op.py_impl(FakeTensorMode)
def assoiciative_scan_fake_tensor_mode(mode, combine_fn, input, dim, lifted_args):
    with mode:
        return [x.clone() for x in input]


@associative_scan_op.py_functionalize_impl
def associative_scan_functionalize(ctx, combine_fn, input, dim, lifted_args):
    unwrapped_input = ctx.unwrap_tensors(input)
    unwrapped_lifted_args = ctx.unwrap_tensors(lifted_args)
    with ctx.redispatch_to_next() as m:
        functional_combine_fn = ctx.functionalize(combine_fn)
        ret = associative_scan_op(
            functional_combine_fn, unwrapped_input, dim, unwrapped_lifted_args
        )
    return ctx.wrap_tensors(ret)


@associative_scan_op.py_impl(torch._C._functorch.TransformType.Vmap)
def associative_scan_batch_rule(interpreter, input, dim, combine_fn):
    input_ = [get_unwrapped(x) for x in input]
    input_bdims = [maybe_get_bdim(x) for x in input]

    batch_size = None
    for inp, bdim in zip(input, input_bdims):
        if bdim is not None:
            batch_size = get_unwrapped(inp).shape[bdim]

    assert batch_size
    input_unwrapped = []
    for x, bdim in zip(input, input_bdims):
        unwrap = get_unwrapped(x)
        if dim is None:
            unwrap = unwrap.unsqueeze(0).expand(batch_size, *x.shape)
        else:
            unwrap = unwrap.movedim(bdim, 0)
        input_unwrapped.append(unwrap)

    res = associative_scan_op(combine_fn, input_unwrapped, dim + 1)
    lvl = interpreter.level()
    return [_add_batch_dim(x, 0, lvl) for x in res]
