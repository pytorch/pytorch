# mypy: allow-untyped-defs
import functools
import itertools
from typing import Callable, List

import torch
import torch._prims_common as utils
import torch._subclasses.functional_tensor
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._C._functorch import (
    _add_batch_dim,
    get_unwrapped,
    is_batchedtensor,
    maybe_get_bdim,
)
from torch._higher_order_ops.utils import (
    _set_compilation_env,
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
    assert len(args) == 2 * num_leaves
    lhs = pytree.tree_unflatten(args[:num_leaves], spec)
    rhs = pytree.tree_unflatten(args[num_leaves:], spec)
    combined = combine_fn(lhs, rhs)
    combined_leaves = pytree.tree_leaves(combined)
    assert num_leaves == len(combined_leaves)
    return combined_leaves



def _interleave(a, b, dim):
    # https://stackoverflow.com/questions/60869537/how-can-i-interleave-5-pytorch-tensors
    if b_trunc := (a.shape[dim] == b.shape[dim] + 1):
        # pad = [0, 0] * b.ndim
        # pad[
        #     (b.ndim - dim - 1) * 2 + 1
        # ] = 1  # +1=always end of dim, pad-order is reversed so start is at end
        pad = [0] * ((b.ndim - dim - 1) * 2 + 1) + [1] + [0] * (b.ndim * 2 - ((b.ndim - dim - 1) * 2 + 2))
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
    combine_mode: str = "pointwise",
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
            This function must be pure, pointwise, and satisfy the associative property.
        input (torch.Tensor): The input tensor, or nested pytree of tensors.
            All inputs are expected to have the same shape.
        dim (int): the dimension to scan over
        reverse (bool): A boolean stating if the scan should be reversed with respect to the dimension.
        combine_mode (str): A string indicating whether the ``combine_fn`` is ``pointwise`` or ``generic``.
            If ``combine_mode=pointwise``, ``combine_fn`` must be pure, may only contain pointwise operations and ``input`` must be CUDA tensors. 
            In all other cases ``combine_mode=generic`` should be used.
            Note: ``combine_mode=pointwise`` is more efficient than ``combine_mode=generic``.


    Example::

        def add(x: torch.Tensor, y: torch.Tensor):
            return x + y

        cumsum = associative_scan(add, x, dim)

    """
    assert callable(combine_fn), "combine_fn must be a callable, but got {combine_fn}"
    assert isinstance(dim, int), "dim must be an int, but got {type(dim)}"
    assert combine_mode in ["pointwise", "generic"]

    if not torch._dynamo.is_compiling():
        with _set_compilation_env(), torch._dynamo.utils.disable_cache_limit():
            return torch.compile(associative_scan, fullgraph=True)(
                combine_fn, input, dim, reverse=reverse
            )

    leaves, spec = pytree.tree_flatten(input)

    if reverse:
        leaves = [torch.flip(elem, [dim]) for elem in leaves]

    assert len(leaves) >= 1, "expected at least 1 input leaf"
    assert all(
        isinstance(x, torch.Tensor) for x in leaves
    ), "input leaves must be a Tensor"
    
    shape = leaves[0].shape
    ndim = len(shape)
    dim = utils.canonicalize_dim(ndim, dim)

    for x in leaves[1:]:
        assert x.shape == shape, "All input tensors must have the same shape"

    out = combine_fn(
        pytree.tree_unflatten(leaves, spec),
        pytree.tree_unflatten(leaves, spec),
    )
    out_leaves, tree_out = pytree.tree_flatten(out)
    # assert spec.num_nodes != tree_out.num_nodes or any(
    #     o.shape != i.shape or o.dtype != i.dtype or o.device != i.device
    #     for o, i in zip(out_leaves, leaves)), "The pytree of the output of the operator needs to match the input pytree"
    

    combine_fn = functools.partial(
        wrap_combine_fn_flat, combine_fn=combine_fn, spec=spec, num_leaves=len(leaves)
    )

    #result_flat = associative_scan_op(combine_fn, leaves, dim)
    if combine_mode == 'generic':
        result_flat = generic_associative_scan(combine_fn, leaves, dim)
    else:
        result_flat = associative_scan_op(combine_fn, leaves, dim)

    if reverse:
        result_flat = [torch.flip(elem, [dim]) for elem in result_flat]

    return pytree.tree_unflatten(result_flat, spec)


def generic_associative_scan(operator, elems_flat, dim=0):
    def _scan(elems):
        """Perform scan on `elems`."""
        num_elems = elems[0].shape[dim]

        if num_elems < 2:
            return elems

        reduced_elems = operator(
            *[elem[slice_along_axis(0, -1, stride=2, dim=dim)] for elem in elems],
            *[elem[slice_along_axis(1, None, stride=2, dim=dim)] for elem in elems]
        )

        # Recursively compute scan for partially reduced tensors.
        odd_elems = _scan(reduced_elems)

        if num_elems % 2 == 0:
            even_elems = operator(
                *[e[slice_along_axis(0, -1, dim=dim)] for e in odd_elems],
                *[e[slice_along_axis(2, None, stride=2, dim=dim)] for e in elems]
            )
        else:
            even_elems = operator(
                *odd_elems,
                *[e[slice_along_axis(2, None, stride=2, dim=dim)] for e in elems]
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
    proxy_mode, func_overload, combine_fn: Callable, input: List[torch.Tensor], dim: int
):
    with disable_proxy_modes_tracing():
        sample_inputs = [
            torch.empty_like(
                x,
                dtype=x.dtype,
                device=x.device,
                requires_grad=x.requires_grad,
            )
            for x in itertools.chain(input, input)
        ]
        combine_graph = reenter_make_fx(combine_fn)(*sample_inputs)

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
        assert (
            o_meta.shape == ()
        ), f"combine_fn must return a scalar tensor but got shape {o_meta.shape}"
        assert (
            o_meta.shape == ()
        ), f"combine_fn must return a scalar tensor but got shape {o_meta.shape}"

    _, combine_graph_name = unique_graph_id(proxy_mode, prefix="scan_combine_graph")

    proxy_mode.tracer.root.register_module(combine_graph_name, combine_graph)

    args = (combine_graph, input, dim)
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, args)
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", func_overload, proxy_args, {}, name="associative_scan"
    )

    with disable_proxy_modes_tracing():
        out = [aten.clone(x) for x in input]

    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


@associative_scan_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def associative_scan_op_dense(combine_fn, input, dim):
    raise NotImplementedError("associative_scan is not implemented for eager")


associative_scan_op.py_impl(DispatchKey.Autograd)(
    autograd_not_implemented(associative_scan_op, deferred_error=True)
)


@associative_scan_op.py_impl(ProxyTorchDispatchMode)
def associative_scan_proxy_mode(mode, combine_fn, input, dim):
    if mode.enable_tracing:
        return trace_associative_scan(mode, associative_scan_op, combine_fn, input, dim)
    else:
        return associative_scan_op(mode, associative_scan_op, combine_fn, input, dim)


@associative_scan_op.py_impl(FakeTensorMode)
def assoiciative_scan_fake_tensor_mode(mode, combine_fn, input, dim):
    with mode:
        return [x.clone() for x in input]


@associative_scan_op.py_functionalize_impl
def associative_scan_functionalize(ctx, combine_fn, input, dim):
    unwrapped_input = ctx.unwrap_tensors(input)
    with ctx.redispatch_to_next() as m:
        functional_combine_fn = ctx.functionalize(combine_fn)
        ret = associative_scan_op(functional_combine_fn, unwrapped_input, dim)
    return ctx.wrap_tensors(ret)


@associative_scan_op.py_impl(torch._C._functorch.TransformType.Vmap)
def associative_scan_batch_rule(interpreter, combine_fn, input, dim):
    input_bdims = [maybe_get_bdim(x) if is_batchedtensor(x) else None for x in input]

    batch_size = None
    for inp, bdim in zip(input, input_bdims):
        if bdim is not None:
            batch_size = get_unwrapped(inp).shape[bdim] if is_batchedtensor(inp) else inp.shape[bdim]

    assert batch_size
    input_unwrapped = []
    for x, bdim in zip(input, input_bdims):
        unwrap = get_unwrapped(x) if is_batchedtensor(x) else x
        if dim is None:
            unwrap = unwrap.unsqueeze(0).expand(batch_size, *x.shape)
        else:
            if bdim is None:
                unwrap = unwrap.unsqueeze(0).expand(batch_size, *x.shape)
            else:
                unwrap = unwrap.movedim(bdim, 0)
        input_unwrapped.append(unwrap)

    with interpreter.lower():
        res = associative_scan_op(combine_fn, input_unwrapped, dim + 1)
    
    lvl = interpreter.level()
    return [_add_batch_dim(x, 0, lvl) for x in res]
