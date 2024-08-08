# mypy: allow-untyped-defs
import functools
import itertools
from typing import Callable, List

import torch
import torch._prims_common as utils
import torch._subclasses.functional_tensor
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._C._functorch import _add_batch_dim, get_unwrapped, maybe_get_bdim
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


def associative_scan(
    combine_fn: Callable[[pytree.PyTree, pytree.PyTree], pytree.PyTree],
    input: pytree.PyTree,
    dim: int,
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


    Example::

        def add(x: torch.Tensor, y: torch.Tensor):
            return x + y

        cumsum = associative_scan(add, x, dim)

    """
    assert callable(combine_fn), "combine_fn must be a callable, but got {combine_fn}"
    assert isinstance(dim, int), "dim must be an int, but got {type(dim)}"

    if not torch._dynamo.is_compiling():
        with _set_compilation_env(), torch._dynamo.utils.disable_cache_limit():
            return torch.compile(associative_scan, fullgraph=True)(
                combine_fn, input, dim
            )

    leaves, spec = pytree.tree_flatten(input)

    assert len(leaves) >= 1, "expected at least 1 input leaf"
    assert all(
        isinstance(x, torch.Tensor) for x in leaves
    ), "input leaves must be a Tensor"
    shape = leaves[0].shape
    ndim = len(shape)
    dim = utils.canonicalize_dim(ndim, dim)

    for x in leaves[1:]:
        assert x.shape == shape, "All input tensors must have the same shape"

    combine_fn = functools.partial(
        wrap_combine_fn_flat, combine_fn=combine_fn, spec=spec, num_leaves=len(leaves)
    )

    result_flat = associative_scan_op(combine_fn, leaves, dim)

    return pytree.tree_unflatten(result_flat, spec)


associative_scan_op = HigherOrderOperator("associative_scan")


def trace_associative_scan(
    proxy_mode, func_overload, combine_fn: Callable, input: List[torch.Tensor], dim: int
):
    with disable_proxy_modes_tracing():
        sample_inputs = [
            torch.full((), False, dtype=x.dtype, device=x.device)
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
        ret = associative_scan_op(combine_fn, unwrapped_input, dim)
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
