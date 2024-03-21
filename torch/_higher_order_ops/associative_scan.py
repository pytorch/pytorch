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
    input: pytree.PyTree,
    dim: int,
    combine_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
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
        input (torch.Tensor): The input tensor
        dim (int): the dimension to scan over
        combine_fn (Callable): A binary callable with type (Tensor, Tensor) -> Tensor,
            which is pure, pointwise, and satisfies the associative property.
            i.e. ``combine_fn(a, combine_fn(b, c)) == combine_fn(combine_fn(a, b), c)``

    Example::

        def add(x: torch.Tensor, y: torch.Tensor):
            return x + y

        cumsum = associative_scan(x, dim, combine_fn=add)

    """

    leaves, spec = pytree.tree_flatten(input)

    torch._check(len(leaves) >= 1, "expected at least 1 input leaf")
    torch._check(
        all(isinstance(x, torch.Tensor) for x in leaves),
        "input leaves must be a Tensor",
    )
    shape = leaves[0].shape
    ndim = len(shape)

    for x in leaves[1:]:
        torch._check(x.shape == shape, "All input tensors must have the same shape")

    dim = utils.canonicalize_dim(ndim, dim)
    torch._check(callable(combine_fn), "combine_fn must be a callable")

    combine_fn = functools.partial(
        wrap_combine_fn_flat, combine_fn=combine_fn, spec=spec, num_leaves=len(leaves)
    )

    if torch._dynamo.is_compiling():
        result_flat = associative_scan_op(input, dim, combine_fn)
    elif not torch._dynamo.is_dynamo_supported():
        raise RuntimeError("associative_scan requires dynamo support.")
    else:
        with _set_compilation_env(), torch._dynamo.utils.disable_cache_limit():
            result_flat = torch.compile(associative_scan_op, fullgraph=True)(
                leaves, dim, combine_fn
            )

    return pytree.tree_unflatten(result_flat, spec)


associative_scan_op = HigherOrderOperator("associative_scan")


def trace_associative_scan(
    proxy_mode, func_overload, input: List[torch.Tensor], dim: int, combine_fn: Callable
):
    pre_dispatch = getattr(proxy_mode, "pre_dispatch", False)

    with disable_proxy_modes_tracing():
        sample_inputs = [
            torch.full((), False, dtype=x.dtype, device=x.device)
            for x in itertools.chain(input, input)
        ]
        combine_graph = reenter_make_fx(combine_fn, pre_dispatch=pre_dispatch)(
            *sample_inputs
        )

    outputs = None
    for node in combine_graph.graph.nodes:
        if node.op == "output":
            assert outputs is None
            assert len(node.args) == 1
            outputs = node.args[0]

    assert outputs is not None
    torch._check(
        len(outputs) == len(input),
        f"expected combine_fn to return {len(input)} results but got {len(outputs)}",
    )

    for i, o in zip(input, outputs):
        o_meta = o.meta["tensor_meta"]
        torch._check(
            o_meta.dtype == i.dtype,
            lambda: (
                f"combine_fn output type mismatch, expected {i.dtype} "
                + f"but got {o_meta.dtype}"
            ),
        )
        torch._check(
            o_meta.shape == (),
            lambda: f"combine_fn must return a scalar tensor but got shape {o_meta.shape}",
        )

    _, combine_graph_name = unique_graph_id(proxy_mode, prefix="scan_combine_graph")

    proxy_mode.tracer.root.register_module(combine_graph_name, combine_graph)

    args = (input, dim, combine_graph)
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, args)
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", func_overload, proxy_args, {}, name="associative_scan"
    )

    with disable_proxy_modes_tracing():
        out = [aten.clone(x) for x in input]

    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


@associative_scan_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def associative_scan_op_dense(input, dim, combine_fn):
    raise NotImplementedError("associative_scan is not implemented for eager")


associative_scan_op.py_impl(DispatchKey.Autograd)(
    autograd_not_implemented(associative_scan_op, deferred_error=True)
)


@associative_scan_op.py_impl(ProxyTorchDispatchMode)
def associative_scan_proxy_mode(mode, input, dim, combine_fn):
    if mode.enable_tracing:
        return trace_associative_scan(mode, associative_scan_op, input, dim, combine_fn)
    else:
        return associative_scan_op(mode, associative_scan_op, input, dim, combine_fn)


@associative_scan_op.py_impl(FakeTensorMode)
def assoiciative_scan_fake_tensor_mode(mode, input, dim, combine_fn):
    with mode:
        return [x.clone() for x in input]


@associative_scan_op.py_functionalize_impl
def associative_scan_functionalize(ctx, input, dim, combine_fn):
    unwrapped_input = ctx.unwrap_tensors(input)
    with ctx.redispatch_to_next() as m:
        ret = associative_scan_op(unwrapped_input, dim, combine_fn)
    return ctx.wrap_tensors(ret)


@associative_scan_op.py_impl(torch._C._functorch.TransformType.Vmap)
def associative_scan_batch_rule(interpreter, input, dim, combine_fn):
    input_ = [get_unwrapped(x) for x in input]
    bdim = [maybe_get_bdim(x) for x in input]

    batch_size = None
    for x in input:
        bdim = maybe_get_bdim(x)
        if bdim is not None:
            batch_size = get_unwrapped(x).shape[bdim]

    assert batch_size
    input_unwrapped = []
    for x in input:
        unwrap = get_unwrapped(x)
        bdim = maybe_get_bdim(x)
        if bdim is None:
            unwrap = unwrap.unsqueeze(0).expand(batch_size, *x.shape)
        else:
            unwrap = unwrap.movedim(bdim, 0)
        input_unwrapped.append(unwrap)

    res = associative_scan_op(input_unwrapped, dim + 1, combine_fn)
    lvl = interpreter.level()
    return [_add_batch_dim(x, 0, lvl) for x in res]
