from typing import Callable

import torch

import torch._prims_common as utils
import torch._subclasses.functional_tensor

import torch.utils._pytree as pytree

from torch._C import DispatchKey
from torch._C._functorch import _add_batch_dim, get_unwrapped, maybe_get_bdim
from torch._higher_order_ops.utils import (
    _maybe_run_with_interpreter,
    _set_compilation_env,
    autograd_not_implemented,
    unique_graph_id,
)

from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    make_fx,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)

aten = torch._ops.ops.aten


def associative_scan(
    combine_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    input: torch.Tensor,
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
        combine_fn (Callable): A binary callable with type (Tensor, Tensor) -> Tensor,
            which is pure, pointwise, and satisfies the associative property.
            i.e. ``combine_fn(a, combine_fn(b, c)) == combine_fn(combine_fn(a, b), c)``
        input (torch.Tensor): The input tensor
        dim (int): the dimension to scan over

    Example::

        def add(x: torch.Tensor, y: torch.Tensor):
            return x + y

        cumsum = associative_scan(add, x, dim)

    """

    assert isinstance(input, torch.Tensor), "input must be a Tensor"
    dim = utils.canonicalize_dim(input.ndim, dim)
    assert callable(combine_fn), "combine_fn must be a callable"

    if torch._dynamo.is_compiling():
        return associative_scan_op(combine_fn, input, dim)

    if not torch._dynamo.is_dynamo_supported():
        raise RuntimeError("associative_scan requires dynamo support.")

    with _set_compilation_env(), torch._dynamo.utils.disable_cache_limit():
        return torch.compile(associative_scan_op, fullgraph=True)(
            combine_fn, input, dim
        )


associative_scan_op = HigherOrderOperator("associative_scan")


def trace_associative_scan(proxy_mode, func_overload, combine_fn, input, dim):
    pre_dispatch = getattr(proxy_mode, "pre_dispatch", False)

    with disable_proxy_modes_tracing():
        sample_inputs = (
            torch.full((), False, dtype=input.dtype, device=input.device),
            torch.full((), False, dtype=input.dtype, device=input.device),
        )
        combine_graph = make_fx(
            _maybe_run_with_interpreter(combine_fn), pre_dispatch=pre_dispatch
        )(*sample_inputs)

    outputs = []
    for node in combine_graph.graph.nodes:
        if node.op == "output":
            outputs.extend(node.args)

    assert (
        len(outputs) == 1
    ), f"expected combine_fn to have 1 output but got {len(outputs)}"

    for o in outputs:
        o_meta = o.meta["tensor_meta"]
        assert o_meta.dtype == input.dtype, (
            f"combine_fn output type mismatch, expected {input.dtype} "
            + f"but got {o_meta.dtype}"
        )
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
        out = aten.clone(input)

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
        return input.clone()


@associative_scan_op.py_functionalize_impl
def associative_scan_functionalize(ctx, combine_fn, input, dim):
    unwrapped_input = ctx.unwrap_tensors(input)
    with ctx.redispatch_to_next() as m:
        ret = associative_scan_op(combine_fn, unwrapped_input, dim)
    return ctx.wrap_tensors(ret)


@associative_scan_op.py_impl(torch._C._functorch.TransformType.Vmap)
def associative_scan_batch_rule(interpreter, combine_fn, input, dim):
    input_ = get_unwrapped(input)
    bdim = maybe_get_bdim(input)
    res = associative_scan_op(combine_fn, input_, dim + (dim >= bdim))
    lvl = interpreter.level()
    return _add_batch_dim(res, bdim, lvl)
