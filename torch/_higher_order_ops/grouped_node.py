# mypy: allow-untyped-defs
from typing import Callable, Tuple, Union

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._higher_order_ops.utils import (
    _has_potential_branch_input_alias,
    _has_potential_branch_input_mutation,
    _set_compilation_env,
    autograd_not_implemented,
    reenter_make_fx,
    UnsupportedAliasMutationException,
)
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree


class GroupedNodeOp(HigherOrderOperator):
    def __init__(self):
        super().__init__("grouped_node")

    def __call__(
        self,
        fn: Callable,
        operands: Tuple[Union[torch.Tensor, int, float, bool]],
        /,
    ):
        if not isinstance(operands, tuple):
            raise RuntimeError(
                f"operands must be a tuple, got {type(operands)}"
            )
        if not all(
            isinstance(t, (torch.Tensor, int, float, bool)) for t in operands
        ):
            raise RuntimeError(
                "operands must be a tuple of tensors, ints, floats, or bools, got "
                f"{operands}"
            )

        return super().__call__(fn, operands)


grouped_node_op = GroupedNodeOp()
# Override grouped_node_op.__module__ to "torch.ops.higher_order" so that in the generated
# graph module, grouped_node node's target is correctedly printed as torch.ops.higher_order.grouped_node
grouped_node_op.__module__ = "torch.ops.higher_order"


def grouped_node(fn, operands):
    if torch.compiler.is_dynamo_compiling():
        return grouped_node_op(fn, operands)

    def _validate_input(fn, operands):
        if not callable(fn):
            raise RuntimeError("Expect fn to be callbale.")

        if not isinstance(operands, (tuple, list)) or pytree.tree_any(
            lambda t: not isinstance(t, torch.Tensor), operands
        ):
            raise RuntimeError(
                "Expect operands to be a tuple of possibly nested dict/list/tuple that only"
                f"consists of tensor leaves, but got {operands}."
            )

    _validate_input(fn, operands)

    with _set_compilation_env(), torch._dynamo.utils.disable_cache_limit():
        return torch.compile(grouped_node_op, backend="eager", fullgraph=True)(
            fn, operands
        )


@grouped_node_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def grouped_node_dense(fn, operands):
    out = fn(*operands)
    assert isinstance(
        out, tuple
    ), f"body_fn should return a tuple but got {type(out)}"
    return out


grouped_node_op.py_impl(DispatchKey.Autograd)(
    autograd_not_implemented(grouped_node_op, deferred_error=True)
)


@grouped_node_op.py_impl(ProxyTorchDispatchMode)
def grouped_node_tracing(mode, fn, operands):
    def _trace_grouped_node(
        proxy_mode, grouped_node_op, fn, operands
    ):
        graph = reenter_make_fx(fn)(*operands)

        next_name = None
        i = 0
        while not next_name:
            candidate = f"grouped_node_graph_{i}"
            if hasattr(proxy_mode.tracer.root, candidate):
                i += 1
            else:
                next_name = candidate
        graph_name = grouped_node
        assert not hasattr(proxy_mode.tracer.root, graph_name)

        proxy_mode.tracer.root.register_module(graph_name, graph)

        args = (graph, operands)

        proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, args)

        out_proxy = proxy_mode.tracer.create_proxy(
            "call_function", grouped_node_op, proxy_args, {}, name="grouped_node"
        )

        out = fn(*operands)
        return track_tensor_tree(
            out, out_proxy, constant=None, tracer=proxy_mode.tracer
        )

    if mode.enable_tracing:
        return _trace_grouped_node(
            mode, grouped_node_op, fn, operands
        )
    else:
        return grouped_node_op(fn, operands)


@grouped_node_op.py_impl(FakeTensorMode)
def grouped_node_fake_tensor_mode(
    mode, fn, operands
):
    with mode:
        return fn(*operands)


@grouped_node_op.py_functionalize_impl
def grouped_node_func(ctx, fn, operands):
    unwrapped_operands = ctx.unwrap_tensors(operands)
    with ctx.redispatch_to_next() as m:
        functional_fn = ctx.functionalize(fn)
        pre_dispatch = hasattr(ctx, "mode") and ctx.mode.pre_dispatch
        ret = grouped_node_op(
            functional_fn,
            unwrapped_operands,
        )
        return ctx.wrap_tensors(ret)
