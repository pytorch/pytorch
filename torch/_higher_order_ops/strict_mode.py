import torch
import torch._subclasses.functional_tensor

import torch.utils._pytree as pytree

from torch._C import DispatchKey
from torch._functorch.utils import exposed_in

from torch._higher_order_ops.utils import _set_compilation_env, autograd_not_implemented
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    make_fx,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)
from torch.utils._python_dispatch import _get_current_dispatch_mode


@exposed_in("torch")
def strict_mode(callable, operands):
    if torch.compiler.is_dynamo_compiling():
        return strict_mode_op(callable, operands)

    with _set_compilation_env():
        with torch._dynamo.utils.disable_cache_limit():
            return torch.compile(strict_mode_op, backend="eager", fullgraph=True)(
                callable, operands
            )


strict_mode_op = HigherOrderOperator("strict_mode")


@strict_mode_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def strict_mode_op_dense(callable, operands):
    mode = _get_current_dispatch_mode()
    assert mode is None, "Mode should never be enabled for CPU/CUDA key"
    return callable(*operands)


strict_mode_op.py_impl(DispatchKey.Autograd)(
    autograd_not_implemented(strict_mode_op, deferred_error=True)
)


@strict_mode_op.py_impl(ProxyTorchDispatchMode)
def inner(mode, callable, operands):
    if mode.enable_tracing:
        return trace_strict_mode(mode, strict_mode_op, callable, operands)
    else:
        return strict_mode_op(callable, operands)


def trace_strict_mode(mode, strict_mode_op, callable, operands):
    pre_dispatch = getattr(mode, "pre_dispatch", False)

    with disable_proxy_modes_tracing():
        graph = make_fx(callable, pre_dispatch=pre_dispatch)(*operands)

    next_name = None
    i = 0
    while not next_name:
        candidate = f"strict_graph_{i}"
        if hasattr(mode.tracer.root, candidate):
            i += 1
        else:
            next_name = candidate

    graph_name = next_name
    mode.tracer.root.register_module(graph_name, graph)

    args = (graph, operands)

    proxy_args = pytree.tree_map(mode.tracer.unwrap_proxy, args)

    out_proxy = mode.tracer.create_proxy(
        "call_function", strict_mode_op, proxy_args, {}, name="strict_mode"
    )

    out = graph(*operands)
    return track_tensor_tree(out, out_proxy, constant=None, tracer=mode.tracer)


@strict_mode_op.py_impl(FakeTensorMode)
def strict_mode_fake_tensor_mode(mode, callable, operands):
    with mode:
        true_outs = callable(*operands)
    return true_outs


@strict_mode_op.py_functionalize_impl
def strict_mode_func(ctx, callable, inputs):
    unwrapped_inputs = ctx.unwrap_tensors(inputs)
    with ctx.redispatch_to_next():
        functional_callable = ctx.functionalize(callable)

        cond_return = strict_mode_op(functional_callable, unwrapped_inputs)
        return ctx.wrap_tensors(cond_return)
