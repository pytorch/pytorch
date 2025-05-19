# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs

import torch
import torch._subclasses.functional_tensor
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._higher_order_ops.utils import (
    _maybe_run_with_interpreter,
    _set_compilation_env,
    autograd_not_implemented,
    reenter_make_fx,
    unique_graph_id,
)
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
    _temp_remove_metadata_torch_function_mode,
    _temp_remove_pre_dispatch_torch_function_mode,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)


class IfOp(HigherOrderOperator):
    def __init__(self):
        super().__init__("if_op")

    def __call__(self, pred, func, *operands):
        return super().__call__(pred, func, *operands)


if_op = IfOp()


def IF(pred, then_func, *operands):
    from torch._dynamo.backends.debugging import (
        make_eager_backend_with_torch_function_mode,
    )

    if torch.compiler.is_dynamo_compiling():
        return if_op(pred, then_func, *operands)

    def _if_op_wrapper(*args, **kwargs):
        return if_op(*args, **kwargs)

    with _set_compilation_env(), torch._dynamo.utils.disable_cache_limit(), _temp_remove_pre_dispatch_torch_function_mode():
        with _temp_remove_metadata_torch_function_mode() as metadata_mode:
            if metadata_mode:
                backend = make_eager_backend_with_torch_function_mode(metadata_mode)
            else:
                backend = "eager"
            return torch.compile(_if_op_wrapper, backend=backend, fullgraph=True)(
                pred, then_func, operands
            )


@if_op.py_impl(FakeTensorMode)
def if_fake_tensor_mode(mode, pred, then_func, *operands):
    with mode:
        return then_func(*operands)


@if_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def if_dense(pred, then_func, *operands):
    # Note: we still wants to execute the subgraph even
    # if pred is False. The reason is that we cannot
    # evaluate the pred in fake tensor mode because of
    # pred is data dependent. Therefore, the graph we
    # captured assume the output is valid so we have
    # to return the same output.
    return then_func(*operands)


if_op.py_autograd_impl(autograd_not_implemented)


@if_op.py_functionalize_impl
def if_func(ctx, pred, then_func, *operands):
    unwrapped_inputs = ctx.unwrap_tensors(operands)
    unwrapped_pred = ctx.unwrap_tensors(pred)
    with ctx.redispatch_to_next():
        functional_then = ctx.functionalize(_maybe_run_with_interpreter(then_func))
        if_return = if_op(unwrapped_pred, functional_then, *unwrapped_inputs)
        return ctx.wrap_tensors(if_return)


@if_op.py_impl(ProxyTorchDispatchMode)
def inner(proxy_mode, pred, then_func, *operands):
    then_graph = reenter_make_fx(then_func)(*operands)
    i, then_name = unique_graph_id(proxy_mode, prefix="then_graph")
    proxy_mode.tracer.root.register_module(then_name, then_graph)
    proxy_args = pytree.tree_map(
        proxy_mode.tracer.unwrap_proxy, (pred, then_graph, *operands)
    )
    out_proxy = proxy_mode.tracer.create_proxy("call_function", if_op, proxy_args, {})
    out = if_op(pred, then_graph, operands)
    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


class MergeOp(HigherOrderOperator):
    def __init__(self):
        super().__init__("merge_op")

    def __call__(self, pred, then_vars, else_vars):
        return super().__call__(pred, then_vars, else_vars)


merge_op = MergeOp()


def MERGE(pred, then_vars, else_vars):
    from torch._dynamo.backends.debugging import (
        make_eager_backend_with_torch_function_mode,
    )

    if torch.compiler.is_dynamo_compiling():
        return merge_op(pred, then_vars, else_vars)

    def _merge_op_wrapper(*args, **kwargs):
        return merge_op(*args, **kwargs)

    with _set_compilation_env(), torch._dynamo.utils.disable_cache_limit(), _temp_remove_pre_dispatch_torch_function_mode():
        with _temp_remove_metadata_torch_function_mode() as metadata_mode:
            if metadata_mode:
                backend = make_eager_backend_with_torch_function_mode(metadata_mode)
            else:
                backend = "eager"
            return torch.compile(_merge_op_wrapper, backend=backend, fullgraph=True)(
                pred, then_vars, else_vars
            )


@merge_op.py_impl(FakeTensorMode)
def merge_fake_tensor_mode(mode, pred, then_vars, else_vars):
    from torch._higher_order_ops.cond import _merge_tensors

    assert len(then_vars) == len(else_vars)
    return [
        _merge_tensors(then_var, else_var, mode)
        for then_var, else_var in zip(then_vars, else_vars)
    ]


@merge_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def merge_dense(pred, then_vars, else_vars):
    return then_vars if pred else else_vars


merge_op.py_autograd_impl(autograd_not_implemented)


@merge_op.py_functionalize_impl
def merge_func(ctx, pred, then_vars, else_vars):
    unwrapped_pred = ctx.unwrap_tensors(pred)
    unwrapped_then_vars = ctx.unwrap_tensors(then_vars)
    unwrapped_else_vars = ctx.unwrap_tensors(else_vars)
    with ctx.redispatch_to_next():
        merge_ret = merge_op(unwrapped_pred, unwrapped_then_vars, unwrapped_else_vars)
        return ctx.wrap_tensors(merge_ret)


@merge_op.py_impl(ProxyTorchDispatchMode)
def merge_proxy_mode(proxy_mode, pred, then_vars, else_vars):
    proxy_args = pytree.tree_map(
        proxy_mode.tracer.unwrap_proxy, (pred, then_vars, else_vars)
    )
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", merge_op, proxy_args, {}
    )
    out = merge_op(pred, then_vars, else_vars)
    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)
