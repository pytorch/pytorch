import torch

import torch.utils._pytree as pytree

from torch._C import DispatchKey
from torch._ops import HigherOrderOperator
from torch._higher_order_ops.utils import autograd_not_implemented, _maybe_run_with_interpreter, _set_compilation_env, _has_potential_branch_input_alias, _has_potential_branch_input_mutation:w

from torch.utils._python_dispatch import _get_current_dispatch_mode
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    make_fx,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)


while_loop_op = HigherOrderOperator("while_loop")

def while_loop(cond_fn, body_fn, operands):
    if torch._dynamo.is_compiling():
        return while_loop_op(cond_fn, body_fn, operands)

    def _validate_input(cond_fn, body_fn, operands):
        if not callable(cond_fn) or not callable(body_fn):
            raise RuntimeError("Expect cond_fn and body_fn to be callbale.")

        if not isinstance(operands, (tuple, list)) or pytree.tree_any(
            lambda t: not isinstance(t, torch.Tensor), operands
        ):
            raise RuntimeError(
                "Expect operands to be a tuple of possibly nested dict/list/tuple that only"
                f"consists of tensor leaves, but got {operands}."
            )

    with _set_compilation_env(), torch._dyanmo.utils.disable_cache_limit():
        return torch.compile(while_loop_op, backend="eager", fullgraph=True)(
            cond_fn, body_fn, operands
        )


@while_loop_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def while_loop_dense(cond_fn, body_fn, operands):
    mode = _get_current_dispatch_mode()
    assert mode is None, "Mode should never be neabled for CPU/CUDA key"
    while cond_fn(*operands):
        operands = body_fn(*operands)
    return operands

while_loop_op.py_impl(DispatchKey.Autograd)(
        autograd_not_implemented(while_loop_op, deferred_error=True)
)

@while_loop_op.py_impl(ProxyTorchDispatchMode)
def while_loop_tracing(mode, cond_fn, body_fn, operands):

    def _trace_while_loop(proxy_mode, while_loop_op, cond_fn, body_fn, operands):
        pre_dispatch = getattr(proxy_mode, "pre_dispatch", False)

        with disable_proxy_modes_tracing():
            cond_graph = make_fx(
                _maybe_run_with_interpreter(cond_fn), pre_dispatch=pre_dispatch
            )(*operands)
            body_graph = make_fx(
                _maybe_run_with_interpreter(body_fn), pre_dispatch=pre_dispatch
            )(*operands)

        next_name = None
        i = 0
        while not next_name:
            candidate = f"while_loop_cond_graph_{i}"
            if hasattr(proxy_mode.tracer.root, candidate):
                i += 1
            else:
                next_name = candidate
        cond_graph_name = next_name
        body_graph_name = f"while_loop_body_graph_{i}"
        assert not hasattr(proxy_mode.tracer.root, body_graph_name)

        proxy_mode.tracer.root.register_module(cond_graph_name, cond_graph)
        proxy_mode.tracer.root.register_module(body_graph_name, body_graph)

        args = (cond_graph, body_graph, operands)

        proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, args)

        out_proxy = proxy_mode.tracer.create_proxy(
            "call_function", while_loop_op, proxy_args, {}, name="while_loop"
        )

        # body_fn return output with the same pytree and tensor meta data as operands
        # so we could just return the output after one iteration.
        out = body_fn(*operands)
        return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)

    if mode.enable_tracing:
        return _trace_while_loop(mode, while_loop_op, cond_fn, body_fn, operands)
    else:
        return while_loop_op(cond_fn, body_fn, operands)

@while_loop_op.py_impl(FakeTensorMode)
def while_loop_fake_tensor_mode(mode, cond_fn, body_fn, operands):
    with mode:
        return while_loop_dense(cond_fn, body_fn, operands)

@while_loop.py_functionalize_impl
def while_loop_func(ctx, cond_fn, body_fn, operands):
    unwrapped_operands = ctx.unwrap_tensors(operands)
    with ctx.redispatch_to_next() as m:
        functional_cond_fn = ctx.functionalize(cond_fn)
        functional_body_fn = ctx.functionalize(body_fn)
        for fn, fn_name in [(functional_cond_fn "cond_fn"), (functional_body_fn, "body_fn")]:
            if _has_potential_branch_input_mutation(branch, unwrapped_inputs):
                raise UnsupportedAliasMutationException(
                    f"torch.while_loop's {fn_name} might be modifying the input!"
                )

        for fn in [functional_cond_fn, functional_body_fn]:
            if _has_potential_branch_input_alias(fn, unwrapped_operands):
                raise UnsupportedAliasMutationException(
                    f"torch.while_loop's {fn_name} might be aliasing the input!"
                )
        ret = while_loop_op(functional_cond_fn, functional_body_fn, unwrapped_operands)
        return ctx.wrap_tensor(ret)
