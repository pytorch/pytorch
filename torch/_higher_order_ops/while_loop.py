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
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)


class WhileLoopOp(HigherOrderOperator):
    def __call__(self, cond_fn, body_fn, operands):
        if not isinstance(cond_fn, torch.fx.GraphModule) or not isinstance(
            body_fn, torch.fx.GraphModule
        ):
            raise RuntimeError(
                "cond_fn and body_fn must be torch.fx.GraphModule, got "
                f"{type(cond_fn)} and {type(body_fn)}"
            )
        if not isinstance(operands, tuple):
            raise RuntimeError("operands must be a tuple, got " f"{type(operands)}")
        if not all(isinstance(t, (torch.Tensor, int, float, bool)) for t in operands):
            raise RuntimeError(
                "operands must be a tuple of tensors, ints, floats, or bools, got "
                f"{operands}"
            )
        return super().__call__(cond_fn, body_fn, operands)


while_loop_op = HigherOrderOperator("while_loop")


def while_loop(cond_fn, body_fn, operands):
    r"""
    Run body_fn(*operands) while cond_fn(*operands) returns a True scalar tensor. Returns the output of body_fn or
    initial operands.

    .. warning::
        `torch.while_loop` is a prototype feature in PyTorch. It has limited support for input and output types and
        doesn't support training currently. Please look forward to a more stable implementation in a future version of PyTorch.
        Read more about feature classification at: https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype

    `while_loop` is a structured control flow operator. It preserves the loop semantic across the torch.compile and torch.export.

    `while_loop` is equivalent to the following:

        def while_loop(cond_fn, body_fn, operands):
            val = operands
            while cond_fn(*val):
                val = body_fn(*val)
            return val

    Args:
        cond_fn (Callable): A callable function that returns a boolean Scalar tensor.

        body_fn (Callable): A callable function that takes the same inputs as `cond_fn` and returns a tuple of tensors

        operands (Tuple of possibly nested dict/list/tuple of tensors): A tuple of inputs to cond_fn and body_fn. It's also
            the initial value of states that are carried across iterations.

    Example:

        def cond_fn(iter, x):
            return iter.sum() < 10

        def body_fn(iter, x):
            return iter + 1, x.sin()

        while_loop(cond_fn, body_fn, (torch.zeros(1), torch.randn(3, 4)))

    Restrictions:

        - body_fn must return tensors with the same metadata (e.g.shape, dtype) as inputs.

        - body_fn and cond_fn must not in-place mutate the operands. A clone before the mutation is required.

        - body_fn and cond_fn must not mutate python varialbles (e.g. list/dict) created outside of the body_fn.

        - body_fn and cond_fn's output cannot aliase any of the inputs. A clone is required.

    .. warning::
        Temporal Limitations:

        - 'while_loop' only supports **inference** right now. Autograd will be supported in the future.

    """
    if torch.compiler.is_dynamo_compiling():
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

    _validate_input(cond_fn, body_fn, operands)

    with _set_compilation_env(), torch._dynamo.utils.disable_cache_limit():
        return torch.compile(while_loop_op, backend="eager", fullgraph=True)(
            cond_fn, body_fn, operands
        )


@while_loop_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def while_loop_dense(cond_fn, body_fn, operands):
    init_val = operands

    def _is_boolean_scalar_tensor(pred):
        return (
            isinstance(pred, torch.Tensor)
            and pred.size() == torch.Size([])
            and pred.dtype == torch.bool
        )

    if not isinstance(operands, tuple):
        raise RuntimeError(f"operands must be a tuple but got {type(operands)}")

    while pred := cond_fn(*init_val):
        if not _is_boolean_scalar_tensor(pred):
            raise RuntimeError(
                f"cond_fn must return a boolean scalar tensor but got {pred}"
            )
        out = body_fn(*init_val)
        assert isinstance(
            out, tuple
        ), f"body_fn should return a tuple but got {type(out)}"
        assert len(out) == len(
            init_val
        ), "body_fn should return the same number of elements as operands"
        init_val = out
    return init_val


while_loop_op.py_impl(DispatchKey.Autograd)(
    autograd_not_implemented(while_loop_op, deferred_error=True)
)


@while_loop_op.py_impl(ProxyTorchDispatchMode)
def while_loop_tracing(mode, cond_fn, body_fn, operands):
    def _trace_while_loop(proxy_mode, while_loop_op, cond_fn, body_fn, operands):
        pre_dispatch = getattr(proxy_mode, "pre_dispatch", False)
        with disable_proxy_modes_tracing():
            cond_graph = reenter_make_fx(cond_fn, pre_dispatch)(*operands)
            body_graph = reenter_make_fx(body_fn, pre_dispatch)(*operands)

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
        return track_tensor_tree(
            out, out_proxy, constant=None, tracer=proxy_mode.tracer
        )

    if mode.enable_tracing:
        return _trace_while_loop(mode, while_loop_op, cond_fn, body_fn, operands)
    else:
        return while_loop_op(cond_fn, body_fn, operands)


@while_loop_op.py_impl(FakeTensorMode)
def while_loop_fake_tensor_mode(mode, cond_fn, body_fn, operands):
    return body_fn(*operands)


@while_loop_op.py_functionalize_impl
def while_loop_func(ctx, cond_fn, body_fn, operands):
    unwrapped_operands = ctx.unwrap_tensors(operands)
    with ctx.redispatch_to_next() as m:
        functional_cond_fn = ctx.functionalize(cond_fn)
        functional_body_fn = ctx.functionalize(body_fn)
        pre_dispatch = hasattr(ctx, "mode") and ctx.mode.pre_dispatch
        for fn, fn_name in [
            (functional_cond_fn, "cond_fn"),
            (functional_body_fn, "body_fn"),
        ]:
            if _has_potential_branch_input_mutation(
                fn, unwrapped_operands, pre_dispatch=pre_dispatch
            ):
                raise UnsupportedAliasMutationException(
                    f"torch.while_loop's {fn_name} might be modifying the input!"
                )

        for fn in [functional_cond_fn, functional_body_fn]:
            if _has_potential_branch_input_alias(
                fn, unwrapped_operands, pre_dispatch=pre_dispatch
            ):
                raise UnsupportedAliasMutationException(
                    f"torch.while_loop's {fn_name} might be aliasing the input!"
                )
        ret = while_loop_op(functional_cond_fn, functional_body_fn, unwrapped_operands)
        return ctx.wrap_tensors(ret)
