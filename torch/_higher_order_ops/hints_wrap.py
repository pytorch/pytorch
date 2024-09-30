# mypy: allow-untyped-defs
import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._higher_order_ops.utils import (
    _has_potential_branch_input_alias,
    _has_potential_branch_input_mutation,
    autograd_not_implemented,
    reenter_make_fx,
    unique_graph_id,
    UnsupportedAliasMutationException,
)
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree


# used for wrapping a function/op with context hints
class HintsWrapper(HigherOrderOperator):
    def __init__(self):
        super().__init__("hints_wrapper")

    def __call__(self, body_fn, args, kwargs, hints):
        r"""
        Call implementation of hints_wrapper

        Args:
            body_fn (Callable): A callable function that is within the scope
             that is being traced.

            args (Tuple of torch.Tensor/int/float/bool): A tuple of inputs to
             body_fn.

            kwargs (dict): Keyword argument to the body_fn.

            hints (dict): A dict of context hints which could be passed to
             backend compiler.
        """
        if not isinstance(args, tuple):
            raise RuntimeError(f"args must be a tuple, got {type(args)}")

        if not all(isinstance(t, (torch.Tensor, int, float, bool)) for t in args):
            raise RuntimeError(
                "args must be a tuple of tensors, ints, floats, or bools, got "
                f"{args}"
            )

        if not isinstance(kwargs, dict):
            raise RuntimeError(f"kwargs must be a dict, got {type(kwargs)}")

        if len(kwargs) > 0:
            raise RuntimeError(
                f"kwargs except for hints are not supported, got {kwargs}"
            )

        if not isinstance(hints, dict):
            raise RuntimeError(f"hints must be a dict, got {type(hints)}")

        for k, v in hints.items():
            if not isinstance(k, str):
                raise RuntimeError(f"hints key must be a str, got {k}.")

            if not isinstance(v, (int, float, bool, str)):
                raise RuntimeError(
                    "hints must be a dict containing int, float, bool or str "
                    f"value, got value {v} for key {k}."
                )

        return super().__call__(body_fn, args, kwargs, hints)


hints_wrapper = HintsWrapper()


@hints_wrapper.py_impl(DispatchKey.CompositeExplicitAutograd)
def hints_wrapper_dense(body_fn, args, kwargs, hints):
    return body_fn(*args, **kwargs)


hints_wrapper.py_impl(DispatchKey.Autograd)(
    autograd_not_implemented(hints_wrapper, deferred_error=True)
)


@hints_wrapper.py_impl(FakeTensorMode)
def hints_wrapper_fake_tensor_mode(mode, body_func, args, kwargs, hints):
    flat_args = pytree.tree_leaves(args)
    with mode:
        return body_func(*flat_args, **kwargs)


@hints_wrapper.py_functionalize_impl
def hints_wrapper_functionalize(ctx, body_fn, args, kwargs, hints):
    unwrapped_args = ctx.unwrap_tensors(args)
    unwrapped_kwargs = ctx.unwrap_tensors(kwargs)
    unwrapped_hints = ctx.unwrap_tensors(hints)
    with ctx.redispatch_to_next():
        functional_body_fn = ctx.functionalize(body_fn)
        pre_dispatch = hasattr(ctx, "mode") and ctx.mode.pre_dispatch
        if _has_potential_branch_input_mutation(
            functional_body_fn, unwrapped_args, pre_dispatch=pre_dispatch
        ):
            raise UnsupportedAliasMutationException(
                "body_fn of hints_wrapper might be modifying the input!"
            )
        if _has_potential_branch_input_alias(
            functional_body_fn, unwrapped_args, pre_dispatch=pre_dispatch
        ):
            raise UnsupportedAliasMutationException(
                "body_fn of hints_wrapper might be aliasing the input!"
            )
        outputs = hints_wrapper(
            functional_body_fn,
            unwrapped_args,
            unwrapped_kwargs,
            unwrapped_hints,
        )
        return ctx.wrap_tensors(outputs)


def trace_hints_wrapper(proxy_mode, hints_wrapper, body_fn, args, kwargs, hints):
    flat_args = tuple(pytree.tree_leaves(args))
    body_graph = reenter_make_fx(body_fn)(*flat_args, **kwargs)

    _, body_graph_name = unique_graph_id(proxy_mode, prefix="hints_wrapper_body_graph")
    proxy_mode.tracer.root.register_module(body_graph_name, body_graph)

    new_args: tuple = (body_graph, flat_args, {})
    # merge hints into kwargs
    new_kwargs = {}
    new_kwargs["hints"] = hints

    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, new_args)
    proxy_kwargs = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, new_kwargs)

    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", hints_wrapper, proxy_args, proxy_kwargs, name="hints_wrapper"
    )

    out = body_fn(*flat_args, **kwargs)
    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


@hints_wrapper.py_impl(ProxyTorchDispatchMode)
def inner(proxy_mode, body_fn, args, kwargs, hints):
    if proxy_mode.enable_tracing:
        return trace_hints_wrapper(
            proxy_mode, hints_wrapper, body_fn, args, kwargs, hints
        )
    else:
        return hints_wrapper(body_fn, args, kwargs, hints)
