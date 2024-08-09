# mypy: allow-untyped-defs
import copy

import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._higher_order_ops.utils import (
    autograd_not_implemented,
    reenter_make_fx,
    unique_graph_id,
)
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree


# used for wrapping a function/op with context hints
class HintsWrapper(HigherOrderOperator):
    def __init__(self):
        super().__init__("hints_wrapper")


hints_wrapper = HintsWrapper()
# Override hints_wrapper.__module__ to "torch.ops.higher_order" so that in the generated
# graph module, hints_wrapper node's target is correctedly printed as torch.ops.higher_order.hints_wrapper
hints_wrapper.__module__ = "torch.ops.higher_order"


class no_hints_kwargs:
    def __init__(self, kwargs):
        self.kwargs = copy.copy(kwargs)

    def __enter__(self):
        if "hints" in self.kwargs:
            del self.kwargs["hints"]
        return self.kwargs

    def __exit__(self, *args):
        pass


@hints_wrapper.py_impl(DispatchKey.CompositeExplicitAutograd)
def hints_wrapper_dense(body_fn, *args, **kwargs):
    with no_hints_kwargs(kwargs) as kw:
        return body_fn(*args, **kw)


hints_wrapper.py_impl(DispatchKey.Autograd)(
    autograd_not_implemented(hints_wrapper, deferred_error=True)
)


@hints_wrapper.py_impl(FakeTensorMode)
def hints_wrapper_fake_tensor_mode(mode, body_func, *args, **kwargs):
    flat_args = pytree.tree_leaves(args)
    with mode:
        with no_hints_kwargs(kwargs) as kw:
            return body_func(*flat_args, **kw)


@hints_wrapper.py_functionalize_impl
def hints_wrapper_functionalize(ctx, body_fn, *args, **kwargs):
    unwrapped_args = ctx.unwrap_tensors(args)
    unwrapped_kwargs = ctx.unwrap_tensors(kwargs)
    hints = dict()
    if "hints" in unwrapped_kwargs:
        hints = unwrapped_kwargs["hints"]
    with ctx.redispatch_to_next():
        functional_body_fn = ctx.functionalize(body_fn)
        outputs = hints_wrapper(
            functional_body_fn,
            unwrapped_args,
            hints=hints,
        )
        return ctx.wrap_tensors(outputs)


def trace_hints_wrapper(proxy_mode, hints_wrapper, body_fn, *args, **kwargs):
    flat_args = tuple(pytree.tree_leaves(args))
    with no_hints_kwargs(kwargs) as kw:
        body_graph = reenter_make_fx(body_fn)(*flat_args, **kw)

    _, body_graph_name = unique_graph_id(proxy_mode, prefix="hints_wrapper_body_graph")
    proxy_mode.tracer.root.register_module(body_graph_name, body_graph)

    new_args = (body_graph, *flat_args)

    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, new_args)
    proxy_kwargs = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, kwargs)

    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", hints_wrapper, proxy_args, proxy_kwargs, name="hints_wrapper"
    )

    with no_hints_kwargs(kwargs) as kw:
        out = body_fn(*flat_args, **kw)
    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


@hints_wrapper.py_impl(ProxyTorchDispatchMode)
def inner(proxy_mode, body_fn, *args, **kwargs):
    if proxy_mode.enable_tracing:
        return trace_hints_wrapper(proxy_mode, hints_wrapper, body_fn, *args, **kwargs)
    else:
        return hints_wrapper(body_fn, *args, **kwargs)
