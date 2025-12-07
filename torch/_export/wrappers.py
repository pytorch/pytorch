# mypy: allow-untyped-defs
import inspect
from contextlib import contextmanager
from functools import wraps

import torch
import torch._custom_ops
from torch._C import DispatchKey
from torch._export.utils import _maybe_find_pre_dispatch_tf_mode_for_export
from torch._higher_order_ops.flat_apply import (
    _ConstantFunction,
    flat_apply,
    to_graphable,
)
from torch._higher_order_ops.strict_mode import strict_mode
from torch._higher_order_ops.utils import autograd_not_implemented
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
    PreDispatchTorchFunctionMode,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import is_traceable_wrapper_subclass_type


class ExportTracepoint(HigherOrderOperator):
    def __init__(self):
        super().__init__("_export_tracepoint")

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


_export_tracepoint = ExportTracepoint()


@_export_tracepoint.py_impl(ProxyTorchDispatchMode)
def export_tracepoint_dispatch_mode(mode, *args, **kwargs):
    p_args, p_kwargs = pytree.tree_map(mode.tracer.unwrap_proxy, (args, kwargs))
    proxy = mode.tracer.create_proxy(
        "call_function", _export_tracepoint, p_args, p_kwargs
    )
    return track_tensor_tree(args, proxy, constant=None, tracer=mode.tracer)


@_export_tracepoint.py_impl(FakeTensorMode)
def export_tracepoint_fake_tensor_mode(mode, *args, **kwargs):
    with mode:
        return args


@_export_tracepoint.py_functionalize_impl
def export_tracepoint_functional(ctx, *args, **kwargs):
    unwrapped_args = ctx.unwrap_tensors(args)
    unwrapped_kwargs = ctx.unwrap_tensors(kwargs)

    with ctx.redispatch_to_next():
        _export_tracepoint(*unwrapped_args, **unwrapped_kwargs)
        return args


_export_tracepoint.py_impl(DispatchKey.Autograd)(
    autograd_not_implemented(_export_tracepoint, deferred_error=True)
)


@_export_tracepoint.py_impl(DispatchKey.CPU)
def export_tracepoint_cpu(*args, **kwargs):
    return args


def _wrap_submodule(mod, path, module_call_specs):
    assert isinstance(mod, torch.nn.Module)
    assert path != ""
    submodule = torch.fx.graph_module._get_attr(mod, path)

    def update_module_call_signatures(path, in_spec, out_spec):
        if path in module_call_specs:
            assert module_call_specs[path]["in_spec"] == in_spec
            assert module_call_specs[path]["out_spec"] == out_spec
        module_call_specs[path] = {"in_spec": in_spec, "out_spec": out_spec}

    def check_flattened(flat_args):
        for a in flat_args:
            if not (isinstance(a, (torch.Tensor, str, int, float, bool)) or a is None):
                raise AssertionError(
                    f"Only Tensors or scalars are supported as pytree flattened inputs, got: {a}"
                )

    def pre_hook(module, args, kwargs):
        flat_args, in_spec = pytree.tree_flatten((args, kwargs))
        check_flattened(flat_args)
        flat_args = _export_tracepoint(*flat_args, kind="module_call_inputs", path=path)
        args, kwargs = pytree.tree_unflatten(flat_args, in_spec)
        return args, kwargs

    def post_hook(module, args, kwargs, res):
        _, in_spec = pytree.tree_flatten((args, kwargs))
        flat_res, out_spec = pytree.tree_flatten(res)
        check_flattened(flat_res)
        flat_res = _export_tracepoint(*flat_res, kind="module_call_outputs", path=path)
        update_module_call_signatures(path, in_spec, out_spec)
        return pytree.tree_unflatten(flat_res, out_spec)

    pre_handle = submodule.register_forward_pre_hook(pre_hook, with_kwargs=True)
    post_handle = submodule.register_forward_hook(post_hook, with_kwargs=True)
    return pre_handle, post_handle


@contextmanager
def _wrap_submodules(f, preserve_signature, module_call_signatures):
    handles = []

    try:
        for path in preserve_signature:
            handles.extend(_wrap_submodule(f, path, module_call_signatures))
        yield
    finally:
        for handle in handles:
            handle.remove()


def _mark_strict_experimental(cls):
    def call(self, *args):
        return strict_mode(self, args)

    cls.__call__ = call
    return cls


def _register_func_spec_proxy_in_tracer(tracer, name, spec):
    """
    This is a wrapper utility method on top of tracer to cache the
    already registered subclass spec attribute. This is useful because
    Subclass.__init__ will be same for each subclass. By default, fx will
    create multiple attributes/proxies for given attribute.
    """
    fx_name = name + "0"
    if hasattr(tracer.root, fx_name):
        assert getattr(tracer.root, fx_name) == spec
        return tracer.create_proxy("get_attr", fx_name, (), {})

    qualname = tracer.get_fresh_qualname(name)
    setattr(tracer.root, qualname, spec)
    return tracer.create_proxy("get_attr", qualname, (), {})


def _emit_flat_apply_call(
    *,
    tracer,
    spec_name: str,
    const_target_for_apply,
    graphable_args,
    track_value,
    call_spec_cache_key: str,
):
    # Flatten to graphable form and record the spec on the FX root
    flat_args, in_spec = to_graphable(graphable_args)
    qualname = tracer.get_fresh_qualname(spec_name)  # type: ignore[union-attr]
    setattr(tracer.root, qualname, in_spec)  # type: ignore[union-attr]
    spec_proxy = tracer.create_proxy("get_attr", qualname, (), {})

    # Reuse/cached ConstantFunction spec on the root
    _, func_spec = pytree.tree_flatten(_ConstantFunction(const_target_for_apply))
    func_spec_proxy = _register_func_spec_proxy_in_tracer(
        tracer, f"{call_spec_cache_key}_const_func_spec", func_spec
    )

    # Map runtime args -> proxies (always via tracer.unwrap_proxy now)
    flat_proxy_args = pytree.tree_map(tracer.unwrap_proxy, flat_args)

    # Emit flat_apply and track result structure
    out_proxy = tracer.create_proxy(
        "call_function", flat_apply, (func_spec_proxy, spec_proxy, *flat_proxy_args), {}
    )
    track_tensor_tree(track_value, out_proxy, constant=None, tracer=tracer)


def _is_init(fn):
    return callable(fn) and fn.__name__ == "__init__"


def mark_subclass_constructor_exportable_experimental(constructor_subclass):
    """
    Experimental decorator that makes subclass to be traceable in export
    with pre-dispatch IR. To make your subclass traceble in export, you need to:
        1. Implement __init__ method for your subclass (Look at DTensor implementation)
        2. Decorate your __init__ method with _mark_constructor_exportable_experimental
        3. Put torch._dynamo_disable decorator to prevent dynamo from peeking into its' impl

    Example:

    class FooTensor(torch.Tensor):
        @staticmethod
        def __new__(cls, elem, *, requires_grad=False):
            # ...
            return torch.Tensor._make_subclass(cls, elem, requires_grad=requires_grad)

        @torch._dynamo_disable
        @mark_subclass_constructor_exportable_experimental
        def __init__(self, elem, ...):
            # ...
    """
    if not _is_init(constructor_subclass):
        raise RuntimeError(
            f"torch._export.wrappers.mark_constructor_exportable_experimental can only be applied on subclass tensor.__init__"
            f"But, you are adding it on {constructor_subclass.__name__} which is not supported. "
            f"If __init__ doesn't exist on your subclass, please add it. Look at DTensor.__init__ implementation for example"
        )

    def wrapper(*args, **kwargs):
        constructor_subclass(*args, **kwargs)

        if not torch.compiler.is_exporting():
            return

        if not is_traceable_wrapper_subclass_type(type(args[0])):
            assert constructor_subclass.__qualname__.endswith("__init__")
            obj_name = constructor_subclass.__qualname__[: -len("__init__")]
            raise RuntimeError(
                f"Can't intercept {obj_name} in export because this object is not a traceable "
                f"tensor subclass. Please look at DTensor.__init__ implementation as an example of proper usage of this API."
            )

        mode = _maybe_find_pre_dispatch_tf_mode_for_export()
        if mode is None:
            return

        assert isinstance(mode, PreDispatchTorchFunctionMode)

        tracer = mode.tracer
        subclass = args[0]
        graphable = (tuple(args[1:]), kwargs)

        spec_name = "_".join(constructor_subclass.__qualname__.lower().split("."))
        call_spec_cache_key = type(subclass).__name__.lower()

        _emit_flat_apply_call(
            tracer=tracer,
            spec_name=spec_name,
            const_target_for_apply=type(subclass),
            graphable_args=graphable,
            track_value=subclass,  # track the constructed subclass instance
            call_spec_cache_key=call_spec_cache_key,
        )
        return

    return wrapper


def allow_in_pre_dispatch_graph(func):
    """
    Experimental decorator that adds user function to export pre-dispatch graph. Note that
    we only support custom autograd function/subclass constructors today. To use this function:
        1. For subclasses:
            1. refer to instructions in mark_subclass_constructor_exportable_experimental
        2. Define apply method on your custom autograd function and apply this decorator.

    Example:

    class MyCoolCustomAutogradFunc(autograd.Function):
        @classmethod
        @torch._export.wrappers.allow_in_pre_dispatch_graph
        def apply(cls, *args, **kwargs):
            return super(MyCoolCustomAutogradFunc, cls).apply(*args, **kwargs)

    """
    if _is_init(func):
        return mark_subclass_constructor_exportable_experimental(func)

    if not (_is_init(func) or func.__name__ == "apply"):
        raise RuntimeError(
            f"torch._export.wrappers.allow_in_pre_dispatch_graph can only be applied on subclass tensor.__init_ "
            f"or custom_autograd_function.apply. "
            f"But, you are adding it on {func.__name__} which is not supported. "
            f"If __init__ doesn't exist on your subclass, please add it. Look at DTensor.__init__ implementation for example. "
            f"If you are adding it on custom autograd function, please add it on apply method. "
            f"If anything else, file an issue on github and we may consider extending our support. "
        )

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not torch.compiler.is_exporting():
            return func(*args, **kwargs)

        if not inspect.isclass(args[0]):
            return func(*args, **kwargs)

        if not issubclass(args[0], torch.autograd.Function):
            return func(*args, **kwargs)

        from torch._ops import _get_dispatch_mode_pre_dispatch

        mode = _get_dispatch_mode_pre_dispatch(torch._C._TorchDispatchModeKey.PROXY)
        if mode is None:
            return func(*args, **kwargs)

        # Sometimes custom autograd functions can call into HOPs that don't have proxy impl
        # at PreDispatch level, so we just dispatch it below to get the concrete result.
        include_to_set = torch._C._dispatch_tls_local_include_set().remove(
            torch._C.DispatchKey.PreDispatch
        )
        exclude_to_set = (
            torch._C._dispatch_tls_local_exclude_set()
            | torch._C.DispatchKeySet(torch._C.DispatchKey.PreDispatch)
        )

        with torch._C._ForceDispatchKeyGuard(include_to_set, exclude_to_set):
            out = func(*args, **kwargs)

        assert mode.pre_dispatch, "Should only do this in predispatch"
        tracer = mode.tracer

        function_cls_name = f"{args[0].__module__}.{args[0].__qualname__}"
        graphable = ((function_cls_name, *args[1:]), kwargs)

        from torch.export.custom_ops import (
            _call_custom_autograd_function_in_pre_dispatch,
        )

        spec_name = "_".join(function_cls_name.split("."))
        call_spec_cache_key = type(
            _call_custom_autograd_function_in_pre_dispatch
        ).__name__.lower()
        _emit_flat_apply_call(
            tracer=tracer,
            spec_name=spec_name,
            const_target_for_apply=_call_custom_autograd_function_in_pre_dispatch,
            graphable_args=graphable,
            track_value=out,
            call_spec_cache_key=call_spec_cache_key,
        )
        return out

    return wrapper
