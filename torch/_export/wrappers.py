from contextlib import contextmanager

import torch
import torch._custom_ops
from torch._C import _ExcludeDispatchKeyGuard, DispatchKey, DispatchKeySet
from torch._functorch.eager_transforms import _unwrap_all_tensors_from_functional
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree
from torch.utils import _pytree as pytree


_export_tracepoint = HigherOrderOperator("_export_tracepoint")


_export_tracepoint.fallthrough(DispatchKey.PythonDispatcher)  # type: ignore[attr-defined]
_export_tracepoint.fallthrough(DispatchKey.PythonTLSSnapshot)  # type: ignore[attr-defined]
_export_tracepoint.fallthrough(DispatchKey.ADInplaceOrView)
_export_tracepoint.fallthrough(DispatchKey.BackendSelect)
_export_tracepoint.fallthrough(DispatchKey.AutocastCPU)  # type: ignore[attr-defined]
_export_tracepoint.fallthrough(DispatchKey.AutogradCPU)


@_export_tracepoint.py_impl(ProxyTorchDispatchMode)
def export_tracepoint_dispatch_mode(mode, *args, **kwargs):
    if not mode.enable_tracing:
        return _export_tracepoint(*args, **kwargs)
    p_args, p_kwargs = pytree.tree_map(mode.tracer.unwrap_proxy, (args, kwargs))
    proxy = mode.tracer.create_proxy(
        "call_function", _export_tracepoint, p_args, p_kwargs
    )
    return track_tensor_tree(args, proxy, constant=None, tracer=mode.tracer)


@_export_tracepoint.py_impl(FakeTensorMode)
def export_tracepoint_fake_tensor_mode(mode, *args, **kwargs):
    with mode:
        return args


@_export_tracepoint.py_impl(DispatchKey.Functionalize)
def export_tracepoint_functionalize(*args, **kwargs):
    reapply_views = torch._C._functionalization_reapply_views_tls()
    unwrapped_args = _unwrap_all_tensors_from_functional(
        args, reapply_views=reapply_views
    )
    unwrapped_kwargs = _unwrap_all_tensors_from_functional(
        kwargs, reapply_views=reapply_views
    )
    with _ExcludeDispatchKeyGuard(DispatchKeySet(DispatchKey.Functionalize)):
        return _export_tracepoint(*unwrapped_args, **unwrapped_kwargs)


@_export_tracepoint.py_impl(DispatchKey.CPU)
def export_tracepoint_cpu(*args, **kwargs):
    return args


def _wrap_submodule(mod, path, module_call_specs):
    assert isinstance(mod, torch.nn.Module)
    assert path != ""
    submodule = mod
    for name in path.split("."):
        if not hasattr(submodule, name):
            raise RuntimeError(f"Couldn't find submodule at path {path}")
        submodule = getattr(submodule, name)

    # TODO(zhxchen17) Remove this decorator.
    @torch._dynamo.assume_constant_result
    def update_module_call_signatures(path, in_spec, out_spec):
        assert path not in module_call_specs
        module_call_specs[path] = {"in_spec": in_spec, "out_spec": out_spec}

    assert "forward" not in submodule.__dict__
    wrapped_forward = submodule.forward

    def check_flattened(flat_args):
        for a in flat_args:
            if not (isinstance(a, (torch.Tensor, str, int, float, bool)) or a is None):
                raise AssertionError(
                    f"Only Tensors or scalars are supported as pytree flattened inputs, got: {a}"
                )

    def wrapper(self, *args, **kwargs):
        flat_args, in_spec = pytree.tree_flatten((args, kwargs))
        check_flattened(flat_args)
        flat_args = _export_tracepoint(*flat_args, kind="module_call_inputs", path=path)
        args, kwargs = pytree.tree_unflatten(flat_args, in_spec)
        res = wrapped_forward(*args, **kwargs)
        flat_res, out_spec = pytree.tree_flatten(res)
        check_flattened(flat_res)
        flat_res = _export_tracepoint(*flat_res, kind="module_call_outputs", path=path)
        update_module_call_signatures(path, in_spec, out_spec)
        return pytree.tree_unflatten(flat_res, out_spec)

    submodule.forward = wrapper.__get__(submodule, type(submodule))
    return submodule


@contextmanager
def _wrap_submodules(f, preserve_signature, module_call_signatures):
    tasks = []

    try:
        for path in preserve_signature:
            tasks.append(_wrap_submodule(f, path, module_call_signatures))
        yield
    finally:
        for submodule in tasks:
            del submodule.__dict__["forward"]
