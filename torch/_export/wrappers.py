from contextlib import contextmanager

import torch
import torch._custom_ops
from torch._C import _ExcludeDispatchKeyGuard, DispatchKey, DispatchKeySet
from torch._export.exported_program import ModuleCallSignature
from torch._functorch.eager_transforms import _unwrap_all_tensors_from_functional
from torch._higher_order_ops.wrap import wrap
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import (
    _get_current_dispatch_mode,
    _pop_mode_temporarily,
)


_export_tracepoint = HigherOrderOperator("_export_tracepoint")


_export_tracepoint.fallthrough(DispatchKey.PythonDispatcher)  # type: ignore[attr-defined]
_export_tracepoint.fallthrough(DispatchKey.PythonTLSSnapshot)  # type: ignore[attr-defined]
_export_tracepoint.fallthrough(DispatchKey.ADInplaceOrView)
_export_tracepoint.fallthrough(DispatchKey.BackendSelect)
_export_tracepoint.fallthrough(DispatchKey.AutocastCPU)  # type: ignore[attr-defined]
_export_tracepoint.fallthrough(DispatchKey.AutogradCPU)


@_export_tracepoint.py_impl(ProxyTorchDispatchMode)
def export_tracepoint_dispatch_mode(*args, **kwargs):
    mode = _get_current_dispatch_mode()
    assert mode is not None, "Mode should always be enabled for python fallback key"
    with _pop_mode_temporarily() as mode:
        if not mode.enable_tracing:
            return _export_tracepoint(*args, **kwargs)
        p_args, p_kwargs = pytree.tree_map(mode.tracer.unwrap_proxy, (args, kwargs))
        proxy = mode.tracer.create_proxy(
            "call_function", _export_tracepoint, p_args, p_kwargs
        )
        return track_tensor_tree(args, proxy, constant=None, tracer=mode.tracer)


@_export_tracepoint.py_impl(FakeTensorMode)
def export_tracepoint_fake_tensor_mode(*args, **kwargs):
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


def _wrap_submodule(mod, path, module_call_signatures):
    assert isinstance(mod, torch.nn.Module)
    assert path != ""
    parent = None
    submodule = mod
    for name in path.split("."):
        parent = submodule
        if not hasattr(submodule, name):
            raise RuntimeError(f"Couldn't find submodule at path {path}")
        submodule = getattr(submodule, name)

    from torch._dynamo import assume_constant_result

    # TODO(zhxchen17) Use pytree output from higher order op directly.
    @assume_constant_result
    def update_module_call_signatures(path, in_spec, out_spec):
        assert path not in module_call_signatures
        module_call_signatures[path] = ModuleCallSignature(
            inputs=[], outputs=[], in_spec=in_spec, out_spec=out_spec
        )

    class WrappedModule:
        def __init__(self):
            self.__class__ = type(
                submodule.__class__.__name__,
                (self.__class__, submodule.__class__),
                {},
            )
            self.__dict__ = submodule.__dict__
            assert not hasattr(self, "module_call_signatures")
            self.module_call_signatures = module_call_signatures

        def forward(self, *args, **kwargs):
            flat_args, in_spec = pytree.tree_flatten((args, kwargs))

            def flat_gm(*flat_args):
                flat_args = _export_tracepoint(
                    *flat_args, kind="module_call_inputs", path=path
                )
                args, kwargs = pytree.tree_unflatten(flat_args, in_spec)
                res = submodule(*args, **kwargs)
                flat_res, out_spec = pytree.tree_flatten(res)
                flat_res = _export_tracepoint(
                    *flat_res, kind="module_call_outputs", path=path
                )
                update_module_call_signatures(path, in_spec, out_spec)
                return flat_res

            flat_res = wrap(flat_gm, *flat_args)
            return pytree.tree_unflatten(
                flat_res, self.module_call_signatures[path].out_spec
            )

    setattr(parent, name, WrappedModule())
    return parent, name, submodule


@contextmanager
def _wrap_submodules(f, preserve_signature, module_call_signatures):
    tasks = []

    try:
        for path in preserve_signature:
            tasks.append(_wrap_submodule(f, path, module_call_signatures))
        yield
    finally:
        for parent, name, submodule in tasks:
            setattr(parent, name, submodule)
