from __future__ import annotations

import warnings
from typing import Any

import torch
import torch.fx as fx
import torch.utils._pytree as pytree
from torch import nn, Tensor
from torch.fx import GraphModule, Proxy
from torch.fx.experimental.proxy_tensor import (
    _GraphAppendingTracerEx,
    _ProxyTensor,
    get_proxy_slot,
    has_proxy_slot,
    set_proxy_slot,
)
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch.utils._python_dispatch import TorchDispatchMode


def _set_real_meta(proxy: Proxy, val: object) -> None:
    if isinstance(val, Tensor):
        proxy.node.meta["val"] = val
        if not val.is_sparse:
            proxy.node.meta["tensor_meta"] = _extract_tensor_metadata(val)


class CudaGraphFxTracer(TorchDispatchMode):
    def __init__(self, tracked_inputs: dict[str, Any]) -> None:
        self.graph = fx.Graph()
        self.tracer = _GraphAppendingTracerEx(self.graph)
        self._placeholder_count = 0
        self._register_tracked_inputs(tracked_inputs)

    def _make_placeholder(self, name: str, tensor: Tensor) -> None:
        if has_proxy_slot(tensor, self.tracer):
            return
        unique_name = f"{name.replace('.', '_')}_{self._placeholder_count}"
        self._placeholder_count += 1
        proxy = self.tracer.create_proxy("placeholder", unique_name, (), {})
        _set_real_meta(proxy, tensor)
        set_proxy_slot(tensor, self.tracer, _ProxyTensor(proxy, None))

    def _register_tracked_inputs(self, tracked_inputs: dict[str, Any]) -> None:
        for name, obj in tracked_inputs.items():
            if isinstance(obj, nn.Module):
                for pname, param in obj.named_parameters():
                    self._make_placeholder(f"{name}.{pname}", param)
                for bname, buf in obj.named_buffers():
                    self._make_placeholder(f"{name}.{bname}", buf)
            elif isinstance(obj, torch.optim.Optimizer):
                for i, group in enumerate(obj.param_groups):
                    for j, p in enumerate(group["params"]):
                        self._make_placeholder(f"{name}.param_groups.{i}.{j}", p)
                        state = obj.state.get(p, {})
                        for sk, sv in state.items():
                            if isinstance(sv, Tensor):
                                self._make_placeholder(f"{name}.state.{i}.{j}.{sk}", sv)
            elif isinstance(obj, Tensor):
                self._make_placeholder(name, obj)
            else:
                raise TypeError(
                    f"tracked_inputs[{name!r}]: expected nn.Module, Optimizer, or Tensor, "
                    f"got {type(obj).__name__}"
                )

    def __torch_dispatch__(
        self,
        func: torch._ops.OpOverload,
        types: tuple[torch._C._TensorMeta, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        kwargs = kwargs or {}

        flat_args, spec = pytree.tree_flatten((args, kwargs))

        proxy_flat = []
        for a in flat_args:
            if isinstance(a, Tensor):
                if not has_proxy_slot(a, self.tracer):
                    warnings.warn(
                        f"Tensor at {a.data_ptr():#x} was not in tracked_inputs; "
                        "adding as implicit placeholder.",
                        stacklevel=2,
                    )
                    self._make_placeholder("_implicit", a)
                proxy_flat.append(get_proxy_slot(a, self.tracer).proxy)
            else:
                proxy_flat.append(a)

        proxy_args, proxy_kwargs = pytree.tree_unflatten(proxy_flat, spec)

        out = func(*args, **kwargs)

        proxy_out = self.tracer.create_proxy(
            "call_function", func, proxy_args, proxy_kwargs
        )
        _track_tensor_tree_real(out, proxy_out, self.tracer)

        return out

    def finalize(self) -> GraphModule:
        self.graph.output(None)
        return GraphModule(nn.Module(), self.graph)


def _track_tensor_tree_real(
    val: object, proxy: Proxy, tracer: _GraphAppendingTracerEx
) -> None:
    if isinstance(val, Tensor):
        _set_real_meta(proxy, val)
        set_proxy_slot(val, tracer, _ProxyTensor(proxy, None))
    elif isinstance(val, (list, tuple)):
        for idx, v in enumerate(val):
            child_proxy = proxy[idx]  # type: ignore[index]
            _track_tensor_tree_real(v, child_proxy, tracer)
