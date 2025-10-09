# mypy: allow-untyped-defs
import contextlib
from typing import Any, Optional

import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.utils._dtype_abbrs import dtype_abbrs
from torch.utils._python_dispatch import (
    _get_current_dispatch_mode,
    _get_current_dispatch_mode_stack,
    TorchDispatchMode,
)
from torch.utils._pytree import tree_map


__all__ = ["DebugMode", "get_active_debug_mode"]

REDISTRIBUTE_FUNC = "redistribute_input"


def _stringify_shape(shape) -> str:
    return f"[{', '.join([str(x) for x in shape])}]"


def _stringify_device_mesh(mesh) -> str:
    return f"DM({', '.join([str(s) for s in mesh.shape])})"


def _stringify_placement(placement) -> str:
    return f"[{', '.join([str(p) for p in placement])}]"


def _tensor_debug_string(tensor) -> str:
    """Convert tensor to debug string representation."""
    if isinstance(tensor, torch.distributed.tensor.DTensor):
        # omitted device mesh
        return f"dt: {dtype_abbrs[tensor.dtype]}{_stringify_shape(tensor.shape)}{_stringify_placement(tensor.placements)}"
    elif isinstance(tensor, FakeTensor):
        return f"ft: {dtype_abbrs[tensor.dtype]}{_stringify_shape(tensor.shape)}"
    elif isinstance(tensor, torch.Tensor):
        return f"t: {dtype_abbrs[tensor.dtype]}{_stringify_shape(tensor.shape)}"
    else:
        raise RuntimeError(f"Unsupported tensor type: {type(tensor)}")


def _arg_to_str(arg) -> str:
    from torch.distributed.tensor._dtensor_spec import DTensorSpec

    def to_str(x):
        if isinstance(x, torch.Tensor):
            return _tensor_debug_string(x)
        elif isinstance(x, DTensorSpec):
            return _stringify_placement(x.placements)
        return x

    arg = tree_map(to_str, arg)
    return str(arg)


class _OperatorCall:
    """Base class for tracking operator calls in DebugMode"""
    def __init__(self, call_depth: int):
        self.call_depth = call_depth

    def __str__(self) -> str:
        raise NotImplementedError("Subclasses must implement __str__")

    __repr__ = __str__


class _OpCall(_OperatorCall):
    """Normal operator call"""
    def __init__(self, op, args: tuple, kwargs: dict, call_depth: int):
        super().__init__(call_depth)
        self.op = op
        self.args = args
        self.kwargs = kwargs

    def __str__(self) -> str:
        args_str = ", ".join(_arg_to_str(arg) for arg in self.args)

        if self.kwargs:
            kwargs_str = ", " + ", ".join(
                f"{k}={_arg_to_str(v)}" for k, v in self.kwargs.items()
            )
        else:
            kwargs_str = ""

        if isinstance(self.op, torch._ops.OpOverload):
            op_name = self.op.__qualname__
        elif hasattr(self.op, "__module__") and hasattr(self.op, "__name__"):
            op_name = f"{self.op.__module__}.{self.op.__name__}"
        else:
            op_name = str(self.op)
        return f"{op_name}({args_str}{kwargs_str})"


class _RedistributeCall(_OperatorCall):
    """Redistribute call from DTensor dispatch"""
    def __init__(self, arg, src_placement, dst_placement, call_depth):
        super().__init__(call_depth)
        self.arg = arg
        self.src_placement = src_placement
        self.dst_placement = dst_placement

    def __str__(self) -> str:
        arg_str = f"{_arg_to_str(self.arg)}"
        src_placement_str = _arg_to_str(self.src_placement)
        dst_placement_str = _arg_to_str(self.dst_placement)
        return f"{REDISTRIBUTE_FUNC}({arg_str}, {src_placement_str} -> {dst_placement_str})"


class _InductorGraphCall(_OperatorCall):
    """Inductor compiled graph call (at runtime)"""
    def __init__(self, post_grad_graph: str, cache_key: str, inputs: tuple, fx_kwargs: dict, call_depth: int):
        super().__init__(call_depth)
        self.post_grad_graph = post_grad_graph
        self.cache_key = cache_key
        self.inputs = inputs
        self.fx_kwargs = fx_kwargs

    def __str__(self) -> str:
        # Base indentation for this call
        base_indent = "  " + "  " * self.call_depth
        inner_indent = base_indent + "  "

        # Runtime args
        args_str = ", ".join(_arg_to_str(arg) for arg in self.inputs)

        # Add call_depth indentation to graph module str
        graph_lines = self.post_grad_graph.strip().split('\n')
        indented_graph = '\n'.join(inner_indent + "  " +  line for line in graph_lines)

        # Format fx_kwargs
        if self.fx_kwargs:
            kwargs_items = [
                f"{k}={v}"
                for k, v in self.fx_kwargs.items()
            ]
            fx_kwargs_str = ", ".join(kwargs_items)
        else:
            fx_kwargs_str = ""

        # Build the full string
        result = f"inductor_graph_call(\n"
        result += f"{inner_indent}inputs: ({args_str})\n"
        result += f"{inner_indent}cache_key: {self.cache_key}\n"
        if fx_kwargs_str:
            result += f"{inner_indent}fx_kwargs: {{{fx_kwargs_str}}}\n"
        result += f"{inner_indent}post_grad_graph:\n"
        result += indented_graph + "\n"
        result += f"{base_indent})"
        return result


class DebugMode(TorchDispatchMode):
    def __init__(
        self,
        *,
        record_torchfunction=False,
        record_faketensor=False,
        record_realtensor=True,
    ):
        super().__init__()
        import torch.distributed.tensor  # noqa: F401

        self.supports_higher_order_operators = True
        self.record_torchfunction = record_torchfunction
        self.record_faketensor = record_faketensor
        self.record_realtensor = record_realtensor

        self.operators = []
        self.call_depth = 0

    # Without this override, running torch.compile under DebugMode
    # will force torch.compile to always use the “eager” backend
    # With this, DebugMode will not take effect on torch.compile
    @classmethod
    def ignore_compile_internals(cls):
        return True

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        self.operators.append(_OpCall(func, args, kwargs, self.call_depth))

        try:
            self.call_depth += 1
            return func(*args, **kwargs)
        finally:
            self.call_depth -= 1

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        # Record the operation with its call depth
        if torch.distributed.tensor.DTensor in types:
            self.operators.append(_OpCall(func, args, kwargs, self.call_depth))
            return NotImplemented
        elif FakeTensor in types or isinstance(
            _get_current_dispatch_mode(), FakeTensorMode
        ):
            if self.record_faketensor:
                if func != torch.ops.prim.device.default:
                    self.operators.append(_OpCall(func, args, kwargs, self.call_depth + 1))
        elif len(types) == 0:
            if self.record_realtensor:
                self.operators.append(_OpCall(func, args, kwargs, self.call_depth + 1))

        result = func(*args, **kwargs)

        return result

    def __enter__(self):
        self.operators = []
        self.call_depth = 0

        if self.record_torchfunction:
            torch._C._push_on_torch_function_stack(self)

        super().__enter__()
        return self

    # pyrefly: ignore  # bad-override
    def __exit__(self, *args):
        super().__exit__(*args)
        if self.record_torchfunction:
            torch._C._pop_torch_function_stack()

    @contextlib.contextmanager
    def record_redistribute_calls(self, arg, src_placement, dst_placement):
        try:
            self.operators.append(
                _RedistributeCall(
                    arg, src_placement, dst_placement, self.call_depth + 1
                )
            )
            self.call_depth += 1
            yield
        finally:
            self.call_depth -= 1

    def record_inductor_graph_call(self, post_grad_graph: str, cache_key: str, inputs: tuple[Any], fx_kwargs: dict[str, Any]):
        self.operators.append(
            _InductorGraphCall(
                post_grad_graph, cache_key, inputs, fx_kwargs, self.call_depth + 1
            )
        )

    def debug_string(self) -> str:
        with torch._C.DisableTorchFunction():
            result = ""
            result += "\n".join(
                "  " + "  " * call.call_depth + str(call)
                for call in self.operators
            )
        return result


def get_active_debug_mode() -> Optional[DebugMode]:
    debug_mode = None
    for mode in _get_current_dispatch_mode_stack():
        if isinstance(mode, DebugMode):
            debug_mode = mode
            break
    return debug_mode
