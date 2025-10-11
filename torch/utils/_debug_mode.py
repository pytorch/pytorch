# mypy: allow-untyped-defs
import contextlib
from typing import Optional

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


def _stringify_attributes(tensor, attributes) -> str:
    pairs = {}
    for attr in attributes:
        if hasattr(tensor, attr):
            pairs[attr] = getattr(tensor, attr)
    if len(pairs) == 0:
        return ""
    return f"{{{', '.join([f'{k}={v}' for k, v in pairs.items()])}}}"


def _tensor_debug_string(tensor, attributes) -> str:
    """Convert tensor to debug string representation."""

    if isinstance(tensor, torch.Tensor):
        tensor_debug_str = f"{dtype_abbrs[tensor.dtype]}{_stringify_shape(tensor.shape)}{_stringify_attributes(tensor, attributes)}"

        if isinstance(tensor, torch.distributed.tensor.DTensor):
            # omitted device mesh
            return f"dt: {tensor_debug_str}{_stringify_placement(tensor.placements)}"
        elif isinstance(tensor, FakeTensor):
            return f"ft: {tensor_debug_str}"
        else:
            return f"t: {tensor_debug_str}"
    else:
        raise RuntimeError(f"Unsupported tensor type: {type(tensor)}")


def _arg_to_str(arg, attributes) -> str:
    from torch.distributed.tensor._dtensor_spec import DTensorSpec

    def to_str(x):
        if isinstance(x, torch.Tensor):
            return _tensor_debug_string(x, attributes)
        elif isinstance(x, DTensorSpec):
            return _stringify_placement(x.placements)
        return x

    arg = tree_map(to_str, arg)
    return str(arg)


def _op_to_str(op, attributes, *args, **kwargs) -> str:
    if op == REDISTRIBUTE_FUNC:
        assert len(args) == 3
        _args = [_arg_to_str(arg, attributes) for arg in args]
        args_str = f"{_args[0]}, {_args[1]} -> {_args[2]}"
    else:
        args_str = ", ".join(_arg_to_str(arg, attributes) for arg in args)

    if kwargs:
        kwargs_str = ", " + ", ".join(
            f"{k}={_arg_to_str(v, attributes)}" for k, v in kwargs.items()
        )
    else:
        kwargs_str = ""

    if isinstance(op, torch._ops.OpOverload):
        op_name = op.__qualname__
    elif hasattr(op, "__module__") and hasattr(op, "__name__"):
        op_name = f"{op.__module__}.{op.__name__}"
    else:
        op_name = str(op)

    return f"{op_name}({args_str}{kwargs_str})"


class DebugMode(TorchDispatchMode):
    def __init__(
        self,
        *,
        record_torchfunction=False,
        record_faketensor=False,
        record_realtensor=True,
        record_tensor_attributes=None,
    ):
        super().__init__()
        import torch.distributed.tensor  # noqa: F401

        self.supports_higher_order_operators = True
        self.record_torchfunction = record_torchfunction
        self.record_faketensor = record_faketensor
        self.record_realtensor = record_realtensor
        self.record_tensor_attributes = record_tensor_attributes or []

        self.reset_logs()

    # Without this override, running torch.compile under DebugMode
    # will force torch.compile to always use the “eager” backend
    # With this, DebugMode will not take effect on torch.compile
    @classmethod
    def ignore_compile_internals(cls):
        return True

    def log_operator(self, op, args, kwargs, depth):
        if not torch.compiler.is_compiling():
            self.operators.append((op, args, kwargs, depth))

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if not (type(func).__name__ == "method-wrapper" and func.__name__ == "__get__"):
            # filter out known noise
            self.log_operator(func, args, kwargs, self.call_depth)

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
            self.log_operator(func, args, kwargs, self.call_depth)
            return NotImplemented
        elif FakeTensor in types or isinstance(
            _get_current_dispatch_mode(), FakeTensorMode
        ):
            if self.record_faketensor:
                if func != torch.ops.prim.device.default:
                    self.log_operator(func, args, kwargs, self.call_depth + 1)
        elif len(types) == 0:
            if self.record_realtensor:
                self.log_operator(func, args, kwargs, self.call_depth + 1)

        result = func(*args, **kwargs)

        return result

    def __enter__(self):
        self.reset_logs()

        if self.record_torchfunction:
            from torch._dynamo.guards import _register_global_guard_filter_fn

            _torch_function_stack_index = torch._C._len_torch_function_stack()
            torch._C._push_on_torch_function_stack(self)

            def guard_filter_fn(guard_entries):
                kw = f"___get_torch_function_mode_stack_at({_torch_function_stack_index})"
                return [
                    (
                        kw not in entry.name
                        or (entry.name == kw and entry.guard_type == "TYPE_MATCH")
                    )
                    and entry.guard_type != "TENSOR_MATCH"
                    for entry in guard_entries
                ]

            self._guard_filter_fn = guard_filter_fn
            _register_global_guard_filter_fn(guard_filter_fn)

        super().__enter__()
        return self

    # pyrefly: ignore  # bad-override
    def __exit__(self, *args):
        super().__exit__(*args)
        if self.record_torchfunction:
            from torch._dynamo.guards import _remove_global_guard_filter_fn

            torch._C._pop_torch_function_stack()
            _remove_global_guard_filter_fn(self._guard_filter_fn)

    def reset_logs(self):
        self.operators = []
        self.call_depth = 0

    @contextlib.contextmanager
    def record_redistribute_calls(self, arg_idx, src_placement, dst_placement):
        try:
            self.log_operator(
                REDISTRIBUTE_FUNC,
                [arg_idx, src_placement, dst_placement],
                {},
                self.call_depth + 1,
            )
            self.call_depth += 1
            yield
        finally:
            self.call_depth -= 1

    def debug_string(self) -> str:
        with torch._C.DisableTorchFunction():
            result = ""
            result += "\n".join(
                "  "
                + "  " * depth
                + _op_to_str(op, self.record_tensor_attributes, *args, **kwargs)
                for op, args, kwargs, depth in self.operators
            )
        return result


def get_active_debug_mode() -> Optional[DebugMode]:
    debug_mode = None
    for mode in _get_current_dispatch_mode_stack():
        if isinstance(mode, DebugMode):
            debug_mode = mode
            break
    return debug_mode
