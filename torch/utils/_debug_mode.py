# mypy: allow-untyped-defs
import contextlib
import weakref
from typing import Optional

import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.utils._dtype_abbrs import dtype_abbrs
from torch.utils._python_dispatch import (
    _get_current_dispatch_mode,
    _get_current_dispatch_mode_stack,
    TorchDispatchMode,
)
from torch.utils._pytree import tree_map, tree_map_only
from torch.utils.weak import WeakIdRef


__all__ = ["DebugMode", "get_active_debug_mode"]

REDISTRIBUTE_FUNC = "redistribute_input"


def _stringify_shape(shape) -> str:
    return f"[{', '.join([str(x) for x in shape])}]"


def _stringify_device_mesh(mesh) -> str:
    return f"DM({', '.join([str(s) for s in mesh.shape])})"


def _stringify_placement(placement) -> str:
    # currently used; defer to _stringify_dtensor_spec
    return f"[{', '.join([str(p) for p in placement])}]"


def _stringify_attributes(tensor, attributes) -> str:
    pairs = {}
    for attr in attributes:
        if hasattr(tensor, attr):
            pairs[attr] = getattr(tensor, attr)
    if len(pairs) == 0:
        return ""
    return f"{{{', '.join([f'{k}={v}' for k, v in pairs.items()])}}}"


def _stringify_dtensor_spec(spec) -> str:
    from torch.distributed.tensor._dtensor_spec import DTensorSpec

    return DTensorSpec.format_shard_order_str(spec.placements, spec.shard_order)


def _tensor_debug_string(tensor, attributes, tensor_memo=None) -> str:
    """Convert tensor to debug string representation."""

    if isinstance(tensor, torch.Tensor):
        if isinstance(tensor, torch.distributed.tensor.DTensor):
            # omitted device mesh
            prefix = "dt"
            base_str = f"{dtype_abbrs[tensor.dtype]}{_stringify_shape(tensor.shape)}"
            base_str += f"{_stringify_attributes(tensor, attributes)}"
            base_str += f"| {_stringify_dtensor_spec(tensor._spec)}"
        else:
            base_str = f"{dtype_abbrs[tensor.dtype]}{_stringify_shape(tensor.shape)}"
            base_str += f"{_stringify_attributes(tensor, attributes)}"
            if isinstance(tensor, FakeTensor):
                prefix = "ft"
            else:
                prefix = "t"
    else:
        raise RuntimeError(f"Unsupported tensor type: {type(tensor)}")

    tensor_id = (
        f"${tensor_memo.get(tensor)}" if tensor_memo and tensor in tensor_memo else ""
    )
    return f"{prefix}{tensor_id}: {base_str}"


def _arg_to_str(arg, attributes, tensor_memo=None) -> str:
    from torch.distributed.tensor._dtensor_spec import DTensorSpec

    def to_str(x):
        if isinstance(x, torch.Tensor):
            return _tensor_debug_string(x, attributes, tensor_memo)
        elif isinstance(x, DTensorSpec):
            return _stringify_dtensor_spec(x)
        return x

    arg = tree_map(to_str, arg)
    return str(arg)


def _op_to_str(op, attributes, *args, output=None, tensor_memo=None, **kwargs) -> str:
    if op == REDISTRIBUTE_FUNC:
        if len(args) == 2:
            args_str = (
                f"{_arg_to_str(args[0], attributes, tensor_memo)}, trace: {args[1]}"
            )
        elif len(args) == 3:
            _args = [_arg_to_str(arg, attributes, tensor_memo) for arg in args]
            args_str = f"{_args[0]}, {_args[1]} -> {_args[2]}"
        else:
            raise RuntimeError(f"Unsupported args for {REDISTRIBUTE_FUNC}: {args}")
    else:
        args_str = ", ".join(_arg_to_str(arg, attributes, tensor_memo) for arg in args)

    if kwargs:
        kwargs_str = ", " + ", ".join(
            f"{k}={_arg_to_str(v, attributes, tensor_memo)}" for k, v in kwargs.items()
        )
    else:
        kwargs_str = ""

    if isinstance(op, torch._ops.OpOverload):
        op_name = op.__qualname__
    elif hasattr(op, "__module__") and hasattr(op, "__name__"):
        op_name = f"{op.__module__}.{op.__name__}"
    else:
        op_name = str(op)

    # Add output annotation
    output_str = ""
    if output is not None:
        output_str = f" -> {_arg_to_str(output, attributes, tensor_memo)}"

    return f"{op_name}({args_str}{kwargs_str}){output_str}"


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

        self.operators = []
        self.call_depth = 0

        self._tensor_memo: dict[WeakIdRef, int] = {}
        self._next_tensor_id = 0
        self._output_info: dict[int, object] = {}

    def _assign_tensor_id(self, tensor):
        o = WeakIdRef(tensor)
        weak_self = weakref.ref(self)

        def del_memo():
            self = weak_self()
            if self is None:
                return
            self._tensor_memo.pop(o, None)

        weakref.finalize(tensor, del_memo)
        if tensor not in self._tensor_memo:
            self._tensor_memo[tensor] = self._next_tensor_id
            self._next_tensor_id += 1

    def _track_tensor_ids(self, obj):
        """Recursively assign IDs to all tensors in a pytree"""
        with torch._C.DisableTorchFunction():
            tree_map_only(torch.Tensor, self._assign_tensor_id, obj)

    def _track_op_output(self, op_index, result):
        """Assign IDs to output tensors and store in output_info"""
        with torch._C.DisableTorchFunction():
            self._track_tensor_ids(result)
            self._output_info[op_index] = result

    # Without this override, running torch.compile under DebugMode
    # will force torch.compile to always use the “eager” backend
    # With this, DebugMode will not take effect on torch.compile
    @classmethod
    def ignore_compile_internals(cls):
        return True

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        # assign ids to input tensors
        self._track_tensor_ids((args, kwargs))

        # Store operator before execution
        op_index = len(self.operators)
        self.operators.append((func, args, kwargs, self.call_depth))

        try:
            self.call_depth += 1
            result = func(*args, **kwargs)
            self._track_op_output(op_index, result)
            return result
        finally:
            self.call_depth -= 1

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        # assign ids to input tensors
        self._track_tensor_ids((args, kwargs))

        # Record the operation with its call depth
        op_index = None
        if torch.distributed.tensor.DTensor in types:
            self.operators.append((func, args, kwargs, self.call_depth))
            return NotImplemented
        elif FakeTensor in types or isinstance(
            _get_current_dispatch_mode(), FakeTensorMode
        ):
            if self.record_faketensor:
                if func != torch.ops.prim.device.default:
                    op_index = len(self.operators)
                    self.operators.append((func, args, kwargs, self.call_depth + 1))
        elif len(types) == 0:
            if self.record_realtensor:
                op_index = len(self.operators)
                self.operators.append((func, args, kwargs, self.call_depth + 1))

        result = func(*args, **kwargs)
        if op_index is not None:  # we logged something, track index -> output
            self._track_op_output(op_index, result)

        return result

    def __enter__(self):
        self.operators = []
        self.call_depth = 0
        self._tensor_memo = {}
        self._next_tensor_id = 0
        self._output_info = {}

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
    def record_redistribute_calls(
        self,
        arg_idx,
        src_placement,
        dst_placement,
        transform_info_str: Optional[str] = None,
    ):
        try:
            arg_list = (
                [arg_idx, transform_info_str]
                if transform_info_str
                else [arg_idx, src_placement, dst_placement]
            )
            self.operators.append(
                (
                    REDISTRIBUTE_FUNC,
                    arg_list,
                    {},
                    self.call_depth + 1,
                )
            )
            self.call_depth += 1
            yield
        finally:
            self.call_depth -= 1

    def debug_string(self, show_outputs: bool = False, show_ids: bool = False) -> str:
        with torch._C.DisableTorchFunction():
            result = ""
            tensor_memo = self._tensor_memo if show_ids else None
            result += "\n".join(
                "  "
                + "  " * depth
                + _op_to_str(
                    op,
                    self.record_tensor_attributes,
                    *args,
                    output=self._output_info.get(idx) if show_outputs else None,
                    tensor_memo=tensor_memo,
                    **kwargs,
                )
                for idx, (op, args, kwargs, depth) in enumerate(self.operators)
            )
        return result


def get_active_debug_mode() -> Optional[DebugMode]:
    debug_mode = None
    for mode in _get_current_dispatch_mode_stack():
        if isinstance(mode, DebugMode):
            debug_mode = mode
            break
    return debug_mode
