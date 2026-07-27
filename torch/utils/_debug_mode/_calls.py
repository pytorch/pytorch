# mypy: allow-untyped-defs
"""
Call record classes for DebugMode: _DebugCall and its subclasses.
"""

import math
from typing import Any, TYPE_CHECKING

import torch

# Import _utils module for mutable globals
from torch.utils._debug_mode import _utils
from torch.utils._debug_mode._utils import (
    _arg_to_str,
    _get_op_name,
    _get_stack_trace,
    _maybe_get_autograd_trace,
    REDISTRIBUTE_FUNC,
    TensorIdTracker,
)
from torch.utils._pytree import tree_all, tree_map


if TYPE_CHECKING:
    from torch._dynamo.device_interface import DeviceInterface


class _DebugCall:
    """Base class for tracking operator calls in DebugMode"""

    def __init__(
        self,
        call_depth: int,
        record: dict[str, Any] | None = None,
        log: dict[str, Any] | None = None,
        stack: bool = False,
    ) -> None:
        self.call_depth = call_depth
        if stack:
            self.stack_trace = _get_stack_trace()
            self.fwd_stack_trace = _maybe_get_autograd_trace()

        # results from dispatch hooks
        self.record = record
        self.log = log
        self.output_str: str | None = None

    def stringify_args(
        self, attributes: list[str], tensor_memo: TensorIdTracker | None = None
    ) -> None:
        """
        To reduce memory consumption, this method stringifies args/kwargs, stores the result, and deletes original args/kwargs.
        """
        raise NotImplementedError(
            "Subclasses must implement stringify_args(), even if no-op"
        )

    def stringify_output(
        self,
        output: Any,
        attributes: list[str],
        tensor_memo: TensorIdTracker | None = None,
    ) -> None:
        """Store stringified version of call output in self.output_str"""
        if tree_all(lambda x: x is None, output):
            return
        output_str = tree_map(lambda x: _arg_to_str(x, attributes, tensor_memo), output)
        self.output_str = f"  ->  {str(output_str)}"

    def render(self, attributes: list[str]) -> str:
        raise NotImplementedError("Subclasses must implement string render()")

    def __repr__(self) -> str:
        return self.render([])


class _OpCall(_DebugCall):
    """Normal operator call"""

    def __init__(
        self,
        op,
        args: tuple,
        kwargs: dict,
        call_depth: int,
        stack: bool = False,
    ) -> None:
        super().__init__(call_depth, stack=stack)
        self.op = op
        self.args = args
        self.kwargs = kwargs

        self.args_str: str | None = None
        self.kwargs_str: str | None = None

    def stringify_args(
        self, attributes: list[str], tensor_memo: TensorIdTracker | None = None
    ) -> None:
        self.args_str = ", ".join(
            _arg_to_str(arg, attributes, tensor_memo) for arg in self.args
        )
        if self.kwargs:
            self.kwargs_str = ", " + ", ".join(
                f"{k}={_arg_to_str(v, attributes, tensor_memo)}"
                for k, v in self.kwargs.items()
            )
        else:
            self.kwargs_str = ""
        del self.args
        del self.kwargs

    def render(self, attributes: list[str]) -> str:
        if self.args_str is not None:
            args_str = self.args_str
        else:
            args_str = ", ".join(_arg_to_str(arg, attributes) for arg in self.args)

        if self.kwargs_str is not None:
            kwargs_str = self.kwargs_str
        else:
            if self.kwargs:
                kwargs_str = ", " + ", ".join(
                    f"{k}={_arg_to_str(v, attributes)}" for k, v in self.kwargs.items()
                )
            else:
                kwargs_str = ""

        if isinstance(self.op, torch._ops.OpOverload):
            op_name = self.op.__qualname__
        elif hasattr(self.op, "__module__") and hasattr(self.op, "__name__"):
            op_name = f"{self.op.__module__}.{self.op.__name__}"
        else:
            op_name = str(self.op)

        base_str = f"{op_name}({args_str}{kwargs_str})"

        if self.output_str:
            base_str += self.output_str
        if self.log:
            base_str += f"  # {self.log}"
        return base_str

    def __iter__(self):
        # for BC; tuple(self) returns (op, args, kwargs, call_depth)
        if self.args_str is not None:
            yield from [self.op, self.args_str, self.kwargs_str, self.call_depth]
        else:
            yield from [self.op, self.args, self.kwargs, self.call_depth]


class _RedistributeCall(_DebugCall):
    def __init__(
        self,
        arg,
        src_placement,
        dst_placement,
        transform_info_str,
        call_depth,
        stack=False,
        is_explicit=False,
    ) -> None:
        super().__init__(call_depth, stack=stack)
        self.arg = arg
        self.src_placement = src_placement
        self.dst_placement = dst_placement
        self.transform_info_str = transform_info_str
        self.is_explicit = is_explicit
        self.is_outer_call = isinstance(arg, int)

        self.arg_str: str | None = None

    def stringify_args(
        self, attributes: list[str], tensor_memo: TensorIdTracker | None = None
    ) -> None:
        self.arg_str = f"{_arg_to_str(self.arg, attributes, tensor_memo)}"
        del self.arg

    def render(self, attributes: list[str]) -> str:
        if self.arg_str is not None:
            arg_str = self.arg_str
        else:
            arg_str = f"{_arg_to_str(self.arg, attributes)}"

        if self.transform_info_str is not None:  # prioritize over src/dst placements
            placement_str = f"trace: {self.transform_info_str}"
        else:
            src_placement_str = _arg_to_str(self.src_placement, attributes)
            dst_placement_str = _arg_to_str(self.dst_placement, attributes)
            placement_str = f"{src_placement_str} -> {dst_placement_str}"

        # DebugMode will add redistribute_input logs at 2 levels,
        # once per redistribute decision, and once per redistributed input.
        # We only annotate [implicit/explicit] logs on the former (outer-level call).
        if self.is_outer_call:
            annotation = " [implicit] "
        elif self.is_explicit:
            annotation = " [explicit] "
        else:
            annotation = ""

        base_str = f"{REDISTRIBUTE_FUNC}{annotation}({arg_str}, {placement_str})"
        if self.output_str:
            base_str += self.output_str
        return base_str

    def __iter__(self):
        # for BC; tuple(self) returns (op, placement info, kwargs, call_depth)
        if self.arg_str is not None:
            arg = self.arg_str
        else:
            arg = self.arg

        yield REDISTRIBUTE_FUNC
        if self.transform_info_str:
            yield [arg, self.transform_info_str]
        else:
            yield [arg, self.src_placement, self.dst_placement]
        yield {}
        yield self.call_depth


class _OutputPlacementCall(_DebugCall):
    """Records output placement for a DTensor op."""

    def __init__(self, placements_str: str, call_depth: int) -> None:
        super().__init__(call_depth)
        self.placements_str = placements_str

    def stringify_args(
        self, attributes: list[str], tensor_memo: TensorIdTracker | None = None
    ) -> None:
        pass  # Already stringified

    def render(self, attributes: list[str]) -> str:
        return f"-> output: {self.placements_str}"


class _TritonKernelCall(_DebugCall):
    """Triton kernel call from Inductor"""

    def __init__(
        self,
        kernel_name: str,
        kwargs: dict[str, Any],
        call_depth: int,
    ):
        super().__init__(call_depth)
        self.kernel_name = kernel_name
        self.kwargs = kwargs
        self.kwargs_str: str | None = None

        self.pre_hashes: dict[str, Any] | None = None
        self.post_hashes: dict[str, Any] | None = None

    def stringify_args(
        self, attributes: list[str], tensor_memo: TensorIdTracker | None = None
    ) -> None:
        # Optionally hash kernel inputs before launch
        if hash_fn := _utils._TRITON_INPUT_HASH_FN:
            self.pre_hashes = {
                k: hash_fn(v)
                for k, v in self.kwargs.items()
                if isinstance(v, torch.Tensor)
            }

        if self.kwargs:
            self.kwargs_str = ", ".join(
                f"{k}={_arg_to_str(v, attributes, tensor_memo)}"
                for k, v in self.kwargs.items()
            )
        else:
            self.kwargs_str = ""

    def render(self, attributes: list[str]) -> str:
        base_str = f"[triton] {self.kernel_name}({self.kwargs_str})"
        if self.pre_hashes:
            pre_hashes_str = ", ".join(f"{k}: {v}" for k, v in self.pre_hashes.items())
            pre_hashes_str = (
                "\n  "
                + "  " * self.call_depth
                + f"# pre-kernel hashes: {{{pre_hashes_str}}}"
            )
        else:
            pre_hashes_str = ""
        if self.post_hashes:
            post_hashes_str = ", ".join(
                f"{k}: {v}" for k, v in self.post_hashes.items()
            )
            post_hashes_str = (
                "\n  "
                + "  " * self.call_depth
                + f"# post-kernel hashes: {{{post_hashes_str}}}"
            )
        else:
            post_hashes_str = ""
        return f"{base_str}{pre_hashes_str}{post_hashes_str}\n"

    def finalize(self, device_interface: "DeviceInterface"):
        # synchronize -> hash/store kernel results
        device_interface.synchronize(device_interface.current_device())
        if _utils._RECORD_TRITON_OUTPUTS:
            self.record = {
                "output": {
                    k: v.clone() if isinstance(v, torch.Tensor) else v
                    for k, v in self.kwargs.items()
                }
            }
        if hash_fn := _utils._TRITON_OUTPUT_HASH_FN:
            self.post_hashes = {
                k: hash_fn(v)
                for k, v in self.kwargs.items()
                if isinstance(v, torch.Tensor)
            }

        # don't store tensors
        del self.kwargs

    def __iter__(self):
        yield from [self.kernel_name, (), self.kwargs_str, self.call_depth]


class _AnnotateCall(_DebugCall):
    """Custom annotation call"""

    def __init__(
        self, tag: Any, header: str, call_depth: int, stack: bool = False
    ) -> None:
        super().__init__(call_depth, stack=stack)
        self.tag = tag
        self.header = header

    def render(self, attributes: list[str]) -> str:
        return f"[{self.header}] {self.tag}"

    def __iter__(self):
        yield from [
            f"[{self.header}] {self.tag}",
            (),
            {},
            self.call_depth,
        ]


def _serialize_debug_value(value: Any) -> dict[str, Any]:
    if value is None or isinstance(value, (bool, int, str)):
        return {"type": "scalar", "value": value}
    if isinstance(value, float):
        if math.isnan(value):
            return {"type": "nonfinite_float", "value": "nan"}
        if math.isinf(value):
            return {
                "type": "nonfinite_float",
                "value": "inf" if value > 0 else "-inf",
            }
        return {"type": "scalar", "value": value}
    if isinstance(value, tuple):
        return {
            "type": "tuple",
            "items": [_serialize_debug_value(item) for item in value],
        }
    if isinstance(value, list):
        return {
            "type": "list",
            "items": [_serialize_debug_value(item) for item in value],
        }
    if isinstance(value, dict):
        items = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    "DebugMode log serialization only supports dicts with string keys, "
                    f"but found key {key!r} of type {type(key).__name__}"
                )
            items.append([key, _serialize_debug_value(item)])
        return {"type": "dict", "items": items}
    raise TypeError(
        "DebugMode log serialization only supports JSON-compatible scalar, "
        f"tuple, list, and dict values, but found {type(value).__name__}"
    )


def _deserialize_debug_value(value: Any) -> Any:
    if not isinstance(value, dict):
        raise ValueError(
            f"Serialized DebugMode value must be a dict, but found {type(value).__name__}"
        )
    value_type = value.get("type")
    if value_type == "scalar":
        return value.get("value")
    if value_type == "nonfinite_float":
        nonfinite_value = value.get("value")
        if nonfinite_value == "nan":
            return float("nan")
        if nonfinite_value == "inf":
            return float("inf")
        if nonfinite_value == "-inf":
            return float("-inf")
        raise ValueError(
            f"Unknown serialized DebugMode non-finite float: {nonfinite_value!r}"
        )
    if value_type == "tuple":
        items = value.get("items")
        if not isinstance(items, list):
            raise ValueError("Serialized DebugMode tuple must contain an items list")
        return tuple(_deserialize_debug_value(item) for item in items)
    if value_type == "list":
        items = value.get("items")
        if not isinstance(items, list):
            raise ValueError("Serialized DebugMode list must contain an items list")
        return [_deserialize_debug_value(item) for item in items]
    if value_type == "dict":
        items = value.get("items")
        if not isinstance(items, list):
            raise ValueError("Serialized DebugMode dict must contain an items list")
        result = {}
        for item in items:
            if not isinstance(item, list) or len(item) != 2:
                raise ValueError(
                    "Serialized DebugMode dict items must be [key, value] pairs"
                )
            key, item_value = item
            if not isinstance(key, str):
                raise ValueError(
                    "Serialized DebugMode dict keys must be strings, "
                    f"but found {type(key).__name__}"
                )
            result[key] = _deserialize_debug_value(item_value)
        return result
    raise ValueError(f"Unknown serialized DebugMode value type: {value_type!r}")


def _serialize_optional_debug_value(value: Any | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return _serialize_debug_value(value)


def _deserialize_optional_debug_value(value: Any | None) -> Any | None:
    if value is None:
        return None
    return _deserialize_debug_value(value)


def _serialize_common_call_fields(call: _DebugCall) -> dict[str, Any]:
    return {
        "call_depth": call.call_depth,
        "log": _serialize_optional_debug_value(call.log),
        "output_str": call.output_str,
        "stack_trace": getattr(call, "stack_trace", None),
        "fwd_stack_trace": getattr(call, "fwd_stack_trace", None),
    }


def _restore_common_call_fields(call: _DebugCall, data: dict[str, Any]) -> None:
    call.log = _deserialize_optional_debug_value(data.get("log"))
    call.output_str = data.get("output_str")
    stack_trace = data.get("stack_trace")
    if stack_trace is not None:
        call.stack_trace = stack_trace
    fwd_stack_trace = data.get("fwd_stack_trace")
    if fwd_stack_trace is not None:
        call.fwd_stack_trace = fwd_stack_trace


def _op_call_args_str(call: _OpCall) -> str:
    if call.args_str is not None:
        return call.args_str
    return ", ".join(_arg_to_str(arg, []) for arg in call.args)


def _op_call_kwargs_str(call: _OpCall) -> str:
    if call.kwargs_str is not None:
        return call.kwargs_str
    if call.kwargs:
        return ", " + ", ".join(
            f"{key}={_arg_to_str(value, [])}" for key, value in call.kwargs.items()
        )
    return ""


def _redistribute_arg_str(call: _RedistributeCall) -> str:
    if call.arg_str is not None:
        return call.arg_str
    return _arg_to_str(call.arg, [])


def _triton_kwargs_str(call: _TritonKernelCall) -> str:
    if call.kwargs_str is not None:
        return call.kwargs_str
    if call.kwargs:
        return ", ".join(
            f"{key}={_arg_to_str(value, [])}" for key, value in call.kwargs.items()
        )
    return ""


def _serialized_str(data: dict[str, Any], field: str) -> str:
    value = data.get(field)
    if not isinstance(value, str):
        raise ValueError(
            f"Serialized DebugMode call field {field!r} must be a string, "
            f"but found {type(value).__name__}"
        )
    return value


def _serialized_optional_str(data: dict[str, Any], field: str) -> str | None:
    value = data.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(
            f"Serialized DebugMode call field {field!r} must be a string or None, "
            f"but found {type(value).__name__}"
        )
    return value


def _serialize_debug_call(call: _DebugCall) -> dict[str, Any]:
    data = _serialize_common_call_fields(call)
    if isinstance(call, _OpCall):
        data.update(
            {
                "type": "op",
                "op_name": _get_call_name(call),
                "args_str": _op_call_args_str(call),
                "kwargs_str": _op_call_kwargs_str(call),
            }
        )
    elif isinstance(call, _RedistributeCall):
        data.update(
            {
                "type": "redistribute",
                "arg_str": _redistribute_arg_str(call),
                "src_placement_str": _arg_to_str(call.src_placement, []),
                "dst_placement_str": _arg_to_str(call.dst_placement, []),
                "transform_info_str": call.transform_info_str,
                "is_explicit": call.is_explicit,
                "is_outer_call": call.is_outer_call,
            }
        )
    elif isinstance(call, _OutputPlacementCall):
        data.update(
            {
                "type": "output_placement",
                "placements_str": call.placements_str,
            }
        )
    elif isinstance(call, _TritonKernelCall):
        data.update(
            {
                "type": "triton_kernel",
                "kernel_name": call.kernel_name,
                "kwargs_str": _triton_kwargs_str(call),
                "pre_hashes": _serialize_optional_debug_value(call.pre_hashes),
                "post_hashes": _serialize_optional_debug_value(call.post_hashes),
            }
        )
    elif isinstance(call, _AnnotateCall):
        data.update(
            {
                "type": "annotate",
                "tag": str(call.tag),
                "header": call.header,
            }
        )
    else:
        raise TypeError(f"Unsupported DebugMode call type: {type(call).__name__}")
    return data


def _deserialize_debug_call(data: Any) -> _DebugCall:
    if not isinstance(data, dict):
        raise ValueError(
            f"Serialized DebugMode call must be a dict, but found {type(data).__name__}"
        )
    call_type = data.get("type")
    call_depth = data.get("call_depth")
    if not isinstance(call_depth, int):
        raise ValueError("Serialized DebugMode call must contain an integer call_depth")

    if call_type == "op":
        call = _OpCall(_serialized_str(data, "op_name"), (), {}, call_depth)
        call.args_str = _serialized_str(data, "args_str")
        call.kwargs_str = _serialized_str(data, "kwargs_str")
    elif call_type == "redistribute":
        arg_str = _serialized_str(data, "arg_str")
        call = _RedistributeCall(
            arg_str,
            _serialized_str(data, "src_placement_str"),
            _serialized_str(data, "dst_placement_str"),
            _serialized_optional_str(data, "transform_info_str"),
            call_depth,
            is_explicit=bool(data.get("is_explicit")),
        )
        call.arg_str = arg_str
        # Serialized redistribute calls only keep rendered args, so restore the
        # outer-call classification instead of relying on __init__'s arg type.
        call.is_outer_call = bool(data.get("is_outer_call"))
    elif call_type == "output_placement":
        call = _OutputPlacementCall(_serialized_str(data, "placements_str"), call_depth)
    elif call_type == "triton_kernel":
        call = _TritonKernelCall(_serialized_str(data, "kernel_name"), {}, call_depth)
        call.kwargs_str = _serialized_str(data, "kwargs_str")
        call.pre_hashes = _deserialize_optional_debug_value(data.get("pre_hashes"))
        call.post_hashes = _deserialize_optional_debug_value(data.get("post_hashes"))
    elif call_type == "annotate":
        call = _AnnotateCall(
            _serialized_str(data, "tag"), _serialized_str(data, "header"), call_depth
        )
    else:
        raise ValueError(f"Unknown serialized DebugMode call type: {call_type!r}")

    _restore_common_call_fields(call, data)
    return call


def _get_call_name(call: _DebugCall) -> str:
    """String identifying _DebugCall (e.g. func, kernel, module name)"""
    if isinstance(call, _OpCall):
        return _get_op_name(call.op)
    elif isinstance(call, _TritonKernelCall):
        return call.kernel_name
    elif isinstance(call, _AnnotateCall):
        return f"[{call.header}] {call.tag}"
    elif isinstance(call, _RedistributeCall):
        return REDISTRIBUTE_FUNC
    else:
        return str(call)
