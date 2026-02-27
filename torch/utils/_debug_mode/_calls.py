# mypy: allow-untyped-defs
"""
Call record classes for DebugMode: _DebugCall and its subclasses.
"""

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
