# mypy: allow-untyped-defs
"""
Utility functions for DebugMode: tensor formatting, hashing, stack traces, and hook runners.
"""

import inspect
import os
import traceback
import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.graph import _parse_stack_trace
from torch.utils._dtype_abbrs import dtype_abbrs
from torch.utils._pytree import tree_map
from torch.utils._traceback import CapturedTraceback
from torch.utils.weak import WeakIdRef


if TYPE_CHECKING:
    from torch.utils._debug_mode._calls import _DebugCall


REDISTRIBUTE_FUNC = "redistribute_input"

# Tracks if we're in inductor benchmarking, and temporarily disables logging
# (for ignoring autotuning kernel launches which don't affect the user-facing result)
_IN_INDUCTOR_BENCHMARK: bool = False
# For record_outputs, log_tensor_hashes hooks for triton kernels.
# Stores kernel outputs in call.record["output"]
_RECORD_TRITON_OUTPUTS: bool = False
# Annotates kernel output hashes, and stores them in call.post_hashes
_TRITON_OUTPUT_HASH_FN: Callable | None = None
# Annotates kernel input hashes, and stores them in call.pre_hashes
_TRITON_INPUT_HASH_FN: Callable | None = None

# registered dispatch call hooks
_DISPATCH_RECORD_HOOKS: list[Callable] = []
_DISPATCH_LOG_HOOKS: list[Callable] = []
_DISPATCH_PRE_LOG_HOOKS: list[Callable] = []


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


def _stringify_dtensor_spec(spec) -> str:
    from torch.distributed.tensor._dtensor_spec import DTensorSpec

    return DTensorSpec.format_shard_order_str(spec.placements, spec.shard_order)


class TensorIdTracker:
    def __init__(self) -> None:
        self.tensor_memo: dict[WeakIdRef, int] = {}
        self.next_tensor_id = 0

    def _id(self, tensor) -> int:
        with torch._C._DisablePythonDispatcher():
            o = WeakIdRef(tensor)

            def del_memo() -> None:
                self.tensor_memo.pop(o, None)

            weakref.finalize(tensor, del_memo)
            if o not in self.tensor_memo:
                self.tensor_memo[o] = self.next_tensor_id
                self.next_tensor_id += 1
            return self.tensor_memo[o]


def _tensor_debug_string(tensor, attributes, tensor_memo=None) -> str:
    """Convert tensor to debug string representation."""

    if isinstance(tensor, torch.Tensor):
        tensor_debug_str = f"{dtype_abbrs[tensor.dtype]}{_stringify_shape(tensor.shape)}{_stringify_attributes(tensor, attributes)}"
        id_str = f"${tensor_memo._id(tensor)}" if tensor_memo is not None else ""
        if isinstance(tensor, torch.distributed.tensor.DTensor):
            # omitted device mesh
            return f"dt{id_str}: {tensor_debug_str}| {_stringify_dtensor_spec(tensor._spec)}"
        elif isinstance(tensor, FakeTensor):
            return f"ft{id_str}: {tensor_debug_str}"
        else:
            return f"t{id_str}: {tensor_debug_str}"
    else:
        raise RuntimeError(f"Unsupported tensor type: {type(tensor)}")


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


def norm_hash_fn(t: torch.Tensor, use_scalar: bool = False) -> torch.Tensor | float:
    """
    from Observer. Computes a hash for a tensor by converting it to float (if needed), making it contiguous,
    replacing NaN/inf values with fixed numbers, and then computing the L1 norm in float64 or complex128.
    This is used to generate a deterministic summary value for tensor comparison.
    """
    with torch._C._DisablePythonDispatcher():
        if not (t.is_floating_point() or t.is_complex()):
            t = t.float()
        t = t.contiguous()

        if t.is_complex():
            t_float = t.to(dtype=torch.complex128)
        else:
            t_float = t.to(dtype=torch.float64)

        out = t_float.norm(p=1)
        if use_scalar:
            return out.item()
        return out


def _compute_rel_diff(hash1, hash2):
    # Relative difference: |hash1 - hash2| / max(|hash1|, |hash2|, eps)
    numerator = abs(hash1 - hash2)
    denominator = max(abs(hash1), abs(hash2), 1e-10)
    return numerator / denominator


def hash_tensor_fn(t: torch.Tensor, use_scalar: bool = False) -> torch.Tensor | int:
    """
    wrapper over torch.hash_tensor
    """
    if isinstance(t, torch.distributed.tensor.DTensor):
        t = t.to_local()

    if t.is_floating_point():
        t_clean = t.to(dtype=torch.float64)
    elif t.is_complex():
        t_clean = t.to(dtype=torch.complex128).view(torch.float64)
    else:
        t_clean = t.to(dtype=torch.int64)

    if t.numel() > 0:
        out = torch.hash_tensor(t_clean)
    else:
        out = torch.zeros((), device=t_clean.device, dtype=torch.uint64)

    if use_scalar:
        return out.item()  # type: ignore[attribute]
    return out


def _get_stack_trace() -> str:
    from torch.fx.experimental.symbolic_shapes import uninteresting_files

    summary = CapturedTraceback.extract().summary()
    summary = summary[:-4]  # filter out DebugMode frames
    summary = [
        frame for frame in summary if frame.filename not in uninteresting_files()
    ]
    summary = traceback.StackSummary.from_list(summary)
    return "".join(summary.format())


def _get_user_stack_trace(stack_trace_str: str) -> str | None:
    # Extract user code stack trace, filtering out torch internals.
    torch_dir = os.path.dirname(inspect.getfile(torch))
    filter_fn = lambda file, name, code: not file.startswith(torch_dir + os.path.sep)  # noqa: E731
    trace = _parse_stack_trace(stack_trace_str, filter_fn=filter_fn)
    if trace:
        return f"File: {trace.file}:{trace.lineno} in {trace.name}, code: {trace.code}"
    return None


def _maybe_get_autograd_trace() -> str | None:
    if torch._C._current_autograd_node() is not None:
        tb = torch._C._current_autograd_node().metadata.get("traceback_")  # type: ignore[attr-defined]
        if tb:
            return "".join(tb)
    return None


def _get_op_name(op) -> str:
    if isinstance(op, torch._ops.OpOverload):
        op_name = op.__qualname__
    elif hasattr(op, "__module__") and hasattr(op, "__name__"):
        op_name = f"{op.__module__}.{op.__name__}"
    else:
        op_name = str(op)
    return op_name


def _run_hook(hook, *args):
    out = hook(*args)
    if out is not None and not isinstance(out, dict):
        raise AssertionError(f"hook must return None or dict, got {type(out).__name__}")
    return out


def _run_dispatch_pre_log_hooks(call: "_DebugCall", func, types, args, kwargs) -> None:
    if _DISPATCH_PRE_LOG_HOOKS:
        for hook in _DISPATCH_PRE_LOG_HOOKS:
            hook_out = _run_hook(hook, func, types, args, kwargs, call)
            if hook_out is not None:
                # Store pre-hook results in call.log
                if call.log is None:
                    call.log = {}
                call.log.update(hook_out)


def _run_dispatch_hooks(call: "_DebugCall", func, types, args, kwargs, result) -> None:
    if _DISPATCH_RECORD_HOOKS:
        record = {}
        for hook in _DISPATCH_RECORD_HOOKS:
            hook_out = _run_hook(hook, func, types, args, kwargs, result)
            if hook_out is not None:
                record.update(hook_out)
        if record:
            call.record = record

    if _DISPATCH_LOG_HOOKS:
        # Preserve existing log from pre-hooks (e.g., input_hash)
        if call.log is None:
            call.log = {}
        for hook in _DISPATCH_LOG_HOOKS:
            hook_out = _run_hook(hook, func, types, args, kwargs, result)
            if hook_out is not None:
                call.log.update(hook_out)
