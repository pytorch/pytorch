# mypy: allow-untyped-defs
"""
DebugMode is a debugging TorchDispatchMode that intercepts and logs runtime calls
to a hierarchical string dump. It logs real tensor, DTensor, and optionally FakeTensor
operations, with some additional handling for DTensor internals.

An example dump from an eager mode DTensor matmul:

    torch.mm(dt$0: f32[8, 8]| S(0), dt$1: f32[8, 32]| S(0))  ->  dt$6: f32[8, 32]| S(0)
      aten::mm(dt$0: f32[8, 8]| S(0), dt$1: f32[8, 32]| S(0))
        redistribute_input(1, S(0) -> R)
          redistribute_input(t$2: f32[1, 32], trace: S(0)->R)
            _c10d_functional::all_gather_into_tensor(t$2: f32[1, 32], 8, 0)  ->  t$3: f32[8, 32]
            _c10d_functional::wait_tensor(t$3: f32[8, 32])  ->  t$3: f32[8, 32]
        aten::mm(t$4: f32[1, 8], t$3: f32[8, 32])  ->  t$5: f32[1, 32]

This mode runs "under" compile, which means it hides itself during compilation, and is re-enabled
at runtime, and DebugMode-related operations won't show up in the compiled region.
DebugMode also provides some visibility into non-torch-dispatch calls (e.g. DTensor redistribute calls,
inductor-generated triton kernels), but requires special handling for these, since dispatch modes
can't intercept them by default.

The mode also provides some extensions for custom debugging (e.g. adding custom dispatch call hooks
via dispatch_hooks), or numerics debugging (e.g. tensor hashing for bitwise equivalence/closeness,
via log_tensor_hashes). These decorators allow annotating string dumps with additional per-call information,
for any region of runtime code.

Usage::

    with DebugMode() as debug_mode:
        result = some_pytorch_operation(tensor_input)
    print(debug_mode.debug_string())
"""

import contextlib
import functools
import inspect
import logging
import os
import traceback
import weakref
from collections.abc import Callable
from typing import Any, TYPE_CHECKING

import torch
from torch._logging import warning_once
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx.graph import _parse_stack_trace
from torch.utils._dtype_abbrs import dtype_abbrs
from torch.utils._python_dispatch import (
    _get_current_dispatch_mode,
    _get_current_dispatch_mode_stack,
    TorchDispatchMode,
)
from torch.utils._pytree import keystr, tree_all, tree_map, tree_map_with_path
from torch.utils._traceback import CapturedTraceback
from torch.utils.weak import WeakIdRef


if TYPE_CHECKING:
    from torch._dynamo.device_interface import DeviceInterface
    from torch.distributed._tools.mod_tracker import ModTracker


log = logging.getLogger(__name__)

__all__ = ["DebugMode", "get_active_debug_mode"]


REDISTRIBUTE_FUNC = "redistribute_input"
# registered dispatch call hooks
_DISPATCH_RECORD_HOOKS: list[Callable] = []
_DISPATCH_LOG_HOOKS: list[Callable] = []
_DISPATCH_PRE_LOG_HOOKS: list[Callable] = []
# Tracks if we're in inductor benchmarking, and temporarily disables logging
# (for ignoring autotuning kernel launches which don't affect the user-facing result)
_IN_INDUCTOR_BENCHMARK = False
# For record_outputs, log_tensor_hashes hooks for triton kernels.
# Stores kernel outputs in call.record["output"]
_RECORD_TRITON_OUTPUTS = False
# Annotates kernel output hashes, and stores them in call.post_hashes
_TRITON_OUTPUT_HASH_FN = None
# Annotates kernel input hashes, and stores them in call.pre_hashes
_TRITON_INPUT_HASH_FN = None


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


_annotate_decorated = False


def _ensure_annotate_decorated():
    """
    Lazily apply dont_skip_tracing decorator to DebugMode._annotate, to avoid circular import/initialization issues.
    """
    global _annotate_decorated
    if not _annotate_decorated:
        DebugMode._annotate = torch._dynamo.dont_skip_tracing(DebugMode._annotate)  # type: ignore[has-type]

        # Mark annotate as side-effectful so aot_eager doesn't DCE it.
        from torch.fx.node import _side_effectful_functions

        _side_effectful_functions.add(torch.ops.debug_mode_ops.annotate.default)

        # Register no-op lowering for inductor backend
        from torch._inductor.lowering import register_lowering

        @register_lowering(torch.ops.debug_mode_ops.annotate)
        def _annotate_lowering(tag: str) -> None:
            warning_once(log, 'DebugMode._annotate() is a no-op for backend="inductor"')
            return None

        _annotate_decorated = True


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
    """Redistribute call from DTensor dispatch"""

    def __init__(
        self,
        arg,
        src_placement,
        dst_placement,
        transform_info_str,
        call_depth,
        stack=False,
    ) -> None:
        super().__init__(call_depth, stack=stack)
        self.arg = arg
        self.src_placement = src_placement
        self.dst_placement = dst_placement
        self.transform_info_str = transform_info_str

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

        base_str = f"{REDISTRIBUTE_FUNC}({arg_str}, {placement_str})"
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
        global _TRITON_INPUT_HASH_FN
        if hash_fn := _TRITON_INPUT_HASH_FN:
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
        global _RECORD_TRITON_OUTPUTS, _TRITON_OUTPUT_HASH_FN
        device_interface.synchronize(device_interface.current_device())
        if _RECORD_TRITON_OUTPUTS:
            self.record = {
                "output": {
                    k: v.clone() if isinstance(v, torch.Tensor) else v
                    for k, v in self.kwargs.items()
                }
            }
        if hash_fn := _TRITON_OUTPUT_HASH_FN:
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


def _run_hook(hook, *args):
    out = hook(*args)
    if out is not None and not isinstance(out, dict):
        raise AssertionError(f"hook must return None or dict, got {type(out).__name__}")
    return out


def _run_dispatch_pre_log_hooks(call: _DebugCall, func, types, args, kwargs) -> None:
    global _DISPATCH_PRE_LOG_HOOKS
    if _DISPATCH_PRE_LOG_HOOKS:
        for hook in _DISPATCH_PRE_LOG_HOOKS:
            hook_out = _run_hook(hook, func, types, args, kwargs, call)
            if hook_out is not None:
                # Store pre-hook results in call.log
                if call.log is None:
                    call.log = {}
                call.log.update(hook_out)


def _run_dispatch_hooks(call: _DebugCall, func, types, args, kwargs, result) -> None:
    global _DISPATCH_RECORD_HOOKS, _DISPATCH_LOG_HOOKS
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


@torch.library.custom_op("debug_mode_ops::annotate", mutates_args=())
def _annotate(tag: str) -> None:
    # This is special-cased in DebugMode.__torch_dispatch__
    return None


@_annotate.register_fake
def _annotate_fake(tag: str) -> None:
    return None


class DebugInterpreter(torch.fx.Interpreter):
    """
    Interpreter class for running aot_eager compiled regions when DebugMode is active,
    instead of using the compiled code. This gives us access to fx.Node metadata to decorate
    and contextualize DebugMode logs (e.g. nn_module_stack, stack_trace, compiled region boundaries).

    Note: this is currently only enabled with DebugMode(run_compile_with_interpreter=True).
    """

    def __init__(self, module, backend):
        super().__init__(module)
        self.mode = get_active_debug_mode()
        if self.mode is None:
            raise RuntimeError("No DebugMode is currently active")

        # for tracking initial nn_module_stack
        self.base_nn_module_stack = list(self.mode.current_nn_module_stack)

        # annotate start of region
        self.backend = backend
        self.mode.operators.append(
            _AnnotateCall(
                "enter", f"{self.backend} region (compile)", self.mode.call_depth
            )
        )

    def run_node(self, n: torch.fx.Node) -> Any:
        if self.mode is None:
            raise RuntimeError("No DebugMode is currently active")

        # handling of nn.Module stack
        if self.mode.record_nn_module and n.op not in ["placeholder", "output"]:
            self.mode._handle_fx_nn_module_stack(
                self.base_nn_module_stack,
                n.meta.get("nn_module_stack", {}),
                n.meta.get("fwd_nn_module_stack", {}),
            )

        # override stack trace with n.meta
        if (
            self.mode.record_stack_trace
            and n.op not in ["placeholder", "output"]
            and (stack_trace := n.meta.get("stack_trace", None)) is not None
        ):
            with self.mode.set_fx_stack_trace(stack_trace):
                return super().run_node(n)
        else:
            return super().run_node(n)

    def run(self, *args, **kwargs):
        if self.mode is None:
            raise RuntimeError("No DebugMode is currently active")
        result = super().run(*args)

        # reset nn.Module stack to pre-compiled region value
        if len(self.mode.current_nn_module_stack) < len(self.base_nn_module_stack):
            warning_once(
                log, "unexpected handling of nn_module_stack in DebugInterpreter"
            )
        while len(self.mode.current_nn_module_stack) > len(self.base_nn_module_stack):
            self.mode._exit_nn_module_call()

        # annotate end of region
        self.mode.operators.append(
            _AnnotateCall(
                "exit", f"{self.backend} region (compile)", self.mode.call_depth
            )
        )

        return result


class DebugMode(TorchDispatchMode):
    def __init__(
        self,
        *,
        record_torchfunction=False,
        record_faketensor=False,
        record_realtensor=True,
        record_tensor_attributes=None,
        record_nn_module=False,
        store_original_args=False,
        record_stack_trace=False,
        record_output=True,
        record_ids=False,
        record_profiler_context=True,
        record_localtensor=True,
        run_compile_with_interpreter=False,
    ) -> None:
        super().__init__()
        import torch.distributed.tensor  # noqa: F401

        _ensure_annotate_decorated()
        self.supports_higher_order_operators = True

        # Pushes DebugMode onto the torchfunction stack, and records __torch_function__ calls as well.
        # WARNING: currently incompatible with torch.compile due to dynamo guard failures.
        self.record_torchfunction = record_torchfunction

        # Records __torch_dispatch__ calls on FakeTensors.
        self.record_faketensor = record_faketensor

        # Records __torch_dispatch__ calls on real tensors.
        self.record_realtensor = record_realtensor

        # Records __torch_dispatch__ calls on LocalTensor.
        self.record_localtensor = record_localtensor

        # Optional list[str] of tensor attributes, to be annotated in the string dump.
        self.record_tensor_attributes = record_tensor_attributes or []

        # Uses ModTracker to record nn.Module entrances.
        # This flag currently has no effect on torch.compiled-regions.
        self.record_nn_module = record_nn_module

        self.module_tracker: ModTracker | None = None
        if self.record_nn_module:
            self.module_tracker_setup()

        # If True, stores call args/kwargs in logs, without immediately stringifying.
        # Defaults to False for memory concerns.
        self.store_original_args = store_original_args

        # For stack trace recording, stores log call stack traces in .stack_trace.
        # For backward graph nodes, will also store the corresponding forward stack traces in .fwd_stack_trace.
        # NOTE: this is only available if autograd tracebacks are being set during the forward pass,
        # e.g. via DebugMode(record_stack_trace=True), or torch.autograd.set_detect_anomaly().
        self.record_stack_trace = record_stack_trace

        # Records call outputs in logs (e.g. for __torch_dispatch__, __torch_function__, redistribute_input)
        self.record_output: bool = record_output

        # Annotates string dumps with graph-style tensor ids, e.g. op($1, $2) -> $3.
        self.record_ids: bool = record_ids

        # Annotates string dumps with profiler.record_function contexts from runtime code.
        # Currently does not preserve contexts inside torch.compile-d regions.
        self.record_profiler_context: bool = record_profiler_context

        # For aot_eager compiled regions, wraps the compiled fx.GraphModule with a DebugInterpreter,
        # and uses it at runtime for node metadata visibility.
        self.run_compile_with_interpreter: bool = run_compile_with_interpreter

        self.reset()

    def reset(self) -> None:
        self.operators = []
        self.call_depth = 0
        self._tensor_memo = TensorIdTracker()
        self._output_info: dict[int, object] = {}
        self.ignored_record_functions = 0
        self.current_nn_module_stack = []
        self.fx_stack_trace = None

    def _track_op_output(self, op_index, result) -> None:
        """Assign IDs to output tensors and store in output_info"""
        self._output_info[op_index] = result

    # Without this override, running torch.compile under DebugMode
    # will force torch.compile to always use the “eager” backend
    # With this, DebugMode will not take effect on torch.compile
    @classmethod
    def ignore_compile_internals(cls) -> bool:
        return True

    def _record_call(self, call) -> None:
        global _IN_INDUCTOR_BENCHMARK
        if _IN_INDUCTOR_BENCHMARK:
            return

        if str(call).startswith("profiler::_record_function"):
            return

        if not self.store_original_args:
            call.stringify_args(
                self.record_tensor_attributes,
                self._tensor_memo if self.record_ids else None,
            )
        if self.fx_stack_trace:
            call.stack_trace = call.fwd_stack_trace = self.fx_stack_trace
        self.operators.append(call)

    def _record_call_output(self, call, output) -> None:
        if not self.record_output:
            return
        call.stringify_output(
            output,
            self.record_tensor_attributes,
            self._tensor_memo if self.record_ids else None,
        )

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        call = _OpCall(
            func, args, kwargs, self.call_depth, stack=self.record_stack_trace
        )
        self._record_call(call)

        try:
            self.call_depth += 1
            result = func(*args, **kwargs)
            self._record_call_output(call, result)
            return result
        finally:
            self.call_depth -= 1

    def _maybe_record_function(self, tag):
        # filter out tags that appear noisy, or aren't runtime-related
        if any(
            tag.startswith(prefix)
            for prefix in [
                # assuming these are from benchmarking, not the actual runtime call
                "CachingAutotuner.",
                "InductorBenchmarker.",
                # inductor compilation
                "compile_fx.<locals>.",
            ]
        ):
            self.ignored_record_functions += 1
            return

        call = _AnnotateCall(
            tag, "record function", self.call_depth, stack=self.record_stack_trace
        )
        self.operators.append(call)
        self.call_depth += 1

    def _maybe_exit_record_function(self):
        if self.ignored_record_functions < 0:
            raise AssertionError(
                f"ignored_record_functions is negative: {self.ignored_record_functions}"
            )
        if self.ignored_record_functions > 0:
            self.ignored_record_functions -= 1
        else:
            self.call_depth -= 1

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        # Handle record_function entries
        if self.record_profiler_context:
            if func == torch.ops.profiler._record_function_enter_new.default:
                if len(args) != 1:
                    raise AssertionError(f"expected 1 arg, got {len(args)}")
                self._maybe_record_function(args[0])
            elif func == torch.ops.profiler._record_function_exit._RecordFunction:
                self._maybe_exit_record_function()

        # Handle DebugMode._annotate()
        if func is torch.ops.debug_mode_ops.annotate.default:
            if len(args) != 1:
                raise AssertionError(f"expected 1 arg, got {len(args)}")
            self._handle_annotate(args[0])
            return

        from torch.distributed._local_tensor import LocalTensor

        # Record the operation with its call depth
        call = None
        if torch.distributed.tensor.DTensor in types:
            call = _OpCall(
                func, args, kwargs, self.call_depth, stack=self.record_stack_trace
            )
            self._record_call(call)
            return NotImplemented
        elif FakeTensor in types or isinstance(
            _get_current_dispatch_mode(), FakeTensorMode
        ):
            if self.record_faketensor:
                if func != torch.ops.prim.device.default:
                    call = _OpCall(
                        func,
                        args,
                        kwargs,
                        self.call_depth + 1,
                        stack=self.record_stack_trace,
                    )
                    self._record_call(call)
        # TODO: check the context manager
        elif LocalTensor in types:
            if self.record_localtensor:
                call = _OpCall(
                    func,
                    args,
                    kwargs,
                    self.call_depth + 1,
                    stack=self.record_stack_trace,
                )
                self._record_call(call)
        elif len(types) == 0:
            if self.record_realtensor:
                call = _OpCall(
                    func,
                    args,
                    kwargs,
                    self.call_depth + 1,
                    stack=self.record_stack_trace,
                )
                self._record_call(call)

        # Run pre-hooks before executing the operation to hash inputs
        # We have to run becore the func() call in case there's any
        # in-place mutation
        if call:
            _run_dispatch_pre_log_hooks(call, func, types, args, kwargs)

        result = func(*args, **kwargs)
        if call:
            self._record_call_output(call, result)
            _run_dispatch_hooks(call, func, types, args, kwargs, result)

        return result

    def __enter__(self):
        self.reset()

        if self.record_torchfunction:
            torch._C._push_on_torch_function_stack(self)

        super().__enter__()
        if self.record_nn_module:
            self.module_tracker.__enter__()  # type: ignore[attribute, union-attr]

        if self.record_stack_trace:
            self.anomaly_for_traces = torch.autograd.set_detect_anomaly(
                True, check_nan=False
            )
            self.anomaly_for_traces.__enter__()
        return self

    # pyrefly: ignore [bad-override]
    def __exit__(self, *args):
        super().__exit__(*args)
        if self.record_nn_module:
            self.module_tracker.__exit__()  # type: ignore[attribute, union-attr]
        if self.record_torchfunction:
            torch._C._pop_torch_function_stack()
        if self.record_stack_trace:
            self.anomaly_for_traces.__exit__(*args)

    @contextlib.contextmanager
    def set_fx_stack_trace(self, stack_trace):
        self.fx_stack_trace = stack_trace
        try:
            yield
        finally:
            self.fx_stack_trace = None

    def _enter_nn_module_call(self, fqn, header):
        call = _AnnotateCall(
            fqn, header, self.call_depth + 1, stack=self.record_stack_trace
        )
        self.operators.append(call)
        self.current_nn_module_stack.append(fqn)
        self.call_depth += 1

    def _exit_nn_module_call(self):
        self.call_depth -= 1
        self.current_nn_module_stack.pop()

    def module_tracker_setup(self) -> None:
        from torch.distributed._tools.mod_tracker import ModTracker

        self.module_tracker = ModTracker()

        # module pre-fw hook: record module call
        def pre_fw_hook(module, input) -> None:
            fqn = self.module_tracker._get_mod_name(module)  # type: ignore[attribute, union-attr]
            self._enter_nn_module_call(fqn, "nn.Mod")

        # module post-fw hook: decrement call depth
        def post_fw_hook(module, input, output) -> None:
            self._exit_nn_module_call()

        self.module_tracker.register_user_hooks(pre_fw_hook, post_fw_hook)

    def _handle_fx_nn_module_stack(
        self,
        base_stack: list[str],
        nn_module_stack: dict[str, tuple[str, Any]] | None,
        fwd_nn_module_stack: dict[str, tuple[str, Any]] | None,
    ) -> None:
        """
        Called when DebugInterpreter observes nn_module_stack or fwd_nn_module_stack metadata
        from executing the compiled GraphModule.

        If the current module stack is mismatched with what's currently tracked in DebugMode
        (current_nn_module_stack), we adjust call depth and add new [nn.Module] log entries accordingly.
        """

        nn_module_stack = nn_module_stack or {}
        fwd_nn_module_stack = fwd_nn_module_stack or {}
        if nn_module_stack and fwd_nn_module_stack:
            raise AssertionError(
                "Expecting at most one of nn_module_stack and fwd_nn_module_stack."
            )

        is_fwd = nn_module_stack
        stack = nn_module_stack if is_fwd else fwd_nn_module_stack

        # forward stack
        current_stack = self.current_nn_module_stack
        new_stack = base_stack + [v[0] for v in stack.values()]

        entered = set(new_stack) - set(current_stack)
        exited = set(current_stack) - set(new_stack)

        # Decrement depth for exited modules
        for _ in exited:
            self._exit_nn_module_call()
        if self.call_depth < 0:
            raise AssertionError("Unexpectedly, DebugMode call_depth is negative")

        # Add [nn.Module] entries for newly entered modules
        for fqn in sorted(entered):
            self._enter_nn_module_call(
                fqn, "nn.Mod (compile)" if is_fwd else "nn.Mod (compile bwd)"
            )

        self.current_nn_module_stack = new_stack

    @contextlib.contextmanager
    def record_redistribute_calls(
        self,
        arg,
        src_placement,
        dst_placement,
        transform_info_str: str | None = None,
    ):
        try:
            self._record_call(
                _RedistributeCall(
                    arg,
                    src_placement=src_placement,
                    dst_placement=dst_placement,
                    transform_info_str=transform_info_str,
                    call_depth=self.call_depth + 1,
                    stack=self.record_stack_trace,
                )
            )
            self.call_depth += 1
            yield
        finally:
            self.call_depth -= 1

    def record_triton_kernel(
        self, kernel_name: str, kwargs: dict[str, Any]
    ) -> _TritonKernelCall:
        call = _TritonKernelCall(kernel_name, kwargs, self.call_depth + 1)
        call.stringify_args(self.record_tensor_attributes)
        self.operators.append(call)
        return call

    def debug_string(self, show_stack_trace: bool | None = None) -> str:
        """
        show_stack_trace: option to display one-line stack trace summaries above groups
                        of operations (similar to gm.print_readable() style).
                        Requires record_stack_trace=True.
                        if None, uses self.record_stack_trace, otherwise overrides it.
        """
        show_stack_trace = (
            self.record_stack_trace if show_stack_trace is None else show_stack_trace
        )

        with torch._C.DisableTorchFunction():
            if not show_stack_trace:
                result = "\n".join(
                    "  "
                    + "  " * op.call_depth
                    + op.render(self.record_tensor_attributes)
                    for op in self.operators
                )
                return result

            # Group operations by stack trace
            lines = []
            prev_stack_summary = None

            for op in self.operators:
                # Get the stack trace: prefer fwd_stack_trace, fallback to stack_trace
                stack_trace = None
                if hasattr(op, "fwd_stack_trace") and op.fwd_stack_trace:
                    stack_trace = op.fwd_stack_trace
                elif hasattr(op, "stack_trace") and op.stack_trace:
                    stack_trace = op.stack_trace

                stack_summary = None
                if stack_trace:
                    stack_summary = _get_user_stack_trace(stack_trace)

                if stack_summary and stack_summary != prev_stack_summary:
                    # add blank line before stack trace comment for readability
                    if lines:  # don't add blank line at the very start
                        lines.append("")
                    indent = "  " * (op.call_depth + 1)
                    lines.append(indent + "# " + stack_summary)
                    prev_stack_summary = stack_summary

                # Add the operation line
                line = (
                    "  "
                    + "  " * op.call_depth
                    + op.render(self.record_tensor_attributes)
                )
                lines.append(line)

            return "\n".join(lines)

    @staticmethod
    @contextlib.contextmanager
    def dispatch_hooks(
        record_hook: Callable | None = None,
        log_hook: Callable | None = None,
        pre_log_hook: Callable | None = None,
    ):
        """
        Allows installing post-hooks on arguments to intercepted __torch_dispatch__ calls;
        hook signatures are expected as (func, types, args, kwargs, result),
        i.e. __torch_dispatch__ args + return value.

        Logging hook outputs are stored in call.log and annotate calls in debug_string(),
        while recording hook outputs are just stored in call.record.
        For now hooks are expected to return dictionaries.

        pre_log_hook signature is (func, types, args, kwargs, call) and is executed before
        the operation. It allows capturing state before in-place mutations.
        """
        global _DISPATCH_RECORD_HOOKS, _DISPATCH_LOG_HOOKS, _DISPATCH_PRE_LOG_HOOKS

        if record_hook:
            _DISPATCH_RECORD_HOOKS.append(record_hook)
        if log_hook:
            _DISPATCH_LOG_HOOKS.append(log_hook)
        if pre_log_hook:
            _DISPATCH_PRE_LOG_HOOKS.append(pre_log_hook)
        try:
            yield
        finally:
            if record_hook:
                _DISPATCH_RECORD_HOOKS.pop()
            if log_hook:
                _DISPATCH_LOG_HOOKS.pop()
            if pre_log_hook:
                _DISPATCH_PRE_LOG_HOOKS.pop()

    @staticmethod
    @contextlib.contextmanager
    def record_outputs():
        """
        Hook for storing cloned output tensors in .record["output"].
        """

        def dispatch_hook(func, types, args, kwargs, result):
            out = tree_map(
                lambda x: x.clone() if isinstance(x, torch.Tensor) else x, result
            )
            return {"output": out}

        global _RECORD_TRITON_OUTPUTS
        try:
            _old_record_triton = _RECORD_TRITON_OUTPUTS
            _RECORD_TRITON_OUTPUTS = True
            with DebugMode.dispatch_hooks(record_hook=dispatch_hook):
                yield
        finally:
            _RECORD_TRITON_OUTPUTS = _old_record_triton

    @staticmethod
    @contextlib.contextmanager
    def log_tensor_hashes(
        hash_fn: Callable | str | list[str] = "norm", hash_inputs: bool = False
    ):
        """
        Installs hook for tensor hash logging.

        hash_fn: One of:
            - Custom-defined hash function
            - String: one of ("norm", "hash_tensor")
                - "norm": uses norm_hash_fn; basically tensor's L1 norm
                - "hash_tensor": uses torch.hash_tensor (XOR sum reduction)
            - List of strings: returns tuple of hashes from above options
        hash_inputs: if True, also hashes tensors in (args, kwargs), storing them in "input_hash".
        Input hashes are captured before the operation executes, so they reflect the state before
        any in-place mutations.
        """

        def hash_fn_option(hash_type):
            if not isinstance(hash_type, str) or hash_type not in [
                "norm",
                "hash_tensor",
            ]:
                raise AssertionError(
                    f"hash_type must be 'norm' or 'hash_tensor', got {hash_type!r}"
                )
            return functools.partial(
                norm_hash_fn if hash_type == "norm" else hash_tensor_fn, use_scalar=True
            )

        if callable(hash_fn):
            fn = hash_fn
        elif isinstance(hash_fn, str):
            fn = hash_fn_option(hash_fn)
        elif isinstance(hash_fn, list):
            fns = [hash_fn_option(fn) for fn in hash_fn]
            fn = lambda x: tuple(fn(x) for fn in fns)  # noqa: E731
        else:
            raise NotImplementedError(
                f"log_tensor_hashes() expected hash_fn to be callable, str, or list[str], but found {type(hash_fn)}"
            )

        def _tree_hash(obj):
            return tree_map(
                lambda x: fn(x) if isinstance(x, torch.Tensor) else None, obj
            )

        def _dispatch_pre_log_hook(func, types, args, kwargs, call):
            """Pre-hook to capture input hashes before operation executes"""
            if "empty" in str(func) or "profiler" in str(func):
                return None

            if hash_inputs:
                # Capture input hashes before the operation
                input_hash = _tree_hash((args, kwargs))
                if not tree_all(lambda x: x is None, input_hash):
                    return {"input_hash": input_hash}
            return None

        def _dispatch_post_hook(func, types, args, kwargs, result):
            """Post-hook to capture output hashes after operation executes"""
            if "empty" in str(func) or "profiler" in str(func):
                return None

            out = {}
            out["hash"] = _tree_hash(result)

            if tree_all(lambda x: x is None, out.values()):
                return None
            return out

        global _TRITON_INPUT_HASH_FN, _TRITON_OUTPUT_HASH_FN
        try:
            if hash_inputs:
                _old_input_hfn = _TRITON_INPUT_HASH_FN
                _TRITON_INPUT_HASH_FN = fn
            _old_output_hfn = _TRITON_OUTPUT_HASH_FN
            _TRITON_OUTPUT_HASH_FN = fn
            with DebugMode.dispatch_hooks(
                log_hook=_dispatch_post_hook,
                pre_log_hook=_dispatch_pre_log_hook if hash_inputs else None,
            ):
                yield
        finally:
            if hash_inputs:
                _TRITON_INPUT_HASH_FN = _old_input_hfn  # type: ignore[assignment]
            _TRITON_OUTPUT_HASH_FN = _old_output_hfn

    @staticmethod
    @contextlib.contextmanager
    def _benchmarking_inductor():
        """
        Context manager for disabling logging during inductor benchmarking,
        so logs don't contain all kernels launched from autotuning.
        """
        global _IN_INDUCTOR_BENCHMARK
        try:
            _IN_INDUCTOR_BENCHMARK = True
            yield
        finally:
            _IN_INDUCTOR_BENCHMARK = False

    @property
    def logs(self):
        return list(self.operators)

    def _handle_annotate(self, tag):
        """Handles DebugMode._annotate()"""
        call = _AnnotateCall(tag, "annotate", self.call_depth, self.record_stack_trace)
        self.operators.append(call)

    @staticmethod
    def _annotate(tag: Any) -> None:
        """
        If an active DebugMode exists, adds an "[annotate] <tag>" entry to the logs. Useful for contextualizing logs.
        Implemented with a custom op.
        """
        torch.ops.debug_mode_ops.annotate(tag)

    @staticmethod
    def check_hash_mismatches(
        logs1: list, logs2: list, compare_inputs: bool = False
    ) -> list[dict]:
        """
        Compares tensor hashes between two DebugMode runs, for checking run-to-run numerical divergence.

        This first validates the two log sequences have identical structure (same operations, input shapes/dtypes, etc.),
        then compares tensor hash values, and returns a list of call outputs where mismatches were found.
        Expects input logs to have been run with log_tensor_hashes, and looks for hashes in .log["hash"] & .log["input_hash"]
        (or .post_hashes & .pre_hashes for triton kernels).

        note: skips checking log pairs where hashes aren't present, but will raise if present in one & not the other.

        Args:
            logs1: logs from the first DebugMode run (from debug_mode.logs)
            logs2: logs from the second DebugMode run
            compare_inputs: If True, also compare input tensor hashes (default: only output checking)

        Returns:
            List of dictionaries describing hash mismatches. Each dict contains:
                - call_type: "torch op" or "triton kernel"
                - call: Operator/kernel name
                - arg_name: For triton kernels, the argument name; None for torch ops
                - pytree_path: For torch ops, the pytree path to the differing tensor; None for kernels
                - hash1: Hash value from the first run
                - hash2: Hash value from the second run
                - rel_diff: Relative difference between hash values
                - is_input_hash: True if this is an input hash, False for output hash

        Raises:
            ValueError: If logs have different lengths, call types, operator names, or call depths

        Usage::

            # Run model first time
            with DebugMode() as debug_mode, DebugMode.log_tensor_hashes():
                model(x)
                logs1 = debug_mode.logs

            # Run again, in exactly the same way
            with DebugMode() as debug_mode, DebugMode.log_tensor_hashes():
                model(x)
                logs2 = debug_mode.logs

            mismatches = DebugMode.check_hash_mismatches(logs1, logs2)
            for m in mismatches:
                print(f"{m['call']}: hash diff {m['rel_diff']:.2e}")
        """
        if len(logs1) != len(logs2):
            raise ValueError(f"Log lengths don't match: {len(logs1)} vs {len(logs2)}")

        difference_info = []
        for i, (log1, log2) in enumerate(zip(logs1, logs2)):
            # check call type
            call1_type = type(log1).__name__
            call2_type = type(log2).__name__
            if call1_type != call2_type:
                raise ValueError(
                    f"Call types don't match at index {i}: {call1_type} vs {call2_type}"
                )
            call_type = call1_type

            # check call name
            op1_name, op2_name = _get_call_name(log1), _get_call_name(log2)
            if op1_name != op2_name:
                raise ValueError(
                    f"Operators don't match at index {i}: {call_type}[{op1_name}] vs {call_type}[{op2_name}]"
                )
            op_name = op1_name

            # check call depth
            if log1.call_depth != log2.call_depth:
                raise ValueError(
                    f"Call depths for {call_type}[{op_name}] don't match at index {i}: {log1.call_depth} vs {log2.call_depth}"
                )

            # Redistribute: call args should be the same
            if isinstance(log1, _RedistributeCall):
                if tuple(log1) != tuple(log2):
                    raise ValueError(
                        f"Redistribute calls don't match at index {i}: {log1} vs {log2}"
                    )

            # Triton kernel: same arg names, arg types
            elif isinstance(log1, _TritonKernelCall):
                if log1.kwargs_str != log2.kwargs_str:
                    raise ValueError(
                        f"Triton kernel call args don't match for {log1.kernel_name} at index {i}:"
                        f"\n\nlog1: {log1.kwargs_str}\n\nlog2: {log2.kwargs_str}"
                    )

                def compare_triton_hashes(hashes1, hashes2, is_input):
                    if set(hashes1.keys()) != set(hashes2.keys()):  # type: ignore[union-attr]
                        raise AssertionError(
                            f"hash key mismatch: {set(hashes1.keys())} vs {set(hashes2.keys())}"
                        )
                    for key in hashes1:
                        if hashes1[key] != hashes2[key]:
                            difference_info.append(
                                {
                                    "call_type": "triton kernel",
                                    "call": op_name,
                                    "arg_name": key,
                                    "pytree_path": None,
                                    "hash1": hashes1[key],
                                    "hash2": hashes2[key],
                                    "rel_diff": _compute_rel_diff(
                                        hashes1[key], hashes2[key]
                                    ),
                                    "is_input_hash": is_input,
                                }
                            )

                # check output hashes
                has_post_1, has_post_2 = (
                    log1.post_hashes is not None,
                    log2.post_hashes is not None,
                )
                if has_post_1 != has_post_2:
                    raise ValueError(
                        f"Triton kernel post-hash presence inconsistent for {log1.kernel_name} "
                        f"at index {i}: log1 has post_hashes={has_post_1}, log2 has post_hashes={has_post_2}"
                    )

                if has_post_1:
                    compare_triton_hashes(
                        log1.post_hashes, log2.post_hashes, is_input=False
                    )

                # maybe check input hashes
                if compare_inputs:
                    has_pre_1, has_pre_2 = (
                        log1.pre_hashes is not None,
                        log2.pre_hashes is not None,
                    )
                    if has_pre_1 != has_pre_2:
                        raise ValueError(
                            f"Triton kernel pre-hash presence inconsistent for {log1.kernel_name} "
                            f"at index {i}: log1 has pre_hashes={has_pre_1}, log2 has pre_hashes={has_pre_2}"
                        )

                    if has_pre_1:
                        compare_triton_hashes(
                            log1.pre_hashes, log2.pre_hashes, is_input=True
                        )

            # regular log calls
            elif isinstance(log1, _OpCall):

                def compare_op_hashes(hashes1, hashes2, is_input):
                    def _helper(keypath, hash1, hash2):
                        if hash1 != hash2:
                            difference_info.append(
                                {
                                    "call_type": "torch op",
                                    "call": op_name,
                                    "arg_name": None,
                                    "pytree_path": keystr(keypath),
                                    "hash1": hash1,
                                    "hash2": hash2,
                                    "rel_diff": _compute_rel_diff(hash1, hash2),
                                    "is_input_hash": is_input,
                                }
                            )

                    tree_map_with_path(_helper, hashes1, hashes2)

                # check output hashes
                has_hash1 = log1.log is not None and "hash" in log1.log
                has_hash2 = log2.log is not None and "hash" in log2.log
                if has_hash1 != has_hash2:
                    raise ValueError(
                        f"Output hash presence inconsistent for triton kernel {call_type}[{op_name}] "
                        f"at index {i}: log1 has hash={has_hash1}, log2 has hash={has_hash2}"
                    )

                if has_hash1:
                    compare_op_hashes(
                        log1.log["hash"],  # type: ignore[union-attr]
                        log2.log["hash"],
                        is_input=False,
                    )

                # maybe check input hashes
                if compare_inputs:
                    has_hash1 = log1.log is not None and "input_hash" in log1.log
                    has_hash2 = log2.log is not None and "input_hash" in log2.log
                    if has_hash1 != has_hash2:
                        raise ValueError(
                            f"Input hash presence inconsistent for triton kernel {call_type}[{op_name}] "
                            f"at index {i}: log1 has input_hash={has_hash1}, log2 has input_hash={has_hash2}"
                        )

                    if has_hash1:
                        compare_op_hashes(
                            log1.log["input_hash"],  # type: ignore[union-attr]
                            log2.log["input_hash"],
                            is_input=True,
                        )

        return difference_info


def get_active_debug_mode() -> DebugMode | None:
    debug_mode = None
    for mode in _get_current_dispatch_mode_stack():
        if isinstance(mode, DebugMode):
            debug_mode = mode
            break
    return debug_mode
