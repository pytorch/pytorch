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
import logging
from collections.abc import Callable
from typing import Any, TYPE_CHECKING

import torch
from torch._logging import warning_once
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

# Import _utils module for mutable globals
from torch.utils._debug_mode import _utils
from torch.utils._debug_mode._calls import (
    _AnnotateCall,
    _get_call_name,
    _OpCall,
    _OutputPlacementCall,
    _RedistributeCall,
    _TritonKernelCall,
)
from torch.utils._debug_mode._utils import (
    _compute_rel_diff,
    _get_user_stack_trace,
    _run_dispatch_hooks,
    _run_dispatch_pre_log_hooks,
    _stringify_dtensor_spec,
    hash_tensor_fn,
    norm_hash_fn,
    TensorIdTracker,
)
from torch.utils._python_dispatch import (
    _get_current_dispatch_mode,
    _get_current_dispatch_mode_stack,
    TorchDispatchMode,
)
from torch.utils._pytree import (
    keystr,
    tree_all,
    tree_map,
    tree_map_only,
    tree_map_with_path,
)


if TYPE_CHECKING:
    from torch.distributed._tools.mod_tracker import ModTracker


log = logging.getLogger(__name__)


# Counter for active DebugMode instances (fast path for get_active_debug_mode)
_ACTIVE_DEBUG_MODE_COUNT = 0

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
    # will force torch.compile to always use the "eager" backend
    # With this, DebugMode will not take effect on torch.compile
    @classmethod
    def ignore_compile_internals(cls) -> bool:
        return True

    def _record_call(self, call) -> None:
        if _utils._IN_INDUCTOR_BENCHMARK:
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

        from torch.distributed._functional_collectives import AsyncCollectiveTensor
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
        elif AsyncCollectiveTensor in types:
            # Record AsyncCollectiveTensor operations so debugging/tracing tools can see them
            if self.record_realtensor:
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
        global _ACTIVE_DEBUG_MODE_COUNT
        _ACTIVE_DEBUG_MODE_COUNT += 1
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
        global _ACTIVE_DEBUG_MODE_COUNT
        _ACTIVE_DEBUG_MODE_COUNT -= 1
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
        is_explicit: bool = False,
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
                    is_explicit=is_explicit,
                )
            )
            self.call_depth += 1
            yield
        finally:
            self.call_depth -= 1

    def record_output_placements(self, output_spec) -> None:
        """Record output placements for a DTensor op as a separate line."""
        if not self.record_output:
            return
        from torch.distributed.tensor._dtensor_spec import DTensorSpec

        placements_str = str(
            tree_map_only(DTensorSpec, _stringify_dtensor_spec, output_spec)
        )
        call = _OutputPlacementCall(placements_str, self.call_depth + 1)
        self._record_call(call)

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
        if record_hook:
            _utils._DISPATCH_RECORD_HOOKS.append(record_hook)
        if log_hook:
            _utils._DISPATCH_LOG_HOOKS.append(log_hook)
        if pre_log_hook:
            _utils._DISPATCH_PRE_LOG_HOOKS.append(pre_log_hook)
        try:
            yield
        finally:
            if record_hook:
                _utils._DISPATCH_RECORD_HOOKS.pop()
            if log_hook:
                _utils._DISPATCH_LOG_HOOKS.pop()
            if pre_log_hook:
                _utils._DISPATCH_PRE_LOG_HOOKS.pop()

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

        try:
            _old_record_triton = _utils._RECORD_TRITON_OUTPUTS
            _utils._RECORD_TRITON_OUTPUTS = True
            with DebugMode.dispatch_hooks(record_hook=dispatch_hook):
                yield
        finally:
            _utils._RECORD_TRITON_OUTPUTS = _old_record_triton

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

        try:
            if hash_inputs:
                _old_input_hfn = _utils._TRITON_INPUT_HASH_FN
                _utils._TRITON_INPUT_HASH_FN = fn
            _old_output_hfn = _utils._TRITON_OUTPUT_HASH_FN
            _utils._TRITON_OUTPUT_HASH_FN = fn
            with DebugMode.dispatch_hooks(
                log_hook=_dispatch_post_hook,
                pre_log_hook=_dispatch_pre_log_hook if hash_inputs else None,
            ):
                yield
        finally:
            if hash_inputs:
                _utils._TRITON_INPUT_HASH_FN = _old_input_hfn  # type: ignore[assignment]
            _utils._TRITON_OUTPUT_HASH_FN = _old_output_hfn

    @staticmethod
    @contextlib.contextmanager
    def _benchmarking_inductor():
        """
        Context manager for disabling logging during inductor benchmarking,
        so logs don't contain all kernels launched from autotuning.
        """
        try:
            _utils._IN_INDUCTOR_BENCHMARK = True
            yield
        finally:
            _utils._IN_INDUCTOR_BENCHMARK = False

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
    # Fast path: if no DebugMode is active, skip the stack walk
    if _ACTIVE_DEBUG_MODE_COUNT == 0:
        return None
    debug_mode = None
    for mode in _get_current_dispatch_mode_stack():
        if isinstance(mode, DebugMode):
            debug_mode = mode
            break
    return debug_mode
