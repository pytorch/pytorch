# mypy: allow-untyped-defs
import contextlib
import traceback
from typing import Any, Callable, Optional, TYPE_CHECKING, Union

import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.utils._dtype_abbrs import dtype_abbrs
from torch.utils._python_dispatch import (
    _get_current_dispatch_mode,
    _get_current_dispatch_mode_stack,
    TorchDispatchMode,
)
from torch.utils._pytree import tree_map
from torch.utils._traceback import CapturedTraceback


if TYPE_CHECKING:
    from torch.distributed._tools.mod_tracker import ModTracker


__all__ = ["DebugMode", "get_active_debug_mode"]


REDISTRIBUTE_FUNC = "redistribute_input"
RECORD_STACK_TRACE = False
CPP_STACK_TRACE = False


_dispatch_record_hooks: list[Callable] = []
_dispatch_log_hooks: list[Callable] = []
_node_record_hooks: list[Callable] = []
_node_log_hooks: list[Callable] = []


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


def _tensor_debug_string(tensor, attributes) -> str:
    """Convert tensor to debug string representation."""

    if isinstance(tensor, torch.Tensor):
        tensor_debug_str = f"{dtype_abbrs[tensor.dtype]}{_stringify_shape(tensor.shape)}{_stringify_attributes(tensor, attributes)}"

        if isinstance(tensor, torch.distributed.tensor.DTensor):
            # omitted device mesh
            return f"dt: {tensor_debug_str}| {_stringify_dtensor_spec(tensor._spec)}"
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
            return _stringify_dtensor_spec(x)
        return x

    arg = tree_map(to_str, arg)
    return str(arg)


def _get_stack_trace(cpp=False) -> str:
    from torch.fx.experimental.symbolic_shapes import uninteresting_files

    summary = CapturedTraceback.extract(cpp=cpp).summary()
    summary = [
        frame for frame in summary if frame.filename not in uninteresting_files()
    ]
    summary = traceback.StackSummary.from_list(summary)
    return ",".join(summary.format())


def _op_to_str(op) -> str:
    if isinstance(op, torch._ops.OpOverload):
        return op.__qualname__
    elif hasattr(op, "__module__") and hasattr(op, "__name__"):
        return f"{op.__module__}.{op.__name__}"
    else:
        return str(op)


class _DebugCall:
    """Base class for tracking operator calls in DebugMode"""

    def __init__(
        self,
        call_depth: int,
        record: Optional[dict[str, Any]] = None,
        log: Optional[dict[str, Any]] = None,
    ):
        global RECORD_STACK_TRACE, CPP_STACK_TRACE

        self.call_depth: int = call_depth
        self.stack_trace: Optional[str] = None
        if RECORD_STACK_TRACE:
            self.stack_trace = _get_stack_trace(cpp=CPP_STACK_TRACE)

        # results from custom hooks
        self.record = record
        self.log = log

    def render(self, attributes: list[str]) -> str:
        raise NotImplementedError("Subclasses must implement string render()")


class _OpCall(_DebugCall):
    """Normal operator call"""

    def __init__(
        self,
        op,
        args: tuple,
        kwargs: dict,
        call_depth: int,
        record: Optional[dict[str, Any]] = None,
        log: Optional[dict[str, Any]] = None,
    ):
        super().__init__(call_depth, record, log)
        self.op = op
        self.args = args
        self.kwargs = kwargs

    def render(self, attributes: list[str]) -> str:
        args_str = ", ".join(_arg_to_str(arg, attributes) for arg in self.args)

        if self.kwargs:
            kwargs_str = ", " + ", ".join(
                f"{k}={_arg_to_str(v, attributes)}" for k, v in self.kwargs.items()
            )
        else:
            kwargs_str = ""

        log_str = f"  # {self.log}" if self.log else ""

        return f"{_op_to_str(self.op)}({args_str}{kwargs_str}){log_str}"

    def __iter__(self):
        # for BC; tuple(self) returns (op, args, kwargs, call_depth)
        yield from [self.op, self.args, self.kwargs, self.call_depth]

    def __repr__(self) -> str:
        return self.render([])


class _RedistributeCall(_DebugCall):
    """Redistribute call from DTensor dispatch"""

    def __init__(
        self,
        arg,
        src_placement,
        dst_placement,
        transform_info_str,
        call_depth,
        record: Optional[dict[str, Any]] = None,
        log: Optional[dict[str, Any]] = None,
    ):
        super().__init__(call_depth, record, log)
        self.arg = arg
        self.src_placement = src_placement
        self.dst_placement = dst_placement
        self.transform_info_str = transform_info_str

    def render(self, attributes: list[str]) -> str:
        arg_str = f"{_arg_to_str(self.arg, attributes)}"
        if self.transform_info_str is not None:  # prioritize over src/dst placements
            placement_str = f"trace: {self.transform_info_str}"
        else:
            src_placement_str = _arg_to_str(self.src_placement, attributes)
            dst_placement_str = _arg_to_str(self.dst_placement, attributes)
            placement_str = f"{src_placement_str} -> {dst_placement_str}"
        return f"{REDISTRIBUTE_FUNC}({arg_str}, {placement_str})"

    def __iter__(self):
        # for BC; tuple(self) returns (op, placement info, kwargs, call_depth)
        yield REDISTRIBUTE_FUNC
        if self.transform_info_str:
            yield [self.arg, self.transform_info_str]
        else:
            yield [self.arg, self.src_placement, self.dst_placement]
        yield {}
        yield self.call_depth

    def __repr__(self) -> str:
        return self.render([])


class _NNModuleCall(_DebugCall):
    """Designates entering an nn.Module's forward method"""

    def __init__(self, module_name: str, call_depth: int):
        super().__init__(call_depth)
        self.module_name = module_name

    def render(self, attributes: list[str]) -> str:
        return f"[nn.Mod] {self.module_name}"


class _FXNodeCall(_DebugCall):
    """FX graph node call"""

    def __init__(
        self,
        node,
        call_depth: int,
        record: Optional[dict[str, Any]] = None,
        log: Optional[dict[str, Any]] = None,
    ):
        super().__init__(call_depth, record, log)
        self.node = node

    def render(self, attributes: list[str]) -> str:
        # Format the node operation
        node = self.node

        if node.op in ["placeholder", "output"]:
            node_str = f"[node] {node.name}: {node.op}"
        else:
            args_str = ", ".join(str(n) for n in node.args)
            if node.kwargs:
                kwargs_str = ", " + ", ".join(
                    f"{k}={n}" for k, n in node.kwargs.items()
                )
            else:
                kwargs_str = ""
            target_str = _op_to_str(node.target)
            node_str = (
                f"[node] {node.name}: {node.op}[{target_str}]({args_str}{kwargs_str})"
            )

        # Check for custom metadata from fx.traceback.annotate
        log = {} if self.log is None else self.log
        if (
            "custom" in self.node.meta
            and isinstance((custom := self.node.meta["custom"]), dict)
            and all(isinstance(x, str) for x in custom.keys())
        ):
            log.update(custom)

        if log:
            log_str = ", ".join(f'"{k}": {v}' for k, v in log.items())
            log_str = f"  # {{{log_str}}}"
        else:
            log_str = ""

        return f"{node_str}{log_str}"


def _run_hook(hook, *args):
    out = hook(*args)
    assert isinstance(out, dict) and all(isinstance(k, str) for k in out.keys())
    return out


def _run_node_hooks(call: _DebugCall, node: torch.fx.Node, result: Any) -> None:
    if _node_record_hooks:
        record = {}
        for hook in _node_record_hooks:
            record.update(_run_hook(hook, node, result))
        call.record = record

    if _node_log_hooks:
        log = {} if call.log is None else call.log
        for hook in _node_log_hooks:
            log.update(_run_hook(hook, node, result))
        call.log = log


def _run_dispatch_hooks(call: _DebugCall, func, types, args, kwargs, result) -> None:
    if _dispatch_record_hooks:
        record = {}
        for hook in _dispatch_record_hooks:
            record.update(_run_hook(hook, func, types, args, kwargs, result))
        call.record = record

    if _dispatch_log_hooks:
        log = {} if call.log is None else call.log
        for hook in _dispatch_log_hooks:
            log.update(_run_hook(hook, func, types, args, kwargs, result))
        call.log = log


def _num_hooks(hooks: Optional[Union[Callable, list[Callable]]]) -> int:
    if hooks is None:
        return 0
    elif isinstance(hooks, list):
        return len(hooks)
    elif callable(hooks):
        return 1
    else:
        raise Exception(  # noqa: TRY002
            f"Received hooks of type {type(hooks)}, expected None, Callable, or list[Callable]."
        )


def _add_hooks(
    hooks: list[Callable], new_hooks: Optional[Union[Callable, list[Callable]]]
) -> None:
    if new_hooks is None:
        return
    elif isinstance(new_hooks, list):
        hooks.extend(new_hooks)
    else:
        hooks.append(new_hooks)


class _DebugInterpreter(torch.fx.Interpreter):
    """Custom FX Interpreter that logs node execution to DebugMode"""

    def __init__(self, module, mode: "DebugMode"):
        super().__init__(module)
        self.parent = mode

    def run_node(self, n):
        # Log the node execution
        call = _FXNodeCall(n, self.parent.call_depth)
        self.parent.operators.append(call)

        # Increment call depth before executing
        self.parent.call_depth += 1
        try:
            # Execute the node using parent's run_node
            result = super().run_node(n)
            _run_node_hooks(call, n, result)
            return result
        finally:
            # Decrement call depth after execution
            self.parent.call_depth -= 1


class DebugMode(TorchDispatchMode):
    def __init__(
        self,
        *,
        record_torchfunction=False,
        record_faketensor=False,
        record_realtensor=True,
        record_tensor_attributes=None,
        record_nn_module=False,
    ):
        super().__init__()
        import torch.distributed.tensor  # noqa: F401

        self.supports_higher_order_operators = True
        self.record_torchfunction = record_torchfunction
        self.record_faketensor = record_faketensor
        self.record_realtensor = record_realtensor
        self.record_tensor_attributes = record_tensor_attributes or []

        self.record_nn_module = record_nn_module

        self.module_tracker: Optional[ModTracker] = None
        if self.record_nn_module:
            self.module_tracker_setup()

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
        call = None
        if torch.distributed.tensor.DTensor in types:
            call = _OpCall(func, args, kwargs, self.call_depth)
            self.operators.append(call)
            _run_dispatch_hooks(call, func, types, args, kwargs, None)
            return NotImplemented
        elif FakeTensor in types or isinstance(
            _get_current_dispatch_mode(), FakeTensorMode
        ):
            if self.record_faketensor:
                if func != torch.ops.prim.device.default:
                    call = _OpCall(func, args, kwargs, self.call_depth + 1)
                    self.operators.append(call)
        elif len(types) == 0:
            if self.record_realtensor:
                call = _OpCall(func, args, kwargs, self.call_depth + 1)
                self.operators.append(call)

        result = func(*args, **kwargs)
        if call:
            _run_dispatch_hooks(call, func, types, args, kwargs, result)

        return result

    def __enter__(self):
        self.operators = []
        self.call_depth = 0

        if self.record_torchfunction:
            torch._C._push_on_torch_function_stack(self)

        super().__enter__()
        if self.record_nn_module:
            self.module_tracker.__enter__()  # type: ignore[attribute, union-attr]
        return self

    # pyrefly: ignore  # bad-override
    def __exit__(self, *args):
        super().__exit__(*args)
        if self.record_nn_module:
            self.module_tracker.__exit__()  # type: ignore[attribute, union-attr]
        if self.record_torchfunction:
            torch._C._pop_torch_function_stack()

    def module_tracker_setup(self):
        from torch.distributed._tools.mod_tracker import ModTracker

        self.module_tracker = ModTracker()

        # module pre-fw hook: record module call
        def pre_fw_hook(module, input):
            fqn = self.module_tracker._get_mod_name(module)  # type: ignore[attribute, union-attr]
            self.operators.append(_NNModuleCall(fqn, self.call_depth + 1))
            self.call_depth += 1

        # module post-fw hook: decrement call depth
        def post_fw_hook(module, input, output):
            self.call_depth -= 1

        self.module_tracker.register_user_hooks(pre_fw_hook, post_fw_hook)

    @staticmethod
    @contextlib.contextmanager
    def dispatch_stack_trace(cpp=False):
        global RECORD_STACK_TRACE, CPP_STACK_TRACE
        old_stack_trace, old_cpp = RECORD_STACK_TRACE, CPP_STACK_TRACE

        RECORD_STACK_TRACE = True
        CPP_STACK_TRACE = cpp
        try:
            yield
        finally:
            RECORD_STACK_TRACE = old_stack_trace
            CPP_STACK_TRACE = old_cpp

    @contextlib.contextmanager
    def record_redistribute_calls(
        self,
        arg,
        src_placement,
        dst_placement,
        transform_info_str: Optional[str] = None,
    ):
        try:
            self.operators.append(
                _RedistributeCall(
                    arg,
                    src_placement=src_placement,
                    dst_placement=dst_placement,
                    transform_info_str=transform_info_str,
                    call_depth=self.call_depth + 1,
                )
            )
            self.call_depth += 1
            yield
        finally:
            self.call_depth -= 1

    def run_graph(self, graph_module, *args, **kwargs):
        interpreter = _DebugInterpreter(graph_module, self)
        with self:
            return interpreter.run(*args, **kwargs)

    @staticmethod
    @contextlib.contextmanager
    def dispatch_hooks(
        record_hook: Optional[Union[Callable, list[Callable]]] = None,
        log_hook: Optional[Union[Callable, list[Callable]]] = None,
    ):
        global _dispatch_record_hooks, _dispatch_log_hooks

        n_record = _num_hooks(record_hook)
        n_log = _num_hooks(log_hook)
        _add_hooks(_dispatch_record_hooks, record_hook)
        _add_hooks(_dispatch_log_hooks, log_hook)
        try:
            yield
        finally:
            if n_record:
                _dispatch_record_hooks = _dispatch_record_hooks[:-n_record]
            if n_log:
                _dispatch_log_hooks = _dispatch_log_hooks[:-n_log]

    @staticmethod
    @contextlib.contextmanager
    def node_hooks(
        record_hook: Optional[Union[Callable, list[Callable]]] = None,
        log_hook: Optional[Union[Callable, list[Callable]]] = None,
    ):
        global _node_record_hooks, _node_log_hooks

        n_record = _num_hooks(record_hook)
        n_log = _num_hooks(log_hook)
        _add_hooks(_node_record_hooks, record_hook)
        _add_hooks(_node_log_hooks, log_hook)
        try:
            yield
        finally:
            if n_record:
                _node_record_hooks = _node_record_hooks[:-n_record]
            if n_log:
                _node_log_hooks = _node_log_hooks[:-n_log]

    @staticmethod
    @contextlib.contextmanager
    def record_outputs():
        def _clone(result):
            return tree_map(
                lambda x: x.clone() if isinstance(x, torch.Tensor) else x, result
            )

        def dispatch_hook(func, types, args, kwargs, result):
            return {"output": _clone(result)}

        def node_hook(node, result):
            return {"output": _clone(result)}

        with DebugMode.dispatch_hooks(dispatch_hook), DebugMode.node_hooks(node_hook):
            yield

    def debug_string(self) -> str:
        with torch._C.DisableTorchFunction():
            result = ""
            result += "\n".join(
                "  " + "  " * op.call_depth + op.render(self.record_tensor_attributes)
                for op in self.operators
            )
        return result


def get_active_debug_mode() -> Optional[DebugMode]:
    debug_mode = None
    for mode in _get_current_dispatch_mode_stack():
        if isinstance(mode, DebugMode):
            debug_mode = mode
            break
    return debug_mode
