# mypy: allow-untyped-defs
import contextlib
from typing import Optional, TYPE_CHECKING

import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.utils._dtype_abbrs import dtype_abbrs
from torch.utils._python_dispatch import (
    _get_current_dispatch_mode,
    _get_current_dispatch_mode_stack,
    TorchDispatchMode,
)
from torch.utils._pytree import tree_map


if TYPE_CHECKING:
    from torch.distributed._tools.mod_tracker import ModTracker


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


class _DebugCall:
    """Base class for tracking operator calls in DebugMode"""

    def __init__(self, call_depth: int):
        self.call_depth = call_depth

    def render(self, attributes: list[str]) -> str:
        raise NotImplementedError("Subclasses must implement string render()")

    def __repr__(self) -> str:
        return self.render([])


class _OpCall(_DebugCall):
    """Normal operator call"""

    def __init__(self, op, args: tuple, kwargs: dict, call_depth: int):
        super().__init__(call_depth)
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

        if isinstance(self.op, torch._ops.OpOverload):
            op_name = self.op.__qualname__
        elif hasattr(self.op, "__module__") and hasattr(self.op, "__name__"):
            op_name = f"{self.op.__module__}.{self.op.__name__}"
        else:
            op_name = str(self.op)

        return f"{op_name}({args_str}{kwargs_str})"

    def __iter__(self):
        # for BC; tuple(self) returns (op, args, kwargs, call_depth)
        yield from [self.op, self.args, self.kwargs, self.call_depth]


class _RedistributeCall(_DebugCall):
    """Redistribute call from DTensor dispatch"""

    def __init__(
        self, arg, src_placement, dst_placement, transform_info_str, call_depth
    ):
        super().__init__(call_depth)
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


class _NNModuleCall(_DebugCall):
    """Designates entering an nn.Module's forward method"""

    def __init__(self, module_name: str, call_depth: int):
        super().__init__(call_depth)
        self.module_name = module_name

    def render(self, attributes: list[str]) -> str:
        return f"[nn.Mod] {self.module_name}"

    def __iter__(self):
        yield from [
            f"[nn.Mod] {self.module_name}",
            (),
            {},
            self.call_depth,
        ]


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
        if torch.distributed.tensor.DTensor in types:
            self.operators.append(_OpCall(func, args, kwargs, self.call_depth))
            return NotImplemented
        elif FakeTensor in types or isinstance(
            _get_current_dispatch_mode(), FakeTensorMode
        ):
            if self.record_faketensor:
                if func != torch.ops.prim.device.default:
                    self.operators.append(
                        _OpCall(func, args, kwargs, self.call_depth + 1)
                    )
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
