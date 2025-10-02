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
from torch.utils._pytree import tree_leaves, tree_map, tree_map_only
from torch.utils.weak import WeakIdRef


__all__ = ["DebugMode", "get_active_debug_mode"]

REDISTRIBUTE_FUNC = "redistribute_input"
UNTRACKED_OUTPUT = object()


def _stringify_shape(shape) -> str:
    return f"[{', '.join([str(x) for x in shape])}]"


def _stringify_device_mesh(mesh) -> str:
    return f"DM({', '.join([str(s) for s in mesh.shape])})"


def _stringify_placement(placement) -> str:
    return f"[{', '.join([str(p) for p in placement])}]"


def _stringify_pytree(tree) -> str:
    if isinstance(tree, (tuple, list)):
        return f"({', '.join([_stringify_pytree(t) for t in tree])})"
    elif isinstance(tree, dict):
        return f"{{{', '.join([f'{k}: {_stringify_pytree(v)}' for k, v in tree.items()])}}}"
    else:
        return str(tree)


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
        if isinstance(x, DTensorSpec):
            return _stringify_placement(x.placements)
        return x

    arg = tree_map(to_str, arg)
    return str(arg)


def _op_to_str(op, *args, **kwargs) -> str:
    # if op == REDISTRIBUTE_FUNC:
    #     assert len(args) == 3
    #     _args = [_arg_to_str(arg) for arg in args]
    #     args_str = f"{_args[0]}, {_args[1]} -> {_args[2]}"
    # else:
    #     args_str = ", ".join(_arg_to_str(arg) for arg in args)
    args_str = ", ".join(_arg_to_str(arg) for arg in args)

    if kwargs:
        kwargs_str = ", " + ", ".join(
            f"{k}={_arg_to_str(v)}" for k, v in kwargs.items()
        )
    else:
        kwargs_str = ""

    op_name = torch.overrides.resolve_name(op)
    # if isinstance(op, torch._ops.OpOverload):
    #     op_name = op.__qualname__
    # elif hasattr(op, "__module__") and hasattr(op, "__name__"):
    #     op_name = f"{op.__module__}.{op.__name__}"
    # else:
    #     op_name = str(op)

    return f"{op_name}({args_str}{kwargs_str})"


class _MemoId:
    def __init__(self, id):
        self.id = id

    def __str__(self):
        return f"${self.id}"


class _OpCall:
    def __init__(self, op, args, kwargs):
        self.op = op
        self.args = args
        self.kwargs = kwargs

    def inputs(self):
        return tree_leaves((self.args, self.kwargs))

    def __str__(self):
        return _op_to_str(self.op, *self.args, **self.kwargs)


class _RedistributeCall:
    def __init__(self, local_t, src_placement, dst_placement):
        self.local_t = local_t
        self.src_placement = src_placement
        self.dst_placement = dst_placement

    def inputs(self):
        return [self.local_t]

    def __str__(self):
        return f"{REDISTRIBUTE_FUNC}({self.local_t} : {self.src_placement} -> {self.dst_placement})"


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
        self.output_info = {}
        self.n_logs = 0
        self.call_depth = 0
        self.memo = {}
        self.t_descs = {}
        self.next_id = 0

    def record_func_call(self, func, args, kwargs, call_depth):
        with torch._C.DisableTorchFunction():
            fmt_args = tree_map_only(torch.Tensor, self._shortid, args)
            fmt_kwargs = tree_map_only(torch.Tensor, self._shortid, kwargs)
            self.operators.append((_OpCall(func, fmt_args, fmt_kwargs), call_depth))
            self.n_logs += 1

    def _shortid(self, t: torch.Tensor) -> _MemoId:
        with torch._C.DisableTorchFunction():
            o = WeakIdRef(t)
            weak_self = weakref.ref(self)

            def del_memo():
                self = weak_self()
                if self is None:
                    return
                self.memo.pop(o, None)

            weakref.finalize(t, del_memo)
            if o not in self.memo:
                self.memo[o] = _MemoId(self.next_id)
                self.t_descs[self.next_id] = _arg_to_str(t)
                self.next_id += 1
            return self.memo[o]

    def record_output(self, log_id, output):
        with torch._C.DisableTorchFunction():
            fmt = tree_map_only(torch.Tensor, self._shortid, output)
            self.output_info[log_id] = fmt

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        log_id = self.n_logs
        self.record_func_call(func, args, kwargs, self.call_depth)

        try:
            self.call_depth += 1
            result = func(*args, **kwargs)
            self.record_output(log_id, result)
            return result
        finally:
            self.call_depth -= 1

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        # Record the operation with its call depth
        log_id = self.n_logs
        if torch.distributed.tensor.DTensor in types:
            self.record_func_call(func, args, kwargs, self.call_depth)
            return NotImplemented
        elif FakeTensor in types or isinstance(
            _get_current_dispatch_mode(), FakeTensorMode
        ):
            if self.record_faketensor:
                if func != torch.ops.prim.device.default:
                    self.record_func_call(func, args, kwargs, self.call_depth + 1)
        elif len(types) == 0:
            if self.record_realtensor:
                self.record_func_call(func, args, kwargs, self.call_depth + 1)

        result = func(*args, **kwargs)
        self.record_output(log_id, result)

        return result

    def __enter__(self):
        self.operators = []
        self.call_depth = 0

        if self.record_torchfunction:
            torch._C._push_on_torch_function_stack(self)

        super().__enter__()
        return self

    def __exit__(self, *args):
        super().__exit__(*args)
        if self.record_torchfunction:
            torch._C._pop_torch_function_stack()

    @contextlib.contextmanager
    def record_redistribute_calls(self, local_t, src_placement, dst_placement):
        try:
            fmt_local_t = self._shortid(local_t)
            self.operators.append((_RedistributeCall(fmt_local_t, src_placement, dst_placement), self.call_depth + 1))
            self.n_logs += 1
            self.call_depth += 1
            yield
        finally:
            self.call_depth -= 1

    @classmethod
    def ignore_compile_internals(cls):
        return True

    def debug_string(self) -> str:
        tracked_ids = set()
        id_map = {}

        def map_id(memo):
            if memo.id not in id_map:
                id_map[memo.id] = _MemoId(len(id_map))
            return id_map[memo.id]

        result = ""
        for log_id, (log, depth) in enumerate(self.operators):
            if result and depth == 0:
                result += "\n"
            output_info = self.output_info.get(log_id, UNTRACKED_OUTPUT)
            for m in log.inputs():
                if isinstance(m, _MemoId) and m.id not in tracked_ids:
                    result += "  " * depth + f"{map_id(m)} = {_arg_to_str(self.t_descs[m.id])}\n"
                    tracked_ids.add(m.id)
            lhs = tree_map(lambda x: map_id(x) if isinstance(x, _MemoId) else "_", output_info)
            lhs = _stringify_pytree(lhs)
            desc = tree_map(lambda x: self.t_descs[x.id] if isinstance(x, _MemoId) else x, output_info)
            desc = _stringify_pytree(desc)
            tree_map_only(_MemoId, lambda x: tracked_ids.add(x.id), output_info)

            if isinstance(log, _OpCall):
                log.args = tree_map_only(_MemoId, map_id, log.args)
                log.kwargs = tree_map_only(_MemoId, map_id, log.kwargs)
            else:
                log.local_t = map_id(log.local_t)
            if all(x == "_" for x in tree_leaves(lhs)):
                result += "  " * depth + f"{log}\n"
            else: 
                result += "  " * depth + f"{lhs} : {desc} = {log}\n"
        # with torch._C.DisableTorchFunction():
        #     result = ""
        #     result += "\n".join(
        #         "  " + "  " * depth + _op_to_str(op, *args, **kwargs)
        #         for op, args, kwargs, depth in self.operators
        #     )
        return result


def get_active_debug_mode() -> Optional[DebugMode]:
    debug_mode = None
    for mode in _get_current_dispatch_mode_stack():
        if isinstance(mode, DebugMode):
            debug_mode = mode
            break
    return debug_mode
