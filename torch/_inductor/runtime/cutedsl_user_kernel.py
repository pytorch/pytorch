"""Compile and launch user-defined CuTeDSL kernels (``@cute.jit`` host functions).

Used by the eager impl of ``cutedsl_kernel_wrapper_mutation`` and by Inductor's
Python wrapper, where each kernel becomes a CuteDSLKernelLauncher: the counterpart
of the CachingAutotuner a user Triton kernel becomes in generated code. Eager
launches go through launchers too, one per host function, argument names and
constants.

A compiled CuTeDSL kernel is specialized to the argument layouts it was compiled for
and checks nothing at launch, so compiles are cached per argument signature: dtype,
shape, stride and alignment of every tensor, the annotation of every runtime scalar,
the value of every constant.
"""

import weakref
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from typing import Any, TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from torch._higher_order_ops.cutedsl_kernel_wrap import CuteDSLHostFnInfo


# host function -> {argument signature -> compiled function}
_compiled: weakref.WeakKeyDictionary[
    Callable[..., Any], dict[tuple[Any, ...], Any]
] = weakref.WeakKeyDictionary()

# host function -> {(runtime argument names, constants' key) -> launcher}: the
# launchers eager launches reuse. They hold their host functions, so this keeps them
# alive, as the side table does for every host function wrap_cutedsl wraps.
_eager_launchers: dict[
    Callable[..., Any], dict[tuple[Any, ...], "CuteDSLKernelLauncher"]
] = {}


def _launch_device(fn: Callable[..., Any], values: Iterable[Any]) -> torch.device:
    """The CUDA device of the tensors among ``values`` (they must share one), else
    the current device."""
    devices = list(
        dict.fromkeys(v.device for v in values if isinstance(v, torch.Tensor))
    )
    if not devices:
        return torch.device("cuda", torch.cuda.current_device())
    if len(devices) > 1 or devices[0].type != "cuda":
        raise ValueError(
            f"CuTeDSL kernel {getattr(fn, '__name__', fn)}: tensor arguments must be "
            f"on one CUDA device, got {', '.join(map(str, devices))}"
        )
    return devices[0]


def _check_launch_args(
    fn: Callable[..., Any], info: "CuteDSLHostFnInfo", names: Collection[str]
) -> None:
    """Check that ``names``, the arguments passed by name, bind to the parameters of
    ``fn`` but the stream, which the launch supplies: a TypeError names the kernel
    and the parameters for a name that is not one, for the stream, and for missing
    parameters without a default."""
    kernel = getattr(fn, "__name__", fn)
    for name in names:
        if name == info.stream_param:
            raise TypeError(
                f"CuTeDSL kernel {kernel}: got a value for its stream parameter "
                f"{name!r}, which the launch supplies"
            )
        if name not in info.params:
            raise TypeError(
                f"CuTeDSL kernel {kernel}: got an unexpected argument {name!r}; its "
                f"parameters are {info.params}"
            )
    missing = [
        p
        for p in info.params
        if p != info.stream_param and p not in names and p not in info.defaults
    ]
    if missing:
        raise TypeError(f"CuTeDSL kernel {kernel}: missing arguments {missing}")


def launch_cutedsl_kernel(
    fn: Callable[..., Any], args: Mapping[str, Any], stream: int | None = None
) -> None:
    """Launch host function ``fn`` with ``args`` (every parameter but the stream,
    by name; missing ones take their defaults) on ``stream``, a raw CUDA stream
    (default: the current stream of the tensors' device)."""
    from torch._higher_order_ops.cutedsl_kernel_wrap import get_host_fn_info
    from torch._higher_order_ops.user_kernel_hop import _snapshot, constant_key

    info = get_host_fn_info(fn)
    _check_launch_args(fn, info, args)
    # Split as ir.UserDefinedCuteDSLKernel.split_args splits a call, so that an eager
    # launch shares its compile with the generated code's launch of the same call.
    runtime_args: dict[str, Any] = {}
    constant_args: dict[str, Any] = {}
    for name in info.params:
        if name == info.stream_param:
            continue
        value = args[name] if name in args else info.defaults[name]
        if isinstance(value, torch.Tensor) or info.is_runtime_scalar(name, value):
            runtime_args[name] = value
        else:
            constant_args[name] = value
    # Keyed on every launch, as the caller may change a constant between launches;
    # the launcher binds snapshots, which keep matching the key.
    names = tuple(runtime_args)
    key = (names, tuple((k, constant_key(v)) for k, v in constant_args.items()))
    launchers = _eager_launchers.setdefault(fn, {})
    launcher = launchers.get(key)
    if launcher is None:
        snapshots = {k: _snapshot(v) for k, v in constant_args.items()}
        launcher = launchers[key] = CuteDSLKernelLauncher(fn, names, snapshots)
    launcher.run(*runtime_args.values(), stream=stream)


class CuteDSLKernelLauncher:
    """Launches host function ``fn`` with constants bound once: one user CuTeDSL
    kernel in Inductor's generated Python wrapper, or the eager launches of ``fn``
    with these argument names and constants (see launch_cutedsl_kernel).

    ``run(*args, stream=...)`` takes the kernel arguments (``arg_names``: tensors and
    runtime scalars, in host-function signature order) positionally. Everything else
    is done here, once: the constants (``constant_args``, and the defaults of the
    parameters neither names) are bound and keyed for the compile cache, and run()
    keeps the compiled functions it uses by device and argument signature, so a
    launch keys only its own arguments. Both users bind snapshots as constants (see
    ir.UserDefinedCuteDSLKernel.split_args), so their key cannot go stale.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        arg_names: Sequence[str],
        constant_args: Mapping[str, Any],
    ) -> None:
        from torch._higher_order_ops.cutedsl_kernel_wrap import (
            get_host_fn_info,
            python_scalar,
        )
        from torch._higher_order_ops.user_kernel_hop import constant_key

        info = get_host_fn_info(fn)
        _check_launch_args(fn, info, [*arg_names, *constant_args])
        self.fn = fn
        self.arg_names = tuple(arg_names)
        self.constant_args = dict(constant_args)
        self._stream_param = info.stream_param
        self._numeric_params = info.numeric_params
        # The parameters run() does not take, as cute.compile takes them, and their
        # part of the compile cache key.
        self._fixed_args: dict[str, Any] = {}
        fixed_key: list[object] = []
        for name in info.params:
            if name == info.stream_param or name in self.arg_names:
                continue
            if name in self.constant_args:
                value = self.constant_args[name]
            else:
                value = info.defaults[name]
            if info.is_runtime_scalar(name, value):
                value = python_scalar(value)
                fixed_key.append((name, info.numeric_params.get(name, type(value))))
            else:
                fixed_key.append((name, constant_key(value)))
            self._fixed_args[name] = value
        self._fixed_key = tuple(fixed_key)
        # The compiled function takes every parameter but the Constexpr ones, in
        # signature order: the fixed ones in place, and where run() puts the stream
        # and its arguments.
        call_params = [p for p in info.params if p not in info.constexpr_params]
        self._call_args = [self._fixed_args.get(p) for p in call_params]
        self._stream_index = call_params.index(info.stream_param)
        self._arg_indices = tuple(
            call_params.index(name) if name in call_params else None
            for name in self.arg_names
        )
        # (device index, *argument signature) -> compiled function
        self._compiled: dict[tuple[Any, ...], Any] = {}

    def __reduce__(self) -> Any:
        # A run of the generated module as a script unpickles its launchers: each is
        # rebuilt from its arguments, without the caches.
        return type(self), (self.fn, self.arg_names, self.constant_args)

    def run(self, *args: Any, stream: int | None = None) -> None:
        """Launch with ``args`` for ``arg_names`` on ``stream``, a raw CUDA stream
        (default: the current stream of the tensors' device)."""
        import cuda.bindings.driver as cuda  # pyrefly: ignore[missing-import]
        from cutlass.cute.runtime import from_dlpack

        from torch._higher_order_ops.cutedsl_kernel_wrap import python_scalar
        from torch._native.cutedsl.launch import supported_alignment

        if len(args) != len(self.arg_names):
            raise AssertionError(
                f"expected {len(self.arg_names)} arguments ({self.arg_names}), "
                f"got {len(args)}"
            )
        device = _launch_device(self.fn, args)
        # Everything runs under the tensors' device: DLPack export requires it.
        with torch.cuda.device(device):
            if stream is None:
                stream = torch._C._cuda_getCurrentRawStream(device.index)
            cu_stream = cuda.CUstream(stream)
            call_args = self._call_args.copy()
            call_args[self._stream_index] = cu_stream
            key: list[object] = [device.index]
            values: list[Any] = []
            for name, index, value in zip(self.arg_names, self._arg_indices, args):
                if isinstance(value, torch.Tensor):
                    align = supported_alignment(value, 16)
                    key.append((name, value.dtype, value.shape, value.stride(), align))
                    value = from_dlpack(value.detach(), assumed_align=align)
                else:
                    # The DSL converts only exact bool/int/float (not an IntEnum or
                    # a numpy float) and takes a cutlass Numeric value as a Python one.
                    value = python_scalar(value)
                    key.append((name, self._numeric_params.get(name, type(value))))
                values.append(value)
                if index is not None:
                    call_args[index] = value
            compiled = self._compiled.get(tuple(key))
            if compiled is None:
                compiled = self._compile(device, tuple(key), values, cu_stream)
            compiled(*call_args)

    def _compile(
        self, device: torch.device, key: tuple[Any, ...], values: list[Any], stream: Any
    ) -> Any:
        """The compiled function for run()'s ``key``, from the compile cache all the
        launches of ``fn`` share (across devices of one capability), compiled for
        ``values`` and ``stream`` if it is missing."""
        import cutlass.cute as cute

        from torch._inductor.codegen.cutedsl.compile_lock import CUTEDSL_COMPILE_LOCK

        capability = torch.cuda.get_device_capability(device)
        shared_key = (capability, self._fixed_key, *key[1:])
        per_fn = _compiled.setdefault(self.fn, {})
        compiled = per_fn.get(shared_key)
        if compiled is None:
            with CUTEDSL_COMPILE_LOCK:
                compiled = per_fn.get(shared_key)
                if compiled is None:
                    compile_args = dict(zip(self.arg_names, values))
                    compile_args[self._stream_param] = stream
                    compiled = cute.compile(self.fn, **self._fixed_args, **compile_args)
                    per_fn[shared_key] = compiled
        self._compiled[key] = compiled
        return compiled
