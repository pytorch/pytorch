"""Compile and launch user-defined CuTeDSL kernels (``@cute.jit`` host functions).

Used by the eager impl of ``cutedsl_kernel_wrapper_mutation`` and by Inductor's
Python wrapper, where each kernel becomes a CuteDSLKernelLauncher: the counterpart
of the CachingAutotuner a user Triton kernel becomes in generated code. Eager
launches go through launchers too, one per host function, argument names and
constants. It lives outside torch._inductor so that an eager launch imports neither
Inductor nor Dynamo.

A compiled CuTeDSL kernel is specialized to the arguments it was compiled for and
checks none of them at launch, so compiles are cached per host function and argument
signature, which is: the compute capability of the device; the dtype, shape, stride
and alignment of every tensor; for every runtime scalar, the cutlass Numeric
annotation of its parameter, or the Python type python_scalar converts it to (bool,
int or float) if the parameter is unannotated; and the constant_key of every
constant, which keys lists, tuples, dicts, dataclass instances, plain objects and
kernel objects (whose class's __call__ is @cute.jit) by type and content, and an
object with state in __slots__ (dataclass fields aside) or in a built-in base, or any
other callable object that is not a dataclass instance, as itself, by identity unless
its class hashes by value (see _object_state in user_kernel_hop.py).

The signature has no device index: devices of one capability share a compile.
cute.compile returns a CudaDialectJitCompiledFunction, which loads its cubin as a
CUDA library, not into a CUDA context, and resolves its kernels on every device when
it is first called (its ``to`` and ``_load_cuda_library`` in
cutlass/cutlass_dsl/cuda_jit_executor.py, in nvidia-cutlass-dsl 4.4 and 4.8).
"""

import contextlib
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from typing import Any

import torch


# host function -> {argument signature -> compiled function}. Nothing is evicted,
# and the host functions are held too: a compile's CUDA library is unloaded once it
# is freed, while a CUDA graph that captured a launch of it still runs its kernel.
# So a host function made for each launch (a closure, a kernel object), or a
# constant keyed by identity (see _object_state in user_kernel_hop.py) made for each
# launch, compiles each time and stays in memory, with its compile, for the life of
# the process.
_compiled: dict[Callable[..., Any], dict[tuple[Any, ...], Any]] = {}

# host function -> {(runtime argument names, constants' key) -> launcher}: the
# launchers eager launches reuse. They hold their host functions, so this keeps them
# alive, as the side table does for every host function wrap_cutedsl wraps. Like
# _compiled, it evicts nothing.
_eager_launchers: dict[
    Callable[..., Any], dict[tuple[Any, ...], "CuteDSLKernelLauncher"]
] = {}


def _launch_device(name: str, values: Iterable[Any]) -> torch.device:
    """The CUDA device of the tensors among ``values`` (they must share one), else
    the current device."""
    devices = list(
        dict.fromkeys(v.device for v in values if isinstance(v, torch.Tensor))
    )
    if not devices:
        return torch.device("cuda", torch.cuda.current_device())
    if len(devices) > 1 or devices[0].type != "cuda":
        raise ValueError(
            f"CuTeDSL kernel {name}: tensor arguments must be on one CUDA device, "
            f"got {', '.join(map(str, devices))}"
        )
    return devices[0]


def _compile(
    fn: Callable[..., Any], args: dict[str, Any], constants: Collection[str]
) -> Any:
    """cute.compile(fn, **args) on copies of the ``constants`` (see _snapshot): the
    host code that the compile runs can change one (cache a layout on a kernel
    object, say), while the compile cache key, the side table's snapshot and the
    caller's object must keep the contents they were keyed by. Not cached by
    jit_cache, whose disk cache keys by qualified name and pickled arguments and
    loads its compiles through tvm-ffi: CuteDSLKernelLauncher caches it."""
    import cutlass.cute as cute

    from torch._higher_order_ops.cutedsl_kernel_wrap import CUTEDSL_CONSTANT_RULES
    from torch._higher_order_ops.user_kernel_hop import _snapshot

    rules = CUTEDSL_CONSTANT_RULES
    copies = {k: _snapshot(v, rules) if k in constants else v for k, v in args.items()}
    return cute.compile(fn, **copies)


def key_constants(
    name: str,
    constant_args: Mapping[str, Any],
    keys: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """The constant_key of each of ``constant_args``, constants of kernel ``name`` by
    parameter, taking the ones in ``keys`` as given. A constant that constant_key
    refuses raises its TypeError or ValueError, naming the kernel and the parameter:
    the eager launch, a launcher and Inductor report it alike. Inductor keys the
    defaults a call leaves out when it defines the call's launcher, or first, in
    ir.UserDefinedCuteDSLKernel.split_args, if one of them cannot be copied."""
    from torch._higher_order_ops.cutedsl_kernel_wrap import CUTEDSL_CONSTANT_RULES
    from torch._higher_order_ops.user_kernel_hop import constant_key

    given = keys or {}
    result: dict[str, object] = {}
    for param, value in constant_args.items():
        if param in given:
            result[param] = given[param]
            continue
        try:
            result[param] = constant_key(value, CUTEDSL_CONSTANT_RULES)
        except (TypeError, ValueError) as e:
            raise type(e)(f"CuTeDSL kernel {name}: parameter {param!r}: {e}") from None
    return result


def launch_cutedsl_kernel(
    fn: Callable[..., Any],
    args: Mapping[str, Any],
    stream: int | None = None,
    constant_keys: Mapping[str, object] | None = None,
) -> None:
    """Launch host function ``fn`` with ``args`` (every parameter but the stream,
    by name; missing ones take their defaults) on ``stream``, a raw CUDA stream
    (default: the current stream of the tensors' device). ``constant_keys`` holds
    the constant_key of constants among ``args`` that the caller has keyed already
    (on the HOP path, the side table's keys)."""
    from torch._higher_order_ops.cutedsl_kernel_wrap import (
        CUTEDSL_CONSTANT_RULES,
        get_host_fn_info,
    )
    from torch._higher_order_ops.user_kernel_hop import _snapshot

    info = get_host_fn_info(fn)
    name = info.name
    info.check_call(name, args.keys())
    # ir.UserDefinedCuteDSLKernel.split_args splits a call with the same rules, so
    # that an eager launch shares its compile with the generated code's launch of
    # the same call.
    runtime_args, constant_args = info.split_args(name, args)
    # Keyed on every launch, as the caller may change a constant between launches;
    # the launcher binds snapshots, which keep matching the key.
    rules = CUTEDSL_CONSTANT_RULES
    keys = key_constants(name, constant_args, constant_keys)
    names = tuple(runtime_args)
    key = (names, tuple(keys.items()))
    launchers = _eager_launchers.setdefault(fn, {})
    launcher = launchers.get(key)
    if launcher is None:
        snapshots = {k: _snapshot(v, rules) for k, v in constant_args.items()}
        launcher = launchers[key] = CuteDSLKernelLauncher(
            fn, names, snapshots, constant_keys=keys
        )
    launcher.run(*runtime_args.values(), stream=stream)


class CuteDSLKernelLauncher:
    """Launches host function ``fn`` with constants bound once: one user CuTeDSL
    kernel in Inductor's generated Python wrapper, or the eager launches of ``fn``
    with these argument names and constants (see launch_cutedsl_kernel).

    ``run(*args, stream=...)`` takes the kernel arguments (``arg_names``: tensors and
    runtime scalars, in host-function signature order) positionally. Everything else
    is done here, once: the constants (``constant_args``, and the defaults of the
    parameters neither names) are bound and keyed for the compile cache, reusing the
    constant_key of those in ``constant_keys``, and run() keeps the compiled
    functions it uses by device and argument signature, so a launch keys only its own
    arguments. Both users bind snapshots as constants (see
    ir.UserDefinedCuteDSLKernel.split_args), so their key cannot go stale.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        arg_names: Sequence[str],
        constant_args: Mapping[str, Any],
        *,
        constant_keys: Mapping[str, object] | None = None,
    ) -> None:
        import cuda.bindings.driver as cuda  # pyrefly: ignore[missing-import]
        from cutlass.cute.runtime import from_dlpack

        from torch._higher_order_ops.cutedsl_kernel_wrap import (
            get_host_fn_info,
            python_scalar,
        )
        from torch._native.cutedsl.launch import stream, supported_alignment

        # Imported here, not at module level (this module imports without cutlass),
        # for run() to use.
        self._cu_stream = cuda.CUstream
        self._from_dlpack = from_dlpack
        self._python_scalar = python_scalar
        self._current_stream = stream
        self._supported_alignment = supported_alignment

        info = get_host_fn_info(fn)
        self._kernel_name = kernel_name = info.name
        info.check_call(kernel_name, [*arg_names, *constant_args])
        self.fn = fn
        self.arg_names = tuple(arg_names)
        self.constant_args = dict(constant_args)
        self._stream_param = info.stream_param
        self._numeric_params = info.numeric_params
        # The parameters run() does not take, as cute.compile takes them, and their
        # part of the compile cache key.
        self._fixed_args: dict[str, Any] = {}
        fixed_key: list[object] = []
        for param in info.params:
            if param == info.stream_param or param in self.arg_names:
                continue
            if param in self.constant_args:
                value = self.constant_args[param]
            else:
                value = info.defaults[param]
            if info.is_runtime_scalar(param, value):
                annotation = info.numeric_params.get(param)
                value = python_scalar(value, annotation)
                fixed_key.append((param, annotation or type(value)))
            else:
                key = key_constants(kernel_name, {param: value}, constant_keys)[param]
                fixed_key.append((param, key))
            self._fixed_args[param] = value
        self._fixed_key = tuple(fixed_key)
        # The compiled function takes every parameter but the Constexpr ones, in
        # signature order: the fixed ones in place, and where run() puts the stream
        # and its arguments.
        call_params = [p for p in info.params if p not in info.constexpr_params]
        self._call_args = [self._fixed_args.get(p) for p in call_params]
        self._stream_index = call_params.index(info.stream_param)
        self._arg_indices = tuple(
            call_params.index(arg) if arg in call_params else None
            for arg in self.arg_names
        )
        # (device index, *argument signature) -> compiled function
        self._compiled: dict[tuple[Any, ...], Any] = {}

    def run(self, *args: Any, stream: int | None = None) -> None:
        """Launch with ``args`` for ``arg_names`` on ``stream``, a raw CUDA stream
        (default: the current stream of the tensors' device)."""
        if len(args) != len(self.arg_names):
            raise AssertionError(
                f"expected {len(self.arg_names)} arguments ({self.arg_names}), "
                f"got {len(args)}"
            )
        device = _launch_device(self._kernel_name, args)
        # DLPack export and the launch need the tensors' device to be current, as it
        # is under the device guard of Inductor's generated code.
        current = device.index == torch.cuda.current_device()
        with contextlib.nullcontext() if current else torch.cuda.device(device):
            if stream is None:
                cu_stream = self._current_stream(device.index)
            else:
                cu_stream = self._cu_stream(stream)
            call_args = self._call_args.copy()
            call_args[self._stream_index] = cu_stream
            key: list[object] = [device.index]
            values: list[Any] = []
            for name, index, value in zip(self.arg_names, self._arg_indices, args):
                if isinstance(value, torch.Tensor):
                    align = self._supported_alignment(value, 16)
                    key.append((name, value.dtype, value.shape, value.stride(), align))
                    value = self._from_dlpack(value.detach(), assumed_align=align)
                else:
                    # The DSL converts only exact bool/int/float (not an IntEnum or
                    # a numpy float) and takes a cutlass Numeric value as a Python one.
                    annotation = self._numeric_params.get(name)
                    value = self._python_scalar(value, annotation)
                    key.append((name, annotation or type(value)))
                values.append(value)
                if index is not None:
                    call_args[index] = value
            compiled = self._compiled.get(tuple(key))
            if compiled is None:
                compiled = self._get_compiled(device, tuple(key), values, cu_stream)
            compiled(*call_args)

    def _get_compiled(
        self, device: torch.device, key: tuple[Any, ...], values: list[Any], stream: Any
    ) -> Any:
        """The compiled function for run()'s ``key``, from the compile cache all the
        launches of ``fn`` share (across devices of one capability), compiled for
        ``values`` and ``stream`` if it is missing."""
        from torch._native.cutedsl.compile_lock import CUTEDSL_COMPILE_LOCK
        from torch._native.cutedsl.hw_caps import caps
        from torch._native.cutedsl.plan_cache import cached_plan

        shared_key = (caps(device.index).cc, self._fixed_key, *key[1:])
        per_fn = _compiled.setdefault(self.fn, {})
        compiled = per_fn.get(shared_key)
        if compiled is None:
            # The bound constants, in generated code the side table's own snapshots,
            # are compiled as copies (see _compile).
            compile_args = dict(self._fixed_args)
            compile_args.update(zip(self.arg_names, values))
            compile_args[self._stream_param] = stream
            with CUTEDSL_COMPILE_LOCK:
                # cached_plan looks the signature up again under the lock, and logs
                # a compile to native_dsl_compile as the native ops' compiles are.
                compiled = cached_plan(
                    per_fn,
                    shared_key,
                    lambda: _compile(self.fn, compile_args, self._fixed_args),
                    op=self._kernel_name,
                )
        self._compiled[key] = compiled
        return compiled
