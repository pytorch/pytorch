"""Compile and launch user-defined CuTeDSL kernels (``@cute.jit`` host functions).

Used by the eager impl of ``cutedsl_kernel_wrapper_mutation``. It lives outside
torch._inductor so that an eager launch imports neither Inductor nor Dynamo.

A compiled CuTeDSL kernel is specialized to the arguments it was compiled for and
checks none of them at launch, so compiles are cached per host function and argument
signature, which is: the compute capability of the device; the dtype, shape, stride
and alignment of every tensor; for every runtime scalar, the cutlass Numeric
annotation of its parameter, or the Python type python_scalar converts it to (bool,
int or float) if the parameter is unannotated; and the constant_key of every
constant, which keys lists, tuples, dicts, dataclass instances, plain objects and
kernel objects (whose class's __call__ is @cute.jit) by type and content, and an
object with state in __slots__ (dataclass fields aside) or any other callable object
that is not a dataclass instance as itself, by identity unless its class hashes by
value (see _object_state in user_kernel_hop.py).

The signature has no device index: devices of one capability share a compile.
cute.compile returns a CudaDialectJitCompiledFunction, which loads its cubin as a
CUDA library, not into a CUDA context, and resolves its kernels on every device when
it is first called (its ``to`` and ``_load_cuda_library`` in
cutlass/cutlass_dsl/cuda_jit_executor.py, in nvidia-cutlass-dsl 4.4 and 4.8).
"""

import weakref
from collections.abc import Callable, Collection, Iterable, Mapping
from typing import Any

import torch


# host function -> {argument signature -> compiled function}. Nothing is evicted: a
# compile's CUDA library is unloaded once it is freed, while a CUDA graph that
# captured a launch of it still runs its kernel. A constant keyed by identity (see
# _object_state in user_kernel_hop.py) made for each launch therefore compiles each
# time and stays in memory, with its compile, as long as the host function does.
_compiled: weakref.WeakKeyDictionary[Callable[..., Any], dict[tuple[Any, ...], Any]] = (
    weakref.WeakKeyDictionary()
)


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
    host code that the compile runs can change one (cache a layout on a config
    object, say), while the compile cache key, the side table's snapshot and the
    caller's object must keep the contents they were keyed by. Not cached by
    jit_cache, whose disk cache keys by qualified name and pickled arguments and
    loads its compiles through tvm-ffi: launch_cutedsl_kernel caches it."""
    import cutlass.cute as cute

    from torch._higher_order_ops.cutedsl_kernel_wrap import CUTEDSL_CONSTANT_RULES
    from torch._higher_order_ops.user_kernel_hop import _snapshot

    rules = CUTEDSL_CONSTANT_RULES
    copies = {k: _snapshot(v, rules) if k in constants else v for k, v in args.items()}
    return cute.compile(fn, **copies)


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
    import cuda.bindings.driver as cuda  # pyrefly: ignore[missing-import]
    from cutlass.cute.runtime import from_dlpack

    from torch._higher_order_ops.cutedsl_kernel_wrap import (
        CUTEDSL_CONSTANT_RULES,
        get_host_fn_info,
        python_scalar,
    )
    from torch._higher_order_ops.user_kernel_hop import constant_key
    from torch._native.cutedsl.compile_lock import CUTEDSL_COMPILE_LOCK
    from torch._native.cutedsl.hw_caps import caps
    from torch._native.cutedsl.launch import (
        stream as current_stream,
        supported_alignment,
    )
    from torch._native.cutedsl.plan_cache import cached_plan

    info = get_host_fn_info(fn)
    name = info.name
    info.check_call(name, args.keys())
    runtime_args, constant_args = info.split_args(name, args)
    device = _launch_device(name, runtime_args.values())
    # Everything runs under the tensors' device: DLPack export requires it.
    with torch.cuda.device(device):
        if stream is None:
            cu_stream = current_stream(device.index)
        else:
            cu_stream = cuda.CUstream(stream)

        key: list[object] = [caps(device.index).cc]
        compile_args: dict[str, Any] = {info.stream_param: cu_stream}
        for param, value in runtime_args.items():
            if isinstance(value, torch.Tensor):
                align = supported_alignment(value, 16)
                key.append((param, value.dtype, value.shape, value.stride(), align))
                value = from_dlpack(value.detach(), assumed_align=align)
            else:
                annotation = info.numeric_params.get(param)
                value = python_scalar(value, annotation)
                key.append((param, annotation or type(value)))
            compile_args[param] = value
        keys = constant_keys or {}
        for param, value in constant_args.items():
            compile_args[param] = value
            if param in keys:
                key.append((param, keys[param]))
                continue
            try:
                key.append((param, constant_key(value, CUTEDSL_CONSTANT_RULES)))
            except (TypeError, ValueError) as e:
                msg = f"CuTeDSL kernel {name}: parameter {param!r}: {e}"
                raise type(e)(msg) from None
        # The compiled function takes every parameter but the Constexpr ones, in
        # signature order.
        call_args = [
            compile_args[p] for p in info.params if p not in info.constexpr_params
        ]

        signature = tuple(key)
        per_fn = _compiled.setdefault(fn, {})
        compiled = per_fn.get(signature)
        if compiled is None:
            with CUTEDSL_COMPILE_LOCK:
                # cached_plan looks the signature up again under the lock, and logs
                # a compile to native_dsl_compile as the native ops' compiles are.
                compiled = cached_plan(
                    per_fn,
                    signature,
                    lambda: _compile(fn, compile_args, constant_args),
                    op=name,
                )
        compiled(*call_args)
