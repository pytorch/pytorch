"""Compile and launch user-defined CuTeDSL kernels (``@cute.jit`` host functions).

Used by the eager impl of ``cutedsl_kernel_wrapper_mutation`` and by Inductor's
Python wrapper, where each kernel becomes a CuteDSLKernelLauncher: the counterpart
of the CachingAutotuner a user Triton kernel becomes in generated code.

A compiled CuTeDSL kernel is specialized to the argument layouts it was compiled for
and checks nothing at launch, so compiles are cached per argument signature: dtype,
shape, stride and alignment of every tensor, the annotation of every runtime scalar,
the value of every constant.
"""

import weakref
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch


# host function -> {argument signature -> compiled function}
_compiled: weakref.WeakKeyDictionary[
    Callable[..., Any], dict[tuple[Any, ...], Any]
] = weakref.WeakKeyDictionary()


def _launch_device(fn: Callable[..., Any], args: Mapping[str, Any]) -> torch.device:
    """The CUDA device of the tensor arguments (they must share one), else the
    current device."""
    devices = list(
        dict.fromkeys(v.device for v in args.values() if isinstance(v, torch.Tensor))
    )
    if not devices:
        return torch.device("cuda", torch.cuda.current_device())
    if len(devices) > 1 or devices[0].type != "cuda":
        raise ValueError(
            f"CuTeDSL kernel {getattr(fn, '__name__', fn)}: tensor arguments must be "
            f"on one CUDA device, got {', '.join(map(str, devices))}"
        )
    return devices[0]


def launch_cutedsl_kernel(
    fn: Callable[..., Any], args: Mapping[str, Any], stream: int | None = None
) -> None:
    """Launch host function ``fn`` with ``args`` (every parameter but the stream,
    by name; missing ones take their defaults) on ``stream``, a raw CUDA stream
    (default: the current stream of the tensors' device)."""
    import cuda.bindings.driver as cuda  # pyrefly: ignore[missing-import]
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    from torch._higher_order_ops.cutedsl_kernel_wrap import (
        get_host_fn_info,
        python_scalar,
    )
    from torch._higher_order_ops.user_kernel_hop import constant_key
    from torch._inductor.codegen.cutedsl.compile_lock import CUTEDSL_COMPILE_LOCK
    from torch._native.cutedsl.launch import supported_alignment

    info = get_host_fn_info(fn)
    device = _launch_device(fn, args)
    # Everything runs under the tensors' device: DLPack export requires it.
    with torch.cuda.device(device):
        if stream is None:
            stream = torch._C._cuda_getCurrentRawStream(device.index)
        cu_stream = cuda.CUstream(stream)

        key: list[object] = [torch.cuda.get_device_capability(device)]
        compile_args: dict[str, Any] = {}
        call_args: list[Any] = []
        for name in info.params:
            if name == info.stream_param:
                value: Any = cu_stream
                key.append(name)
            else:
                value = args[name] if name in args else info.defaults[name]
                if isinstance(value, torch.Tensor):
                    align = supported_alignment(value, 16)
                    key.append((name, value.dtype, value.shape, value.stride(), align))
                    value = from_dlpack(value.detach(), assumed_align=align)
                elif info.is_runtime_scalar(name, value):
                    # The DSL converts only exact bool/int/float (not an IntEnum or
                    # a numpy float) and takes a cutlass Numeric value as a Python one.
                    value = python_scalar(value)
                    key.append((name, info.numeric_params.get(name, type(value))))
                else:
                    key.append((name, constant_key(value)))
            compile_args[name] = value
            if name not in info.constexpr_params:
                call_args.append(value)

        per_fn = _compiled.setdefault(fn, {})
        compiled = per_fn.get(tuple(key))
        if compiled is None:
            with CUTEDSL_COMPILE_LOCK:
                compiled = per_fn.get(tuple(key))
                if compiled is None:
                    compiled = cute.compile(fn, **compile_args)
                    per_fn[tuple(key)] = compiled
        compiled(*call_args)


class CuteDSLKernelLauncher:
    """One user CuTeDSL kernel in Inductor's generated Python wrapper.

    ``run(*args, stream=...)`` takes the kernel arguments (``arg_names``: tensors and
    runtime scalars, in host-function signature order) positionally; the constants
    were bound when the wrapper was generated.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        arg_names: Sequence[str],
        constant_args: Mapping[str, Any],
    ) -> None:
        self.fn = fn
        self.arg_names = tuple(arg_names)
        self.constant_args = dict(constant_args)

    def run(self, *args: Any, stream: int) -> None:
        if len(args) != len(self.arg_names):
            raise AssertionError(
                f"expected {len(self.arg_names)} arguments ({self.arg_names}), "
                f"got {len(args)}"
            )
        launch_cutedsl_kernel(
            self.fn, {**dict(zip(self.arg_names, args)), **self.constant_args}, stream
        )
