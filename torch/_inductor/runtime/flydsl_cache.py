"""Helpers for routing FlyDSL JIT artifacts through TorchInductor's cache."""

from __future__ import annotations

import contextlib
import os
import threading
from pathlib import Path
from typing import Any, TYPE_CHECKING

from torch._inductor.runtime.cache_dir_utils import cache_dir


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence


# Serialize cold compiles process-wide; warm cache hits bypass this lock.
_compiled_cache_lock = threading.Lock()


class LazyFlyDSLLauncher:
    """Build generated FlyDSL JIT launchers once per process and device."""

    def __init__(self, factory: Callable[[], Any]) -> None:
        self._factory = factory
        self._launchers: dict[tuple[int, str | None, int | None], Any] = {}
        self._lock = threading.Lock()

    def get(self, device: Any = None) -> Any:
        device = getattr(device, "device", device)
        device_type = getattr(device, "type", None)
        device_index = getattr(device, "index", None)
        if device_type == "cuda" and device_index is None:
            import torch

            device_index = torch.cuda.current_device()
        key = (os.getpid(), device_type, device_index)
        launcher = self._launchers.get(key)
        if launcher is None:
            with self._lock:
                launcher = self._launchers.get(key)
                if launcher is None:
                    launcher = self._factory()
                    self._launchers[key] = launcher
        return launcher


@contextlib.contextmanager
def temporary_env(updates: dict[str, str | None]):
    """Temporarily override process environment variables."""
    old_values = {key: os.environ.get(key) for key in updates}
    os.environ.update(
        {key: value for key, value in updates.items() if value is not None}
    )
    try:
        yield
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _device_gpu_arch(device: Any) -> str | None:
    """The arch to compile for, taken from the device the kernel will run on.

    Left to itself, FlyDSL finds the arch by running ``rocm_agent_enumerator``
    and silently falls back to gfx942 if that fails. It does fail in Inductor's
    subprocess workers when their inherited ``PYTHONPATH`` does not match the
    enumerator's interpreter. gfx942 then gets code it cannot run: the gfx950
    kernels either fail instruction selection or, for MX formats, target a
    chip with no MX support. The device already reports its arch, so no
    subprocess is needed. An explicit ``FLYDSL_GPU_ARCH`` still wins; returning
    None leaves the environment as it is.
    """
    if os.environ.get("FLYDSL_GPU_ARCH"):
        return None
    if getattr(device, "type", None) != "cuda":
        return None
    import torch

    arch = getattr(torch.cuda.get_device_properties(device), "gcnArchName", None)
    return str(arch).split(":", 1)[0] if arch else None


def run_cached_flydsl(
    jit_func: Any,
    *compile_args: Any,
    constexpr_param: Any,
    compiler: Callable[..., Any],
    dispatch_args: tuple[Any, ...],
    compile_args_factory: Callable[[], tuple[Any, ...]] | None = None,
    layout_key: tuple[Any, ...] = (),
) -> Any:
    """Cache a layout-dynamic FlyDSL dispatcher by constexpr param.

    `compile_args` are only read on the compile path -- a cache hit dispatches
    straight through `dispatch_args`. Building them eagerly therefore wastes
    work on every call after the first for a given param: the FlyDSL argument
    descriptors (`from_torch_tensor(...).mark_layout_dynamic()`) are
    constructed and immediately discarded. Callers that can defer the work
    should pass `compile_args_factory` instead, a zero-argument callable
    invoked only when a compile is actually needed. `layout_key` must cover
    whatever the compiled dispatcher bakes in about the other arguments.
    """
    if compile_args_factory is not None and compile_args:
        raise TypeError("pass compile_args or compile_args_factory, not both")
    device = getattr(dispatch_args[0], "device", None)
    cache_key = (
        os.getpid(),
        getattr(device, "index", None),
        constexpr_param.__cache_signature__(),
        layout_key,
    )
    compiled_cache = getattr(jit_func, "_compiled_cache", None)
    if compiled_cache is not None:
        compiled = compiled_cache.get(cache_key)
        if compiled is not None:
            compiled(*dispatch_args)
            return compiled

    dispatch_after_wait = False
    with _compiled_cache_lock:
        compiled_cache = getattr(jit_func, "_compiled_cache", None)
        if compiled_cache is None:
            compiled_cache = {}
            jit_func._compiled_cache = compiled_cache

        compiled = compiled_cache.get(cache_key)
        if compiled is None:
            # compile() executes this invocation; dispatching again would double-run.
            args = (
                compile_args_factory()
                if compile_args_factory is not None
                else compile_args
            )
            with temporary_env({"FLYDSL_GPU_ARCH": _device_gpu_arch(device)}):
                compiled = compiler(jit_func, *args)
            compiled_cache[cache_key] = compiled
        else:
            dispatch_after_wait = True

    if dispatch_after_wait:
        compiled(*dispatch_args)
    return compiled


def _cache_dir() -> Path:
    return Path(cache_dir()) / "flydsl_compile_cache"


def configure_flydsl_cache_dir() -> str:
    """Route FlyDSL's disk cache through TorchInductor's cache root by default.

    FlyDSL has its own disk cache controlled by ``FLYDSL_RUNTIME_CACHE_DIR``.
    Inductor-generated kernels should participate in Inductor cache cleanup and
    subprocess warming, so default FlyDSL to an Inductor-owned subdirectory --
    but respect an explicit ``FLYDSL_RUNTIME_CACHE_DIR`` the user already set.
    """
    existing = os.environ.get("FLYDSL_RUNTIME_CACHE_DIR")
    if existing is not None:
        return existing
    resolved = str(_cache_dir())
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = resolved
    return resolved


def run_flydsl_template(
    jit_func: Any,
    dispatch_args: tuple[Any, ...],
    *,
    compiler: Callable[..., Any],
    tensor_converter: Callable[[Any], Any],
) -> Any:
    """Compile or dispatch a generated FlyDSL template through Inductor's cache."""
    if len(dispatch_args) < 2:
        raise ValueError("FlyDSL template dispatch requires tensors and a stream")

    configure_flydsl_cache_dir()
    tensor_args = dispatch_args[:-1]
    stream = dispatch_args[-1]
    flydsl_compile_only = os.environ.get("FLYDSL_COMPILE_ONLY") == "1"
    compile_only = os.environ.get("COMPILE_ONLY", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if compile_only != flydsl_compile_only:
        raise RuntimeError(
            "COMPILE_ONLY and FLYDSL_COMPILE_ONLY must only be enabled together "
            "by the FlyDSL precompile path"
        )

    constexpr_param = tensor_converter(tensor_args[0])

    def make_compile_args() -> tuple[Any, ...]:
        return (
            constexpr_param,
            *(tensor_converter(tensor) for tensor in tensor_args[1:]),
            stream,
        )

    if flydsl_compile_only:
        return compiler(jit_func, *make_compile_args())
    return run_cached_flydsl(
        jit_func,
        constexpr_param=constexpr_param,
        compiler=compiler,
        dispatch_args=dispatch_args,
        compile_args_factory=make_compile_args,
        layout_key=tuple(
            _tensor_layout_key(arg, tensor_converter is flydsl_runtime_layout_arg)
            for arg in tensor_args[1:]
        ),
    )


def _tensor_layout_key(arg: Any, runtime_layout: bool) -> tuple[Any, ...]:
    """What a compiled kernel fixes about a tensor argument: under a runtime
    layout its first unit-stride dim, otherwise its whole layout."""
    if not callable(getattr(arg, "stride", None)):
        return (type(arg),)
    tensor = arg
    stride = tuple(tensor.stride())
    if runtime_layout and 1 in stride:
        return (tensor.dtype, len(stride), stride.index(1))
    return (tensor.dtype, tuple(tensor.shape), stride)


class FlyDSLHostLauncher:
    """Run a launcher that packs its own arguments and compiles on its first
    call, such as the split backward's. That call compiles for the device's
    arch into Inductor's FlyDSL cache; ``precompile`` only warms the cache."""

    def __init__(self, launch: Callable[..., Any]) -> None:
        self._launch = launch
        self._compiled = False

    def __call__(self, *args: Any, device: Any, **kwargs: Any) -> Any:
        if self._compiled:
            return self._launch(*args, **kwargs)
        configure_flydsl_cache_dir()
        with (
            _compiled_cache_lock,
            temporary_env({"FLYDSL_GPU_ARCH": _device_gpu_arch(device)}),
        ):
            result = self._launch(*args, **kwargs)
        self._compiled = True
        return result

    def precompile(
        self, *args: Any, flydsl_gpu_arch: str | None = None, **kwargs: Any
    ) -> None:
        updates: dict[str, str | None] = {
            "COMPILE_ONLY": "1",
            "FLYDSL_COMPILE_ONLY": "1",
            "FLYDSL_GPU_ARCH": flydsl_gpu_arch,
        }
        configure_flydsl_cache_dir()
        with _compiled_cache_lock, temporary_env(updates):
            self._launch.compile(*args, **kwargs)  # type: ignore[attr-defined]


def fill_flydsl_capture_table(
    launchers: Sequence[LazyFlyDSLLauncher],
    slots: int,
    table: Any,
    captures: Sequence[Any],
    stream: Any,
    *,
    compiler: Callable[..., Any],
    tensor_converter: Callable[[Any], Any],
) -> None:
    """Write capture base addresses to ``table``; launcher ``i`` takes
    ``(table, *captures[i * slots:][:slots], stream)`` padded with ``table``."""
    for first, launcher in zip(range(0, len(captures), slots), launchers):
        chunk = tuple(captures[first : first + slots])
        run_flydsl_template(
            launcher.get(table.device),
            (table, *chunk, *(table,) * (slots - len(chunk)), stream),
            compiler=compiler,
            tensor_converter=tensor_converter,
        )


def precompile_flydsl_template(
    kernel: Callable[..., Any],
    tensor_names: Sequence[str],
    shapes: Mapping[str, Sequence[int]],
    strides: Mapping[str, Sequence[int]],
    dtypes: Mapping[str, str],
    flydsl_gpu_arch: str | None = None,
) -> None:
    """Warm a generated FlyDSL template from its ordered tensor argument names."""
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode

    updates: dict[str, str | None] = {
        "COMPILE_ONLY": "1",
        "FLYDSL_COMPILE_ONLY": "1",
    }
    if flydsl_gpu_arch is not None:
        updates["FLYDSL_GPU_ARCH"] = flydsl_gpu_arch

    names = tuple(dict.fromkeys((*tensor_names, "output")))
    with _compiled_cache_lock, temporary_env(updates), FakeTensorMode():
        tensors = {
            name: torch.empty_strided(
                tuple(shapes[name]),
                tuple(strides[name]),
                dtype=getattr(torch, dtypes[name]),
                device="cpu",
            )
            for name in names
        }
        kernel(
            *(tensors[name] for name in tensor_names),
            tensors["output"],
            stream=0,
        )


def flydsl_tensor_arg(tensor: Any) -> Any:
    import flydsl.compiler as flyc

    return flyc.from_torch_tensor(tensor).mark_layout_dynamic()


def flydsl_runtime_layout_arg(tensor: Any) -> Any:
    """Pass ``tensor`` with a runtime layout, so that one compiled kernel takes
    every size; a tensor without a unit stride keeps its static layout."""
    import flydsl.compiler as flyc

    arg = flyc.from_torch_tensor(tensor)
    return arg.mark_layout_dynamic() if 1 in tensor.stride() else arg


def precompile_flydsl(
    kernel: Callable[..., Any],
    shapes: Mapping[str, Sequence[int]],
    strides: Mapping[str, Sequence[int]],
    dtypes: Mapping[str, str],
    flydsl_gpu_arch: str | None = None,
) -> None:
    """Warm the disk cache without allocating tensors or launching a kernel."""
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode

    updates: dict[str, str | None] = {"COMPILE_ONLY": "1"}
    if flydsl_gpu_arch is not None:
        updates["FLYDSL_GPU_ARCH"] = flydsl_gpu_arch
    # Cold launches must not observe FlyDSL's process-global compile-only mode.
    # Warm launches call the cached dispatcher directly and do not read it.
    with _compiled_cache_lock, temporary_env(updates), FakeTensorMode():
        tensors = {
            name: torch.empty_strided(
                tuple(shape),
                tuple(strides[name]),
                dtype=getattr(torch, dtypes[name]),
                device="cpu",
            )
            for name, shape in shapes.items()
        }
        configure_flydsl_cache_dir()
        kernel(**tensors, stream=0, compile_only=True)
