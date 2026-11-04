"""Compile and launch user-defined FlyDSL kernels (``@flyc.jit`` host functions).

Used by the eager impl of ``flydsl_kernel_wrapper_mutation`` (launch_flydsl_kernel)
and by Inductor's Python wrapper, where each kernel becomes a FlyDSLKernelLauncher:
the counterpart of the CachingAutotuner a user Triton kernel becomes in generated
code.

A compiled FlyDSL kernel (a CompiledFunction) is specialized to the argument layouts
it was compiled for and checks nothing at launch, so compiles are cached per
argument signature: dtype, shape, stride and alignment of every tensor, the type of
every runtime scalar, the value of every constant, the device's target and the
device itself (a CompiledFunction is bound to the device of its first call), and the
module globals, environment variables and compile hints the compile reads.

Also the FlyDSL compile helpers that every PT2 compile of a FlyDSL kernel uses. Each
compile checks the flydsl version (require_flydsl, from torch._library.flydsl, where
wrap_flydsl checks it without importing Inductor); goes through a fresh private
JitFunction (fresh_private_jit), never the user's; compiles from metadata-only
placeholders (meta_placeholder), because FlyDSL compiles AND launches when given
device tensors; and runs under FLYDSL_COMPILE_LOCK with the environment FlyDSL reads
pinned (flydsl_env).
"""

import contextlib
import dis
import functools
import inspect
import operator
import os
import threading
import types
import weakref
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from typing import Any, NamedTuple

import torch
from torch._higher_order_ops.user_kernel_hop import constant_key, ConstantRules
from torch._library.flydsl import flydsl_version_ok, require_flydsl
from torch._native.utils.tensor import const_data_ptr
from torch.utils._ordered_set import OrderedSet


__all__ = [
    "FLYDSL_COMPILE_LOCK",
    "FlyDSLKernelLauncher",
    "compile_host_fn",
    "device_arch",
    "flydsl_env",
    "flydsl_version_ok",
    "fresh_private_jit",
    "launch_flydsl_kernel",
    "meta_placeholder",
    "require_flydsl",
]


# Every FlyDSL compile, trace and export PT2 starts runs under this lock: FlyDSL's
# compiler is not known to be thread-safe, and flydsl_env edits os.environ.
FLYDSL_COMPILE_LOCK = threading.RLock()


@functools.cache
def _gcn_arch(index: int) -> str:
    return torch.cuda.get_device_properties(index).gcnArchName.split(":")[0]


def device_arch(device: torch.device | int | None = None) -> str:
    """The FlyDSL target of a GPU (default: the current one): its gcnArchName without
    the feature flags, "gfx950" for "gfx950:sramecc+:xnack-"."""
    from torch.cuda._utils import _get_device_index

    return _gcn_arch(_get_device_index(device, optional=True))


def fresh_private_jit(fn: Any) -> Any:
    """A new private JitFunction for user JitFunction ``fn``, which PT2 compiles
    instead of ``fn``: a copy of the user's function as it was before FlyDSL's AST
    rewrite (``fn._original_func``, which has no annotations) with the annotations of
    ``fn.func``, so FlyDSL classifies the parameters as it does for ``fn``.

    A new one for every compile: FlyDSL checks the module globals a JitFunction uses
    against their values at its first compile and raises when a script has rebound
    one (even a name the kernel only uses as an attribute, like ``x`` in
    ``fx.thread_idx.x``), and serves a second compile with the same arguments from
    its cache, bound to the device of the first launch. ``fn`` itself is never
    compiled by PT2, whose compile cache keys on the values of the globals the kernel
    does read instead (launch_flydsl_kernel)."""
    import flydsl.compiler as flyc

    original = fn._original_func
    copy = types.FunctionType(
        original.__code__,
        original.__globals__,
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    copy.__annotations__ = dict(fn.func.__annotations__)
    copy.__kwdefaults__ = original.__kwdefaults__
    copy.__qualname__ = original.__qualname__
    copy.__module__ = original.__module__
    private = flyc.jit(copy)
    # Hints set on fn (flyc.compile[hints](fn, ...) keeps them) apply to PT2's compiles.
    private.compile_hints.update(fn.compile_hints)
    return private


def _rocm_home() -> str | None:
    rocm = os.environ.get("ROCM_PATH") or os.environ.get("ROCM_HOME")
    if rocm:
        return rocm
    from torch.utils.cpp_extension import ROCM_HOME

    return ROCM_HOME


def _set_env(values: Mapping[str, str | None]) -> None:
    for name, value in values.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


@contextlib.contextmanager
def flydsl_env(arch: str, *, for_export: bool = False) -> Iterator[None]:
    """Pin what FlyDSL reads from the environment when it compiles: the target
    ``arch`` (unset, FlyDSL asks rocm_agent_enumerator for the machine's first GPU or
    silently targets gfx942), no disk cache (it can return stale kernels across
    processes), and neither COMPILE_ONLY nor FLYDSL_RUNTIME_RUN_ONLY (with either, a
    compile returns nothing). ``for_export``: also put the ROCm LLVM tools first on
    PATH, where CompiledFunction.export_to_c looks for a linker and objcopy. Restores
    the environment on exit. Enter it under FLYDSL_COMPILE_LOCK: os.environ is
    process-wide."""
    overrides: dict[str, str | None] = {
        "ARCH": arch,
        "FLYDSL_GPU_ARCH": arch,
        "FLYDSL_RUNTIME_ENABLE_CACHE": "0",
        "COMPILE_ONLY": None,
        "FLYDSL_RUNTIME_RUN_ONLY": None,
    }
    if for_export:
        rocm = _rocm_home()
        if rocm is not None:
            llvm_bin = os.path.join(rocm, "llvm", "bin")
            path = os.environ.get("PATH")
            overrides["PATH"] = os.pathsep.join((llvm_bin, path)) if path else llvm_bin
    saved = {name: os.environ.get(name) for name in overrides}
    try:
        _set_env(overrides)
        yield
    finally:
        _set_env(saved)


def meta_placeholder(
    dtype: torch.dtype,
    sizes: Sequence[int],
    strides: Sequence[int],
    align: int,
    dynamic_sizes: Collection[int] = (),
    dynamic_strides: Collection[int] = (),
) -> Any:
    """The compile value of a tensor argument: a FlyDSL tensor argument for a meta
    tensor with ``sizes`` and ``strides`` whose data pointer is aligned to ``align``
    bytes. The layout is static in the compiled kernel except for the dims in
    ``dynamic_sizes`` / ``dynamic_strides`` (whose entries in ``sizes`` / ``strides``
    are hints). A meta tensor has no data, so compiling from it launches nothing."""
    import flydsl.compiler as flyc

    placeholder = flyc.from_torch_tensor(
        torch.empty_strided(sizes, strides, dtype=dtype, device="meta"),
        assumed_align=align,
    )
    if dynamic_sizes:
        placeholder = placeholder.mark_shape_dynamic(list(dynamic_sizes))
    if dynamic_strides:
        placeholder = placeholder.mark_stride_dynamic(list(dynamic_strides))
    return placeholder


def compile_host_fn(fn: Any, compile_args: Mapping[str, Any], arch: str) -> Any:
    """Compile user host function ``fn`` for target ``arch`` and ``compile_args``
    (the compile value of every parameter by name: meta placeholders for tensors,
    Python scalars, Constexpr values, a stream) through a fresh private JitFunction,
    under FLYDSL_COMPILE_LOCK and flydsl_env(arch). Returns the CompiledFunction; the
    kernel is not launched (a host function without tensor arguments, which FlyDSL
    would launch, is refused)."""
    from torch._higher_order_ops.flydsl_kernel_wrap import (
        FlyDSLKernel,
        get_host_fn_info,
    )

    require_flydsl()
    import flydsl.compiler as flyc
    from flydsl.compiler.jit_function import _is_compile_placeholder

    info = get_host_fn_info(fn)
    if not any(_is_compile_placeholder(v) for v in compile_args.values()):
        raise ValueError(
            f"FlyDSL kernel {FlyDSLKernel(fn, ()).name}: host functions without a "
            "tensor argument are not supported (FlyDSL launches one when it compiles "
            "it, and PT2 compiles a kernel before launching it)"
        )
    # In signature order, as FlyDSL binds them; keyword-only parameters by name.
    args = [compile_args[p] for p in info.params if p not in info.kwonly_params]
    kwargs = {p: compile_args[p] for p in info.params if p in info.kwonly_params}
    with FLYDSL_COMPILE_LOCK, flydsl_env(arch):
        return flyc.compile(fresh_private_jit(fn), *args, **kwargs)


class _TensorSpec(NamedTuple):
    """The layout a tensor argument is compiled for (meta_placeholder's arguments)."""

    dtype: torch.dtype
    sizes: tuple[int, ...]
    strides: tuple[int, ...]
    align: int


def _loaded_globals(code: types.CodeType) -> Iterator[str]:
    # The names code (and the functions and comprehensions defined in it) loads as
    # globals; attribute names (x in fx.thread_idx.x) are not among them.
    for instruction in dis.get_instructions(code):
        if instruction.opname in ("LOAD_GLOBAL", "LOAD_NAME"):
            yield instruction.argval
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            yield from _loaded_globals(const)


def _global_refs(fn: Any) -> tuple[tuple[str, dict[str, Any]], ...]:
    """The module globals user JitFunction ``fn`` loads, directly or through the
    kernels, host functions and Python helpers it calls (the ones FlyDSL's own globals
    check follows), as (name, module globals) pairs."""
    from flydsl.compiler.jit_function import (
        _get_underlying_func,
        _is_user_function,
        JitFunction,
    )
    from flydsl.compiler.kernel_function import KernelFunction

    root = fn._original_func
    try:
        root_file = inspect.getfile(root)
    except (TypeError, OSError):
        root_file = ""
    refs: dict[tuple[str, int], tuple[str, dict[str, Any]]] = {}
    seen: OrderedSet[int] = OrderedSet()

    def visit(value: Any) -> None:
        func = _get_underlying_func(value)
        if func is None or id(func) in seen:
            return
        if not isinstance(value, (JitFunction, KernelFunction)) and not (
            _is_user_function(func, root_file)
        ):
            return
        seen.add(id(func))
        module_globals = func.__globals__
        for name in _loaded_globals(func.__code__):
            # Also a name that is not a global yet (a builtin, or one defined later).
            refs.setdefault((name, id(module_globals)), (name, module_globals))
            if name in module_globals:
                visit(module_globals[name])
        for cell in func.__closure__ or ():
            try:
                value = cell.cell_contents
            except ValueError:  # an empty cell
                continue
            visit(value)

    visit(fn)
    return tuple(refs.values())


_SIMPLE_TYPES = (bool, int, float, str, bytes, type(None))


def _is_simple(value: object) -> bool:
    # A constant FlyDSL bakes into the kernel by value.
    if type(value) is tuple:
        return all(_is_simple(v) for v in value)
    return type(value) in _SIMPLE_TYPES


class _Keep:
    """Keeps a value alive inside a key without taking part in the key's equality."""

    __slots__ = ("value",)

    def __init__(self, value: object) -> None:
        self.value = value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Keep)

    def __hash__(self) -> int:
        return 0


def _global_value_key(value: object) -> object:
    # Simple constants (and tuples of them) by value; anything else (a kernel, a
    # module, an object) by identity, kept alive so that its id is not reused.
    if type(value) is tuple:
        return (tuple, tuple(_global_value_key(v) for v in value))
    if type(value) in _SIMPLE_TYPES:
        return constant_key(value, ConstantRules())
    return ("id", id(value), _Keep(value))


class _GlobalsSnapshot:
    """The module globals a user JitFunction reads (_global_refs: names and module
    globals dicts), their values, and the key of those values. Immutable."""

    def __init__(
        self,
        names: tuple[str, ...],
        dicts: tuple[dict[str, Any], ...],
        values: tuple[object, ...],
    ) -> None:
        self.names = names
        self.dicts = dicts
        self.values = values
        self.key: tuple[tuple[str, object], ...] = tuple(
            zip(names, map(_global_value_key, values))
        )

    def current_values(self) -> tuple[object, ...]:
        # A name that is not a global (yet) reads as None.
        return tuple(map(dict.get, self.dicts, self.names))

    @classmethod
    def discover(cls, fn: Any) -> "_GlobalsSnapshot":
        refs = _global_refs(fn)
        names = tuple(name for name, _ in refs)
        dicts = tuple(module_globals for _, module_globals in refs)
        return cls(names, dicts, tuple(map(dict.get, dicts, names)))


# User JitFunction -> its latest _GlobalsSnapshot.
_globals_snapshots: "weakref.WeakKeyDictionary[Any, _GlobalsSnapshot]" = (
    weakref.WeakKeyDictionary()
)


def _globals_key(fn: Any) -> tuple[tuple[str, object], ...]:
    """The values of the module globals user JitFunction ``fn`` reads, as a key: a
    launch after one is rebound to another value compiles anew (FlyDSL's own check
    raises instead, and also counts attribute names such as x in fx.thread_idx.x).
    Cheap when none was rebound: one identity check per global."""
    snapshot = _globals_snapshots.get(fn)
    if snapshot is None:
        snapshot = _globals_snapshots[fn] = _GlobalsSnapshot.discover(fn)
        return snapshot.key
    values = snapshot.current_values()
    if not all(map(operator.is_, values, snapshot.values)):
        changed = [
            v
            for new, old in zip(values, snapshot.values)
            if new is not old
            for v in (new, old)
        ]
        if all(map(_is_simple, changed)):
            snapshot = _GlobalsSnapshot(snapshot.names, snapshot.dicts, values)
        else:
            # Discovered anew: a rebound kernel or helper may read other globals.
            snapshot = _GlobalsSnapshot.discover(fn)
        _globals_snapshots[fn] = snapshot
    return snapshot.key


@functools.cache
def _key_readers() -> tuple[
    Callable[[], tuple[object, ...]], Callable[[], dict[str, Any]]
]:
    """Readers of what else a FlyDSL compile depends on: the environment variables
    that change it (FlyDSL's own reader, or a faster equivalent over os.environ's
    encoded mapping, which os.environ updates in place; its values stay bytes) and
    the calling thread's compile hints."""
    from flydsl.compiler import jit_function
    from flydsl.compiler.kernel_function import CompilationContext

    data = getattr(os.environ, "_data", None)
    if not isinstance(data, dict):
        return jit_function._cache_invalidating_env_values, (
            CompilationContext.get_compile_hints
        )
    names = tuple(map(os.fsencode, jit_function._CACHE_INVALIDATING_ENV_VARS))
    return lambda: tuple(map(data.get, names)), CompilationContext.get_compile_hints


def _tensor_alignment(value: torch.Tensor) -> int:
    """The alignment a tensor argument is compiled for, by CuTeDSL's rule
    (torch._native.cutedsl.launch.supported_alignment): the largest power of two up
    to 16 that divides the data pointer, at least the element size."""
    ptr = const_data_ptr(value)
    align = 16
    while align > value.element_size() and ptr % align:
        align //= 2
    return align


def _check_int_range(
    fn: Callable[..., Any], name: str, value: Any, annotation: Any
) -> None:
    """Refuse an int runtime scalar that does not fit the integer type of its
    parameter (``annotation``, the FlyDSL Numeric type; an unannotated int is an
    Int32): FlyDSL would wrap it like a C cast."""
    width = 32 if annotation is None else getattr(annotation, "width", 0)
    if not (
        isinstance(value, int)
        and not isinstance(value, bool)
        and (annotation is None or getattr(annotation, "is_integer", False))
        and width > 1
    ):
        return
    signed = annotation is None or bool(getattr(annotation, "signed", True))
    low = -(1 << (width - 1)) if signed else 0
    high = (1 << (width - 1)) - 1 if signed else (1 << width) - 1
    if not low <= value <= high:
        from torch._higher_order_ops.flydsl_kernel_wrap import FlyDSLKernel

        type_name = "Int32" if annotation is None else annotation.__name__
        raise ValueError(
            f"FlyDSL kernel {FlyDSLKernel(fn, ()).name}: {name}={value} does not "
            f"fit {type_name} [{low}, {high}], and FlyDSL would wrap it"
        )


def _launch_key_extras(fn: Any) -> list[tuple[str, object]]:
    """The launch-key entries for what FlyDSL's own cache key holds besides the
    arguments and the target: the module globals the kernel reads, the environment
    variables that change a compile, and the compile hints (fn's, which
    fresh_private_jit copies, merged with the calling thread's)."""
    env, thread_hints = _key_readers()
    extras: list[tuple[str, object]] = [("globals", _globals_key(fn)), ("env", env())]
    if fn.compile_hints or thread_hints():
        hints = fn._effective_compile_hints()
        extras.append(
            (
                "hints",
                tuple(
                    sorted(
                        (hint, type(v).__module__, type(v).__qualname__, repr(v))
                        for hint, v in hints.items()
                    )
                ),
            )
        )
    return extras


_HostFn = Callable[..., Any]
_Signature = tuple[Any, ...]
# host function -> {argument signature -> compiled function}
_compiled: "weakref.WeakKeyDictionary[_HostFn, dict[_Signature, Any]]" = (
    weakref.WeakKeyDictionary()
)


def _launch_device(fn: Callable[..., Any], args: Mapping[str, Any]) -> torch.device:
    """The CUDA device of the tensor arguments (they must share one), else the
    current device."""
    devices = list(
        dict.fromkeys(v.device for v in args.values() if isinstance(v, torch.Tensor))
    )
    if not devices:
        return torch.device("cuda", torch.cuda.current_device())
    if len(devices) > 1 or devices[0].type != "cuda":
        from torch._higher_order_ops.flydsl_kernel_wrap import FlyDSLKernel

        raise ValueError(
            f"FlyDSL kernel {FlyDSLKernel(fn, ()).name}: tensor arguments must be on "
            f"one CUDA device, got {', '.join(map(str, devices))}"
        )
    return devices[0]


def launch_flydsl_kernel(
    fn: Any,
    args: Mapping[str, Any],
    stream: int | None = None,
) -> None:
    """Launch host function ``fn`` with ``args`` (every parameter but the stream,
    by name; missing ones take their defaults) on ``stream``, a raw HIP stream
    (default: the current stream of the tensors' device), compiling it once per
    argument signature (see the module docstring)."""
    import flydsl.expr as fx

    from torch._higher_order_ops.flydsl_kernel_wrap import (
        _constant_key,
        get_host_fn_info,
        python_scalar,
    )

    info = get_host_fn_info(fn)
    device = _launch_device(fn, args)
    # Everything runs under the tensors' device: a CompiledFunction is bound to the
    # device current at its first call.
    with torch.cuda.device(device):
        if stream is None:
            stream = torch._C._cuda_getCurrentRawStream(device.index)
        arch = device_arch(device)

        # The device too: a CompiledFunction is bound to the device of its first call.
        key: list[object] = [(arch, device.index)]
        compile_args: dict[str, Any] = {}
        call_args: list[Any] = []
        for name in info.params:
            if name == info.stream_param:
                # A stream is not part of a specialization: compile with the null
                # stream, call with the raw one.
                compile_value: Any = fx.Stream(None)
                value: Any = stream
                key.append(name)
            else:
                value = args[name] if name in args else info.defaults[name]
                if isinstance(value, torch.Tensor):
                    align = _tensor_alignment(value)
                    key.append((name, value.dtype, value.shape, value.stride(), align))
                    # Compiled from a meta placeholder with the tensor's layout (made
                    # on a cache miss); called with the tensor itself.
                    compile_value = _TensorSpec(
                        value.dtype, tuple(value.shape), tuple(value.stride()), align
                    )
                elif info.is_runtime_scalar(name, value):
                    # FlyDSL converts only exact bool/int/float (not an IntEnum or a
                    # numpy float) and takes its Numeric values as Python ones.
                    value = compile_value = python_scalar(value)
                    _check_int_range(fn, name, value, info.numeric_params.get(name))
                    key.append((name, info.numeric_params.get(name, type(value))))
                else:
                    compile_value = value
                    key.append((name, _constant_key(value)))
            compile_args[name] = compile_value
            # A CompiledFunction takes every parameter positionally, in signature
            # order, Constexpr ones included (it ignores their values).
            call_args.append(value)
        key += _launch_key_extras(fn)

        per_fn = _compiled.setdefault(fn, {})
        compiled = per_fn.get(tuple(key))
        if compiled is None:
            with FLYDSL_COMPILE_LOCK:
                compiled = per_fn.get(tuple(key))
                if compiled is None:
                    placeholders = {
                        name: meta_placeholder(*value)
                        if isinstance(value, _TensorSpec)
                        else value
                        for name, value in compile_args.items()
                    }
                    compiled = compile_host_fn(fn, placeholders, arch)
                    per_fn[tuple(key)] = compiled
        compiled(*call_args)


class FlyDSLKernelLauncher:
    """One user FlyDSL kernel in Inductor's generated Python wrapper.

    ``run(*args, stream=...)`` takes the kernel arguments (``arg_names``: tensors and
    runtime scalars, in host-function signature order) positionally; the constants
    were bound when the wrapper was generated. It launches through
    launch_flydsl_kernel, which compiles each argument signature once.
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
        launch_flydsl_kernel(
            self.fn,
            {**dict(zip(self.arg_names, args)), **self.constant_args},
            stream,
        )
