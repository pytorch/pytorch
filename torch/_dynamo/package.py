"""
This module provides the infrastructure for creating and managing compile package
for torch.compile. We mainly have two abstractions here:
  - CompilePackage: Overarching data structure for store and lookup a list of compiled codes.
  - CodeCacheEntry: Data structure for a single code being compiled by torch.compile.
The caching behavior is always under user control explicitly so that a stronger guarantee can
be provided about cache hit for a specific compiled model. Users can load the compile package
from a different process or host.
"""

import abc
import ast
import contextlib
import dataclasses
import functools
import hashlib
import importlib
import inspect
import itertools
import json
import logging
import os
import pickle
import platform
import shutil
import sys
import threading
import types
import weakref
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from contextlib import nullcontext
from typing import Any, NewType, Optional, TYPE_CHECKING, Union
from typing_extensions import Never

import torch
from torch._dynamo.exc import PackageError
from torch._dynamo.graph_utils import _graph_device_type
from torch.utils.weak import WeakIdKeyDictionary

from .bytecode_transformation import (
    COMPILED_FN_PREFIX,
    get_code_keys,
    is_compiled_fn_name,
)
from .types import FrameAction, FrameExecStrategy
from .utils import CleanupHook, counters, dynamo_timed, increment_frame


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from .guards import GuardManagerWrapper, GuardsState


_CODE_CACHE = WeakIdKeyDictionary()

# code object -> the live CompilePackages that installed entries on it, and the
# live CompilePackages that skip_code()d it. Weak on both sides: a package
# dropped without unloading must not block a later one. A serving process can
# load and unload from several threads, so the complete operations and registry
# mutations are serialized.
_PRECOMPILE_INSTALLERS: WeakIdKeyDictionary = WeakIdKeyDictionary()
_SKIP_INSTALLERS: WeakIdKeyDictionary = WeakIdKeyDictionary()
# When both are needed, acquire the operation lock before the registry lock.
_PACKAGE_INSTALL_LOCK = threading.RLock()
_INSTALLER_REGISTRY_LOCK = threading.Lock()


def _register_installer(
    registry: WeakIdKeyDictionary, code: types.CodeType, package: "CompilePackage"
) -> None:
    with _INSTALLER_REGISTRY_LOCK:
        registry.setdefault(code, weakref.WeakSet()).add(package)


@dataclasses.dataclass
class _SkipInstallerState:
    owners: weakref.WeakSet["CompilePackage"]
    prior_strategy: FrameExecStrategy
    generation: int


_PACKAGE_SKIP_STRATEGY = FrameExecStrategy(FrameAction.SKIP, FrameAction.DEFAULT)


# Distinguishes "the name is unbound" from "it is bound to None", so uninstall()
# can tell whether the binding it wrote is still the one there.
_ABSENT_GLOBAL = object()


@dataclasses.dataclass(frozen=True)
class _InstalledGlobal:
    """A global install() wrote into a module, and the value it wrote."""

    name: str
    value: object


def _code_cache(fn: Callable[..., Any]) -> Callable[..., Any]:
    def _(
        cls: type[Any], code: Union["SerializedCode", types.CodeType]
    ) -> Union["SerializedCode", types.CodeType]:
        if code in _CODE_CACHE:
            return _CODE_CACHE[code]
        res = fn(cls, code)
        _CODE_CACHE[code] = res
        return res

    return _


@dataclasses.dataclass(frozen=True)
class SerializedCode:
    co_argcount: int
    co_posonlyargcount: int
    co_kwonlyargcount: int
    co_nlocals: int
    co_stacksize: int
    co_flags: int
    co_code: bytes
    co_consts: tuple[Any, ...]
    co_names: tuple[str, ...]
    co_varnames: tuple[str, ...]
    co_filename: str
    co_name: str
    co_firstlineno: int
    co_cellvars: tuple[str, ...]
    co_freevars: tuple[str, ...]
    co_linetable: bytes | None = None
    co_qualname: str | None = None
    co_exceptiontable: bytes | None = None
    co_lnotab: str | None = None

    @classmethod
    @_code_cache
    def from_code_object(cls, code: types.CodeType) -> "SerializedCode":
        kwargs = {key: getattr(code, key) for key in get_code_keys()}
        kwargs["co_consts"] = tuple(
            cls.from_code_object(c) if isinstance(c, types.CodeType) else c
            for c in kwargs["co_consts"]
        )
        return cls(**kwargs)

    @classmethod
    @_code_cache
    def to_code_object(cls, serialized_code: "SerializedCode") -> types.CodeType:
        kwargs = {key: getattr(serialized_code, key) for key in get_code_keys()}
        kwargs["co_consts"] = tuple(
            cls.to_code_object(c) if isinstance(c, SerializedCode) else c
            for c in kwargs["co_consts"]
        )
        return types.CodeType(
            *kwargs.values(),
        )


@dataclasses.dataclass
class _GuardedCodeCacheEntry:
    """
    Contains the serializable information associated with a single compilation in dynamo.
    To restore an execution of compiled code, we will need to serialize the following data:
      - Dynamo bytecode for mapping Python inputs/outputs.
      - Dynamo guards.
    """

    guards_state: bytes
    dynamo_code: SerializedCode


def load_guards_state(guards_state: bytes) -> Any:
    try:
        import torch.distributed.fsdp._fully_shard._fully_shard as _fully_shard

        ctx = _fully_shard.disable_fsdp_module_new_init()
    except ImportError:
        ctx = nullcontext()  # type: ignore[assignment]
    with ctx:
        return pickle.loads(guards_state)


def load_guard_manager(
    guards_state: "GuardsState",
    target_code: types.CodeType,
    runtime_global_scope: Any,
) -> "GuardManagerWrapper":
    from .output_graph import OutputGraphCommon

    return torch._dynamo.guards.CheckFunctionManager(
        target_code,
        OutputGraphCommon(guards_state.output_graph),
        shape_code_parts=guards_state.shape_code_parts,
        runtime_global_scope=runtime_global_scope,
        guard_build_local_state=getattr(guards_state, "local_state", None),
    ).guard_manager


_BackendId = NewType("_BackendId", str)  # __compiled_fn
_FunctionId = NewType("_FunctionId", str)  # __resume_at


def _backend_ids_from_code(code: types.CodeType) -> Iterator[_BackendId]:
    for name in code.co_names:
        if is_compiled_fn_name(name):
            yield _BackendId(name)
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            yield from _backend_ids_from_code(const)


@dataclasses.dataclass(frozen=True)
class InlinedSource:
    module: str
    firstlineno: int
    lastlineno: int
    checksum: str
    content: str


@functools.cache
def _get_module_content(module: types.ModuleType) -> str:
    return inspect.getsource(module)


def _defining_module_name(code: types.CodeType) -> str | None:
    """
    The sys.modules key whose source actually contains ``code``.

    Two things make this harder than ``inspect.getmodule``. It can hand back a
    module that merely re-exports the code from a private implementation file --
    ``collections.abc`` is three lines of ``from _collections_abc import *`` --
    and hashing this code's line range against that module reads lines that are
    not there. And ``__name__`` is not necessarily importable back to the same
    object: ``_collections_abc`` sets its own ``__name__`` to "collections.abc",
    which imports to the shim instead. Load-time revalidation re-imports this
    name, so return the key, not ``__name__``.

    Real models inline through such modules constantly, so give up and skip the
    checksum rather than record one against the wrong file.
    """
    module = inspect.getmodule(code)
    if module is not None and getattr(module, "__file__", None) == code.co_filename:
        name = getattr(module, "__name__", None)
        if name is not None and sys.modules.get(name) is module:
            return name
    return _scan_sys_modules_for_file(code.co_filename)


@functools.cache
def _scan_sys_modules_for_file(filename: str) -> str | None:
    """
    Memoized because the fallback is O(len(sys.modules)) and this runs per
    inlined code object during capture, on the shared caching_precompile path.
    A hit is stable; a miss is recomputed only when a new filename shows up.
    """
    for key, candidate in list(sys.modules.items()):
        if getattr(candidate, "__file__", None) == filename:
            return key
    return None


@dataclasses.dataclass
class SourceInfo:
    inlined_sources: set[InlinedSource]

    def add_code(self, code: types.CodeType) -> None:
        module_name = _defining_module_name(code)
        if module_name is None:
            return
        module = sys.modules[module_name]
        sourcelines, firstlineno = inspect.getsourcelines(code)
        lastlineno = firstlineno + len(sourcelines)
        source = "".join(sourcelines)
        if source != "".join(_get_sourcelines(module, firstlineno, lastlineno)):
            raise AssertionError(
                f"Source mismatch for {module.__name__} "
                f"(line {firstlineno}-{lastlineno})"
            )
        self.inlined_sources.add(
            InlinedSource(
                module=module_name,
                firstlineno=firstlineno,
                lastlineno=lastlineno,
                checksum=_hash_source(source),
                content=_get_module_content(module),
            )
        )


@dataclasses.dataclass
class _DynamoCodeCacheEntry:
    """
    Contains the serializable information associated with a single code object
    in dynamo. To restore an execution of compiled code, we will need the following
    ingredients:
      1. The "original" code object, which serves as the entry point for eager
         execution, i.e. the code only executed when there's no cache entry hit.
      2. The python module name this code object belongs to, for identifying the
         enclosing global scope to inject compiled and resume functions.
      3. A list of function names that pointing to this code object. There could be
         multiple function objects pointing to the same code such as recursive functions.
      4. A list of guarded code that eval frame dispatches to.
      5. A list of imported module objects unioned from all compiled branches.
      6. A list of "backends" (compiled fx graph) unioned from all compiled branches.
      7. A string path used to access the original code object users defined.
         A code object can be accessed by "{python_module}.{function_name}.{code_source}" .
      8. A boolean flag indicating whether the function is installed to global scope.
      9. A boolean flag indicating whether the function has a compile id.
      10. Whether or not this code entry was bypassed
    """

    python_code: SerializedCode
    python_module: str
    function_names: list[_FunctionId]
    guarded_codes: list[_GuardedCodeCacheEntry]
    import_sources: dict[str, str]
    backend_ids: list[_BackendId]
    code_source: str | None
    install_to_global: bool
    has_compile_id: bool = False
    bypassed: bool = False


def _resume_global_renames(
    entries: Iterable[_DynamoCodeCacheEntry],
) -> dict[str, str]:
    """
    Pick a global name for every resume function that is unique to the code it
    names, rather than to the process that captured it.

    ``__resume_at_<offset>_<n>`` comes from a counter that restarts in every
    capture process, so two artifacts captured separately both claim, say,
    ``__resume_at_16_3``. A serving process installs both into the same module
    dict: the second one wins and the first model silently runs the second's
    continuation. Unlike ``__compiled_fn`` names, which carry a uuid, these
    names carry nothing that distinguishes the artifact.
    """
    renames: dict[str, str] = {}
    for entry in entries:
        if not entry.install_to_global:
            continue
        digest = hashlib.sha256(pickle.dumps(entry.python_code)).hexdigest()[:16]
        for name in entry.function_names:
            renames[name] = f"{name}_{digest}"
    return renames


def _rename_globals(code: types.CodeType, renames: dict[str, str]) -> types.CodeType:
    """
    Rewrite ``co_names`` so LOAD_GLOBAL follows the renamed bindings. Indices
    into ``co_names`` are preserved, so the bytecode itself is untouched.
    """
    if not renames:
        return code
    consts = tuple(
        _rename_globals(c, renames) if isinstance(c, types.CodeType) else c
        for c in code.co_consts
    )
    names = tuple(renames.get(name, name) for name in code.co_names)
    if names == code.co_names and all(
        new is old for new, old in zip(consts, code.co_consts)
    ):
        return code
    return code.replace(co_names=names, co_consts=consts)


def _lookup_code(entry: _DynamoCodeCacheEntry) -> types.CodeType:
    if len(entry.function_names) != 1:
        raise AssertionError(
            f"Expected exactly one function name, got {len(entry.function_names)}"
        )
    fn: Any = sys.modules[entry.python_module]
    parts = entry.function_names[0].split(".")
    for part in parts:
        fn = getattr(fn, part)
    if entry.code_source:
        parts = entry.code_source.split(".")
        for part in parts:
            if part.endswith("]"):
                index_begin = part.rfind("[")
                if not isinstance(index_begin, int):
                    raise AssertionError(
                        f"Expected int for index_begin, got {type(index_begin)}"
                    )
                if index_begin < 0:
                    raise AssertionError(
                        f"Expected non-negative index_begin, got {index_begin}"
                    )
                attr = getattr(fn, part[:index_begin], None)
                if attr is None:
                    raise PackageError(f"Cannot find source for code entry {entry}")
                fn = attr[ast.literal_eval(part[index_begin + 1 : -1])]
            else:
                fn = getattr(fn, part)
    else:
        raise PackageError(f"Cannot find source for code entry {entry}")
    if not isinstance(fn, types.CodeType):
        raise AssertionError(
            f"Expected CodeType, got {type(fn)} for code entry {entry}"
        )
    return fn


def _raise_resolution_error(code: types.CodeType, scope: Any) -> Never:
    raise PackageError(
        f"Cannot resolve a fully qualified name for {code}. Lookup scope: {scope}"
    )


def _get_code_source(code: types.CodeType) -> tuple[str, str]:
    """
    Given a code object, return a fully qualified name which will be used as
    a serialized handle to access the code object from the new process.
    This is normally a straightforward process, but there are some corner cases:
    1. When a function is defined with decorator, then this function will be captured
       inside a closure with the wrapper object.
    2. When a function is defined as a nested function, then the code object will be
       stored on the co_consts field of the parent code object by Python compiler.
    This function handles all of the corner cases above.
    """

    module = inspect.getmodule(code)
    if module is None:
        raise PackageError(f"Cannot find module for code {code}")

    toplevel: Any = module
    if sys.version_info >= (3, 11):
        parts = code.co_qualname.split(".")

        for part in parts:
            if not hasattr(toplevel, part):
                _raise_resolution_error(code, toplevel)
            toplevel = getattr(toplevel, part)
            if inspect.isfunction(toplevel) or inspect.ismethod(toplevel):
                break
    seen = set()

    def _find_code_source(obj: Any) -> str | None:
        nonlocal toplevel
        nonlocal seen
        if obj in seen:
            return None

        seen.add(obj)

        if inspect.iscode(obj):
            if obj is code:
                return ""

            for i, const in enumerate(obj.co_consts):
                if (res := _find_code_source(const)) is not None:
                    return f".co_consts[{i}]{res}"

        if inspect.ismethod(obj):
            if (res := _find_code_source(obj.__func__)) is not None:
                toplevel = obj
                return f".__func__{res}"

        if inspect.isfunction(obj):
            if (res := _find_code_source(obj.__code__)) is not None:
                toplevel = obj
                return f".__code__{res}"
            if obj.__closure__ is not None:
                for i, cell in enumerate(obj.__closure__):
                    try:
                        cell_contents = cell.cell_contents
                    except ValueError:
                        continue
                    if not (
                        inspect.isfunction(cell_contents)
                        or inspect.iscode(cell_contents)
                        or inspect.ismethod(cell_contents)
                    ):
                        continue
                    if (res := _find_code_source(cell_contents)) is not None:
                        toplevel = obj
                        return f".__closure__[{i}].cell_contents{res}"

        if sys.version_info < (3, 11):
            if inspect.ismodule(obj):
                for value in obj.__dict__.values():
                    if not (
                        inspect.isfunction(value)
                        or inspect.isclass(value)
                        or inspect.ismethod(value)
                    ):
                        continue
                    if (res := _find_code_source(value)) is not None:
                        return res

            if inspect.isclass(obj):
                for name in itertools.chain(obj.__dict__.keys(), dir(obj)):
                    try:
                        value = getattr(obj, name)
                    except AttributeError:
                        continue
                    if not (
                        inspect.isfunction(value)
                        or inspect.isclass(value)
                        or inspect.ismethod(value)
                    ):
                        continue
                    if (res := _find_code_source(value)) is not None:
                        if value.__name__ != name:
                            _raise_resolution_error(code, toplevel)
                        return res
        return None

    code_source = _find_code_source(toplevel)
    if code_source is None:
        _raise_resolution_error(code, toplevel)
    return toplevel.__qualname__, code_source.strip(".")


@dataclasses.dataclass(frozen=True)
class SystemInfo:
    """
    System information including Python, PyTorch, and GPU details.
    This information is used to ensure compiled artifacts can only be loaded
    with compatible system configurations.
    """

    python_version: str
    torch_version: str
    toolkit_version: str | None
    triton_version: tuple[int, int] | None
    gpu_name: str | None
    CHECK_GPUS = ("cuda", "xpu")

    @classmethod
    def current(cls) -> "SystemInfo":
        """Create a SystemInfo instance with current system information."""
        # Get GPU name if CUDA or XPU is available
        gpu_name = None
        from torch.utils._triton import get_triton_version

        gpu_name, toolkit_version = None, None
        for device_type in cls.CHECK_GPUS:
            if getattr(torch, device_type).is_available():
                try:
                    gpu_name = getattr(torch, device_type).get_device_name()
                    toolkit_version = getattr(torch.version, device_type)
                    break
                except Exception:
                    pass

        return cls(
            python_version=platform.python_version(),
            torch_version=torch.__version__,
            toolkit_version=toolkit_version,
            triton_version=get_triton_version((0, 0)),
            gpu_name=gpu_name,
        )

    def check_compatibility(
        self, other: "SystemInfo", device_type: str = "cpu"
    ) -> None:
        """
        Check if this SystemInfo is compatible with another SystemInfo.
        Raises RuntimeError if incompatible.
        """
        if self.python_version != other.python_version:
            raise RuntimeError(
                f"Compile package was created with a different Python version: {self.python_version}"
            )

        if self.torch_version != other.torch_version:
            raise RuntimeError(
                f"Compile package was created with a different PyTorch version: {self.torch_version}"
            )
        if device_type in self.CHECK_GPUS:
            if not getattr(torch, device_type).is_available():
                raise RuntimeError(f"{device_type} is not available")

            if self.toolkit_version != other.toolkit_version:
                raise RuntimeError(
                    f"Compile package was created with a different toolkit version: {self.toolkit_version}"
                )

            if (
                other.triton_version != (0, 0)
                and self.triton_version != other.triton_version
            ):
                raise RuntimeError(
                    f"Compile package was created with a different Triton version: {self.triton_version}"
                )

            # Check GPU name if CUDA/XPU was used
            if other.gpu_name is not None and self.gpu_name != other.gpu_name:
                raise RuntimeError(
                    f"Compile package was created with different GPU: "
                    f"cached={self.gpu_name}, current={other.gpu_name}"
                )


@dataclasses.dataclass
class _DynamoCacheEntry:
    codes: list[_DynamoCodeCacheEntry]
    source_info: SourceInfo
    device_type: str
    system_info: SystemInfo = dataclasses.field(default_factory=SystemInfo.current)
    fn_name: str | None = None
    fn_first_lineno: int | None = None

    @property
    def backend_ids(self) -> set[_BackendId]:
        return {backend_id for code in self.codes for backend_id in code.backend_ids}

    def check_versions(self) -> None:
        """Check if the current system is compatible with the system used to create this cache entry."""
        current_system_info = SystemInfo.current()
        self.system_info.check_compatibility(current_system_info, self.device_type)

    def debug_info(self) -> dict[str, Any]:
        if len(self.codes) == 0:
            raise AssertionError("Expected at least one code entry")
        return {
            "num_codes": str(len(self.codes)),
            "fn_name": self.fn_name,
            "fn_first_lineno": self.fn_first_lineno,
            "device_type": self.device_type,
            "backend_ids": list(self.backend_ids),
        }


from torch.compiler._cache import (
    CacheArtifact,
    CacheArtifactFactory,
    CacheArtifactRecorder,
)


@CacheArtifactFactory.register
class PrecompileCacheArtifact(CacheArtifact):
    def populate_cache(self) -> None:
        DynamoCache._write_to_local_cache(self.content, self.key)

    @staticmethod
    def type() -> str:
        return "precompile"


@dataclasses.dataclass
class PrecompileCacheEntry:
    """
    A full cache entry for caching precompile, for a toplevel torch.compile.
    Consists of a _DynamoCacheEntry, which contains all the dynamo related contents,
    and a set of backends content. In general, the backend content here will always
    be of type precompile_context.BackendCacheArtifact
    """

    dynamo: _DynamoCacheEntry
    backends: dict[_BackendId, Any]

    @staticmethod
    def from_cache_entry(
        cache_entry: _DynamoCacheEntry, backends: dict[_BackendId, Any]
    ) -> Optional["PrecompileCacheEntry"]:
        backend_content: dict[_BackendId, Any] = {}

        for code in cache_entry.codes:
            for backend_id in code.backend_ids:
                if backend_id not in backends:
                    logger.warning("Backend not found")
                    debug_str = json.dumps(
                        {
                            "entry": cache_entry.debug_info(),
                            "missing_backend": backend_id,
                        }
                    )
                    torch._logging.trace_structured(
                        "artifact",
                        metadata_fn=lambda: {
                            "name": "dynamo_cache_bypass",
                            "encoding": "json",
                        },
                        payload_fn=lambda: debug_str,
                        expect_trace_id=False,
                    )
                    code.bypassed = True
                    break
                else:
                    backend_content[backend_id] = backends[backend_id]

        return PrecompileCacheEntry(dynamo=cache_entry, backends=backend_content)


def _hash_source(source: str) -> str:
    sha256_hash = hashlib.sha256()
    sha256_hash.update(source.encode())
    return sha256_hash.hexdigest()


def _get_sourcelines(
    m: types.ModuleType, firstlineno: int, lastlineno: int
) -> list[str]:
    return inspect.getsourcelines(m)[0][firstlineno - 1 : lastlineno - 1]


def _hash_sourcelines(m: types.ModuleType, firstlineno: int, lastlineno: int) -> str:
    return _hash_source("".join(_get_sourcelines(m, firstlineno, lastlineno)))


def _compile_frame_context(
    code: types.CodeType,
) -> contextlib.AbstractContextManager[None]:
    from torch._dynamo.convert_frame import get_compile_id, log_dynamo_start
    from torch._guards import compile_context, CompileContext

    # Each code represents a new compile frame
    # recompiles on the same frame are all saved
    # under the same cache entry, so we don't have recompile ids
    # i.e. If cold start had 0/0, 0/1, 1/0, 1/1, these would be
    # collapsed into 0/0, 1/0 on warm.
    # pyrefly: ignore [deprecated]
    @contextlib.contextmanager
    def _ctx() -> Iterator[None]:
        increment_frame()
        compile_id = get_compile_id(frame_state={})
        with (
            compile_context(CompileContext(compile_id)),
            dynamo_timed(
                "_compile.compile_inner",
                phase_name="entire_frame_compile",
                dynamo_compile_column_us="dynamo_cumulative_compile_time_us",
                # TODO: save all relevant compilation metrics
                metadata={
                    "frame_key": str(torch._dynamo.utils.curr_frame),
                    "co_name": code.co_name,
                    "co_filename": code.co_filename,
                    "co_firstlineno": code.co_firstlineno,
                },
            ),
        ):
            log_dynamo_start(code)
            yield

    return _ctx()


class CompilePackage:
    """
    CompilePackage is considered a low level component and should not be directly exposed to
    end users. It has the following interface:

    1. `CompilePackage.__init__()` which optionally takes previously serialized dynamo states.
        a. when `dynamo` argument is None, it will construct a brand new CompilePackage object.
        b. when `dynamo` argument is not None, it will load a pre-compiled dynamo state.
    2. `package.save()` which dumps the dynamo and backend states to a DynamoCacheEntry object.
    3. `package.install(backends) which will handle all the side-effectful global scope
        updates with compiled functions and resume functions.
    """

    def __init__(
        self,
        fn: Callable[..., Any] | None,
        dynamo: _DynamoCacheEntry | None = None,
        ignore_inlined_sources: bool = False,
        serialization_guard_filter_fn: Callable[[Sequence[Any]], Sequence[bool]]
        | None = None,
    ) -> None:
        self._innermost_fn = None
        self._codes: dict[types.CodeType, _DynamoCodeCacheEntry] = {}

        self._current_entry: _DynamoCodeCacheEntry | None = None
        self._installed_globals: dict[types.ModuleType, list[_InstalledGlobal]] = {}
        # Code objects we registered precompile entries on, so uninstall() can
        # clear all of them. install() covers resume functions and any frame
        # reached through code_source, not just the entry frame.
        self._installed_precompile_codes: list[types.CodeType] = []
        self._skipped_codes: list[types.CodeType] = []
        # Frames whose capture was cut short by the recompile limit. Deliberately
        # runtime-only and NOT serialized: it describes this capture session, not
        # the artifact, and it must not affect what install() serves.
        self._truncated_frames: set[str] = set()
        # A frame can enter Dynamo yet produce no guarded code for one exercised
        # variant (for example, an unsupported or empty resume path). Keep that
        # distinct from resume code that was generated but never executed.
        self._uncovered_frames: set[str] = set()
        # device_type that model compiled with.
        self._device_type = "cpu"

        # For debugging/testing purpose only.
        self._cached_backends: dict[_BackendId, Any] = {}
        self._source_info: SourceInfo = SourceInfo(inlined_sources=set())
        self._resume_codes: set[types.CodeType] = set()
        # Runtime guards stay intact; this filter applies only to the guard
        # state recorded in the package.
        self.serialization_guard_filter_fn = serialization_guard_filter_fn
        self._initialized = False
        if fn is not None:
            self.initialize(fn, dynamo, ignore_inlined_sources)
            self.uninstall()
            self.validate()

    def is_initialized(self) -> bool:
        return self._initialized

    def initialize(
        self,
        fn: Any,
        dynamo: _DynamoCacheEntry | None = None,
        ignore_inlined_sources: bool = False,
    ) -> None:
        from .eval_frame import innermost_fn

        if self._initialized:
            raise AssertionError("CompilePackage is already initialized")
        # A load that raises is retried on the SAME object -- eval_frame's
        # caching_precompile path falls back to initialize(fn, None) -- so every
        # field a load writes has to be reset here rather than trusted to still
        # hold its __init__ value.
        self._source_info = SourceInfo(inlined_sources=set())
        self._codes = {}
        self._device_type = "cpu"
        self._cached_backends = {}
        self._resume_codes = set()
        self._truncated_frames = set()
        self._uncovered_frames = set()
        self._innermost_fn = innermost_fn(fn)  # type: ignore[assignment]
        if self._innermost_fn is None:
            raise AssertionError("innermost_fn returned None")
        if dynamo is not None:
            if not isinstance(dynamo, _DynamoCacheEntry):
                raise AssertionError(f"Expected _DynamoCacheEntry, got {type(dynamo)}")
            dynamo.check_versions()
            if not ignore_inlined_sources:
                for code in dynamo.source_info.inlined_sources:
                    m = importlib.import_module(code.module)
                    checksum = _hash_sourcelines(m, code.firstlineno, code.lastlineno)
                    if checksum != code.checksum:
                        raise RuntimeError(
                            f"Source code changes detected for {code.module} (line {code.firstlineno} - line {code.lastlineno})"
                        )

                self._source_info = dynamo.source_info

            main, *codes = dynamo.codes
            self._codes = {self._innermost_fn.__code__: main}
            for code in codes:
                self._codes[SerializedCode.to_code_object(code.python_code)] = code
            # Restore what the artifact was captured with. Recomputing it from
            # whatever this process happens to recompile lets a load, one cpu
            # recompile and a re-save downgrade a cuda artifact to "cpu", which
            # silently disables every GPU check for whoever loads it next.
            # Written last, after every raise above: update_device_type only
            # widens cpu -> accelerator, so a value a FAILED load left here
            # could never be corrected and would be re-saved as this capture's.
            self._device_type = dynamo.device_type
        else:
            self._add_function(
                self._innermost_fn.__code__, self._innermost_fn.__module__
            )
        self._initialized = True

    def _add_function(
        self,
        python_code: types.CodeType,
        python_module: str,
        function_name: _FunctionId | None = None,
        code_source: str | None = None,
        install_to_global: bool = False,
    ) -> None:
        if python_code not in self._codes:
            code = _DynamoCodeCacheEntry(
                python_code=SerializedCode.from_code_object(python_code),
                python_module=python_module,
                function_names=[],
                guarded_codes=[],
                import_sources={},
                backend_ids=[],
                code_source=code_source,
                install_to_global=install_to_global,
            )
            self._codes[python_code] = code
        else:
            code = self._codes[python_code]
            if code.python_module != python_module:
                raise AssertionError(
                    f"python_module mismatch: {code.python_module} != {python_module}"
                )
            if code.install_to_global != install_to_global:
                raise AssertionError(
                    f"install_to_global mismatch: {code.install_to_global} != {install_to_global}"
                )
            if code.code_source != code_source:
                raise AssertionError(
                    f"code_source mismatch: {code.code_source} != {code_source}"
                )

        if function_name is not None:
            code.function_names.append(function_name)

    @property
    def cached_backends(self) -> dict[_BackendId, Any]:
        return self._cached_backends

    @functools.cached_property
    def source_id(self) -> str:
        if self._innermost_fn is None:
            raise AssertionError("_innermost_fn is not set")
        return CompilePackage.source_id_from_fn(self._innermost_fn)

    def _add_user_function(self, code: types.CodeType) -> None:
        function_name, code_source = _get_code_source(code)
        module = inspect.getmodule(code)
        if module is None:
            raise PackageError(f"Cannot find module for code {code}")
        self._add_function(
            code,
            module.__name__,
            function_name=_FunctionId(function_name),
            code_source=code_source,
        )

    @contextlib.contextmanager
    def code_context(self, code: types.CodeType) -> Generator[None, None, None]:
        if self._current_entry is not None:
            raise AssertionError("_current_entry is already set in code_context")

        # Sometimes user code cannot be inlined in dynamo resulting in extra user code
        # being compiled. We should record these as when they are actually invoked.
        if code not in self._codes:
            self._add_user_function(code)

        entry = self._codes[code]
        guarded_codes_before = len(entry.guarded_codes)
        self._current_entry = entry
        try:
            yield
        finally:
            entry.has_compile_id = True
            if len(entry.guarded_codes) == guarded_codes_before and not entry.bypassed:
                self._uncovered_frames.add(code.co_name)
            self._current_entry = None

    def add_guarded_code(
        self,
        guards_state: bytes,
        dynamo_code: types.CodeType,
    ) -> None:
        if self._current_entry is None:
            raise AssertionError("_current_entry is not set in add_guarded_code")
        if self._current_entry.bypassed:
            return
        guarded_code_entry = _GuardedCodeCacheEntry(
            guards_state=guards_state,
            dynamo_code=SerializedCode.from_code_object(dynamo_code),
        )
        self._current_entry.guarded_codes.append(guarded_code_entry)
        for backend_id in _backend_ids_from_code(dynamo_code):
            self._add_backend_id(backend_id)

    def add_inlined_source(self, sources: list[types.CodeType]) -> None:
        if self._current_entry is None:
            raise AssertionError("_current_entry is not set in add_inlined_source")
        if self._current_entry.bypassed:
            return
        for code in sources:
            if code in self._resume_codes:
                continue
            self._source_info.add_code(code)

    def update_device_type(self, graph: torch.fx.Graph | None) -> None:
        # One package holds every graph Dynamo compiles for this callable and
        # this single field gates all of the GPU checks in
        # SystemInfo.check_compatibility, so a cpu graph -- the scalar epilogue
        # after an .item() break, say -- must not erase an accelerator that an
        # earlier graph recorded.
        if self._device_type == "cpu":
            self._device_type = _graph_device_type(graph)

    def has_current_entry(self) -> bool:
        return self._current_entry is not None

    def mark_current_entry_truncated(self) -> None:
        """
        Record that this frame hit the recompile limit, so callers building an
        artifact can tell the capture is missing variants. Unlike bypassing, the
        variants already captured stay installable -- a truncated frame still
        serves what it covers and recompiles for the rest.

        Only the frame that hit the limit lands here, so ``truncated_frames`` is
        a LOWER BOUND: the limit also puts everything called beneath this frame
        into run-only mode, and those frames stop capturing without ever
        re-entering Dynamo to report it.
        """
        if self._current_entry is None:
            raise AssertionError(
                "_current_entry is not set in mark_current_entry_truncated"
            )
        code = self._current_entry.python_code
        self._truncated_frames.add(
            f"{code.co_name} ({code.co_filename}:{code.co_firstlineno})"
        )

    @property
    def truncated_frames(self) -> frozenset[str]:
        return frozenset(self._truncated_frames)

    @property
    def uncovered_frames(self) -> frozenset[str]:
        return frozenset(self._uncovered_frames)

    def guarded_code_count(self, code: types.CodeType) -> int:
        entry = self._codes.get(code)
        return 0 if entry is None else len(entry.guarded_codes)

    def code_objects(self) -> tuple[types.CodeType, ...]:
        return tuple(self._codes)

    def bypass_current_entry(self) -> None:
        if self._current_entry is None:
            raise AssertionError("_current_entry is not set in bypass_current_entry")
        self._current_entry.bypassed = True

    def add_resume_function(
        self,
        python_code: types.CodeType,
        python_module: str,
        function_name: str,
    ) -> None:
        self._add_function(
            python_code,
            python_module,
            function_name=_FunctionId(function_name),
            install_to_global=True,
        )
        self._resume_codes.add(python_code)

    def add_import_source(self, alias: str, module_name: str) -> None:
        if self._current_entry is None:
            raise AssertionError("_current_entry is not set in add_import_source")
        self._current_entry.import_sources[alias] = module_name

    def _add_backend_id(
        self, backend_id: _BackendId, backend: Any | None = None
    ) -> None:
        if self._current_entry is None:
            raise AssertionError("_current_entry is not set in add_backend_id")
        if backend_id not in self._current_entry.backend_ids:
            self._current_entry.backend_ids.append(backend_id)
        if backend is not None:
            self._cached_backends[backend_id] = backend

    def add_backend_id(self, backend_id: str, backend: Any | None = None) -> None:
        if not backend_id.startswith(f"{COMPILED_FN_PREFIX}_"):
            raise AssertionError(
                f"backend_id must start with '{COMPILED_FN_PREFIX}_', got '{backend_id}'"
            )
        self._add_backend_id(_BackendId(backend_id), backend)

    def validate(self) -> None:
        if self._current_entry is not None:
            raise AssertionError("_current_entry should be None during validate")
        if self._innermost_fn is None:
            raise AssertionError("_innermost_fn is not set during validate")
        if not self._initialized:
            raise AssertionError("CompilePackage is not initialized during validate")
        if next(iter(self._codes)) is not self._innermost_fn.__code__:
            raise AssertionError(
                "First code entry does not match _innermost_fn.__code__"
            )

    def _install_global(self, module: types.ModuleType, name: str, value: Any) -> None:
        # A pre-reset compile in this process may still own `name` via a
        # CleanupHook that hasn't fired yet. We're taking over the binding now,
        # so that hook must not delete it once its code object is collected.
        CleanupHook.disown(module.__dict__, name)
        module.__dict__[name] = value
        self._installed_globals.setdefault(module, []).append(
            _InstalledGlobal(name, value)
        )

    def uninstall(self) -> None:
        with _PACKAGE_INSTALL_LOCK:
            self._uninstall()

    def _uninstall(self) -> None:
        from torch._C._dynamo.eval_frame import (
            _debug_get_precompile_entries,
            _reset_precompile_entries,
        )

        if self._innermost_fn is None:
            raise AssertionError("_innermost_fn is not set in uninstall")
        # This namespace is shared with plain torch.compile and with any other
        # package loaded for the same module, so remove a name only while it is
        # still bound to what we wrote: one something else has rebound since
        # belongs to that writer now, and popping it leaves live consumers of
        # the name with a NameError. Deliberately no attempt to put back what we
        # displaced -- the only writer that displaces one of these names is a
        # second package installing the same artifact, whose load orphaned the
        # first package's value already, so restoring it on the later unload
        # would leave a compiled backend bound in the module forever.
        for module, installed in self._installed_globals.items():
            for installed_global in installed:
                if (
                    module.__dict__.get(installed_global.name, _ABSENT_GLOBAL)
                    is installed_global.value
                ):
                    del module.__dict__[installed_global.name]

        self._installed_globals = {}

        from torch._C._dynamo.eval_frame import compare_and_set_code_exec_strategy

        for code in self._skipped_codes:
            with _INSTALLER_REGISTRY_LOCK:
                state = _SKIP_INSTALLERS.get(code)
                if state is None:
                    continue
                state.owners.discard(self)
                if state.owners:
                    continue
                del _SKIP_INSTALLERS[code]
                # The generation check and write happen in one C++ call. A
                # same-valued skip or a different strategy installed after us
                # therefore wins, including a write racing with this unload.
                compare_and_set_code_exec_strategy(
                    code, state.generation, state.prior_strategy
                )
        self._skipped_codes = []

        # _reset_precompile_entries clears every entry on a code object and there
        # is no per-entry removal, so this also drops any other live package's
        # entries for the same frame. It runs on the entry code object even when
        # this package installed nothing on it, because __init__ calls uninstall()
        # to start from a clean slate -- which is what loading a second artifact
        # for another instance of the same class does, so that path has to warn
        # too. Warn rather than raise: refusing would deadlock two packages that
        # share a frame, since neither could ever go first.
        for code in dict.fromkeys(
            [self._innermost_fn.__code__, *self._installed_precompile_codes]
        ):
            with _INSTALLER_REGISTRY_LOCK:
                others = [
                    p for p in _PRECOMPILE_INSTALLERS.get(code, ()) if p is not self
                ]
            if others and _debug_get_precompile_entries(code):
                logger.warning(
                    "Clearing the precompile entries on code object %s (%s:%d), "
                    "which %d other loaded package(s) also installed on. Entries "
                    "can only be cleared en masse, so those packages stop serving "
                    "this frame. Their callers then fall through to whatever "
                    "entries remain: if a surviving package's guards accept the "
                    "call, it is served THAT package's graph, silently and with "
                    "no error, which is what happens when the guard that told "
                    "the two apart was an identity guard precompile had to drop. "
                    "Only a call no remaining entry matches recompiles, or "
                    "raises under fail_on_recompile. Loading a second artifact "
                    "for the same function, or for another instance of the same "
                    "class, lands here.",
                    code.co_name,
                    code.co_filename,
                    code.co_firstlineno,
                    len(others),
                )
            _reset_precompile_entries(code)
            with _INSTALLER_REGISTRY_LOCK:
                installers = _PRECOMPILE_INSTALLERS.get(code)
                if installers is not None:
                    installers.discard(self)
        self._installed_precompile_codes = []

    def install(self, backends: dict[_BackendId, Any]) -> None:
        """
        Sync the package states to the compiled function. This includes the following actions:
          1. Clean up the previously installed states.
          2. Install the compiled functions to global scopes.
          3. Install the precompiled cache entries to ExtraStates on the code object.
        """
        with _PACKAGE_INSTALL_LOCK:
            self._uninstall()
            try:
                self._install_codes(backends)
            except Exception:
                # A half-installed package is worse than an unloaded one: some
                # frames serve precompiled code and some do not, and because
                # install() raised, the caller has no handle to undo it. The
                # expected way to get here is after_deserialization() rejecting an
                # artifact on a serving host that does not match the capture host.
                self._uninstall()
                raise

    def reset_after_failed_install(self) -> None:
        """Make an install-clean package reusable for a cold-cache fallback."""
        with _PACKAGE_INSTALL_LOCK:
            if (
                self._installed_globals
                or self._installed_precompile_codes
                or self._skipped_codes
            ):
                raise AssertionError("failed install left package state installed")
            self._initialized = False

    def _install_codes(self, backends: dict[_BackendId, Any]) -> None:
        from torch._C._dynamo.eval_frame import _load_precompile_entry

        from .convert_frame import input_codes
        from .output_graph import get_builtins_dict

        # Resume functions are bound under a name unique to their code, not
        # under the name the capture process happened to mint. Every reference
        # to them lives in some frame's dynamo bytecode, remapped below.
        resume_renames = _resume_global_renames(self._codes.values())
        for code, entry in self._codes.items():
            context = (
                _compile_frame_context(code)
                if entry.has_compile_id
                else contextlib.nullcontext()
            )
            with context:
                module = sys.modules[entry.python_module]
                for alias, module_name in entry.import_sources.items():
                    # Deliberately not recorded for uninstall. An import alias
                    # is a module object under a name derived from the module,
                    # so every writer writes the same value, and plain
                    # torch.compile (symbolic_convert.import_source) installs it
                    # permanently. Taking it back out breaks whoever else in
                    # this module resolved the same alias.
                    module.__dict__[alias] = importlib.import_module(module_name)
                target_code = code
                if entry.install_to_global:
                    for function_name in entry.function_names:
                        installed_name = resume_renames[function_name]
                        if code.co_freevars:
                            # Resume functions with freevars need a factory
                            # that takes a closure tuple, matching
                            # install_resume_function_global in output_graph.py.
                            f_globals = module.__dict__
                            fn_name = installed_name

                            def _make_fn(
                                closure: tuple[types.CellType, ...],
                                _code: types.CodeType = code,
                                _globals: dict[str, Any] = f_globals,
                                _name: str = fn_name,
                            ) -> types.FunctionType:
                                return types.FunctionType(
                                    _code, _globals, _name, None, closure
                                )

                            self._install_global(module, installed_name, _make_fn)
                        else:
                            fn = types.FunctionType(
                                code, module.__dict__, installed_name
                            )
                            self._install_global(module, installed_name, fn)
                if entry.code_source:
                    target_code = _lookup_code(entry)

                if entry.bypassed:
                    # If the entry is bypassed, do not install backends
                    # or guarded codes.
                    continue

                input_codes.add(target_code)
                for backend_id in entry.backend_ids:
                    if backend_id not in backends:
                        raise RuntimeError(
                            f"Backend {backend_id} is not found in the given backends"
                        )
                    with dynamo_timed(
                        "after_deserialization", phase_name="backend_compile"
                    ):
                        backend = backends[backend_id].after_deserialization()
                        self._install_global(
                            module,
                            backend_id,
                            torch._dynamo.disable(backend),
                        )

                if len(entry.guarded_codes) == 0:
                    # Legacy and transparent-cache artifacts can contain a frame
                    # with no guarded code. It must run eager so covered child
                    # frames can still dispatch.
                    # Remember it, and register as one of the packages holding
                    # the skip, so uninstall() can restore the frame without
                    # un-skipping it under another package that still needs it.
                    self._skipped_codes.append(target_code)
                    with _INSTALLER_REGISTRY_LOCK:
                        state = _SKIP_INSTALLERS.get(target_code)
                        current_generation = None
                        if state is not None:
                            from torch._C._dynamo.eval_frame import (
                                get_code_exec_strategy_token,
                            )

                            _, current_generation = get_code_exec_strategy_token(
                                target_code
                            )
                        if state is None or current_generation != state.generation:
                            from torch._C._dynamo.eval_frame import (
                                set_code_exec_strategy_with_token,
                            )

                            prior_strategy, generation = (
                                set_code_exec_strategy_with_token(
                                    target_code, _PACKAGE_SKIP_STRATEGY
                                )
                            )
                            state = _SkipInstallerState(
                                owners=(
                                    weakref.WeakSet() if state is None else state.owners
                                ),
                                prior_strategy=prior_strategy,
                                generation=generation,
                            )
                            _SKIP_INSTALLERS[target_code] = state
                        state.owners.add(self)

                for guarded_code in entry.guarded_codes:
                    with dynamo_timed("precompile_load_guards"):
                        guards_state = load_guards_state(guarded_code.guards_state)
                    runtime_global_scope = sys.modules[entry.python_module].__dict__
                    # The installed builtins dict might be absent from the runtime
                    # while loading guards. Populate it if it's missing.
                    if (
                        builtin_dict_name
                        := guards_state.output_graph.name_of_builtins_dict_key_in_fglobals
                    ):
                        # A pre-reset compile's CleanupHook may still own this
                        # name even when we're about to leave its value alone
                        # below (same dict object every compile in this
                        # module), so it must not delete it once collected.
                        CleanupHook.disown(runtime_global_scope, builtin_dict_name)
                        builtins_dict = get_builtins_dict(runtime_global_scope)
                        if builtin_dict_name in runtime_global_scope:
                            if (
                                runtime_global_scope[builtin_dict_name]
                                is not builtins_dict
                            ):
                                raise AssertionError(
                                    f"Builtins dict mismatch for key '{builtin_dict_name}'"
                                )
                        else:
                            # Recorded, so uninstall() takes it back out. The
                            # name carries the capture process's unique_id
                            # counter, so leaving it behind makes the first
                            # local compile that mints the same name die in
                            # CleanupHook.create. That collision still happens
                            # WHILE the artifact is installed: fixing it means
                            # not minting the name off a process-local counter,
                            # which is output_graph's call, not this loader's.
                            self._install_global(
                                module, builtin_dict_name, builtins_dict
                            )
                    if not isinstance(guards_state, torch._dynamo.guards.GuardsState):
                        raise AssertionError(
                            f"Expected GuardsState, got {type(guards_state)}"
                        )
                    with dynamo_timed("precompile_build_guards"):
                        guard_manager = load_guard_manager(
                            guards_state, target_code, runtime_global_scope
                        )
                    if target_code not in self._installed_precompile_codes:
                        self._installed_precompile_codes.append(target_code)
                    _register_installer(_PRECOMPILE_INSTALLERS, target_code, self)
                    _load_precompile_entry(
                        target_code,
                        guard_manager,
                        _rename_globals(
                            SerializedCode.to_code_object(guarded_code.dynamo_code),
                            resume_renames,
                        ),
                    )

    def cache_entry(self) -> _DynamoCacheEntry:
        self.validate()
        if self._innermost_fn is None:
            raise AssertionError("_innermost_fn is not set in cache_entry")
        return _DynamoCacheEntry(
            codes=list(self._codes.values()),
            source_info=self._source_info,
            device_type=self._device_type,
            fn_name=self._innermost_fn.__qualname__,
            fn_first_lineno=self._innermost_fn.__code__.co_firstlineno,
        )

    @staticmethod
    def source_id_from_fn(fn: Callable[..., Any]) -> str:
        from .eval_frame import innermost_fn

        innermost_fn_ = innermost_fn(fn)

        sha256_hash = hashlib.sha256()
        sha256_hash.update(innermost_fn_.__qualname__.encode())
        sha256_hash.update(str(innermost_fn_.__code__.co_firstlineno).encode())
        return sha256_hash.hexdigest()


_Backends = dict[_BackendId, Any]


class DynamoStore(abc.ABC):
    """
    A DynamoStore tracks active CompilePackages, and provides methods to store and retrieve them.

    This is an abstract base class for different storage implementations.
    """

    def record_package(self, package: CompilePackage) -> None:
        """
        Records a package to PrecompileContext, so that it can be serialized later.
        """
        from torch._dynamo.precompile_context import PrecompileContext

        cache_entry = package.cache_entry()
        PrecompileContext.record_dynamo_cache_entry(
            cache_entry=cache_entry, key=package.source_id
        )

    def record_eager_backend(self, backend_id: _BackendId, backend: Any) -> None:
        """
        Records eager fx graphs to PrecompileContext for testing purposes.
        """
        from torch._dynamo.precompile_context import (
            EagerCacheArtifact,
            PrecompileContext,
        )

        result = EagerCacheArtifact(key=backend_id, content=backend)
        PrecompileContext.record_artifact(result)

    @abc.abstractmethod
    def clear(self) -> None: ...

    @abc.abstractmethod
    def write(
        self,
        cache_entry: PrecompileCacheEntry,
        path: str,
    ) -> None:
        """
        Abstract method to write dynamo cache entry and backends to storage.

        Args:
            dynamo: The dynamo cache entry to write
            backends: Dictionary of backend content to write
            path: Path or key to identify where to write the data
        """
        ...

    def save_cache_entry(self, cache_entry: _DynamoCacheEntry, key: str) -> None:
        """
        Saves a package to a given path. Grabs backends from PrecompileContext.
        """
        from torch._dynamo.precompile_context import (
            BackendCacheArtifact,
            PrecompileContext,
        )

        backend_content: _Backends = {}
        for backend_id in cache_entry.backend_ids:
            serialized_backend = PrecompileContext.serialize_artifact_by_key(backend_id)
            if serialized_backend is None:
                raise RuntimeError(
                    f"Backend {backend_id} is not found in the given backends"
                )
            if not isinstance(serialized_backend, BackendCacheArtifact):
                raise AssertionError(
                    f"Expected BackendCacheArtifact, got {type(serialized_backend)}"
                )
            backend_content[backend_id] = serialized_backend

        entry = PrecompileCacheEntry(cache_entry, backend_content)

        self.write(entry, key)

    def save_package(self, package: CompilePackage, key: str) -> None:
        """
        Saves a package to a given path. Grabs backends from PrecompileContext.
        """
        self.record_package(package)
        cache_entry = package.cache_entry()
        self.save_cache_entry(cache_entry, key)

    @abc.abstractmethod
    def read(self, path: str) -> PrecompileCacheEntry:
        """
        Abstract method to read dynamo cache entry and backends from storage.

        Args:
            path: Path or key to identify where to read the data from

        Returns:
            A tuple containing (dynamo_cache_entry, backend_content)
        """
        ...

    def load_cache_entry(self, key: str) -> PrecompileCacheEntry:
        from torch._dynamo.precompile_context import (
            BackendCacheArtifact,
            PrecompileContext,
        )

        precompile_entry = self.read(key)
        for backend in precompile_entry.backends.values():
            if not isinstance(backend, BackendCacheArtifact):
                raise AssertionError(
                    f"Expected BackendCacheArtifact, got {type(backend)}"
                )
            PrecompileContext.record_artifact(backend)

        return precompile_entry

    def load_package(
        self, fn: Any, key: str
    ) -> tuple[CompilePackage, dict[_BackendId, Any]]:
        """
        Loads a package from a given path and returns it plus a list of deserialized backends
        """
        entry = self.load_cache_entry(key)
        package = CompilePackage(fn, entry.dynamo)
        return package, entry.backends


class InMemoryDynamoStore(DynamoStore):
    """
    A DynamoStore implementation that keeps state about CompilePackages in memory.
    """

    def __init__(self) -> None:
        self.packages: dict[str, PrecompileCacheEntry] = {}

    def clear(self) -> None:
        self.packages.clear()

    def write(
        self,
        cache_entry: PrecompileCacheEntry,
        path: str,
    ) -> None:
        """
        Store the dynamo cache entry and backends in memory instead of writing to disk.
        """
        self.packages[path] = cache_entry

    def read(self, path: str) -> PrecompileCacheEntry:
        """
        Read dynamo cache entry and backends from memory.
        """
        if path not in self.packages:
            raise RuntimeError(f"No package found with key {path}")

        return self.packages[path]


class DiskDynamoStore(DynamoStore):
    """
    A DynamoStore implementation that keeps state about CompilePackages on disk.
    """

    def __init__(self, path_prefix: str = "") -> None:
        """
        Initialize a DiskDynamoStore with a path prefix.

        Args:
            path_prefix: Prefix directory for where to put CompilePackages on disk
        """
        self._path_prefix = path_prefix

    def path_prefix(self) -> str:
        return self._path_prefix

    def clear(self) -> None:
        """
        Clear all CompilePackages from disk.
        """
        if self.path_prefix():
            shutil.rmtree(self.path_prefix(), ignore_errors=True)

    def write(
        self,
        cache_entry: PrecompileCacheEntry,
        path: str,
    ) -> None:
        """
        Write dynamo cache entry and backends to disk.
        """
        try:
            pickled_content: bytes = pickle.dumps(cache_entry)
            CacheArtifactRecorder(PrecompileCacheArtifact.type(), path).record(
                pickled_content
            )
            self._write_to_local_cache(pickled_content, path)
        except Exception as e:
            raise RuntimeError(f"Failed to save package to {path}: {e}") from e

    def _write_to_local_cache(self, pickled_content: bytes, path: str) -> None:
        from torch._inductor.codecache import write_atomic

        path = os.path.join(self.path_prefix(), path) if self.path_prefix() else path
        try:
            os.makedirs(path, exist_ok=True)
            write_atomic(os.path.join(path, "entry"), pickled_content)
        except Exception as e:
            raise RuntimeError(f"Failed to save package to {path}: {e}") from e

    def read(self, path: str) -> PrecompileCacheEntry:
        """
        Read dynamo cache entry and backends from disk.
        """
        path = os.path.join(self.path_prefix(), path) if self.path_prefix() else path
        try:
            with open(os.path.join(path, "entry"), "rb") as f:
                pickled_content = f.read()
                entry = pickle.loads(pickled_content)
                return entry
        except Exception as e:
            raise RuntimeError(f"Failed to load package from path {path}: {e}") from e


class DiskDynamoCache(DiskDynamoStore):
    """
    Special DiskDynamoStore which adds some helper functions for automatically
    tracking paths of packages
    """

    def save(self, package: CompilePackage) -> None:
        """
        Saves a package to a given path. Grabs backends from PrecompileContext.
        """
        key = package.source_id
        logger.info("Saving CompilePackage for %s", package.source_id)
        super().save_package(package, key)

    def load(self, fn: Callable[..., Any]) -> PrecompileCacheEntry | None:
        """
        Loads a package from a given path and returns it plus a list of deserialized backends
        """
        key = CompilePackage.source_id_from_fn(fn)
        logger.info("Loading CompilePackage for %s", key)
        path = os.path.join(self.path_prefix(), key)
        if os.path.exists(path):
            try:
                result = super().load_cache_entry(key)
                counters["dynamo_cache"]["dynamo_cache_hit"] += 1
                return result
            except Exception:
                counters["dynamo_cache"]["dynamo_cache_error"] += 1
                logger.warning("Failed to load package from path %s", exc_info=True)
                return None
        logger.info("No package found for %s", key)
        counters["dynamo_cache"]["dynamo_cache_miss"] += 1
        return None

    def load_and_install_package(self, fn: Callable[..., Any]) -> CompilePackage | None:
        """
        Load directly into a package and install backends
        """
        results = self.load(fn)
        if results is None:
            return None
        else:
            package = CompilePackage(fn, results.dynamo)
            package.install(results.backends)
            return package

    def path_prefix(self) -> str:
        return os.path.join(cache_dir(), "dynamo")


def cache_dir() -> str:
    from torch._inductor.runtime.cache_dir_utils import cache_dir

    return cache_dir()


DynamoCache = DiskDynamoCache(os.path.join(cache_dir(), "dynamo"))
