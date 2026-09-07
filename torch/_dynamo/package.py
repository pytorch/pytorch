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
import io
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
import uuid
import weakref
from collections import deque
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from contextlib import nullcontext
from typing import Any, NewType, Optional, TYPE_CHECKING, Union
from typing_extensions import Never

import torch
from torch._dynamo.exc import PackageError
from torch._dynamo.graph_utils import _graph_device_types
from torch.utils.weak import WeakIdKeyDictionary

from .bytecode_transformation import (
    _reserve_unique_id_through,
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

# code object -> the live CompilePackages that skip_code()d it. Weak on both
# sides: a package dropped without unloading must not block a later one.
_SKIP_INSTALLERS: WeakIdKeyDictionary = WeakIdKeyDictionary()
# When both are needed, acquire the operation lock before the registry lock.
_PACKAGE_INSTALL_LOCK = threading.RLock()
_INSTALLER_REGISTRY_LOCK = threading.Lock()


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


@dataclasses.dataclass
class _GlobalBinding:
    """One value bound under a name, and the live packages that installed it."""

    value: object
    owners: weakref.WeakSet["CompilePackage"]


# module -> name -> STACK of _GlobalBinding, oldest first.
#
# Several live packages can need one name at once. Two loads of the SAME
# artifact write the same value and share a binding; two loads of different
# artifacts displace each other and get separate ones. Either way the name must
# survive until its last owner leaves, and an unload that pops the top has to
# REBIND to whatever is underneath rather than delete -- an earlier package is
# still serving and still reads that name from this module. Deleting is only
# right when the stack empties.
_GLOBAL_BINDINGS: WeakIdKeyDictionary = WeakIdKeyDictionary()


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


class _Missing:
    def __init__(self, reason: str | None = None) -> None:
        self._reason = reason

    def __repr__(self) -> str:
        return f"_Missing({self._reason})"

    def __str__(self) -> str:
        return f"_Missing({self._reason})"

    # Sometimes _Missing object is used as the callable with functools.partial,
    # so we add a dummy __call__ here to bypass TypeError from partial().
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return _Missing()


# The persistent id GuardsStatePickler.persistent_id emits for a pruned value
# the C pickler never routes through reducer_override; _GuardsStateUnpickler
# turns it back into _Missing. Any other persistent id in a guards state is a bug.
_PRUNED_VALUE_PID = "missing values"


class FunctionPicklerBase(pickle.Pickler):
    """Reducers shared by GuardsStatePickler and AOTCompilePickler.

    Both rebuild the same kinds of objects that pickle cannot do by reference:
    code objects, closure cells, python modules, bound methods, and functions
    rebuilt from their code object. Each subclass keeps its own dispatch and
    decides what a rebuilt function carries; this class fixes HOW it is rebuilt
    so a fix in one pickler cannot be missed in the other.

    Defaults, __doc__, __dict__, and the globals snapshot travel as pickle STATE, applied
    after memoization, so `wrapper.me = wrapper` and module-scope cycles end.
    A closure cell is a reduce ARGUMENT: a function closing over itself is
    reduced twice, and save_reduce's recursive-object fallback (present in both
    the C and the pure-Python pickler) drops the outer copy.
    """

    @classmethod
    def _unpickle_code(cls, serialized_code: SerializedCode) -> types.CodeType:
        return SerializedCode.to_code_object(serialized_code)

    @classmethod
    def _unpickle_python_module(cls, name: str) -> types.ModuleType:
        return importlib.import_module(name)

    @classmethod
    def _unpickle_bound_method(cls, func: Any, base: Any) -> types.MethodType:
        return types.MethodType(func, base)

    @classmethod
    def _unpickle_empty_cell(cls) -> types.CellType:
        return types.CellType()

    @staticmethod
    def _set_cell_contents(cell: types.CellType, state: tuple[Any]) -> None:
        # The contents travel wrapped in a 1-tuple: pickle skips the state step
        # entirely when the state object is None, and None is an ordinary cell
        # value that must not come back as an empty cell.
        cell.cell_contents = state[0]

    @classmethod
    def _build_function(
        cls,
        f_globals: dict[str, Any],
        module: str | None,
        code: types.CodeType,
        qualname: str,
        name: str,
        closure: tuple[types.CellType, ...] | None,
    ) -> types.FunctionType:
        fn = types.FunctionType(code, f_globals, name, None, closure)
        # FunctionType derives __module__ from f_globals["__name__"], so any
        # scope that is not the real module dict leaves it None and a guard
        # rooted at fn.__module__ rebuilds against that. Leave that None in
        # place rather than assigning it back (which the stub rejects).
        if module is not None:
            fn.__module__ = module
        fn.__qualname__ = qualname
        return fn

    @classmethod
    def _unpickle_fn_from_module(
        cls,
        module: str | None,
        code: types.CodeType,
        qualname: str,
        name: str,
        closure: tuple[types.CellType, ...] | None,
    ) -> types.FunctionType:
        # functools.wraps copies __module__, so this scope can be a different
        # file from the one the function lives in; a pickler that guards
        # __globals__ sends the snapshot variant instead. A module that only
        # existed in sys.modules at save (exec-created, transformers_modules.*)
        # gets an empty scope: a guard never calls the rebuilt function.
        f_globals: dict[str, Any]
        try:
            # A <locals>/exec function can carry __module__ is None (bare
            # globals with no __name__); import_module(None) would raise
            # AttributeError, not ImportError, so guard it into the empty scope.
            f_globals = importlib.import_module(module).__dict__ if module else {}
        except ImportError:
            f_globals = {}
        return cls._build_function(f_globals, module, code, qualname, name, closure)

    @classmethod
    def _unpickle_fn_from_snapshot(
        cls,
        module: str | None,
        code: types.CodeType,
        qualname: str,
        name: str,
        closure: tuple[types.CellType, ...] | None,
    ) -> types.FunctionType:
        # The scope arrives as pickle STATE, through _apply_function_state.
        return cls._build_function({}, module, code, qualname, name, closure)

    @staticmethod
    def _apply_function_state(fn: types.FunctionType, state: tuple[Any, ...]) -> None:
        (
            defaults,
            kwdefaults,
            attributes,
            globals_snapshot,
            doc,
            annotations,
            type_params,
        ) = state
        fn.__defaults__ = defaults
        fn.__kwdefaults__ = kwdefaults
        # FunctionType took __doc__/__annotations__/__type_params__ from the code
        # object; functools.wraps overwrote them on the live function and a guard
        # rooted there rebakes, so restore what the reducer captured.
        fn.__doc__ = doc
        fn.__annotations__ = annotations
        if type_params is not None:
            fn.__type_params__ = type_params
        fn.__dict__.update(attributes)
        if globals_snapshot is not None:
            fn.__globals__.update(globals_snapshot)

    @staticmethod
    def _read_raw_annotations(obj: Any, *, resolve: bool = False) -> dict[str, Any]:
        # Reading obj.__annotations__ directly forces PEP 649 lazy evaluation on
        # 3.14+, raising NameError for a TYPE_CHECKING-only name. The guard
        # pickler wants the unevaluated shape, so it takes FORWARDREF and prunes
        # the proxies later. A caller that must SERIALIZE the annotations passes
        # resolve=True instead: it gets real values, and an empty dict when a
        # name will not resolve, because a ForwardRef -- even nested in
        # list[Bar] -- is not picklable. This resolves the whole set or nothing;
        # a caller that also needs per-value picklability filters on top.
        if sys.version_info >= (3, 14):
            import annotationlib

            if resolve:
                try:
                    return annotationlib.get_annotations(
                        obj, format=annotationlib.Format.VALUE
                    )
                except Exception:
                    return {}
            return annotationlib.get_annotations(
                obj, format=annotationlib.Format.FORWARDREF
            )
        return obj.__annotations__

    def _reduce_cell(self, cell: types.CellType) -> tuple[Any, ...]:
        try:
            contents = cell.cell_contents
        except ValueError:
            # A free variable only assigned on a path that did not run.
            return type(self)._unpickle_empty_cell, ()
        return (
            type(self)._unpickle_empty_cell,
            (),
            (contents,),
            None,
            None,
            type(self)._set_cell_contents,
        )

    def _reduce_bound_method(self, method: types.MethodType) -> tuple[Any, ...] | None:
        # pickle rebuilds a bound method by getattr() on self at load, which is
        # wrong when that does not resolve back to the same function; those
        # carry the function and self explicitly.
        func = method.__func__
        inner = getattr(method.__self__, func.__name__, None)
        if inspect.ismethod(inner):
            inner = inner.__func__
        if func is inner:
            return None
        return type(self)._unpickle_bound_method, (func, method.__self__)

    def _reduce_function(
        self,
        fn: types.FunctionType,
        *,
        defaults: tuple[Any, ...] | None,
        kwdefaults: dict[str, Any] | None,
        closure: tuple[types.CellType, ...] | None,
        attributes: dict[str, Any],
        annotations: dict[str, Any],
        type_params: tuple[Any, ...] | None,
        globals_snapshot: dict[str, Any] | None = None,
    ) -> tuple[Any, ...]:
        # annotations/type_params are passed in rather than read off fn: the
        # guard pickler prunes what no guard reads, so an unpicklable local class
        # in an annotation cannot fail the whole dump (a failure there silently
        # bypasses the package).
        args = (fn.__module__, fn.__code__, fn.__qualname__, fn.__name__, closure)
        if globals_snapshot is None:
            unpickle = type(self)._unpickle_fn_from_module
        else:
            unpickle = type(self)._unpickle_fn_from_snapshot
        state = (
            defaults,
            kwdefaults,
            attributes,
            globals_snapshot,
            fn.__doc__,
            annotations,
            type_params,
        )
        return unpickle, args, state, None, None, type(self)._apply_function_state


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


class _GuardsStateUnpickler(pickle.Unpickler):
    def persistent_load(self, pid: Any) -> Any:
        if pid != _PRUNED_VALUE_PID:
            raise pickle.UnpicklingError(f"unknown guards state persistent id {pid!r}")
        return _Missing(pid)


def load_guards_state(guards_state: bytes) -> Any:
    try:
        import torch.distributed.fsdp._fully_shard._fully_shard as _fully_shard

        ctx = _fully_shard.disable_fsdp_module_new_init()
    except ImportError:
        ctx = nullcontext()  # type: ignore[assignment]
    with ctx:
        return _GuardsStateUnpickler(io.BytesIO(guards_state)).load()


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
        explicit_capture=True,
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


@dataclasses.dataclass
class _PreparedInstall:
    """What install() would have computed, computed early by prepare()."""

    backends: dict[_BackendId, Any]
    managers: dict[tuple[types.CodeType, int], Any]
    guard_states: dict[tuple[types.CodeType, int], Any]


@dataclasses.dataclass(frozen=True)
class InlinedSource:
    module: str
    firstlineno: int
    lastlineno: int
    checksum: str


def _defining_module_name(code: types.CodeType) -> str | None:
    """
    The sys.modules key whose source actually contains ``code``.

    ``inspect.getmodule(code)`` maps ``co_filename`` to the ``__name__`` of the
    module owning that file, and a private implementation file can set its own
    ``__name__`` to the public name -- ``_collections_abc`` says
    "collections.abc" -- so getmodule hands back the re-exporting shim, whose
    file does not contain the code, and hashing the code's line range against
    it fails the Source mismatch check. Nor is that ``__name__`` importable
    back to the same object: it imports to the shim, and load-time
    revalidation re-imports this name, so return the key, not ``__name__``.

    Real models inline through such modules, so give up and skip the checksum
    rather than record one against the wrong file.
    """
    module = inspect.getmodule(code)
    if module is not None and getattr(module, "__file__", None) == code.co_filename:
        name = getattr(module, "__name__", None)
        if name is not None and sys.modules.get(name) is module:
            return name
    return _scan_sys_modules_for_file(code.co_filename)


# filename -> (len(sys.modules) when scanned, module key or None).
_MODULE_KEY_BY_FILE: dict[str, tuple[int, str | None]] = {}


def _scan_sys_modules_for_file(filename: str) -> str | None:
    """
    Memoized because the fallback is O(len(sys.modules)) and this runs per
    inlined code object during capture, on the shared caching_precompile path.

    A hit is cached for as long as the key still names a module with this
    file. A MISS is only cached while sys.modules has not changed size, because
    the usual reason for one is that the module has not been imported yet --
    caching that permanently, which functools.cache would, silently drops the
    source checksum for every lazily imported file for the rest of the process.

    Length is an ABA check, not a version: equal-size churn between two calls
    keeps a stale miss, and ``del sys.modules[m]; import m`` -- the ordinary
    force-reimport idiom -- is exactly that. sys.modules exposes no mutation
    counter to use instead. A stale MISS costs this file's checksum, so a later
    edit to it is not caught at load; worth knowing when hunting a checksum that
    should have fired.

    A cached HIT is revalidated against sys.modules before it is returned, for
    the cost of one dict lookup rather than the scan the memo exists to avoid.
    Trusting it instead made ``del sys.modules[m]`` -- with no re-import, so the
    ABA check above cannot see it -- hand back a dead name that ``add_code``
    then raised KeyError on.
    """
    generation = len(sys.modules)
    cached = _MODULE_KEY_BY_FILE.get(filename)
    if cached is not None:
        cached_generation, cached_key = cached
        if cached_key is None:
            if cached_generation == generation:
                return None
        elif getattr(sys.modules.get(cached_key), "__file__", None) == filename:
            return cached_key
    found = None
    for key, candidate in list(sys.modules.items()):
        if getattr(candidate, "__file__", None) == filename:
            found = key
            break
    _MODULE_KEY_BY_FILE[filename] = (generation, found)
    return found


@dataclasses.dataclass
class SourceInfo:
    inlined_sources: set[InlinedSource]

    def add_code(self, code: types.CodeType) -> None:
        module_name = _defining_module_name(code)
        if module_name is None:
            return
        module = sys.modules.get(module_name)
        if module is None:
            return
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
    # Why Dynamo gave up on this frame. Known at the bypass site and otherwise
    # only reachable via tlparse, which leaves a refusal downstream guessing.
    bypass_reason: str | None = None


def _resume_global_renames(
    entries: Iterable[_DynamoCodeCacheEntry], install_token: str
) -> dict[str, str]:
    """
    Pick a global name for every resume function that the installing package
    owns exclusively, so two artifacts that share a capture-time name do not
    collide in one module dict.

    ``__resume_at_<offset>_<n>`` comes from a counter that restarts in every
    capture process, so two artifacts captured separately both claim, say,
    ``__resume_at_16_3``. A serving process installs both into the same module
    dict: the second one wins and the first model silently runs the second's
    continuation. Unlike ``__compiled_fn`` names, which carry a uuid, these
    names carry nothing that distinguishes the artifact.

    The per-install token is what actually separates them: it is unique to the
    loaded package. The code digest is only a readability hint so the name
    still says which code it belongs to; it is NOT a stability guarantee
    (pickling a code object is not byte-stable across processes -- constants
    pass through by reference and a frozenset's byte order is
    PYTHONHASHSEED-dependent), and nothing relies on it being one.
    """
    renames: dict[str, str] = {}
    for entry in entries:
        if not entry.install_to_global:
            continue
        digest = hashlib.sha256(pickle.dumps(entry.python_code)).hexdigest()[:16]
        for name in entry.function_names:
            renames[name] = f"{name}_{digest}_{install_token}"
    return renames


def _rename_globals(code: types.CodeType, renames: dict[str, str]) -> types.CodeType:
    """
    Rewrite ``co_names`` so LOAD_GLOBAL follows the renamed bindings. Indices
    into ``co_names`` are preserved, so the bytecode itself is untouched.

    ``co_names`` is one table shared with LOAD_ATTR/STORE_ATTR/IMPORT_NAME/
    LOAD_NAME, and every matching slot is rewritten, not just LOAD_GLOBAL's.
    That is safe only because the rename keys are ``__resume_at_*`` names minted
    by ``unique_id``, which no attribute or import name plausibly collides with.
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


def _descriptor_functions(obj: Any) -> list[tuple[str, Any]]:
    """The functions a descriptor wraps, as (attribute name, function) pairs.

    ``getattr`` on the CLASS returns the descriptor itself, not the function
    inside it, so a code object defined under ``@property`` resolves to a
    ``property`` object that nothing downstream can descend into. The attribute
    name is what makes the path round-trip: the loader replays it with plain
    ``getattr``, and ``property.fget`` is an ordinary attribute.
    """
    if isinstance(obj, property):
        wrapped = (("fget", obj.fget), ("fset", obj.fset), ("fdel", obj.fdel))
    elif isinstance(obj, functools.cached_property):
        wrapped = (("func", obj.func),)
    else:
        return []
    # A pybind11 property wraps an instancemethod, not a Python function: no
    # code object to name and not even hashable, so keep only real functions
    # the loader can round-trip through getattr.
    return [(name, fn) for name, fn in wrapped if inspect.isfunction(fn)]


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
            if (
                inspect.isfunction(toplevel)
                or inspect.ismethod(toplevel)
                or _descriptor_functions(toplevel)
            ):
                # Stop at a descriptor too, and let _find_code_source unwrap it.
                # Walking past one cannot work: the remaining parts of a
                # qualname like "C.prop.<locals>.inner" are not attributes of
                # the property object.
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

        for attr, wrapped in _descriptor_functions(obj):
            if (res := _find_code_source(wrapped)) is not None:
                # No `toplevel = obj` here: the recursive call sets it to the
                # wrapped function, whose __qualname__ is the descriptor's own
                # dotted name, which is what the loader walks to.
                return f".{attr}{res}"

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
                    # A descriptor is what getattr on the CLASS returns for
                    # anything defined under @property or @cached_property, so
                    # excluding it here hides every code object inside one.
                    wrapped = _descriptor_functions(value)
                    if not (
                        inspect.isfunction(value)
                        or inspect.isclass(value)
                        or inspect.ismethod(value)
                        or wrapped
                    ):
                        continue
                    if (res := _find_code_source(value)) is not None:
                        # A descriptor has no __name__; the functions it wraps
                        # carry the attribute's name instead. An alias (or a
                        # property built from a differently named function) is
                        # skipped so the definition's own name resolves it.
                        actual = wrapped[0][1].__name__ if wrapped else value.__name__
                        if actual != name:
                            continue
                        return res
        return None

    code_source = _find_code_source(toplevel)
    if code_source is None:
        _raise_resolution_error(code, toplevel)
    return toplevel.__qualname__, code_source.strip(".")


_CpuCodegenTarget = tuple[str, str, int, tuple[str, ...], int | None, str | None]


def _current_cpu_codegen_target() -> _CpuCodegenTarget | None:
    """(machine, vec_isa, vec_isa_width, vec_isa_macro, simdlen, march): what inductor bakes into CPU code.

    ``pick_vec_isa`` dry-compiles a probe with the C++ toolchain, so call this
    only when the artifact can hold native CPU code. None means the host has no
    usable CPU codegen target: the probe raised, or it picked no valid vector
    ISA (``pick_vec_isa`` never raises for a missing compiler; it returns
    ``invalid_vec_isa``), so it can neither produce nor run a vectorized
    inductor CPU kernel.
    """
    from torch._inductor import config as inductor_config, cpu_vec_isa

    try:
        vec_isa = cpu_vec_isa.pick_vec_isa()
    except Exception:
        logger.warning(
            "Could not determine the CPU vector ISA, so no CPU codegen target "
            "is recorded and none will be checked.",
            exc_info=True,
        )
        return None
    if isinstance(vec_isa, cpu_vec_isa.InvalidVecISA):
        return None

    return (
        platform.machine(),
        str(vec_isa),
        vec_isa.bit_width(),
        tuple(vec_isa.build_macro()),
        inductor_config.cpp.simdlen,
        inductor_config.cpp.march,
    )


def _cpu_codegen_target_problem(
    cached: _CpuCodegenTarget, current: _CpuCodegenTarget | None
) -> str | None:
    """Why code generated for ``cached`` cannot be built and run here, or None.

    The artifact carries kernel source tiled for the ISA pick_vec_isa() made at
    codegen, and the loading host compiles that source with the flags of its
    own pick_vec_isa(). The two must agree, so every component is compared
    exactly (march is the unresolved config knob, so two hosts recording None
    compare equal though -march=native expands differently -- benign, since the
    loading host supplies the actual flags); a wider host ISA is not a superset
    here, its masked loads
    zero-fill the lanes the narrower tiling never wrote. The ISA name and its
    bit width must both agree: VecSVE(128) and VecSVE(256) share the name
    "asimd", so the name alone would accept a kernel tiled for the wrong width.
    The build macros disambiguate further: VecNEON and VecSVE(128) share both
    the name "asimd" and a 128-bit width but compile with different capability
    macros, so name and width alone would accept a kernel tiled for the wrong
    one.
    """
    if current is None:
        # No current tuple to compare against, so no component-level reason is
        # available -- the host simply reports no target of its own.
        return (
            "This host reports no CPU codegen target (no C++ toolchain or no "
            "supported vector ISA), so it cannot reproduce the target the "
            "artifact's CPU kernels were built for."
        )
    machine, vec_isa, vec_isa_width, vec_isa_macro, simdlen, march = cached
    if machine != current[0]:
        return f"The artifact was built for machine {machine!r}, this host is {current[0]!r}."
    if (vec_isa, vec_isa_width, vec_isa_macro) != (current[1], current[2], current[3]):
        return (
            f"The artifact's CPU kernels were generated for vector ISA {vec_isa!r} "
            f"({vec_isa_width}-bit); this host would compile them for {current[1]!r} "
            f"({current[2]}-bit). Set ATEN_CPU_CAPABILITY or "
            "torch._inductor.config.cpp.simdlen so the host picks the same ISA."
        )
    if simdlen != current[4]:
        return f"The artifact was built with simdlen={simdlen!r}, this host uses {current[4]!r}."
    if march != current[5]:
        return f"The artifact was built with march={march!r}, this host uses {current[5]!r}."
    return None


# Registered backends that generate no native code, so an artifact of theirs
# has no baked vector width to protect and must not be gated on one. This is a
# blacklist on purpose: anything unrecognised -- including a user's own
# callable, whose compiler_name is just its __name__ -- is assumed to emit
# code, because a false rejection at load is recoverable and silently running a
# kernel built for another ISA is not.
_NO_NATIVE_CODE_BACKENDS = frozenset(
    {
        "aot_eager",
        "aot_eager_decomp_partition",
        "aot_eager_decomp_partition_crossref",
        "aot_eager_decomp_partition_with_mode",
        "aot_eager_default_partitioner",
        "eager",
        "eager_debug",
        "eager_noexcept",
        "pre_dispatch_eager",
    }
)


def emits_native_code(backend_name: str) -> bool:
    return backend_name not in _NO_NATIVE_CODE_BACKENDS


@dataclasses.dataclass(frozen=True)
class SystemInfo:
    """
    System information including Python, PyTorch, CPU codegen, and GPU details.
    This information is used to ensure compiled artifacts can only be loaded
    with compatible system configurations.
    """

    python_version: str
    torch_version: str
    toolkit_version: str | None
    triton_version: tuple[int, int] | None
    gpu_name: str | None
    cpu_codegen_target: _CpuCodegenTarget | None = None
    CHECK_GPUS = ("cuda", "xpu")

    @classmethod
    def current(cls, *, cpu_codegen: bool = True) -> "SystemInfo":
        """Create a SystemInfo instance with current system information.

        ``cpu_codegen=False`` skips the C++ toolchain probe behind
        ``cpu_codegen_target``.
        """
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
            cpu_codegen_target=_current_cpu_codegen_target() if cpu_codegen else None,
        )

    def check_compatibility(
        self,
        other: "SystemInfo",
        device_type: str = "cpu",
        *,
        check_codegen: bool = True,
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
        # A cached None means the artifact predates this field, not "no vector
        # ISA"; for a release build that is every artifact already on disk.
        if (
            check_codegen
            and device_type == "cpu"
            and self.cpu_codegen_target is not None
        ):
            problem = _cpu_codegen_target_problem(
                self.cpu_codegen_target, other.cpu_codegen_target
            )
            if problem is not None:
                raise RuntimeError(
                    "Compile package was created for a CPU codegen target this host "
                    f"cannot run: cached={self.cpu_codegen_target}, "
                    f"current={other.cpu_codegen_target}. {problem}"
                )
        if device_type in self.CHECK_GPUS:
            # Device EXISTENCE is not a native-code question: an artifact
            # holding cuda tensors cannot run without cuda whatever backend
            # produced it, so this check stays outside check_codegen. Only the
            # toolkit/Triton/GPU-model checks below describe generated code and
            # are skipped for a backend that emits none.
            if not getattr(torch, device_type).is_available():
                raise RuntimeError(f"{device_type} is not available")

            if not check_codegen:
                return

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
    # device_type keeps the collapsed accelerator-wins value for BC; a mixed
    # cpu+accelerator capture still holds native CPU code, so keep the full set.
    device_types: frozenset[str] | None = None
    requires_native_backend_compatibility: bool = True
    fn_name: str | None = None
    fn_first_lineno: int | None = None

    @property
    def backend_ids(self) -> set[_BackendId]:
        return {backend_id for code in self.codes for backend_id in code.backend_ids}

    def check_versions(self) -> None:
        """Check if the current system is compatible with the system used to create this cache entry."""
        device_types = self.device_types or frozenset((self.device_type,))
        check_codegen = self.requires_native_backend_compatibility
        current_system_info = SystemInfo.current(
            cpu_codegen=(
                check_codegen
                and "cpu" in device_types
                and self.system_info.cpu_codegen_target is not None
            )
        )
        # cpu first, so a codegen-target refusal is not masked by a GPU error.
        for device_type in sorted(device_types):
            self.system_info.check_compatibility(
                current_system_info,
                device_type,
                check_codegen=check_codegen,
            )

    def debug_info(self) -> dict[str, Any]:
        if len(self.codes) == 0:
            raise AssertionError("Expected at least one code entry")
        return {
            "num_codes": str(len(self.codes)),
            "fn_name": self.fn_name,
            "fn_first_lineno": self.fn_first_lineno,
            "device_type": self.device_type,
            "device_types": sorted(self.device_types or frozenset((self.device_type,))),
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
        # Non-mutating: the entry handed in may be the live one still serving
        # this process, so a code whose backend is missing is bypassed on a copy.
        codes: list[_DynamoCodeCacheEntry] = []
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
                    code = dataclasses.replace(code, bypassed=True)
                    break
                backend_content[backend_id] = backends[backend_id]
            codes.append(code)

        dynamo = dataclasses.replace(cache_entry, codes=codes)
        return PrecompileCacheEntry(dynamo=dynamo, backends=backend_content)


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


@dataclasses.dataclass
class _DeadPackageState:
    installed_globals: dict[types.ModuleType, list[_InstalledGlobal]]
    skipped_codes: list[types.CodeType]


# Registry-held state of packages that died without uninstall(), awaiting
# cleanup. The finalize callback can fire from GC at an arbitrary allocation --
# including under _INSTALLER_REGISTRY_LOCK on this very thread -- so it must
# never BLOCK on that lock; states it cannot clean immediately are parked here
# and drained by the next install()/uninstall()/_claim_global(). deque.append is
# atomic, so parking itself needs no lock.
_DEAD_PACKAGES: deque[_DeadPackageState] = deque()


def _cleanup_dead_packages(blocking: bool) -> None:
    from torch._C._dynamo.eval_frame import compare_and_set_code_exec_strategy

    while _DEAD_PACKAGES:
        if not _INSTALLER_REGISTRY_LOCK.acquire(blocking=blocking):
            return
        rebinds = []
        try:
            try:
                dead = _DEAD_PACKAGES.popleft()
            except IndexError:
                return
            # Pruning ownerless frames and rebinding to the survivor is the
            # tail of _uninstall()'s own logic (see there for the rationale).
            # Liveness is decided by ITERATING each WeakSet: when this runs
            # from the dead owner's own finalize, its entry can still be
            # pending removal and len() would count it, but iteration yields
            # only live members.
            for module, installed in dead.installed_globals.items():
                for installed_global in installed:
                    by_name = _GLOBAL_BINDINGS.get(module) or {}
                    stack = by_name.get(installed_global.name) or []
                    stack[:] = [b for b in stack if any(True for _ in b.owners)]
                    survivor = stack[-1].value if stack else _ABSENT_GLOBAL
                    if not stack:
                        by_name.pop(installed_global.name, None)
                    rebinds.append((module, installed_global, survivor))
            for code in dead.skipped_codes:
                state = _SKIP_INSTALLERS.get(code)
                if state is None or any(True for _ in state.owners):
                    continue
                del _SKIP_INSTALLERS[code]
                compare_and_set_code_exec_strategy(
                    code, state.generation, state.prior_strategy
                )
        finally:
            _INSTALLER_REGISTRY_LOCK.release()
        for module, installed_global, survivor in rebinds:
            name = installed_global.name
            current = module.__dict__.get(name, _ABSENT_GLOBAL)
            if survivor is _ABSENT_GLOBAL:
                if current is installed_global.value:
                    del module.__dict__[name]
            elif current is not survivor and (
                current is installed_global.value or current is _ABSENT_GLOBAL
            ):
                module.__dict__[name] = survivor


def _uninstall_abandoned_package(
    installed_globals: dict[types.ModuleType, list[_InstalledGlobal]],
    skipped_codes: list[types.CodeType],
    region_skipped_codes: list[types.CodeType],
    precompile_codes: list[types.CodeType],
    region_id: int,
    owner: object,
) -> None:
    # weakref.finalize callback for a CompilePackage that died while still
    # installed: its precompile entries, skip strategies and installed globals
    # must not outlive it, or every reload of one artifact grows the frame
    # cache and the module globals without bound. GC can fire this
    # mid-guard-evaluation or under this module's own locks, so nothing here
    # may block: entry teardown parks in C++ when the cache lock is
    # unavailable, the strategy calls take only a C++ mutex no Python runs
    # under, and the registry-held state parks above when the registry lock is.
    from torch._C._dynamo.eval_frame import (
        _reset_precompile_entries_for_owner,
        set_code_region_exec_strategy,
    )

    for code in precompile_codes:
        _reset_precompile_entries_for_owner(code, region_id, owner)
    default_strategy = FrameExecStrategy(FrameAction.DEFAULT, FrameAction.DEFAULT)
    for code in region_skipped_codes:
        set_code_region_exec_strategy(code, region_id, default_strategy)
    _DEAD_PACKAGES.append(_DeadPackageState(installed_globals, skipped_codes))
    _cleanup_dead_packages(blocking=False)


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
        *,
        serialization_guard_filter_fn: Callable[[Sequence[Any]], Sequence[bool]]
        | None = None,
        explicit_capture: bool = False,
        serving: bool = False,
        requires_native_backend_compatibility: bool = True,
    ) -> None:
        self._innermost_fn = None
        self._codes: dict[types.CodeType, _DynamoCodeCacheEntry] = {}
        # Suffix minted once here (never refreshed) and appended to every
        # resume function's global name, so two packages holding byte-identical
        # resume code -- two loads of one artifact, or two artifacts of one
        # script captured in separate processes -- do not take each other's
        # name. Distinct from _install_owner, which is re-minted per install.
        # See _resume_global_renames.
        self._resume_name_token = uuid.uuid4().hex
        # Identity token stamped onto every precompile entry this package
        # installs, so uninstall() can remove its own and leave a neighbour
        # package's entries on a shared code object alone.
        self._install_owner = object()
        # Uninstalls an installed package that dies without uninstall(),
        # so repeated loads of one artifact cannot grow the frame cache and
        # module globals without bound. Registered by install().
        self._uninstall_finalizer: weakref.finalize[..., CompilePackage] | None = None
        self._prepared: _PreparedInstall | None = None

        self._current_entry: _DynamoCodeCacheEntry | None = None
        self._installed_globals: dict[types.ModuleType, list[_InstalledGlobal]] = {}
        # Code objects holding this package's region state, so uninstall() can
        # clear all of them. install() covers resume functions and any frame
        # reached through code_source, not just the entry frame; code_context()
        # adds the live frames an uncovered call compiled inside the region.
        self._installed_precompile_codes: list[types.CodeType] = []
        # One of those codes that actually received entries, used to notice a
        # torch._dynamo.reset() wiping the install out from under us. A frame
        # with no guarded code is installed but gets no entries, so it cannot
        # serve as the probe.
        self._installed_precompile_probe: types.CodeType | None = None
        self._installed_precompile_region_id = -1
        self._skipped_codes: list[types.CodeType] = []
        self._region_skipped_codes: list[types.CodeType] = []
        # Frames whose capture was cut short by the recompile limit. Deliberately
        # runtime-only and NOT serialized: it describes this capture session, not
        # the artifact, and it must not affect what install() serves.
        self._truncated_frames: set[str] = set()
        self._device_types: set[str] = set()
        self._system_info: SystemInfo | None = None
        # Set when the CPU codegen target changed between compiles of one
        # capture. The compile itself must not fail over it (the ambient
        # caching_precompile path reaches update_device_type on every user
        # compile), but the mixed-target package can never be serialized.
        self._cpu_codegen_target_drift: str | None = None
        self._default_requires_native_backend_compatibility = (
            requires_native_backend_compatibility
        )
        self._requires_native_backend_compatibility = (
            self._default_requires_native_backend_compatibility
        )

        # For debugging/testing purpose only.
        self._cached_backends: dict[_BackendId, Any] = {}
        self._source_info: SourceInfo = SourceInfo(inlined_sources=set())
        self._resume_codes: set[types.CodeType] = set()
        # Runtime guards stay intact; this filter applies only to the guard
        # state recorded in the package.
        self._serialization_guard_filter_fn = serialization_guard_filter_fn
        # A torch.compiler.precompile capture or serve, as opposed to the
        # ambient caching_precompile cache: only the filtered copy of the guards
        # is recorded, and the package is never auto-persisted.
        self._explicit_capture = explicit_capture
        # Serves a loaded artifact. A frame it does not cover still compiles
        # and counts toward the recompile limit, but nothing will ever save
        # this package, so its guards are neither serialized nor held to the
        # strictness of a capture.
        self._serving = serving
        self._initialized = False
        if fn is not None:
            self.initialize(fn, dynamo, ignore_inlined_sources)
            self.validate()

    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def serialization_guard_filter_fn(
        self,
    ) -> Callable[[Sequence[Any]], Sequence[bool]] | None:
        return self._serialization_guard_filter_fn

    @property
    def explicit_capture(self) -> bool:
        return self._explicit_capture

    @property
    def serving(self) -> bool:
        return self._serving

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
        self._prepared = None
        self._codes = {}
        self._device_types = set()
        self._system_info = None
        self._cpu_codegen_target_drift = None
        self._requires_native_backend_compatibility = (
            self._default_requires_native_backend_compatibility
        )
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
            # Written last so a failed load cannot leak into a cold-cache fallback.
            self._device_types = set(dynamo.device_types or (dynamo.device_type,))
            self._system_info = dynamo.system_info
            # OR, never replace: a loaded entry that did not require native
            # backend compatibility must not relax a host that does, or the ISA
            # check fails open on a kernel built for another target.
            self._requires_native_backend_compatibility = (
                self._requires_native_backend_compatibility
                or dynamo.requires_native_backend_compatibility
            )
        else:
            module_name = (
                _defining_module_name(self._innermost_fn.__code__)
                or self._innermost_fn.__module__
            )
            self._add_function(self._innermost_fn.__code__, module_name)
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

        # A call the artifact does not cover compiles INSIDE the installed
        # region and leaves its cache entries on the LIVE code object, which for
        # a resume function is not the reconstructed twin _codes is keyed by.
        # The two compare EQUAL, so match on identity: region_codes() has to
        # hand a region-wide clear the LIVE code, or the live frame keeps one
        # entry per load until accumulated_recompile_limit refuses to compile it.
        if self._installed_precompile_region_id >= 0 and not any(
            installed is code for installed in self._installed_precompile_codes
        ):
            self._installed_precompile_codes.append(code)

        entry = self._codes[code]
        self._current_entry = entry
        try:
            yield
        finally:
            entry.has_compile_id = True
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
        # An empty variant contributes no device, and a SymInt-first graph is not
        # "cpu": either misread bakes a cpu_codegen_target into a pure-accelerator
        # capture, which then refuses to load on a host with a different ISA.
        device_types = _graph_device_types(graph)
        if not device_types:
            return
        # Computing cpu_codegen_target runs the C++ toolchain, so only a capture
        # that emits CPU native code pays for it; a later cpu graph backfills it.
        needs_cpu_codegen = (
            self._requires_native_backend_compatibility and "cpu" in device_types
        )
        if self._system_info is None:
            self._system_info = SystemInfo.current(cpu_codegen=needs_cpu_codegen)
        elif needs_cpu_codegen:
            # Re-read per cpu compile, not the whole SystemInfo: the toolchain
            # probe is cached, and the inductor config it folds in can change
            # between compiles of one process.
            current_target = _current_cpu_codegen_target()
            if self._system_info.cpu_codegen_target is None:
                self._system_info = dataclasses.replace(
                    self._system_info, cpu_codegen_target=current_target
                )
            elif self._system_info.cpu_codegen_target != current_target:
                # Never fail the compile: the ambient caching_precompile path
                # runs through here. refuse_unserializable() refuses the
                # mixed-target package at serialization boundaries instead.
                if self._cpu_codegen_target_drift is None:
                    self._cpu_codegen_target_drift = (
                        "CPU codegen target changed during capture: "
                        f"first={self._system_info.cpu_codegen_target}, "
                        f"current={current_target}"
                    )
                    logger.warning(
                        "%s; this package will not be serialized.",
                        self._cpu_codegen_target_drift,
                    )
        self._device_types.update(device_types)

    @property
    def current_entry(self) -> _DynamoCodeCacheEntry | None:
        return self._current_entry

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
        # Entered Dynamo yet holds no guarded code, which is exactly what
        # install() skip_code()s. Resume code that was generated but never
        # executed has no compile id and is not a gap; a frame that hit the
        # recompile limit has working variants and is reported as truncated.
        return frozenset(
            code.co_name
            for code, entry in self._codes.items()
            if entry.has_compile_id and not entry.guarded_codes and not entry.bypassed
        )

    def guarded_code_count(self, code: types.CodeType) -> int:
        entry = self._codes.get(code)
        return 0 if entry is None else len(entry.guarded_codes)

    def code_objects(self) -> tuple[types.CodeType, ...]:
        return tuple(self._codes)

    def region_codes(self) -> tuple[types.CodeType, ...]:
        """
        Every live code object an isolated region of this package can hold
        state on.

        A frame reached through code_source is installed onto the code the
        RUNNING program resolves that name to, not onto the reconstructed twin
        in _codes, and the two compare EQUAL, so this is deliberately not
        deduplicated: any set or dict keyed by value collapses the pair and
        drops exactly the live code. Call it before uninstall(), which forgets
        what it installed onto. _region_skipped_codes is a strict subset of
        _installed_precompile_codes, so it needs no separate entry here.
        """
        return (*self._codes, *self._installed_precompile_codes)

    def bypass_current_entry(self, reason: str | None = None) -> None:
        if self._current_entry is None:
            raise AssertionError("_current_entry is not set in bypass_current_entry")
        self._current_entry.bypassed = True
        # A bypassed entry is never installed, so what it already registered
        # would only be serialized for nothing.
        self._current_entry.backend_ids.clear()
        self._current_entry.guarded_codes.clear()
        self._current_entry.bypass_reason = reason

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

    def _install_global(
        self, module: types.ModuleType, name: str, value: object
    ) -> None:
        # A pre-reset compile in this process may still own `name` via a
        # CleanupHook that hasn't fired yet. We're taking over the binding now,
        # so that hook must not delete it once its code object is collected.
        CleanupHook.disown(module.__dict__, name)
        module.__dict__[name] = value
        self._claim_global(module, name, value)

    def _claim_global(self, module: types.ModuleType, name: str, value: Any) -> None:
        self._installed_globals.setdefault(module, []).append(
            _InstalledGlobal(name, value)
        )
        with _INSTALLER_REGISTRY_LOCK:
            by_name = _GLOBAL_BINDINGS.setdefault(module, {})
            stack = by_name.setdefault(name, [])
            if not stack or stack[-1].value is not value:
                stack.append(_GlobalBinding(value=value, owners=weakref.WeakSet()))
            stack[-1].owners.add(self)
        # After the claim, not before: a dead package sharing this value must
        # find us on the binding, or the drain deletes what we just wrote.
        _cleanup_dead_packages(blocking=True)

    def claim_region_global(self, scope: dict[str, Any], name: str, value: Any) -> None:
        """
        Take over a global a compile inside this package's installed region just
        wrote, so uninstall() removes it with the rest.

        A call the artifact does not cover falls back to an ordinary Dynamo
        compile inside the region. OutputGraph installs that compile's globals
        and anchors them to a CleanupHook on the transformed code object, which
        the package never sees and which does not fire when the region goes
        away, so unclaimed they stay in the served module for the life of the
        process. Only while installed: the same path runs during capture, for
        globals the capture session's own compiled callable still reads.
        """
        if self._installed_precompile_region_id < 0:
            return
        module_name = scope.get("__name__")
        module = sys.modules.get(module_name) if isinstance(module_name, str) else None
        if module is None or module.__dict__ is not scope:
            return
        with _PACKAGE_INSTALL_LOCK:
            self._claim_global(module, name, value)

    def uninstall(self) -> None:
        with _PACKAGE_INSTALL_LOCK:
            _cleanup_dead_packages(blocking=True)
            self._uninstall()

    def _uninstall(self) -> None:
        from torch._C._dynamo.eval_frame import _reset_precompile_entries_for_owner

        if self._innermost_fn is None:
            raise AssertionError("_innermost_fn is not set in uninstall")
        if self._uninstall_finalizer is not None:
            self._uninstall_finalizer.detach()
            self._uninstall_finalizer = None
        # This namespace is shared with plain torch.compile and with any other
        # package loaded for the same module, so a name goes only when BOTH
        # hold: it is still bound to what we wrote (something that has rebound
        # since owns it now, and popping it leaves live consumers with a
        # NameError), and we are its last live owner. Two packages loaded from
        # one artifact -- the ordinary replica shape -- write the same value
        # under the same name, and dropping it on the first unload broke the
        # one still serving. Deliberately no attempt to put back what we
        # displaced: a writer that displaced this name orphaned our value when
        # it did so, and restoring it later would leave a compiled backend
        # bound in the module forever.
        for module, installed in self._installed_globals.items():
            for installed_global in installed:
                name = installed_global.name
                # Deregister FIRST, unconditionally. Whether our value is still
                # the one bound decides what to do with the namespace, not
                # whether we are still an owner: a package whose binding was
                # displaced by a later load must still drop its claim, or the
                # last unload finds a phantom owner and leaves the name behind.
                with _INSTALLER_REGISTRY_LOCK:
                    by_name = _GLOBAL_BINDINGS.get(module) or {}
                    stack = by_name.get(name) or []
                    for binding in stack:
                        # The frame we actually joined, not merely the first one
                        # holding this value: a later load can stack a value an
                        # earlier one already installed, and deregistering from
                        # that earlier frame leaves us an owner of ours forever.
                        if binding.value is installed_global.value:
                            if self in binding.owners:
                                binding.owners.discard(self)
                                break
                    stack[:] = [b for b in stack if b.owners]
                    survivor = stack[-1].value if stack else _ABSENT_GLOBAL
                    if not stack:
                        by_name.pop(name, None)
                current = module.__dict__.get(name, _ABSENT_GLOBAL)
                if survivor is _ABSENT_GLOBAL:
                    # Nobody left. Remove it only if what is bound is still
                    # ours; anything else belongs to whoever wrote it.
                    if current is installed_global.value:
                        del module.__dict__[name]
                elif current is not survivor and (
                    current is installed_global.value or current is _ABSENT_GLOBAL
                ):
                    # An owner remains; put the name back to THEIR value. Only
                    # when what is bound is ours or gone, though: a bystander
                    # that rebound it owns it now, exactly as in the delete case
                    # above, and overwriting it would hand a package's value to
                    # whatever else in this module reads the name.
                    module.__dict__[name] = survivor

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

        if self._region_skipped_codes:
            from torch._C._dynamo.eval_frame import set_code_region_exec_strategy

            default_strategy = FrameExecStrategy(
                FrameAction.DEFAULT, FrameAction.DEFAULT
            )
            for code in self._region_skipped_codes:
                set_code_region_exec_strategy(
                    code, self._installed_precompile_region_id, default_strategy
                )
        self._region_skipped_codes = []

        for code in self._installed_precompile_codes:
            _reset_precompile_entries_for_owner(
                code, self._installed_precompile_region_id, self._install_owner
            )
        self._installed_precompile_codes = []
        self._installed_precompile_probe = None
        self._installed_precompile_region_id = -1

    def _deserialize_backends(
        self, backends: dict[_BackendId, Any]
    ) -> dict[_BackendId, Any]:
        """
        Deserialize outside the install lock, since an inductor artifact can be
        slow to load, but only the backends install will actually reach: a
        bypassed entry installs nothing, so loading its artifact is wasted work
        and can fail the whole install over a graph that serves nothing.
        """
        needed = {
            backend_id
            for entry in self._codes.values()
            if not entry.bypassed
            for backend_id in entry.backend_ids
        }
        deserialized = {}
        for backend_id, artifact in backends.items():
            if backend_id not in needed:
                continue
            with dynamo_timed("after_deserialization", phase_name="backend_compile"):
                deserialized[backend_id] = artifact.after_deserialization()
        return deserialized

    def install(
        self,
        backends: dict[_BackendId, Any],
        *,
        isolate_recompiles_id: int = -1,
    ) -> None:
        """
        Sync the package states to the compiled function. This includes the following actions:
          1. Clean up the previously installed states.
          2. Install the compiled functions to global scopes.
          3. Install the precompiled cache entries to ExtraStates on the code object.
        """
        prepared = self._prepared
        self._prepared = None
        deserialized_backends = (
            prepared.backends
            if prepared is not None
            else self._deserialize_backends(backends)
        )
        with _PACKAGE_INSTALL_LOCK:
            _cleanup_dead_packages(blocking=True)
            self._uninstall()
            # A fresh owner identity per install: the uninstall above may
            # have PARKED its eviction (lock contended, or run from inside a
            # lookup), and a parked eviction keyed on the old owner must not
            # take the entries this install is about to add.
            self._install_owner = object()
            self._installed_precompile_region_id = isolate_recompiles_id
            try:
                self._install_codes(
                    deserialized_backends,
                    prepared.managers if prepared is not None else {},
                    prepared.guard_states if prepared is not None else {},
                )
            except BaseException:
                # A half-installed package is worse than an unloaded one: some
                # frames serve precompiled code and some do not, and because
                # install() raised, the caller has no handle to undo it. The
                # expected way to get here is after_deserialization() rejecting an
                # artifact on a serving host that does not match the capture host.
                try:
                    self._uninstall()
                except BaseException:
                    logger.exception("Failed to roll back a partial package install")
                raise

    def installed_entries_dropped(self) -> bool:
        """
        True when the precompile entries install() loaded are gone.

        torch._dynamo.reset() clears every code object install() touched -- they
        all go through convert_frame.input_codes -- while leaving the installed
        globals behind, so the next call recompiles instead of serving.

        Scoped to the region install() used rather than to the code object as a
        whole: lookup() never serves a precompile entry across regions, so a
        second artifact loaded onto the same function after the reset is not
        coverage for this one. A served call runs this every time, so it asks
        C++ a yes/no question instead of materializing a wrapper per entry.
        """
        from torch._C._dynamo.eval_frame import _has_precompile_entries

        probe = self._installed_precompile_probe
        return probe is not None and not _has_precompile_entries(
            probe, self._installed_precompile_region_id
        )

    def reset_after_failed_install(self) -> None:
        """Make an install-clean package reusable for a cold-cache fallback."""
        with _PACKAGE_INSTALL_LOCK:
            if (
                self._installed_globals
                or self._installed_precompile_codes
                or self._skipped_codes
                or self._region_skipped_codes
            ):
                raise AssertionError("failed install left package state installed")
            self._initialized = False

    def prepare(self, backends: dict[_BackendId, Any]) -> None:
        """Do install()'s pure half now, so its failures land here.

        Deserializing the backends and building the guard trees touches nothing
        the interpreter can see -- a guard manager reads its example values from
        the state it was pickled with, and only STORES the runtime scope -- but
        they are where an artifact that does not fit this host says so. Running
        them at load costs nothing extra, because install() consumes what this
        leaves rather than redoing it, and it moves the failure off the first
        served call.
        """
        managers = {}
        guard_states = {}
        for code, entry in self._codes.items():
            if entry.bypassed or not entry.guarded_codes:
                continue
            target_code = _lookup_code(entry) if entry.code_source else code
            scope = sys.modules[entry.python_module].__dict__
            for index, guarded_code in enumerate(entry.guarded_codes):
                try:
                    guards_state = load_guards_state(guarded_code.guards_state)
                    guard_states[(target_code, index)] = guards_state
                    managers[(target_code, index)] = load_guard_manager(
                        guards_state,
                        target_code,
                        scope,
                    )
                except Exception as e:
                    # Name the frame and the variant: several frames can guard
                    # the same source, so without them a failure here cannot be
                    # told apart from one the capture reported dropping.
                    raise RuntimeError(
                        f"{entry.python_module}.{target_code.co_name} "
                        f"variant {index}: {type(e).__name__}: {e}"
                    ) from e
        self._prepared = _PreparedInstall(
            backends=self._deserialize_backends(backends),
            managers=managers,
            guard_states=guard_states,
        )

    def _install_codes(
        self,
        backends: dict[_BackendId, Any],
        prebuilt: dict[tuple[types.CodeType, int], Any] | None = None,
        prebuilt_states: dict[tuple[types.CodeType, int], Any] | None = None,
    ) -> None:
        from torch._C._dynamo.eval_frame import _load_precompile_entry

        from .convert_frame import input_codes
        from .output_graph import get_builtins_dict

        # Resume functions are bound under a name unique to their code and to
        # this package, not under the name the capture process happened to
        # mint. Every reference to them lives in some frame's dynamo bytecode,
        # remapped below.
        renames = _resume_global_renames(self._codes.values(), self._resume_name_token)
        # Registered before anything is installed, so a failed install is still
        # torn down when the package dies. The callback must not capture self
        # (it would never fire); it works off the containers the loop below
        # fills in place, which uninstall() rebinds rather than mutates, so an
        # explicit uninstall + reinstall cannot be undone by a stale finalizer.
        self._uninstall_finalizer = weakref.finalize(
            self,
            _uninstall_abandoned_package,
            self._installed_globals,
            self._skipped_codes,
            self._region_skipped_codes,
            self._installed_precompile_codes,
            self._installed_precompile_region_id,
            self._install_owner,
        )
        # Not at interpreter exit: module dicts and the frame cache are torn down.
        self._uninstall_finalizer.atexit = False
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
                        installed_name = renames[function_name]
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
                # Dedup on identity: code objects compare structurally, so two
                # distinct frames with identical bytecode would collapse under
                # ``in``. input_codes above already keys on id() for the same
                # reason.
                if not any(target_code is c for c in self._installed_precompile_codes):
                    # Deliberately NOT clearing the region here. A frame reached
                    # through code_source is shared -- a library block two
                    # loaded models both call -- and several packages may hold
                    # entries for it in one region, which lookup handles by
                    # evaluating each entry's guards. Clearing the region would
                    # evict a live neighbour, and since lookup is region-exact
                    # the neighbour cannot be served by what is left. This
                    # package's own stale entries are already gone: install()
                    # runs uninstall() first, which removes exactly the ones it
                    # owns.
                    self._installed_precompile_codes.append(target_code)
                if entry.guarded_codes and self._installed_precompile_probe is None:
                    self._installed_precompile_probe = target_code
                for backend_id in entry.backend_ids:
                    if backend_id not in backends:
                        raise RuntimeError(
                            f"Backend {backend_id} is not found in the given backends"
                        )
                    self._install_global(
                        module,
                        backend_id,
                        torch._dynamo.disable(backends[backend_id]),
                    )

                if len(entry.guarded_codes) == 0:
                    # Legacy and transparent-cache artifacts can contain a frame
                    # with no guarded code. It must run eager so covered child
                    # frames can still dispatch.
                    # Remember it, and register as one of the packages holding
                    # the skip, so uninstall() can restore the frame without
                    # un-skipping it under another package that still needs it.
                    if self._installed_precompile_region_id >= 0:
                        from torch._C._dynamo.eval_frame import (
                            set_code_region_exec_strategy,
                        )

                        self._region_skipped_codes.append(target_code)
                        set_code_region_exec_strategy(
                            target_code,
                            self._installed_precompile_region_id,
                            _PACKAGE_SKIP_STRATEGY,
                        )
                        continue
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

                for _index, guarded_code in enumerate(entry.guarded_codes):
                    guards_state = (prebuilt_states or {}).get((target_code, _index))
                    if guards_state is None:
                        with dynamo_timed("precompile_load_guards"):
                            guards_state = load_guards_state(guarded_code.guards_state)
                    runtime_global_scope = sys.modules[entry.python_module].__dict__
                    # The installed builtins dict might be absent from the runtime
                    # while loading guards. Populate it if it's missing.
                    if (
                        builtin_dict_name
                        := guards_state.output_graph.name_of_builtins_dict_key_in_fglobals
                    ):
                        _, separator, suffix = builtin_dict_name.rpartition("_")
                        if separator and suffix.isdigit():
                            _reserve_unique_id_through(int(suffix))
                        # A pre-reset compile's CleanupHook may still own this
                        # name even when we're about to leave its value alone
                        # below (same dict object every compile in this
                        # module), so it must not delete it once collected.
                        CleanupHook.disown(runtime_global_scope, builtin_dict_name)
                        builtins_dict = get_builtins_dict(runtime_global_scope)
                        # Dict and stack read under one lock, so a finalizer on
                        # this thread can only park between them. A binding's
                        # existence, not its liveness, says a package minted the
                        # name: a dead owner's binding is pruned by the drain.
                        with _INSTALLER_REGISTRY_LOCK:
                            bound = runtime_global_scope.get(
                                builtin_dict_name, _ABSENT_GLOBAL
                            )
                            owned_by_a_package = any(
                                binding.value is builtins_dict
                                for binding in (_GLOBAL_BINDINGS.get(module) or {}).get(
                                    builtin_dict_name, ()
                                )
                            )
                        if bound is not _ABSENT_GLOBAL and bound is not builtins_dict:
                            raise AssertionError(
                                f"Builtins dict mismatch for key '{builtin_dict_name}'"
                            )
                        # Recorded, so uninstall() takes it back out. The
                        # artifact's counter was reserved above, so local
                        # compiles cannot mint the same name while loaded.
                        # Joining a set another package already owns matters as
                        # much as creating the binding: two loads of one
                        # artifact record the same name, and the first unload
                        # would otherwise delete a key the other one's bytecode
                        # reads. A name a PLAIN compile minted has no owner to
                        # join and is left alone, since claiming it would make
                        # our unload delete what that compile reads.
                        if bound is _ABSENT_GLOBAL or owned_by_a_package:
                            self._install_global(
                                module, builtin_dict_name, builtins_dict
                            )
                    if not isinstance(guards_state, torch._dynamo.guards.GuardsState):
                        raise AssertionError(
                            f"Expected GuardsState, got {type(guards_state)}"
                        )
                    # Keyed by the code object install() itself resolved, so a
                    # prepare that resolved a different one falls back to
                    # building rather than serving a stale tree.
                    guard_manager = (prebuilt or {}).get((target_code, _index))
                    if guard_manager is None:
                        with dynamo_timed("precompile_build_guards"):
                            guard_manager = load_guard_manager(
                                guards_state, target_code, runtime_global_scope
                            )
                    _load_precompile_entry(
                        target_code,
                        guard_manager,
                        _rename_globals(
                            SerializedCode.to_code_object(guarded_code.dynamo_code),
                            renames,
                        ),
                        self._installed_precompile_region_id,
                        self._install_owner,
                    )

    def code_entries(self) -> Iterable["_DynamoCodeCacheEntry"]:
        """The per-frame entries, for a caller that edits them before they are
        packaged. Unlike cache_entry(), this does not require a complete
        capture."""
        return self._codes.values()

    def refuse_unserializable(self) -> None:
        """Raise PackageError if this package can never be serialized -- its
        CPU codegen target drifted mid-capture, so it mixes native code for
        two targets. Called at serialization boundaries only: introspection
        (summary(), backend-id enumeration, teardown) still builds a
        cache_entry() from such a package without raising.
        """
        if self._cpu_codegen_target_drift is not None:
            raise PackageError(
                f"{self._cpu_codegen_target_drift}; the package mixes native "
                "code for two targets and cannot be serialized."
            )

    def cache_entry(self) -> _DynamoCacheEntry:
        self.validate()
        if self._innermost_fn is None:
            raise AssertionError("_innermost_fn is not set in cache_entry")
        device_types = frozenset(self._device_types or ("cpu",))
        device_type = next(
            (device for device in sorted(device_types) if device != "cpu"),
            "cpu",
        )
        return _DynamoCacheEntry(
            codes=list(self._codes.values()),
            source_info=self._source_info,
            device_type=device_type,
            device_types=device_types,
            # _system_info is None only when no graph was ever captured, so
            # there is nothing baked and no reason to run the toolchain probe.
            system_info=self._system_info or SystemInfo.current(cpu_codegen=False),
            requires_native_backend_compatibility=(
                self._requires_native_backend_compatibility
            ),
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

        package.refuse_unserializable()
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

    def load_and_install_package(
        self, fn: Callable[..., Any], *, isolate_recompiles_id: int = -1
    ) -> CompilePackage | None:
        """
        Load directly into a package and install backends.

        ``isolate_recompiles_id`` must be the region the caller will look up in:
        precompile entries match their own region only, so installing into the
        default bucket for an isolated caller loads the artifact and then serves
        nothing from it.
        """
        results = self.load(fn)
        if results is None:
            return None
        else:
            package = CompilePackage(fn, results.dynamo)
            package.install(
                results.backends, isolate_recompiles_id=isolate_recompiles_id
            )
            return package

    def path_prefix(self) -> str:
        return os.path.join(cache_dir(), "dynamo")


def cache_dir() -> str:
    from torch._inductor.runtime.cache_dir_utils import cache_dir

    return cache_dir()


DynamoCache = DiskDynamoCache(os.path.join(cache_dir(), "dynamo"))
