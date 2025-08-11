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
import logging
import os
import pickle
import platform
import shutil
import sys
import types
from collections.abc import Generator, Iterator
from typing import Any, Callable, NewType, Optional
from typing_extensions import Never

import torch
import torch._inductor.package
from torch._dynamo.exc import PackageError
from torch._dynamo.precompile_context import PrecompileCacheArtifact, PrecompileContext
from torch._inductor.runtime.cache_dir_utils import cache_dir
from torch.compiler._cache import CacheArtifactFactory

from .bytecode_transformation import get_code_keys
from .utils import dynamo_timed, increment_frame


logger = logging.getLogger(__name__)


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
    co_linetable: Optional[bytes] = None
    co_qualname: Optional[str] = None
    co_exceptiontable: Optional[bytes] = None
    co_lnotab: Optional[str] = None

    @classmethod
    @functools.cache
    def from_code_object(cls, code: types.CodeType) -> "SerializedCode":
        kwargs = {key: getattr(code, key) for key in get_code_keys()}
        kwargs["co_consts"] = tuple(
            cls.from_code_object(c) if isinstance(c, types.CodeType) else c
            for c in kwargs["co_consts"]
        )
        return cls(**kwargs)

    @classmethod
    @functools.cache
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


_BackendId = NewType("_BackendId", str)  # __compiled_fn
_FunctionId = NewType("_FunctionId", str)  # __resume_at


@dataclasses.dataclass(frozen=True)
class InlinedSource:
    module: str
    firstlineno: int
    lastlineno: int
    checksum: str


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
      6. A list of "backends" (compiled fx graph) unioned from all compield branches.
      7. A string path used to access the original code object users defined.
         A code object can be accessed by "{python_module}.{function_name}.{code_source}" .
      8. A boolean flag indicating whether the function is installed to global scope.
      9. A boolean flag indicating whether the function has a compile id.
    """

    python_code: SerializedCode
    python_module: str
    function_names: list[_FunctionId]
    guarded_codes: list[_GuardedCodeCacheEntry]
    import_sources: dict[str, str]
    backend_ids: list[_BackendId]
    code_source: Optional[str]
    install_to_global: bool
    has_compile_id: bool = False


def _lookup_code(entry: _DynamoCodeCacheEntry) -> types.CodeType:
    assert len(entry.function_names) == 1
    fn: Any = sys.modules[entry.python_module]
    parts = entry.function_names[0].split(".")
    for part in parts:
        fn = getattr(fn, part)
    if entry.code_source:
        parts = entry.code_source.split(".")
        for part in parts:
            if part.endswith("]"):
                index_begin = part.rfind("[")
                assert isinstance(index_begin, int) and index_begin >= 0
                attr = getattr(fn, part[:index_begin], None)
                if attr is None:
                    raise PackageError(f"Cannot find source for code entry {entry}")
                fn = attr[ast.literal_eval(part[index_begin + 1 : -1])]
            else:
                fn = getattr(fn, part)
    else:
        raise PackageError(f"Cannot find source for code entry {entry}")
    assert isinstance(fn, types.CodeType)
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
            if inspect.isfunction(toplevel):
                break
    seen = set()

    def _find_code_source(obj: Any) -> Optional[str]:
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
                    ):
                        continue
                    if (res := _find_code_source(cell_contents)) is not None:
                        toplevel = obj
                        return f".__closure__[{i}].cell_contents{res}"

        if sys.version_info < (3, 11):
            if inspect.ismodule(obj):
                for value in obj.__dict__.values():
                    if not (inspect.isfunction(value) or inspect.isclass(value)):
                        continue
                    if (res := _find_code_source(value)) is not None:
                        return res

            if inspect.isclass(obj):
                for name, value in obj.__dict__.items():
                    value = getattr(obj, name)
                    if not (inspect.isfunction(value) or inspect.isclass(value)):
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


@dataclasses.dataclass
class _DynamoCacheEntry:
    codes: list[_DynamoCodeCacheEntry]
    inlined_sources: set[InlinedSource]
    python_version: str = platform.python_version()
    torch_version: str = torch.__version__

    @property
    def backend_ids(self) -> set[_BackendId]:
        return {backend_id for code in self.codes for backend_id in code.backend_ids}


@CacheArtifactFactory.register
class _DynamoCacheArtifact(PrecompileCacheArtifact[_DynamoCacheEntry]):
    @staticmethod
    def type() -> str:
        return "precompile_dynamo"

    def after_deserialization(self) -> _DynamoCacheEntry:
        return pickle.loads(self.content)


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
    @contextlib.contextmanager
    def _ctx() -> Iterator[None]:
        increment_frame()
        compile_id = get_compile_id(frame_state={})
        log_dynamo_start(code)
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
        fn: Optional[Callable[..., Any]],
        dynamo: Optional[_DynamoCacheEntry] = None,
        ignore_inlined_sources: bool = False,
    ) -> None:
        self._innermost_fn = None
        self._codes: dict[types.CodeType, _DynamoCodeCacheEntry] = {}

        self._current_entry: Optional[_DynamoCodeCacheEntry] = None
        self._installed_globals: dict[types.ModuleType, list[str]] = {}

        # For debugging/testing purpose only.
        self._cached_backends: dict[_BackendId, Any] = {}
        self._inlined_sources: set[InlinedSource] = set()
        self._resume_codes: set[types.CodeType] = set()
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
        dynamo: Optional[_DynamoCacheEntry] = None,
        ignore_inlined_sources: bool = False,
    ) -> None:
        from .eval_frame import innermost_fn

        assert not self._initialized
        self._inlined_sources = set()
        self._innermost_fn = innermost_fn(fn)  # type: ignore[assignment]
        assert self._innermost_fn is not None
        if dynamo is not None:
            assert isinstance(dynamo, _DynamoCacheEntry)
            if dynamo.python_version != platform.python_version():
                raise RuntimeError(
                    f"Compile package was created with a different Python version: {dynamo.python_version}"
                )
            if dynamo.torch_version != torch.__version__:
                raise RuntimeError(
                    f"Compile package was created with a different PyTorch version: {dynamo.torch_version}"
                )
            if not ignore_inlined_sources:
                for code in dynamo.inlined_sources:
                    m = importlib.import_module(code.module)
                    checksum = _hash_sourcelines(m, code.firstlineno, code.lastlineno)
                    if checksum != code.checksum:
                        raise RuntimeError(
                            f"Source code changes detected for {code.module} (line {code.firstlineno} - line {code.lastlineno})"
                        )

                self._inlined_sources = dynamo.inlined_sources

            main, *codes = dynamo.codes
            self._codes = {self._innermost_fn.__code__: main}
            for code in codes:
                self._codes[SerializedCode.to_code_object(code.python_code)] = code
        else:
            self._add_function(
                self._innermost_fn.__code__, self._innermost_fn.__module__
            )
        self._initialized = True

    def _add_function(
        self,
        python_code: types.CodeType,
        python_module: str,
        function_name: Optional[_FunctionId] = None,
        code_source: Optional[str] = None,
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
            assert code.python_module == python_module
            assert code.install_to_global == install_to_global
            assert code.code_source == code_source

        if function_name is not None:
            code.function_names.append(function_name)

    @property
    def cached_backends(self) -> dict[_BackendId, Any]:
        return self._cached_backends

    @functools.cached_property
    def source_id(self) -> str:
        assert self._innermost_fn is not None
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
        assert self._current_entry is None

        # Sometimes user code cannot be inlined in dynamo resulting in extra user code
        # being compiled. We should record these as when they are actually invoked.
        if code not in self._codes:
            self._add_user_function(code)

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
        assert self._current_entry is not None
        guarded_code_entry = _GuardedCodeCacheEntry(
            guards_state=guards_state,
            dynamo_code=SerializedCode.from_code_object(dynamo_code),
        )
        self._current_entry.guarded_codes.append(guarded_code_entry)

    def add_inlined_source(self, sources: list[types.CodeType]) -> None:
        for code in sources:
            if code in self._resume_codes:
                continue
            module = inspect.getmodule(code)
            if module is None:
                continue
            source = inspect.getsource(code)
            lastlineno = code.co_firstlineno + len(inspect.getsourcelines(code)[0])
            assert source == "".join(
                _get_sourcelines(module, code.co_firstlineno, lastlineno)
            )
            self._inlined_sources.add(
                InlinedSource(
                    module=module.__name__,
                    firstlineno=code.co_firstlineno,
                    lastlineno=lastlineno,
                    checksum=_hash_source(source),
                )
            )

    def add_resume_function(
        self,
        python_code: types.CodeType,
        python_module: str,
        function_name: Optional[str],
    ) -> None:
        self._add_function(
            python_code,
            python_module,
            function_name=_FunctionId(function_name) if function_name else None,
            install_to_global=True,
        )
        self._resume_codes.add(python_code)

    def add_import_source(self, alias: str, module_name: str) -> None:
        assert self._current_entry is not None
        self._current_entry.import_sources[alias] = module_name

    def add_backend_id(self, backend_id: str, backend: Optional[Any] = None) -> None:
        assert self._current_entry is not None
        assert backend_id.startswith("__compiled_fn_")  # sanity check
        backend_id = _BackendId(backend_id)
        self._current_entry.backend_ids.append(backend_id)
        if backend is not None:
            self._cached_backends[backend_id] = backend

    def validate(self) -> None:
        assert self._current_entry is None
        assert self._innermost_fn is not None
        assert self._initialized
        assert next(iter(self._codes)) is self._innermost_fn.__code__

    def _install_global(self, module: types.ModuleType, name: str, value: Any) -> None:
        module.__dict__[name] = value
        self._installed_globals.setdefault(module, []).append(name)

    def uninstall(self) -> None:
        from torch._C._dynamo.eval_frame import _reset_precompile_entries

        assert self._innermost_fn is not None
        for module, names in self._installed_globals.items():
            for name in names:
                module.__dict__.pop(name)

        self._installed_globals = {}

        _reset_precompile_entries(self._innermost_fn.__code__)

    def install(self, backends: dict[_BackendId, Any]) -> None:
        """
        Sync the package states to the compiled function. This includes the following actions:
          1. Clean up the previously installed states.
          2. Install the compiled functions to global scopes.
          3. Install the precompiled cache entries to ExtraStates on the code object.
        """
        from torch._C._dynamo.eval_frame import _load_precompile_entry

        from .output_graph import get_builtins_dict

        self.uninstall()
        for code, entry in self._codes.items():
            context = (
                _compile_frame_context(code)
                if entry.has_compile_id
                else contextlib.nullcontext()
            )
            with context:
                module = sys.modules[entry.python_module]
                for alias, module_name in entry.import_sources.items():
                    self._install_global(
                        module, alias, importlib.import_module(module_name)
                    )
                target_code = code
                if entry.install_to_global:
                    for function_name in entry.function_names:
                        fn = types.FunctionType(code, module.__dict__, function_name)
                        self._install_global(module, function_name, fn)
                if entry.code_source:
                    target_code = _lookup_code(entry)

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
                    # Dynamo generates empty graph for trivial functions, should just skip them
                    # in these cases.
                    torch._dynamo.eval_frame.skip_code(target_code)

                for guarded_code in entry.guarded_codes:
                    guards_state = pickle.loads(guarded_code.guards_state)
                    runtime_global_scope = sys.modules[entry.python_module].__dict__
                    # The installed builtins dict might be absent from the runtime
                    # while loading guards. Populate it if it's missing.
                    if (
                        builtin_dict_name
                        := guards_state.output_graph.name_of_builtins_dict_key_in_fglobals
                    ):
                        builtins_dict = get_builtins_dict(runtime_global_scope)
                        if builtin_dict_name in runtime_global_scope:
                            assert (
                                runtime_global_scope[builtin_dict_name] is builtins_dict
                            )
                        else:
                            runtime_global_scope[builtin_dict_name] = builtins_dict
                    assert isinstance(guards_state, torch._dynamo.guards.GuardsState)
                    check_fn_manager = torch._dynamo.guards.CheckFunctionManager(
                        target_code,
                        guards_state.output_graph,
                        guards_serialization_mode="load",
                        shape_code_parts=guards_state.shape_code_parts,
                        runtime_global_scope=runtime_global_scope,
                    )
                    _load_precompile_entry(
                        target_code,
                        check_fn_manager.guard_manager,
                        SerializedCode.to_code_object(guarded_code.dynamo_code),
                    )

    def cache_entry(self) -> _DynamoCacheEntry:
        self.validate()
        return _DynamoCacheEntry(
            codes=list(self._codes.values()), inlined_sources=self._inlined_sources
        )

    @staticmethod
    def source_id_from_fn(fn: Callable[..., Any]) -> str:
        from .eval_frame import innermost_fn

        innermost_fn_ = innermost_fn(fn)

        sha256_hash = hashlib.sha256()
        sha256_hash.update(innermost_fn_.__qualname__.encode())
        sha256_hash.update(str(innermost_fn_.__code__.co_firstlineno).encode())
        return sha256_hash.hexdigest()


@CacheArtifactFactory.register
class EagerCacheArtifact(PrecompileCacheArtifact[Any]):
    @staticmethod
    def type() -> str:
        return "precompile_eager"

    def after_deserialization(self) -> Any:
        return pickle.loads(self.content)


_Backends = dict[_BackendId, PrecompileCacheArtifact[Any]]


class DynamoStore(abc.ABC):
    """
    A DynamoStore tracks active CompilePackages, and provides methods to store and retrieve them.

    This is an abstract base class for different storage implementations.
    """

    def record_package(self, package: CompilePackage) -> None:
        """
        Records a package to PrecompileContext, so that it can be serialized later.
        """
        cache_entry = package.cache_entry()
        pickled_result = pickle.dumps(cache_entry)
        PrecompileContext.record_artifact(
            _DynamoCacheArtifact.type(), key=package.source_id, content=pickled_result
        )

    def record_eager_backend(self, backend_id: _BackendId, backend: Any) -> None:
        """
        Records eager fx graphs to PrecompileContext for testing purposes.
        """
        pickled_result = pickle.dumps(backend)
        PrecompileContext.record_artifact(
            EagerCacheArtifact.type(), key=backend_id, content=pickled_result
        )

    @abc.abstractmethod
    def clear(self) -> None: ...

    @abc.abstractmethod
    def write(
        self,
        dynamo: _DynamoCacheEntry,
        backends: _Backends,
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
        backend_content: _Backends = {}
        for backend_id in cache_entry.backend_ids:
            serialized_backend = PrecompileContext.serialize_artifact_by_key(backend_id)
            if serialized_backend is None:
                raise RuntimeError(
                    f"Backend {backend_id} is not found in the given backends"
                )
            assert isinstance(serialized_backend, PrecompileCacheArtifact)
            backend_content[backend_id] = serialized_backend

        self.write(cache_entry, backend_content, key)

    def save_package(self, package: CompilePackage, key: str) -> None:
        """
        Saves a package to a given path. Grabs backends from PrecompileContext.
        """
        self.record_package(package)
        cache_entry = package.cache_entry()
        self.save_cache_entry(cache_entry, key)

    @abc.abstractmethod
    def read(self, path: str) -> tuple[_DynamoCacheEntry, _Backends]:
        """
        Abstract method to read dynamo cache entry and backends from storage.

        Args:
            path: Path or key to identify where to read the data from

        Returns:
            A tuple containing (dynamo_cache_entry, backend_content)
        """
        ...

    def load_cache_entry(
        self, key: str
    ) -> tuple[_DynamoCacheEntry, dict[_BackendId, Any]]:
        cache_entry, backend_content = self.read(key)
        for backend_id, backend in backend_content.items():
            PrecompileContext.record_artifact(
                backend.type(), key=backend.key, content=backend.content
            )
            backend_content[backend_id] = backend

        return cache_entry, backend_content

    def load_package(
        self, fn: Any, key: str
    ) -> tuple[CompilePackage, dict[_BackendId, Any]]:
        """
        Loads a package from a given path and returns it plus a list of deserialized backends
        """
        cache_entry, backend_content = self.load_cache_entry(key)
        package = CompilePackage(fn, cache_entry)
        return package, backend_content


class InMemoryDynamoStore(DynamoStore):
    """
    A DynamoStore implementation that keeps state about CompilePackages in memory.
    """

    def __init__(self) -> None:
        self.packages: dict[str, tuple[_DynamoCacheEntry, _Backends]] = {}

    def clear(self) -> None:
        self.packages.clear()

    def write(
        self,
        dynamo: _DynamoCacheEntry,
        backends: _Backends,
        path: str,
    ) -> None:
        """
        Store the dynamo cache entry and backends in memory instead of writing to disk.
        """
        self.packages[path] = (dynamo, backends)

    def read(self, path: str) -> tuple[_DynamoCacheEntry, _Backends]:
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

    def __init__(self, path_prefix: str = ""):
        """
        Initialize a DiskDynamoStore with a path prefix.

        Args:
            path_prefix: Prefix directory for where to put CompilePackages on disk
        """
        self.path_prefix = path_prefix

    def clear(self) -> None:
        """
        Clear all CompilePackages from disk.
        """
        if self.path_prefix:
            shutil.rmtree(self.path_prefix, ignore_errors=True)

    def write(
        self,
        dynamo: _DynamoCacheEntry,
        backends: _Backends,
        path: str,
    ) -> None:
        """
        Write dynamo cache entry and backends to disk.
        """
        path = os.path.join(self.path_prefix, path) if self.path_prefix else path
        try:
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, "dynamo"), "wb") as dynamo_path:
                pickle.dump(dynamo, dynamo_path)
            with open(os.path.join(path, "backends"), "wb") as backend_path:
                pickle.dump(backends, backend_path)
        except Exception as e:
            raise RuntimeError(f"Failed to save package to {path}: {e}") from e

    def read(self, path: str) -> tuple[_DynamoCacheEntry, _Backends]:
        """
        Read dynamo cache entry and backends from disk.
        """
        path = os.path.join(self.path_prefix, path) if self.path_prefix else path
        try:
            with open(os.path.join(path, "dynamo"), "rb") as dynamo_path:
                cache_entry = pickle.load(dynamo_path)
            with open(os.path.join(path, "backends"), "rb") as backend_path:
                backend_content = pickle.load(backend_path)
            return cache_entry, backend_content
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

    def load(
        self, fn: Callable[..., Any]
    ) -> Optional[tuple[_DynamoCacheEntry, dict[_BackendId, Any]]]:
        """
        Loads a package from a given path and returns it plus a list of deserialized backends
        """
        key = CompilePackage.source_id_from_fn(fn)
        logger.info("Loading CompilePackage for %s", key)
        path = os.path.join(self.path_prefix, key)
        if os.path.exists(path):
            try:
                result = super().load_cache_entry(key)
                return result
            except Exception as e:
                logger.warning("Failed to load package from path %s: %s", path, str(e))
                return None
        logger.info("No package found for %s", key)
        return None

    def load_and_install_package(
        self, fn: Callable[..., Any]
    ) -> Optional[CompilePackage]:
        """
        Load directly into a package and install backends
        """
        results = self.load(fn)
        if results is None:
            return None
        else:
            (entry, backends) = results
            package = CompilePackage(fn, entry)
            package.install(backends)
            return package


DynamoCache = DiskDynamoCache(os.path.join(cache_dir(), "dynamo"))
