"""
This module provides the infrastructure for creating and managing compile package
for torch.compile. We mainly have two abstractions here:
  - CompilePackage: Overarching data structure for store and lookup a list of compiled codes.
  - CodeCacheEntry: Data structure for a single code being compiled by torch.compile.
The caching behavior is always under user control explicitly so that a stronger guarantee can
be provided about cache hit for a specific compiled model. Users can load the compile package
from a different process or host.
"""

import contextlib
import dataclasses
import functools
import hashlib
import importlib
import logging
import os
import pickle
import platform
import sys
import types
from collections.abc import Generator
from typing import Any, NewType, Optional

import torch
import torch._inductor.package
from torch._dynamo.precompile_context import PrecompileCacheArtifact, PrecompileContext
from torch.compiler._cache import CacheArtifactFactory

from .bytecode_transformation import get_code_keys


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
    """

    python_code: SerializedCode
    python_module: str
    function_names: list[_FunctionId]
    guarded_codes: list[_GuardedCodeCacheEntry]
    import_sources: dict[str, str]
    backend_ids: list[_BackendId]


@dataclasses.dataclass
class _DynamoCacheEntry:
    codes: list[_DynamoCodeCacheEntry]
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

    def __init__(self, fn: Any, dynamo: Optional[_DynamoCacheEntry] = None) -> None:
        self._innermost_fn = None
        self._codes: dict[types.CodeType, _DynamoCodeCacheEntry] = {}

        self._current_entry: Optional[_DynamoCodeCacheEntry] = None
        self._installed_globals: dict[types.ModuleType, list[str]] = {}

        # For debugging/testing purpose only.
        self._cached_backends: dict[_BackendId, Any] = {}

        self._initialize(fn, dynamo)
        self.uninstall()
        self.validate()

    def _initialize(self, fn: Any, dynamo: Optional[_DynamoCacheEntry] = None) -> None:
        from .eval_frame import innermost_fn

        self._innermost_fn = innermost_fn(fn)
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

            main, *codes = dynamo.codes
            self._codes = {self._innermost_fn.__code__: main}
            for code in codes:
                self._codes[SerializedCode.to_code_object(code.python_code)] = code
        else:
            self._add_function(
                self._innermost_fn.__code__, self._innermost_fn.__module__
            )

    def _add_function(
        self,
        python_code: types.CodeType,
        python_module: str,
        name: Optional[_FunctionId] = None,
    ) -> None:
        if python_code not in self._codes:
            code = _DynamoCodeCacheEntry(
                python_code=SerializedCode.from_code_object(python_code),
                python_module=python_module,
                function_names=[],
                guarded_codes=[],
                import_sources={},
                backend_ids=[],
            )
            self._codes[python_code] = code
        else:
            code = self._codes[python_code]
            assert code.python_module == python_module

        if name is not None:
            code.function_names.append(name)

    @property
    def cached_backends(self) -> dict[_BackendId, Any]:
        return self._cached_backends

    @functools.cached_property
    def source_id(self) -> str:
        assert self._innermost_fn is not None
        sha256_hash = hashlib.sha256()
        sha256_hash.update(self._innermost_fn.__qualname__.encode())
        sha256_hash.update(str(self._innermost_fn.__code__.co_firstlineno).encode())
        return sha256_hash.hexdigest()

    @contextlib.contextmanager
    def code_context(self, code: types.CodeType) -> Generator[None, None, None]:
        assert self._current_entry is None

        entry = self._codes[code]
        self._current_entry = entry
        try:
            yield
        finally:
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

    def add_resume_function(
        self,
        python_code: types.CodeType,
        python_module: str,
        name: Optional[str],
    ) -> None:
        self._add_function(
            python_code, python_module, _FunctionId(name) if name else None
        )

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

        self.uninstall()

        for code, entry in self._codes.items():
            module = sys.modules[entry.python_module]
            for alias, module_name in entry.import_sources.items():
                self._install_global(
                    module, alias, importlib.import_module(module_name)
                )
            for function_name in entry.function_names:
                fn = types.FunctionType(code, module.__dict__, function_name)
                self._install_global(module, function_name, fn)
            for backend_id in entry.backend_ids:
                if backend_id not in backends:
                    raise RuntimeError(
                        f"Backend {backend_id} is not found in the given backends"
                    )
                backend = backends[backend_id]
                self._install_global(
                    module,
                    backend_id,
                    torch._dynamo.disable(backend),
                )

        for code, entry in self._codes.items():
            for guarded_code in entry.guarded_codes:
                guards_state = pickle.loads(guarded_code.guards_state)
                assert isinstance(guards_state, torch._dynamo.guards.GuardsState)
                check_fn_manager = torch._dynamo.guards.CheckFunctionManager(
                    code,
                    guards_state.output_graph,
                    guards_serialization_mode="load",
                    shape_code_parts=guards_state.shape_code_parts,
                )
                _load_precompile_entry(
                    code,
                    check_fn_manager.guard_manager,
                    SerializedCode.to_code_object(guarded_code.dynamo_code),
                )

    def cache_entry(self) -> _DynamoCacheEntry:
        self.validate()
        return _DynamoCacheEntry(codes=list(self._codes.values()))


@CacheArtifactFactory.register
class EagerCacheArtifact(PrecompileCacheArtifact[Any]):
    @staticmethod
    def type() -> str:
        return "precompile_eager"

    def after_deserialization(self) -> Any:
        return pickle.loads(self.content)


class DynamoStore:
    """
    A DynamoStore tracks active CompilePackages, and provides methods to store and retrieve them.
    """

    def record_package(self, package: CompilePackage) -> None:
        """Records a package to PrecompileContext, so that it can be serialized later."""
        cache_entry = package.cache_entry()
        pickled_result = pickle.dumps(cache_entry)
        PrecompileContext.record_artifact(
            _DynamoCacheArtifact.type(), key=package.source_id, content=pickled_result
        )

    def record_eager_backend(self, backend_id: _BackendId, backend: Any) -> None:
        """Records eager fx graphs to PrecompileContext for testing purposes."""
        pickled_result = pickle.dumps(backend)
        PrecompileContext.record_artifact(
            EagerCacheArtifact.type(), key=backend_id, content=pickled_result
        )

    def save_package(self, package: CompilePackage, path: str) -> None:
        """Saves a package to a given path. Grabs backends from PrecompileContext."""
        backend_content = {}
        cache_entry = package.cache_entry()
        for backend_id in cache_entry.backend_ids:
            serialized_backend = PrecompileContext.serialize_artifact_by_key(backend_id)
            if serialized_backend is None:
                raise RuntimeError(
                    f"Backend {backend_id} is not found in the given backends"
                )
            backend_content[backend_id] = serialized_backend
        try:
            with open(os.path.join(path, "dynamo"), "wb") as dynamo_path:
                pickle.dump(cache_entry, dynamo_path)
            with open(os.path.join(path, "backends"), "wb") as backend_path:
                pickle.dump(backend_content, backend_path)
        except Exception as e:
            raise RuntimeError(f"Failed to save package to {path}: {e}") from e

    def load_package(
        self, fn: Any, path: str
    ) -> tuple[CompilePackage, dict[_BackendId, Any]]:
        """Loads a package from a given path and returns it plus a list of deserialized backends"""
        try:
            with open(os.path.join(path, "dynamo"), "rb") as dynamo_path:
                cache_entry = pickle.load(dynamo_path)
            with open(os.path.join(path, "backends"), "rb") as backend_path:
                backend_content = pickle.load(backend_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load package from path {path}: {e}") from e
        for backend_id, backend in backend_content.items():
            backend_content[backend_id] = backend.after_deserialization()
        package = CompilePackage(fn, cache_entry)
        return package, backend_content
