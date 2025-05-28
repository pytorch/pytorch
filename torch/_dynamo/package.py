"""
This module provides the infrastructure for creating and managing compile package
for torch.compile. We mainly have two abstractions here:
  - CompilePackage: Overarching data structure for store and lookup a list of precompiles.
  - FunctionState: Data structure for a single function compiled by torch.compile.
  - GuardedCodeState: Artifacts related to one particular dynamo compilation.
The caching behavior is always under user control explicitly so that a stronger guarantee can
be provided about cache hit for a specific compiled model. Users can load the compile package
from a different process or even host but cautions should be taken that compile package will
only check a subset of the original Dynamo guards so there might be soundness problems.
"""

import contextlib
import dataclasses
import importlib
import json
import logging
import os
import pickle
import platform
import sys
import types
import typing
from collections.abc import Generator
from typing import Any, NewType, Optional

import torch
import torch._inductor.package

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
    co_linetable: bytes
    co_cellvars: tuple[str, ...]
    co_freevars: tuple[str, ...]
    co_qualname: Optional[str] = None
    co_exceptiontable: Optional[bytes] = None

    @classmethod
    def from_code_object(cls, code: types.CodeType) -> "SerializedCode":
        kwargs = {key: getattr(code, key) for key in get_code_keys()}
        kwargs["co_consts"] = tuple(
            cls.from_code_object(c) if isinstance(c, types.CodeType) else c
            for c in kwargs["co_consts"]
        )
        return cls(**kwargs)

    @classmethod
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
class _GuardedCodeState:
    """
    Contains all the serializable information associated with a single
    compilation in dynamo. To restore an execution of compiled code, we will
    need to serialize the following (not exhaustive):
      - Dynamo bytecode for mapping Python inputs/outputs.
      - Dynamo guards.
    """

    dynamo_code: SerializedCode
    guards_state: bytes


_BackendState = typing.Union[torch.fx.GraphModule,]
_BackendId = NewType("_BackendId", str)  # __compiled_fn
_FunctionId = NewType("_FunctionId", str)  # __resume_at


@dataclasses.dataclass
class _CodeState:
    python_code: SerializedCode
    python_module: str
    function_names: list[_FunctionId]
    guarded_codes: list[_GuardedCodeState]
    import_sources: dict[str, str]
    backend_ids: list[_BackendId]


@dataclasses.dataclass
class _DynamoState:
    code_states: list[_CodeState]


class _PackageWriter:
    def __init__(self, path: str, backend_type: str):
        if not (os.path.exists(path) and os.path.isdir(path)):
            raise RuntimeError(f"Compile package path '{path}' doesn't exist.")
        self.path = path
        self.backend_type = backend_type
        self.backend_ids: list[str] = []

    def _write_pickle(self, data, *path: str):
        with open(os.path.join(self.path, *path) + ".pickle", "wb") as f:
            pickle.dump(data, f)

    def _write_json(self, data, *path: str):
        with open(os.path.join(self.path, *path) + ".json", "w") as f:
            json.dump(data, f)

    def write_package_info(self, package_info):
        self._write_json(package_info, "package_info")

    def write_dynamo(self, dynamo_state: _DynamoState):
        self._write_pickle(dynamo_state, "dynamo")

    def write_backend(self, backend_id: _BackendId, backend: _BackendState):
        os.makedirs(os.path.join(self.path, backend_id), exist_ok=True)
        if self.backend_type == "eager":
            self._write_pickle(backend, backend_id, "fx_graph")
        else:
            raise NotImplementedError(f"Unsupported backend {self.backend_type}")
        self.backend_ids.append(backend_id)


class _PackageReader:
    def __init__(self, path: str, backend_type: str):
        self.path = path
        self.backend_type = backend_type
        self.loaded_graph_states = {}

    def _read_pickle(self, *path):
        with open(os.path.join(self.path, *path) + ".pickle", "rb") as f:
            return pickle.load(f)

    def _read_json(self, *path):
        with open(os.path.join(self.path, *path) + ".json") as f:
            return json.load(f)

    def read_package_info(self):
        return self._read_json("package_info")

    def read_backend(self, backend_id: str) -> _BackendState:
        if self.backend_type == "eager":
            return self._read_pickle(backend_id, "fx_graph")
        else:
            raise NotImplementedError(f"Unsupported backend {self.backend_type}")

    def read_dynamo(self) -> _DynamoState:
        return self._read_pickle("dynamo")


class _CompilePackage:
    """
    The main entry point of compile package system. This data structure should be created
    per torch.compile() call and propagated through the layers to collect compiled
    artifacts from Dynamo, AOTAutograd and Inductor. This essentially maintains a
    list of (guards, compiled code) which will be looked up in order when a set of
    new inputs are passed to compiled object.
    """

    def __init__(self, backend_type: str):
        self._innermost_fn = None
        self._backend_type = backend_type
        self._code_states: dict[types.CodeType, _CodeState] = {}
        self._backends: dict[_BackendId, _BackendState] = {}

        self._current_code: Optional[_CodeState] = None
        self._installed_globals: dict[types.ModuleType, list[str]] = {}

    def _add_function(
        self,
        python_code: types.CodeType,
        python_module: str,
        name: Optional[_FunctionId] = None,
    ) -> None:
        if python_code not in self._code_states:
            code_state = _CodeState(
                python_code=SerializedCode.from_code_object(python_code),
                python_module=python_module,
                function_names=[],
                guarded_codes=[],
                import_sources={},
                backend_ids=[],
            )
            self._code_states[python_code] = code_state
        else:
            code_state = self._code_states[python_code]
            assert code_state.python_module == python_module

        if name is not None:
            code_state.function_names.append(name)

    def initialize(self, fn):
        from .eval_frame import innermost_fn

        if self._innermost_fn is not None:
            raise RuntimeError("Compile package is already initialized.")
        self._innermost_fn = innermost_fn(fn)
        self._add_function(self._innermost_fn.__code__, self._innermost_fn.__module__)

    @contextlib.contextmanager
    def function_context(self, code: types.CodeType) -> Generator[None, None, None]:
        assert self._current_code is None

        code_state = self._code_states[code]
        self._current_code = code_state
        try:
            yield
        finally:
            self._current_code = None

    def add_guarded_code(
        self,
        dynamo_code: types.CodeType,
        guards_state: bytes,
    ) -> None:
        guarded_code_state = _GuardedCodeState(
            SerializedCode.from_code_object(dynamo_code), guards_state
        )
        self._current_code.guarded_codes.append(guarded_code_state)

    def add_resume_function(
        self,
        python_code: types.CodeType,
        python_module: str,
        name: Optional[str],
    ) -> None:
        self._add_function(
            python_code, python_module, _FunctionId(name) if name else None
        )

    def add_import_source(self, alias: str, module_name: str):
        self._current_function.import_sources[alias] = module_name

    def add_backend(self, name: str, backend: _BackendState) -> None:
        backend_id = _BackendId(name)
        self._backends[backend_id] = backend
        self._current_code.backend_ids.append(backend_id)

    def unimplemented(self, msg: str) -> None:
        raise NotImplementedError(
            f"Feature not implemented yet for compile package: {msg}."
        )

    def validate(self) -> None:
        assert self._current_code is None
        for fn in self._code_states.values():
            for backend_id in fn.backend_ids:
                assert backend_id in self._backends
        assert next(iter(self._code_states)) is self._innermost_fn.__code__

    def _install_global(self, module: types.ModuleType, name: str, value):
        module.__dict__[name] = value
        self._installed_globals.setdefault(module, []).append(name)

    def _apply(self) -> None:
        from torch._C._dynamo.eval_frame import (
            _load_precompile_entry,
            _reset_precompile_entries,
        )

        for module, names in self._installed_globals.items():
            for name in names:
                module.__dict__.pop(name)

        self._installed_globals = {}

        _reset_precompile_entries(self._innermost_fn.__code__)

        for code, state in self._code_states.items():
            module = sys.modules[state.python_module]
            for alias, module_name in state.import_sources.items():
                self._install_global(
                    module, alias, importlib.import_module(module_name)
                )
            for function_name in state.function_names:
                fn = types.FunctionType(code, module.__dict__, function_name)
                self._install_global(module, function_name, fn)
            for backend_id in state.backend_ids:
                self._install_global(module, backend_id, self._backends[backend_id])

        for code, state in self._code_states.items():
            for guarded_code in state.guarded_codes:
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

    def save(self, path) -> None:
        self.validate()
        os.makedirs(path, exist_ok=True)
        writer = _PackageWriter(path, self._backend_type)
        for backend_id, backend in self._backends.items():
            writer.write_backend(backend_id, backend)
        writer.write_dynamo(_DynamoState(code_states=list(self._code_states.values())))
        writer.write_package_info(
            {
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "backend_type": self._backend_type,
                "backend_ids": writer.backend_ids,
            }
        )

    def load(self, path) -> None:
        reader = _PackageReader(path, self._backend_type)
        package_info = reader.read_package_info()

        if package_info["backend_type"] != self._backend_type:
            raise RuntimeError(
                f"Compile package was created with a different backend: {package_info['backend']}"
            )

        if package_info["python_version"] != platform.python_version():
            raise RuntimeError(
                f"Compile package was created with a different Python version: {package_info['python_version']}"
            )

        self._backends = {}
        for backend_id in package_info["backend_ids"]:
            backend = reader.read_backend(backend_id)
            self._backends[backend_id] = torch._dynamo.decorators.disable(backend)

        dynamo_state = reader.read_dynamo()
        code_state, *code_states = dynamo_state.code_states
        self._code_states = {self._innermost_fn.__code__: code_state}
        for code_state in code_states:
            self._code_states[SerializedCode.to_code_object(code_state.python_code)] = (
                code_state
            )

        self.validate()

        self._apply()
