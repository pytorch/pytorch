"""
Python execution state recording and replay functionality.

This module provides mechanisms for capturing and replaying Python execution state:

- ModuleRecord: Tracks module access patterns and attribute usage
- DummyModule: Lightweight module substitute for replay
- ExecutionRecord: Manages execution context including globals, locals and builtins
- ExecutionRecorder: Records variable states and module access during execution

The module enables serialization and reproduction of Python execution environments,
particularly useful for debugging and testing frameworks that need to capture
and recreate specific program states.
"""

import dataclasses
from dataclasses import field
from types import CellType, CodeType, ModuleType
from typing import Any, IO
from typing_extensions import Self

from torch.utils._import_utils import import_dill


dill = import_dill()


@dataclasses.dataclass
class ModuleRecord:
    module: ModuleType
    accessed_attrs: dict[str, Any] = field(default_factory=dict)


@dataclasses.dataclass
class DummyModule:
    name: str
    is_torch: bool = False
    value: object = None

    @property
    def __name__(self) -> str:
        return self.name


@dataclasses.dataclass
class ExecutionRecord:
    code: CodeType
    closure: tuple[CellType]
    globals: dict[str, Any] = field(default_factory=dict)
    locals: dict[str, Any] = field(default_factory=dict)
    builtins: dict[str, Any] = field(default_factory=dict)
    code_options: dict[str, Any] = field(default_factory=dict)

    def dump(self, f: IO[str]) -> None:
        assert dill is not None, "replay_record requires `pip install dill`"
        dill.dump(self, f)

    @classmethod
    def load(cls, f: IO[bytes]) -> Self:
        assert dill is not None, "replay_record requires `pip install dill`"
        return dill.load(f)


@dataclasses.dataclass
class ExecutionRecorder:
    LOCAL_MOD_PREFIX = "___local_mod_"

    code: CodeType
    closure: tuple[CellType]
    globals: dict[str, Any] = field(default_factory=dict)
    locals: dict[str, Any] = field(default_factory=dict)
    builtins: dict[str, Any] = field(default_factory=dict)
    code_options: dict[str, Any] = field(default_factory=dict)
    name_to_modrec: dict[str, ModuleRecord] = field(default_factory=dict)

    def add_local_var(self, name: str, var: Any) -> None:
        if isinstance(var, ModuleType):
            self.locals[name] = self._add_mod(var)
        else:
            self.locals[name] = var

    def add_global_var(self, name: str, var: Any) -> None:
        if isinstance(var, ModuleType):
            self.globals[name] = self._add_mod(var)
        else:
            self.globals[name] = var

    def add_local_mod(self, name: str, mod: ModuleType) -> None:
        assert isinstance(mod, ModuleType)
        self.add_global_var(name, mod)

    def record_module_access(self, mod: ModuleType, name: str, val: Any) -> None:
        if isinstance(val, ModuleType):
            self.name_to_modrec[mod.__name__].accessed_attrs[name] = self._add_mod(val)
            return

        if mod.__name__ in self.name_to_modrec:
            self.name_to_modrec[mod.__name__].accessed_attrs[name] = val

    def get_record(self) -> ExecutionRecord:
        return ExecutionRecord(
            self.code,
            self.closure,
            ExecutionRecorder._resolve_modules(self.globals),
            ExecutionRecorder._resolve_modules(self.locals),
            self.builtins.copy(),
            self.code_options.copy(),
        )

    def _add_mod(self, mod: ModuleType) -> ModuleRecord:
        if mod.__name__ not in self.name_to_modrec:
            self.name_to_modrec[mod.__name__] = ModuleRecord(mod)

        return self.name_to_modrec[mod.__name__]

    @classmethod
    def _resolve_modules(cls, vars: dict[str, Any]) -> dict[str, Any]:
        def resolve_module(var: Any) -> Any:
            if not isinstance(var, ModuleRecord):
                return var

            dummy_mod = DummyModule(var.module.__name__)
            for attr_name, attr_value in var.accessed_attrs.items():
                attr_value = resolve_module(attr_value)
                dummy_mod.__setattr__(attr_name, attr_value)

            return dummy_mod

        return {k: resolve_module(v) for k, v in vars.items()}
