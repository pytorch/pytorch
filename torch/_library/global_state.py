from __future__ import annotations

from typing import Any, cast, TYPE_CHECKING, TypeVar
from weakref import WeakKeyDictionary, WeakValueDictionary


if TYPE_CHECKING:
    from collections.abc import Callable


def _get_none() -> None:
    return None


_T = TypeVar("_T")


class LibraryGlobalState:
    """
    Central storage for Python-side torch.library registries.

    These containers are intentionally still re-exported from their historical
    modules for compatibility. New shared registry state should live here so
    lifecycle and cleanup behavior is easier to audit.
    """

    def __init__(self) -> None:
        # Keep this module dependency-free: most concrete registry value types
        # are defined in modules that import global_state, so annotations here
        # intentionally use Any to avoid circular imports.
        self.impls: set[str] = set()
        self.defs: set[str] = set()
        self.keep_alive: list[Any] = []

        self.simple_registry: Any | None = None
        self.global_ctx_getter: Callable[[], Any] = _get_none

        self.custom_opdefs: WeakValueDictionary[str, Any] = WeakValueDictionary()
        self.custom_opdef_to_lib: dict[str, Any] = {}
        self.legacy_custom_op_registry: dict[str, Any] = {}

        self.fake_class_registry: Any | None = None
        self.triton_ops_to_kernels: dict[str, list[object]] = {}

        self.opaque_types: WeakKeyDictionary[Any, Any] = WeakKeyDictionary()
        self.opaque_types_by_name: dict[str, Any] = {}

    def get_or_create_simple_registry(self, factory: type[_T]) -> _T:
        # These registries are lazy because their concrete classes live in
        # modules that import global_state.
        if self.simple_registry is None:
            self.simple_registry = factory()
        return cast(_T, self.simple_registry)

    def get_or_create_fake_class_registry(self, factory: type[_T]) -> _T:
        # These registries are lazy because their concrete classes live in
        # modules that import global_state.
        if self.fake_class_registry is None:
            self.fake_class_registry = factory()
        return cast(_T, self.fake_class_registry)


library_state = LibraryGlobalState()


__all__ = ["LibraryGlobalState", "library_state"]
