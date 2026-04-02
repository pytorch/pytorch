# Owner(s): ["module: dsl-native-ops"]

import functools
import logging
from typing import Protocol

from packaging.version import Version

from .registry import _OpFn


log = logging.getLogger(__name__)


class DSLModuleProtocol(Protocol):
    """Complete interface for DSL utility modules"""

    def runtime_available(self) -> bool: ...
    def runtime_version(self) -> Version | None: ...

    def deregister_op_overrides(self) -> None: ...

    def register_op_override(
        self,
        lib_symbol: str,
        op_symbol: str,
        dispatch_key: str,
        impl: _OpFn,
        *,
        allow_multiple_override: bool = False,
        unconditional_override: bool = False,
    ) -> None: ...


class DSLRegistry:
    """Registry for DSL modules - calls their existing API functions dynamically"""

    def __init__(self):
        self._dsl_modules: dict[str, DSLModuleProtocol] = {}

    def _validate_dsl_name(self, name: str) -> None:
        """Validate DSL name at runtime"""
        if not isinstance(name, str):
            raise TypeError(f"DSL name must be string, got {type(name).__name__}")

        if not name.strip():
            raise ValueError("DSL name cannot be empty or whitespace")

    def register_dsl(self, name: str, dsl_module: DSLModuleProtocol) -> None:
        """Register a DSL module with required interface"""
        # Runtime validation for name and module interface
        self._validate_dsl_name(name)

        # Validate that module implements the protocol
        required_methods = [
            "runtime_available",
            "runtime_version",
            "register_op_override",
            "deregister_op_overrides",
        ]
        missing_methods = [
            method for method in required_methods if not hasattr(dsl_module, method)
        ]
        if missing_methods:
            raise TypeError(
                f"DSL module '{name}' missing required methods: {missing_methods}"
            )

        # Handle duplicate registration case
        if name in self._dsl_modules:
            existing_module = self._dsl_modules[name]
            if existing_module is dsl_module:
                # Same module re-registering - this is OK (import-time registration)
                log.debug(
                    "DSL '%s' re-registered with same module",
                    name,
                )
                return
            else:
                # Different module object but same name - warn and allow (for testing)
                # This can happen when tests import modules directly
                log.warning(
                    "DSL '%s' re-registered with different module object (possibly from test imports)",
                    name,
                )
                # Continue to allow the registration

        # No cast needed - already properly typed
        self._dsl_modules[name] = dsl_module

        # Clear caches to prevent stale results after registration
        self.is_dsl_available.cache_clear()
        self.get_dsl_version.cache_clear()
        self.list_available_dsls.cache_clear()
        self.list_all_dsls.cache_clear()

        log.info("Successfully registered DSL: %s", name)

    @functools.cache  # noqa: B019
    def is_dsl_available(self, dsl_name: str) -> bool:
        """Check if DSL is available by calling its runtime_available()"""
        dsl_module = self._dsl_modules.get(dsl_name)
        if dsl_module is None:
            return False
        try:
            return dsl_module.runtime_available()
        except ImportError:
            log.debug("DSL %s import error", dsl_name, exc_info=True)
            return False
        except Exception:
            log.exception("Error checking availability for DSL %s", dsl_name)
            return False

    @functools.cache  # noqa: B019
    def get_dsl_version(self, dsl_name: str) -> Version | None:
        """Get DSL version by calling its runtime_version()"""
        dsl_module = self._dsl_modules.get(dsl_name)
        if dsl_module is None:
            return None
        try:
            return dsl_module.runtime_version()
        except Exception:
            log.debug("Error getting version for DSL %s", dsl_name, exc_info=True)
            return None

    @functools.cache  # noqa: B019
    def list_available_dsls(self) -> tuple[str, ...]:
        """Get names of currently available DSLs"""
        available = []
        for name in self._dsl_modules:
            if self.is_dsl_available(name):  # Use cached method
                available.append(name)
        return tuple(available)

    @functools.cache  # noqa: B019
    def list_all_dsls(self) -> tuple[str, ...]:
        """Get all registered DSL names (available or not)"""
        return tuple(self._dsl_modules.keys())


# Global registry instance
dsl_registry = DSLRegistry()
