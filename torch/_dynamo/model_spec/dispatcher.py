"""
Dispatcher generation for ModelSpec.

This module will contain the DispatcherFunction class in later phases.
For now, it's a placeholder to support imports.
"""

from dataclasses import dataclass
from typing import Any, Callable


__all__ = ["DispatcherFunction", "CompiledVariant"]


@dataclass
class CompiledVariant:
    """
    A single compiled variant for specific conditions.

    This will be fully implemented later.
    """

    compiled_fn: Callable[..., Any]
    metadata: dict[str, Any]


class DispatcherFunction:
    """
    Executable dispatcher generated from ModelSpec.

    This will be fully implemented later.
    For now, it's a placeholder.
    """

    def __init__(
        self,
        spec: Any,
        compile_fn: Callable[..., Any],
        compiled_variants: dict[str, Any],
    ) -> None:
        self.spec = spec
        self.compile_fn = compile_fn
        self.compiled_variants = compiled_variants

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "DispatcherFunction.__call__ will be implemented later"
        )

    def dump_guards(self) -> None:
        raise NotImplementedError(
            "DispatcherFunction.dump_guards will be implemented later"
        )

    def print_readable(self) -> None:
        raise NotImplementedError(
            "DispatcherFunction.print_readable will be implemented later"
        )
