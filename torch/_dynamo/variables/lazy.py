from __future__ import annotations

import collections
import functools
import inspect
from typing import Any, TYPE_CHECKING

from .. import config
from ..utils import is_function_or_wrapper
from .base import VariableTracker, VariableTrackerMeta


if TYPE_CHECKING:
    from collections.abc import Callable
    from typing_extensions import Self

    from .tensor import SymNodeVariable


class LazyCache:
    """Container to cache the real VariableTracker"""

    def __init__(self, value: Any, source: Any) -> None:
        if not isinstance(value, LazySymNodeFormatString):
            assert source
        self.value = value
        self.source = source
        self.name_hint: str | None = None
        self.vt: VariableTracker | None = None

    def realize(self) -> None:
        assert self.vt is None
        from ..symbolic_convert import InstructionTranslator
        from . import builder

        tx = InstructionTranslator.current_tx()

        if isinstance(self.value, LazySymNodeFormatString):
            self.vt = builder.SourcelessBuilder.create(tx, self.value)
        else:
            # Pass allow_lazy_constant=False to prevent VariableBuilder from
            # returning LazyConstantVariable, which would cause infinite recursion
            # when LazyVariableTracker.realize() returns LazyConstantVariable.
            self.vt = builder.VariableBuilder(
                tx, self.source, allow_lazy_constant=False
            )(self.value)

        if self.name_hint is not None:
            self.vt.set_name_hint(self.name_hint)

        del self.value
        del self.source
        del self.name_hint


class LazyVariableTracker(VariableTracker, metaclass=VariableTrackerMeta):
    """
    A structure that defers the creation of the actual VariableTracker
    for a given underlying value until it is accessed.

    The `realize` function invokes VariableTracker.build() to produce the real object.
    Once a LazyVariableTracker has been realized, internal bookkeeping will
    prevent double realization.

    This object should be utilized for processing containers, or objects that
    reference other objects where we may not want to take on creating all the
    VariableTrackers right away.
    """

    # Flag to prevent implicit realization in isinstance checks (inherited by subclasses)
    _no_implicit_realize = True
    _nonvar_fields = {"_cache", *VariableTracker._nonvar_fields}

    @staticmethod
    def create(value: Any, source: Any, **options: Any) -> VariableTracker:
        if type(value) in LazyConstantVariable.supported_types:
            return LazyConstantVariable.create(value, source, **options)
        return LazyVariableTracker(LazyCache(value, source), source=source, **options)

    def __init__(self, _cache: LazyCache, **kwargs: Any) -> None:
        assert isinstance(_cache, LazyCache)
        super().__init__(**kwargs)
        self._cache = _cache

    def realize(self) -> VariableTracker:
        """Force construction of the real VariableTracker"""
        if self._cache.vt is None:
            self._cache.realize()
            assert self._cache.vt is not None
        return self._cache.vt

    def lazy_isinstance(self, cls: type) -> bool:
        """Check isinstance after realizing, used by ImplicitRealizingVariableTrackerMeta"""
        return type.__instancecheck__(cls, self.realize())

    def unwrap(self) -> VariableTracker | Self:
        """Return the real VariableTracker if it already exists"""
        if self.is_realized():
            assert self._cache.vt is not None
            return self._cache.vt
        return self

    def is_realized(self) -> bool:
        return self._cache.vt is not None

    def clone(self, **kwargs: Any) -> VariableTracker:
        assert kwargs.get("_cache", self._cache) is self._cache
        if kwargs.get("source", self.source) is not self.source:
            self.realize()
        return VariableTracker.clone(self.unwrap(), **kwargs)

    def peek_type(self) -> type[Any]:
        assert not self.is_realized()
        return type(self._cache.value)

    def peek_value(self) -> Any:
        assert not self.is_realized()
        return self._cache.value

    def set_name_hint(self, name: str) -> None:
        if self.is_realized():
            self._cache.vt.set_name_hint(name)  # type: ignore[union-attr]
        else:
            self._cache.name_hint = name

    def __str__(self) -> str:
        variable_info = "LazyVariableTracker("
        if self.is_realized():
            variable_info += f"realized: {repr(self.unwrap())})"
        else:
            variable_info += f"unrealized: {self.peek_type()})"

        return variable_info

    def __getattr__(self, item: str) -> Any:
        return getattr(self.realize(), item)

    def get_handler_type_for_dispatch(self) -> type:
        """Return the VariableTracker type to use for builtin handler dispatch.

        For regular LazyVariableTracker (not LazyConstantVariable), we return
        LazyVariableTracker itself so that _make_handler knows it needs to realize
        the arguments before calling the handler. LazyConstantVariable overrides
        this to return ConstantVariable since it can stay lazy.
        """
        return LazyVariableTracker

    # most methods are auto-generated below, these are the ones we want to exclude
    visit = VariableTracker.visit  # type: ignore[assignment]
    __repr__ = __str__

    @classmethod
    def realize_all(
        cls,
        value: Any,
        cache: dict[int, tuple[Any, Any]] | None = None,
        *,
        allow_lazy_constant: bool = False,
    ) -> Any:
        """
        Walk an object and realize all LazyVariableTrackers inside it.
        """
        if cache is None:
            cache = {}

        idx = id(value)
        if idx in cache:
            return cache[idx][0]

        value_cls = type(value)
        if issubclass(value_cls, LazyVariableTracker):
            # Allow LazyConstantVariable to stay lazy when returning from a frame
            keep_lazy = allow_lazy_constant and isinstance(value, LazyConstantVariable)
            if keep_lazy:
                result = value
            else:
                result = cls.realize_all(
                    value.realize(), cache, allow_lazy_constant=allow_lazy_constant
                )
        elif issubclass(value_cls, VariableTracker):
            # update value in-place
            result = value
            value_dict = value.__dict__
            nonvars = value._nonvar_fields
            for key in value_dict:
                if key not in nonvars:
                    value_dict[key] = cls.realize_all(
                        value_dict[key], cache, allow_lazy_constant=allow_lazy_constant
                    )
        elif value_cls is list:
            result = [
                cls.realize_all(v, cache, allow_lazy_constant=allow_lazy_constant)
                for v in value
            ]
        elif value_cls is tuple:
            result = tuple(
                cls.realize_all(v, cache, allow_lazy_constant=allow_lazy_constant)
                for v in value
            )
        elif value_cls in (dict, collections.OrderedDict):
            result = {
                k: cls.realize_all(v, cache, allow_lazy_constant=allow_lazy_constant)
                for k, v in list(value.items())
            }
        else:
            result = value

        # save `value` to keep it alive and ensure id() isn't reused
        cache[idx] = (result, value)
        return result

    def is_hashable(self) -> bool:
        # Checks that the underlying value is hashable without realizing the VT.
        # This is used by ConstDictVariable tracker to find if the key LazyVT
        # can be hashed.
        def _helper(value: Any) -> bool:
            # TODO: Add support for more types
            return (
                inspect.isbuiltin(value)
                or issubclass(type(value), type)
                or is_function_or_wrapper(value)
            )

        assert not self.is_realized()
        value = self._cache.value
        if isinstance(value, tuple):
            return all(_helper(v) for v in value)
        return _helper(value)

    def original_value(self) -> Any:
        # Returns the value without realizing the VT.
        assert not self.is_realized()
        return self._cache.value

    def original_source(self) -> Any:
        # Returns the source without realizing the VT.
        assert not self.is_realized()
        return self._cache.source


class LazyConstantVariable(LazyVariableTracker):
    """
    A lazy variable tracker for constants (int, float, bool, str) that defers
    guarding until the value is actually used in a way that requires it.

    This allows constants that are just passed through (e.g., returned without
    being used in control flow or math) to avoid unnecessary recompilation when
    their values change.

    Guards are installed lazily:
    - TYPE_MATCH guard is installed when type-based methods (python_type, is_tensor,
      lazy_isinstance) are called
    - CONSTANT_MATCH guard is installed on full realization (e.g., used in control
      flow or math), which subsumes any TYPE_MATCH guard
    """

    supported_types = (int, float, bool, str)
    _nonvar_fields = {"_type_guard_installed", *LazyVariableTracker._nonvar_fields}

    @staticmethod
    def create(  # pyrefly: ignore[bad-override]
        value: Any,
        source: Any,
        **options: Any,
    ) -> VariableTracker:
        from ..source import is_constant_source
        from .constant import ConstantVariable

        assert type(value) in LazyConstantVariable.supported_types
        assert source is not None

        # If the source doesn't support guards (e.g., ConstantSource), fall back
        # to creating a regular ConstantVariable directly
        if is_constant_source(source):
            return ConstantVariable.create(value, source=source, **options)

        return LazyConstantVariable(LazyCache(value, source), source=source, **options)

    def __init__(self, _cache: LazyCache, **kwargs: Any) -> None:
        super().__init__(_cache, **kwargs)
        self._type_guard_installed = False

    def _ensure_type_guard(self) -> None:
        """Install TYPE_MATCH guard if not already installed and not realized."""
        if self._type_guard_installed or self.is_realized():
            return

        from ..guards import GuardBuilder, install_guard

        assert self.source is not None
        install_guard(self.source.make_guard(GuardBuilder.TYPE_MATCH))
        self._type_guard_installed = True

    def realize(self) -> VariableTracker:
        """Force construction of the real VariableTracker."""
        if self.is_realized():
            return super().realize()

        from torch._guards import TracingContext

        from ..guards import GuardBuilder, install_guard
        from .constant import ConstantVariable

        tracing_context = TracingContext.get()
        assert self.source is not None

        # Realize first to see what we get
        result = super().realize()

        # Only remove TYPE_MATCH if we're installing CONSTANT_MATCH
        # (which subsumes it). For SymNodeVariable, keep TYPE_MATCH.
        if isinstance(result, ConstantVariable):
            if self._type_guard_installed:
                tracing_context.guards_context.dynamo_guards.remove_guards_with_source(
                    self.source
                )
            constant_guard = self.source.make_guard(GuardBuilder.CONSTANT_MATCH)
            install_guard(constant_guard)

        return result

    def python_type(self) -> type:
        """Return the Python type without triggering realization."""
        if self.is_realized():
            return super().python_type()
        self._ensure_type_guard()
        return self.peek_type()

    def is_tensor(self) -> bool:
        """Primitive constants are never tensors."""
        self._ensure_type_guard()
        return False

    def is_constant_none(self) -> bool:
        self._ensure_type_guard()
        return False

    def _maybe_realize_for_type(self) -> type | None:
        """Check if we need to realize to determine the VariableTracker type.

        Returns None if we can determine the type without realization (and installs
        TYPE_MATCH guard). Returns the realized type if realization was needed.

        With specialize_int=False or specialize_float=False, ints/floats may become
        either ConstantVariable or SymNodeVariable, so we must realize to know.
        For bool/str, we always know it will be ConstantVariable.
        """
        if self.is_realized():
            return type(self.realize())

        value_type = self.peek_type()

        # When specialize_int/specialize_float is False, ints/floats may become
        # SymNodeVariable. Must realize to determine the actual type.
        if not config.specialize_int and value_type is int:
            return type(self.realize())
        if not config.specialize_float and value_type is float:
            return type(self.realize())

        # For bool/str, or when specializing ints/floats, we know it will be
        # ConstantVariable. Install TYPE_MATCH guard and return None.
        self._ensure_type_guard()
        return None

    def get_handler_type_for_dispatch(self) -> type:
        """Return the VariableTracker type to use for builtin handler dispatch.

        This allows builtins like isinstance() and type() to find the correct
        handler without triggering full realization of the lazy constant.
        """
        from .constant import ConstantVariable

        realized_type = self._maybe_realize_for_type()
        return realized_type if realized_type is not None else ConstantVariable

    def lazy_isinstance(self, cls: type) -> bool:
        """Check isinstance without triggering realization when possible.

        LazyConstantVariable only wraps primitive types (int, float, bool, str)
        which always realize to ConstantVariable, so we can answer isinstance
        checks by checking if the target class is ConstantVariable or a parent.

        However, when specialize_int=False or specialize_float=False, integers
        and floats may realize to SymNodeVariable instead of ConstantVariable,
        so we must fall back to full realization for those cases.
        """
        from .constant import ConstantVariable
        from .tensor import SymNodeVariable

        # If already realized, delegate to the parent which does the regular check
        if self.is_realized():
            return super().lazy_isinstance(cls)

        # LazyConstantVariable can only realize to ConstantVariable or SymNodeVariable.
        # If cls is not a parent of either, we can answer False without realization.
        if not issubclass(ConstantVariable, cls) and not issubclass(
            SymNodeVariable, cls
        ):
            self._ensure_type_guard()
            return False

        realized_type = self._maybe_realize_for_type()
        if realized_type is not None:
            return issubclass(realized_type, cls)
        return issubclass(ConstantVariable, cls)


class LazySymNodeFormatString:
    def __init__(
        self, sym_node_variable: SymNodeVariable, fmt_spec_var: VariableTracker
    ) -> None:
        from .constant import ConstantVariable

        self.sym_node_var = sym_node_variable
        self.fmt_var = ConstantVariable.create(
            "{:" + fmt_spec_var.as_python_constant() + "}"
        )

    def __repr__(self) -> str:
        return str.format(
            self.fmt_var.as_python_constant(),
            str(self.sym_node_var.evaluate_expr()),
        )


def _create_realize_and_forward(
    name: str,
) -> Callable[[LazyVariableTracker, Any, Any], Any]:
    @functools.wraps(getattr(VariableTracker, name))
    def realize_and_forward(
        self: LazyVariableTracker, *args: Any, **kwargs: Any
    ) -> Any:
        return getattr(self.realize(), name)(*args, **kwargs)

    return realize_and_forward


def _populate() -> None:
    for name, value in VariableTracker.__dict__.items():
        if name not in LazyVariableTracker.__dict__:
            if callable(value):
                setattr(LazyVariableTracker, name, _create_realize_and_forward(name))


_populate()
