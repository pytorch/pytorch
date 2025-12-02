import collections
import functools
import inspect
from collections.abc import Callable, Hashable
from typing import Any, Optional, Union
from typing_extensions import Self

from ..utils import is_function_or_wrapper
from .base import VariableTracker, VariableTrackerMeta
from .tensor import SymNodeVariable


class LazyCache:
    """Container to cache the real VariableTracker"""

    def __init__(self, value: Any, source: Any) -> None:
        if not isinstance(value, LazySymNodeFormatString):
            assert source
        self.value = value
        self.source = source
        self.name_hint: Optional[str] = None
        self.vt: Optional[VariableTracker] = None

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
            # pyrefly: ignore [missing-attribute]
            self.vt.set_name_hint(self.name_hint)

        del self.value
        del self.source
        del self.name_hint


# Flag to prevent implicit realization in isinstance checks (inherited by subclasses)
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

    _no_implicit_realize = True
    _nonvar_fields = {"_cache", *VariableTracker._nonvar_fields}

    @staticmethod
    def create(value: Any, source: Any, **options: Any) -> "LazyVariableTracker":
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

    def unwrap(self) -> Union[VariableTracker, Self]:
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

    # most methods are auto-generated below, these are the ones we want to exclude
    visit = VariableTracker.visit  # type: ignore[assignment]
    __repr__ = __str__

    @classmethod
    def realize_all(
        cls,
        value: Any,
        cache: Optional[dict[int, tuple[Any, Any]]] = None,
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
            # Only return True for types that are KNOWN to be safely hashable
            # without needing the full VT machinery. For tensors and modules,
            # we need to go through the proper VT path to get consistent hashing
            # (e.g., using FakeTensor for TensorVariable).
            # Note: isinstance(value, Hashable) is too broad - it includes tensors
            # and modules which need special handling.
            return (
                isinstance(value, (int, float, bool, str, type(None), frozenset))
                or inspect.isbuiltin(value)
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

    When realized, calls the realize_fn to create the appropriate VT with guards.
    If no realize_fn is provided, creates a ConstantVariable with CONSTANT_MATCH guard.
    """

    _nonvar_fields = {
        "_realize_fn",
        *LazyVariableTracker._nonvar_fields,
    }

    @staticmethod
    def create(  # pyrefly: ignore[bad-override]
        value: Any,
        source: Any,
        realize_fn: Optional[Callable[[], VariableTracker]] = None,
        **options: Any,
    ) -> "LazyConstantVariable":
        return LazyConstantVariable(
            LazyCache(value, source), realize_fn=realize_fn, source=source, **options
        )

    def __init__(
        self,
        _cache: LazyCache,
        realize_fn: Optional[Callable[[], VariableTracker]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(_cache, **kwargs)
        self._realize_fn = realize_fn

    def realize(self) -> VariableTracker:
        """Force construction of the real VariableTracker with guards."""
        if self._cache.vt is None:
            if self._realize_fn is not None:
                # Use custom realize function (e.g., for wrap_symint/wrap_symfloat)
                self._cache.vt = self._realize_fn()
            else:
                # Default: use VariableBuilder to create the appropriate VT with guards.
                # Pass allow_lazy_constant=False to prevent returning another
                # LazyConstantVariable (which would cause infinite recursion).
                from ..symbolic_convert import InstructionTranslator
                from . import builder

                tx = InstructionTranslator.current_tx()
                value = self._cache.value
                source = self._cache.source
                self._cache.vt = builder.VariableBuilder(
                    tx, source, allow_lazy_constant=False
                )(value)

            # Clean up cache (mirroring LazyCache.realize() cleanup pattern)
            if self._cache.name_hint is not None:
                self._cache.vt.set_name_hint(self._cache.name_hint)

            del self._cache.value
            del self._cache.source
            del self._cache.name_hint
            self._realize_fn = None

        return self._cache.vt  # pyrefly: ignore[bad-return]

    def is_hashable(self) -> bool:
        if not self.is_realized():
            value = self._cache.value
            return isinstance(value, Hashable)
        return super().is_hashable()

    def reconstruct(self, codegen: Any) -> None:
        if self.is_realized():
            assert self._cache.vt is not None
            self._cache.vt.reconstruct(codegen)
        else:
            from .constant import ConstantVariable

            ConstantVariable.create(self.peek_value()).reconstruct(codegen)


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
