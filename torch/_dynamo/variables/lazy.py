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
            # pyrefly: ignore [missing-attribute]
            self.vt.set_name_hint(self.name_hint)

        del self.value
        del self.source
        del self.name_hint


class ComputedLazyCache:
    """Container to cache the real VariableTracker for computed lazy constants.

    Unlike LazyCache, this doesn't use VariableBuilder since computed lazy
    constants have no source. It creates a ConstantVariable directly.

    Stores the operands (`args`), `op` (the operator function), and `reconstruct_fn`
    (bytecode generator) to allow recomputing the value at runtime. This enables
    calling functions with different input values without recompiling.
    """

    def __init__(
        self,
        value: Any,
        lazy_vars: list[LazyConstantVariable],
        args: list[VariableTracker],
        op: Callable[..., Any],
        reconstruct_fn: Callable[[Any, list[VariableTracker]], None],
    ) -> None:
        self.value = value
        self.lazy_vars = lazy_vars
        self.args = args
        self.op = op
        self.reconstruct_fn = reconstruct_fn
        self.name_hint: str | None = None
        self.vt: VariableTracker | None = None

    def realize(self) -> None:
        assert self.vt is None
        from ..symbolic_convert import InstructionTranslator
        from .builtin import BuiltinVariable
        from .constant import ConstantVariable
        from .tensor import SymNodeVariable

        # Realize all source LazyConstantVariables (this installs their guards)
        # and check if any became symbolic
        any_symbolic = False
        for lazy_var in self.lazy_vars:
            realized = lazy_var.realize()
            if isinstance(realized, SymNodeVariable):
                any_symbolic = True

        if any_symbolic:
            # One of the sources became symbolic (due to automatic_dynamic_shapes).
            # We need to recompute the result symbolically instead of using
            # the pre-computed constant value.
            tx = InstructionTranslator.current_tx()
            realized_args = [arg.realize() for arg in self.args]
            self.vt = BuiltinVariable(self.op).call_function(tx, realized_args, {})
        else:
            # All sources are constants, use the pre-computed value
            self.vt = ConstantVariable.create(self.value)

        if self.name_hint is not None:
            assert self.vt is not None
            self.vt.set_name_hint(self.name_hint)

        del self.value
        del self.lazy_vars
        del self.args
        del self.op
        del self.reconstruct_fn
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

    def __init__(self, _cache: LazyCache | ComputedLazyCache, **kwargs: Any) -> None:
        assert isinstance(_cache, (LazyCache, ComputedLazyCache))
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
            # Allow LazyConstantVariable and ComputedLazyConstantVariable to stay
            # lazy when returning from a frame
            keep_lazy = allow_lazy_constant and isinstance(
                value, (LazyConstantVariable, ComputedLazyConstantVariable)
            )
            if keep_lazy:
                if isinstance(value, ComputedLazyConstantVariable):
                    # Check if already realized (attributes deleted after realization)
                    if value._cache.vt is not None:
                        # Already realized - recurse on the realized value
                        result = cls.realize_all(
                            value.realize(),
                            cache,
                            allow_lazy_constant=allow_lazy_constant,
                        )
                    else:
                        # Not yet realized - install TYPE_MATCH guards on source lazy vars.
                        # The reconstruct_fn will regenerate the value from the inputs at runtime.
                        # pyrefly: ignore[missing-attribute]
                        for lazy_var in value._cache.lazy_vars:
                            lazy_var._ensure_type_guard()
                        result = value
                else:
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
        # pyrefly: ignore[missing-attribute]
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

        For LazyConstantVariable, we return LazyConstantVariable (this class) so
        that the handler lookup in _make_handler can recognize it and handle
        it specially via ComputedLazyConstantVariable.

        We don't call _maybe_realize_for_type() here because that would realize
        the variable and install guards, defeating the purpose of lazy evaluation.
        The _make_handler code checks for LazyConstantVariable in arg_types and
        creates ComputedLazyConstantVariable with proper reconstruct_fn.
        """
        # If already realized, return the actual type
        if self.is_realized():
            return type(self.realize())

        # For unrealized LazyConstantVariable, return LazyConstantVariable
        # so that _make_handler recognizes it and handles it lazily.
        # Install TYPE_MATCH guard (not CONSTANT_MATCH) to avoid
        # unnecessary recompilation when values change.
        self._ensure_type_guard()
        return LazyConstantVariable

    def lazy_isinstance(self, cls: type) -> bool:
        """Check isinstance without triggering realization when possible.

        LazyConstantVariable only wraps primitive types (int, float, bool, str)
        which usually realize to ConstantVariable. However, when specialize_int
        or specialize_float is False, int/float values may realize to
        SymNodeVariable instead, so we must realize in those cases.
        """
        from .constant import ConstantVariable
        from .tensor import SymNodeVariable

        # If already realized, just check the realized type
        if self.is_realized():
            return type.__instancecheck__(cls, self.realize())

        # Check if this lazy variable might realize to SymNodeVariable
        # instead of ConstantVariable due to specialize_int/specialize_float
        value_type = self.peek_type()
        might_be_symnode = (value_type is int and not config.specialize_int) or (
            value_type is float and not config.specialize_float
        )

        if might_be_symnode:
            # We don't know if this will become ConstantVariable or SymNodeVariable.
            # Check if we can answer without realizing:
            const_match = issubclass(ConstantVariable, cls)
            sym_match = issubclass(SymNodeVariable, cls)

            if const_match and sym_match:
                # Both types would match, so answer is True
                self._ensure_type_guard()
                return True
            if not const_match and not sym_match:
                # Neither type would match, so answer is False
                return False
            # Only one would match - must realize to know which type we get
            return type.__instancecheck__(cls, self.realize())

        self._ensure_type_guard()
        return issubclass(ConstantVariable, cls)

    def try_peek_constant(self) -> tuple[bool, bool, Any]:
        """Peek at the constant value without triggering realization.

        LazyConstantVariable wraps primitive constants, so we can always peek
        at the underlying value without installing guards.

        Note: If already realized, the realized variable might be a SymNodeVariable
        (when specialize_int=False), which is not a constant. In that case, we
        delegate to the realized variable's try_peek_constant.
        """
        if self.is_realized():
            realized = self.realize()
            return realized.try_peek_constant()
        return (True, True, self.peek_value())

    def is_python_hashable(self) -> bool:
        """LazyConstantVariable wraps primitive types which are all hashable."""
        # Primitive types (int, float, bool, str) are always hashable
        return True

    def get_python_hash(self) -> int:
        """Get hash without triggering realization.

        We can compute the hash from the peeked value. This installs a TYPE_MATCH
        guard since the hash behavior depends on the type.
        """
        if self.is_realized():
            return self.realize().get_python_hash()
        self._ensure_type_guard()
        return hash(self.peek_value())

    def is_python_equal(self, other: VariableTracker) -> bool:
        """Check equality with proper guard handling.

        Same cache (same source) means we can return True without guards.
        Otherwise, we must realize to install guards for correctness.
        """
        if (
            isinstance(other, LazyConstantVariable)
            and not other.is_realized()
            and self._cache is other._cache
        ):
            return True
        return self.realize().is_python_equal(other)


class ComputedLazyConstantVariable(LazyVariableTracker):
    """
    A lazy variable tracker for computed constants (results of operations between
    LazyConstantVariable/ConstantVariable operands) that defers guard installation
    until the value is actually needed.

    The value is computed eagerly at creation time (using peek_value() on lazy
    operands), but guard installation is deferred. This allows chains of operations
    on lazy constants to remain "unguarded" until the final result is used in a way
    that requires guards (e.g., control flow, comparison, or tensor operations).

    When realized, it realizes all referenced LazyConstantVariables (which installs
    their CONSTANT_MATCH guards) and returns a ConstantVariable with the pre-computed
    value.

    Unlike LazyConstantVariable, ComputedLazyConstantVariable has no source or guards
    of its own - it derives guards from the LazyConstantVariables it references.
    """

    _cache: ComputedLazyCache  # pyrefly: ignore[bad-override]

    @staticmethod
    def create(
        op: Callable[..., Any],
        args: list[VariableTracker],
        reconstruct_fn: Callable[[Any, list[VariableTracker]], None],
    ) -> ComputedLazyConstantVariable:
        """Create a ComputedLazyConstantVariable for the given operation.

        Args:
            op: The operator function (e.g., operator.add)
            args: The operands (LazyConstantVariable, ConstantVariable, or
                  ComputedLazyConstantVariable)
            reconstruct_fn: Function that generates bytecode to recompute the result
                  at runtime from the source operands.

        Returns:
            A ComputedLazyConstantVariable that will defer guard installation.
        """
        # Collect all LazyConstantVariables that need to be realized
        lazy_vars: list[LazyConstantVariable] = []

        def get_value(arg: VariableTracker) -> Any:
            if isinstance(arg, ComputedLazyConstantVariable):
                # pyrefly: ignore[missing-attribute]
                lazy_vars.extend(arg._cache.lazy_vars)
                return arg._cache.value
            elif isinstance(arg, LazyConstantVariable):
                lazy_vars.append(arg)
                if arg.is_realized():
                    return arg.realize().as_python_constant()
                return arg.peek_value()
            else:
                # ConstantVariable or other constant-like types
                return arg.as_python_constant()

        # Compute the value eagerly
        value = op(*[get_value(arg) for arg in args])

        # Verify the result is a valid constant type that ConstantVariable can handle.
        # If not, raise an exception so the caller can fall back to realizing args.
        from .constant import ConstantVariable

        if not ConstantVariable.is_base_literal(value):
            raise TypeError(
                f"ComputedLazyConstantVariable cannot wrap value of type {type(value)}"
            )

        return ComputedLazyConstantVariable(
            ComputedLazyCache(value, lazy_vars, args, op, reconstruct_fn)
        )

    def __init__(self, _cache: ComputedLazyCache, **kwargs: Any) -> None:
        assert isinstance(_cache, ComputedLazyCache)
        # Call VariableTracker.__init__ directly with no source
        VariableTracker.__init__(self, **kwargs)
        self._cache = _cache

    def python_type(self) -> type:
        """Return the Python type of the computed result."""
        if self.is_realized():
            assert self._cache.vt is not None
            return self._cache.vt.python_type()
        return type(self._cache.value)

    def is_tensor(self) -> bool:
        """Computed constants are never tensors."""
        return False

    def is_constant_none(self) -> bool:
        if self.is_realized():
            assert self._cache.vt is not None
            return self._cache.vt.is_constant_none()
        return self._cache.value is None

    def lazy_isinstance(self, cls: type) -> bool:
        """Check isinstance without triggering realization."""
        from .constant import ConstantVariable

        return issubclass(ConstantVariable, cls)

    def is_python_constant(self) -> bool:
        return True

    def as_python_constant(self) -> Any:
        """Return the computed constant value.

        If this ComputedLazyConstantVariable has a reconstruct_fn, the value
        depends on the source lazy variables. In that case, we must realize
        the source lazy variables to install their guards. This ensures that
        if the source values change, recompilation will be triggered.

        Without this, operations like f-strings that use as_python_constant()
        would return stale cached values without installing any guards.
        """
        if self.is_realized():
            assert self._cache.vt is not None
            return self._cache.vt.as_python_constant()

        # The value depends on source lazy vars. Realize them to install guards.
        for lazy_var in self._cache.lazy_vars:
            lazy_var.realize()

        return self._cache.value

    def try_peek_constant(self) -> tuple[bool, bool, Any]:
        """Peek at the constant value without triggering realization.

        ComputedLazyConstantVariable stores its computed value eagerly,
        so we can always peek without installing guards.
        """
        if self.is_realized():
            assert self._cache.vt is not None
            return (True, False, self._cache.vt.as_python_constant())
        return (True, True, self._cache.value)

    def reconstruct(self, codegen: Any) -> None:
        """Generate bytecode to reconstruct the computed value.

        This uses the reconstruct_fn provided at creation time to generate
        bytecode that recomputes the value from the operands. This allows
        the function to be called with different input values without
        recompiling.
        """
        if self.is_realized():
            # If realized, just load the constant
            assert self._cache.vt is not None
            codegen.append_output(
                codegen.create_load_const(self._cache.vt.as_python_constant())
            )
        else:
            # Use the reconstruct function to generate bytecode
            self._cache.reconstruct_fn(codegen, self._cache.args)

    def original_source(self) -> Any:
        # ComputedLazyConstantVariable has no source
        return None

    def get_handler_type_for_dispatch(self) -> type:
        """Return the VariableTracker type to use for builtin handler dispatch.

        For ComputedLazyConstantVariable, we return this class itself so that
        the handler lookup in _make_handler can recognize it as constant-like
        and handle it lazily. This is similar to LazyConstantVariable.
        """
        if self.is_realized():
            return type(self.realize())
        return ComputedLazyConstantVariable

    def __repr__(self) -> str:
        if self.is_realized():
            return f"ComputedLazyConstantVariable(realized: {self._cache.vt})"
        return f"ComputedLazyConstantVariable(value={self._cache.value!r})"

    def __str__(self) -> str:
        return self.__repr__()


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
