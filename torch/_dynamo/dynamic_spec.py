"""Dynamic shape specification types for ``torch.compile`` and ``torch.export``.

Provides class `IntSpec` for fine-grained control over whether an integer
(dimension size or scalar argument) is treated as static, backed, or unbacked
during compilation.

Backed vs. unbacked
-------------------
``torch.compile`` provides two kinds of dynamic shapes: ``backed`` and
``unbacked``. ``torch.compile`` guards on ``backed`` dynamic shapes and does
not provide a guarantee that no guards will be added to them. User code,
dynamo, inductor, and autograd all can add guards when tracing through
branching, e.g. ``if x.size() > 10``. Moreover, for 0/1 specializations,
backed symbols are specialized unconditionally to ``0``, ``1``, or ``>=2``
even without encountering a branching on those ranges.

On the contrary, ``unbacked`` dynamic shapes are guaranteed not to be guarded
on and are not 0/1 specialized. However, there is a possibility of throwing a
data-dependent error when a branch that requires their value is encountered
and no explicit unbacked handling is defined. The framework is converging to
a state where it won't throw DDE but rather pick general paths. One downside
of using unbacked is missed optimization opportunities due to either perf
bugs or picking general paths, or using a fixed non-example input-based hint.
An example of picking general paths is assuming input not contiguous in
functions called ``contiguous()`` and ``reshape()`` when it cannot be
symbolically proven, with a change of introducing a clone.

For more info see
https://dev-discuss.pytorch.org/t/backed-to-unbacked-from-guardable-to-guardless-shapes-in-pytorch/3333.
"""

import enum
from typing import Any, ClassVar


__all__ = ["IntSpecType", "IntSpec"]


class IntSpecType(enum.Enum):
    """How an integer should be treated during compilation.

    STATIC: compile-time constant; triggers recompilation if the value changes.
    BACKED: symbolic with guards and 0/1 specialization permitted.
    UNBACKED: symbolic, no guards, no 0/1 specialization; may raise a data-dependent error on branching.
    """

    STATIC = "static"
    BACKED = "backed"
    UNBACKED = "unbacked"


class IntSpec:
    """Shape specification for a single integer (dimension size or scalar arg).

    Create via a classmethod factory or the constructor directly:

        IntSpec.static("x", value=10)
        IntSpec.backed("batch", min=1, max=64, guarding_hint=32)
        IntSpec.unbacked("seq", min=1, max=2048, optimization_hint=512)
        IntSpec("x", IntSpecType.STATIC, value=10)

    ``type`` is fixed at construction; all other fields are mutable via
    fluent setters that double as getters (no arg = read, one arg = write):

        spec = IntSpec.backed("batch", min=1, max=64)
        spec.guarding_hint(32)   # set, returns self
        spec.guarding_hint()     # get, returns 32
        spec.min(1).max(64)      # chain
    """

    _name: str | None
    _type: IntSpecType
    _value: int | None
    _min: int | None
    _max: int | None
    _guarding_hint: int | None
    _optimization_hint: int | None

    __slots__ = (
        "_name",
        "_type",
        "_value",
        "_min",
        "_max",
        "_guarding_hint",
        "_optimization_hint",
    )

    def __init__(
        self,
        name: str | None,
        type: IntSpecType,
        *,
        min: int | None = None,
        max: int | None = None,
        value: int | None = None,
        guarding_hint: int | None = None,
        optimization_hint: int | None = None,
    ) -> None:
        if not isinstance(type, IntSpecType):
            raise TypeError(f"IntSpec type must be an IntSpecType, got {type!r}")
        self._type = type
        self._name = name
        self._min = min
        self._max = max
        self._value = value
        self._guarding_hint = guarding_hint
        self._optimization_hint = optimization_hint
        self._validate()

    def __setattr__(self, key: str, value: Any) -> None:
        # ``_type`` is the only pinned slot — it drives the per-mode
        # validation rules and integration-level dispatch (BACKED vs.
        # UNBACKED).
        if key == "_type" and hasattr(self, "_type"):
            raise AttributeError("IntSpec type is immutable; cannot reassign")
        object.__setattr__(self, key, value)

    def __delattr__(self, key: str) -> None:
        raise AttributeError(f"IntSpec attribute {key!r} cannot be deleted")

    _MODE_KWARG_HINT: ClassVar[dict[IntSpecType, tuple[str, str]]] = {
        IntSpecType.STATIC: ("static", "value"),
        IntSpecType.BACKED: ("backed", "guarding_hint"),
        IntSpecType.UNBACKED: ("unbacked", "optimization_hint"),
    }

    @staticmethod
    def _check_name(value: Any, type_: IntSpecType) -> None:
        if value is not None and not isinstance(value, str):
            factory, kwarg = IntSpec._MODE_KWARG_HINT[type_]
            raise TypeError(
                f"IntSpec.name must be str or None, got "
                f"{value.__class__.__name__}; if you meant to pass a "
                f"value/hint, use a keyword argument "
                f"(e.g. IntSpec.{factory}({kwarg}={value!r}))"
            )

    @staticmethod
    def _check_int_field(field_name: str, value: Any) -> None:
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool)
        ):
            raise TypeError(
                f"IntSpec.{field_name} must be int or None, got "
                f"{value.__class__.__name__}"
            )

    # -- validation --------------------------------------------------------
    #
    # Single entry point: type checks (name, int fields), per-mode rules,
    # and cross-field invariants like ``min <= max``. Run on every
    # construction and on every fluent set (via ``_try_set``).

    def _validate(self) -> None:
        IntSpec._check_name(self._name, self._type)
        IntSpec._check_int_field("min", self._min)
        IntSpec._check_int_field("max", self._max)
        IntSpec._check_int_field("value", self._value)
        IntSpec._check_int_field("guarding_hint", self._guarding_hint)
        IntSpec._check_int_field("optimization_hint", self._optimization_hint)
        if self._type is IntSpecType.STATIC:
            if self._min is not None or self._max is not None:
                raise ValueError(
                    "min/max are only valid for BACKED/UNBACKED IntSpec, not STATIC"
                )
            if self._guarding_hint is not None:
                raise ValueError("guarding_hint is only valid for BACKED IntSpec")
            if self._optimization_hint is not None:
                raise ValueError("optimization_hint is only valid for UNBACKED IntSpec")
        elif self._type is IntSpecType.BACKED:
            if self._value is not None:
                raise ValueError("value is only valid for STATIC IntSpec")
            if self._optimization_hint is not None:
                raise ValueError("optimization_hint is only valid for UNBACKED IntSpec")
        else:  # UNBACKED
            if self._value is not None:
                raise ValueError("value is only valid for STATIC IntSpec")
            if self._guarding_hint is not None:
                raise ValueError("guarding_hint is only valid for BACKED IntSpec")
        if self._min is not None and self._max is not None and self._min > self._max:
            raise ValueError(
                f"min must be <= max, got min={self._min}, max={self._max}"
            )

    @classmethod
    def static(cls, name: str | None = None, *, value: int | None = None) -> "IntSpec":
        """Construct a STATIC `IntSpec`.

        ``value`` pins a concrete size; if ``None`` the value is taken from
        the example input at compile time.
        """
        return cls(name, type=IntSpecType.STATIC, value=value)

    @classmethod
    def backed(
        cls,
        name: str | None = None,
        *,
        min: int | None = None,
        max: int | None = None,
        guarding_hint: int | None = None,
    ) -> "IntSpec":
        """Construct a BACKED `IntSpec`.

        ``guarding_hint`` is the concrete value the symbolic shape
        environment substitutes when a hint is needed for reasoning or
        codegen.
        """
        return cls(
            name,
            type=IntSpecType.BACKED,
            min=min,
            max=max,
            guarding_hint=guarding_hint,
        )

    @classmethod
    def unbacked(
        cls,
        name: str | None = None,
        *,
        min: int | None = None,
        max: int | None = None,
        optimization_hint: int | None = None,
    ) -> "IntSpec":
        """Construct an UNBACKED `IntSpec`.

        ``optimization_hint`` is used by downstream codegen (e.g. inductor
        autotuning) only; it never participates in symbolic reasoning.
        """
        return cls(
            name,
            type=IntSpecType.UNBACKED,
            min=min,
            max=max,
            optimization_hint=optimization_hint,
        )

    # -- fluent setters ----------------------------------------------------
    #
    # Each setter mutates in place, revalidates, and returns ``self`` for
    # chaining. Per-mode validity is enforced on each set, so e.g.
    # ``IntSpec.static("x").guarding_hint(10)`` raises ``ValueError``.

    def _try_set(self, slot: str, new_value: Any) -> None:
        # Atomic: if ``_validate`` rejects the new state, roll back so the
        # spec stays in a consistent state for the caller.
        old = getattr(self, slot)
        setattr(self, slot, new_value)
        try:
            self._validate()
        except Exception:
            setattr(self, slot, old)
            raise

    def name(self, value: str) -> "IntSpec":
        self._try_set("_name", value)
        return self

    def min(self, value: int) -> "IntSpec":
        self._try_set("_min", value)
        return self

    def max(self, value: int) -> "IntSpec":
        self._try_set("_max", value)
        return self

    def value(self, value: int) -> "IntSpec":
        self._try_set("_value", value)
        return self

    def guarding_hint(self, value: int) -> "IntSpec":
        self._try_set("_guarding_hint", value)
        return self

    def optimization_hint(self, value: int) -> "IntSpec":
        self._try_set("_optimization_hint", value)
        return self

    # -- dunder ------------------------------------------------------------

    def __repr__(self) -> str:
        parts: list[str] = []
        for slot in self.__slots__:
            val = getattr(self, slot)
            if slot == "_type":
                parts.append(f"type={val.name}")
            elif slot == "_name":
                if val is not None:
                    parts.append(f"name={val!r}")
            elif val is not None:
                parts.append(f"{slot[1:]}={val}")
        return f"IntSpec({', '.join(parts)})"
