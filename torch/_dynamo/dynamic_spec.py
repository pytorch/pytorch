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
from collections.abc import Iterator
from typing import Any, ClassVar, TypeAlias

from torch._dynamo.source import LocalSource


__all__ = [
    "IntSpecType",
    "IntSpec",
    "TensorSpec",
    "ParamsSpec",
    "ShapesSpec",
    "lookup_spec_from_dynamo_source",
]

# Type alias for leaf specs (individual argument specifications)
LeafSpec: TypeAlias = "TensorSpec | IntSpec | None"
# Any spec — what public APIs accept. TODO: expand to LeafSpec | ObjectSpec | ListSpec | DictSpec
IntermediateSpec: TypeAlias = LeafSpec


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

    ``min`` and ``max`` are assumptions about the value range, translated to
    ``torch._check`` calls on the newly created symbolic variables during
    compilation.

    ``type`` is fixed at construction; all other fields are mutable via
    fluent setters that return ``self`` for chaining:

        spec = IntSpec.backed("batch", min=1, max=64)
        spec.guarding_hint(32)   # set, returns self
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
        # Auto-generate a name when the user doesn't supply one.
        if name is None:
            name = f"_intspec_{type.value}_{id(self):x}"
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

        ``guarding_hint`` overrides the hint used by the shape environment
        (assumes my first example input is ``guarding_hint``). Affects
        branching decisions and optimization choices. Changing
        ``guarding_hint`` will cause FxGraphCache misses.
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

        ``optimization_hint`` is used to guide guardless optimizations for
        unbacked symbols, accessed by the ``optimization_hint`` API
        (e.g. inductor autotuning, graph partitioning). It never
        participates in guard generation or symbolic reasoning. Changing
        ``optimization_hint`` will cause FxGraphCache misses.
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


class TensorSpec:
    """Per-dimension shape specification for a tensor.

    A list-like container of ``IntSpec | None`` with length equal to the
    tensor's dim. ``None`` entries inherit the default dynamism policy from
    the compile context.

    Construct from any of:

    - ``int`` — number of dims; all entries start as ``None``.
    - ``list`` / ``tuple`` of ``IntSpec | None`` — dim is inferred from
      length, entries used as-is.
    - ``dict[int, IntSpec | None]`` — sparse per-dim spec; dim is inferred
      from ``max(keys) + 1``. Empty dict rejected.

    Example::
        TensorSpec(3)  # rank 3, all None
        TensorSpec([IntSpec.backed("batch"), None])  # rank 2, dim 0 backed
        TensorSpec({0: IntSpec.backed("batch")})  # rank 1, dim 0 backed
    """

    def __init__(
        self,
        arg: int
        | list[IntSpec | None]
        | tuple[IntSpec | None, ...]
        | dict[int, IntSpec | None],
    ) -> None:
        self._sparse = False
        if isinstance(arg, int):
            self._dim = arg
            self._specs: list[IntSpec | None] = [None] * arg
        elif isinstance(arg, (list, tuple)):
            self._dim = len(arg)
            self._specs = list(arg)
        elif isinstance(arg, dict):
            self._sparse = True
            self._dim = max(arg.keys()) + 1
            self._specs = [None] * self._dim
            for k, v in arg.items():
                self._specs[k] = v
        else:
            raise TypeError(
                f"TensorSpec expects int / list / tuple / dict, "
                f"got {type(arg).__name__}"
            )

    def dim(self, index: int, spec: IntSpec) -> "TensorSpec":
        """Set the spec at ``index`` and return ``self`` for chaining."""
        self._specs[index] = spec
        return self

    def __getitem__(self, index: int) -> IntSpec | None:
        if index >= self._dim:
            if not self._sparse:
                raise IndexError(
                    f"TensorSpec has {self._dim} dims but got index {index}; "
                    f"tensor rank doesn't match the spec"
                )
            return None
        return self._specs[index]

    def __setitem__(self, index: int, spec: IntSpec | None) -> None:
        self._specs[index] = spec

    def __len__(self) -> int:
        return self._dim

    def __iter__(self) -> Iterator[IntSpec | None]:
        return iter(self._specs)

    def __repr__(self) -> str:
        entries = ", ".join(repr(spec) for spec in self._specs)
        return f"TensorSpec([{entries}])"

    # No ``__eq__`` / ``__hash__``: matches :class:`IntSpec`'s design — specs
    # are immutable compile-time inputs compared via ``repr()`` when needed.
    # Value-based equality would force cache keys to drift with object
    # identity and conflict with the AOT-snapshot invariant.


class ParamsSpec:
    """Specification for the arguments of a compiled function.

    Describes the dynamic shape behavior for named arguments, *args, and
    **kwargs of a ``torch.compile``-wrapped function::

        def f(x, y, *args, **kwargs):
        #    ^^^^  named_args
        #           ^^^^^  varargs
        #                   ^^^^^^  varkw

    Example::

        ParamsSpec({"x": TensorSpec(3), "y": IntSpec.backed("y")})
    """

    def __init__(
        self,
        named_args: dict[str, IntermediateSpec] | None = None,
        *,
        varargs: list[IntermediateSpec] | None = None,
        varkw: dict[str, IntermediateSpec] | None = None,
    ) -> None:
        self._named_args: dict[str, LeafSpec] = dict(named_args) if named_args else {}
        if varargs is not None:
            raise NotImplementedError("varargs is not supported yet")
        if varkw is not None:
            raise NotImplementedError("varkw is not supported yet")
        self._varargs: list[IntermediateSpec] | None = None
        self._varkw: dict[str, IntermediateSpec] | None = None

    def __repr__(self) -> str:
        parts: list[str] = []
        if self._named_args:
            parts.append(f"named_args={self._named_args!r}")
        if self._varargs is not None:
            parts.append(f"varargs={self._varargs!r}")
        if self._varkw is not None:
            parts.append(f"varkw={self._varkw!r}")
        return f"ParamsSpec({', '.join(parts)})"


class ShapesSpec:
    """Top-level shape specification for a ``torch.compile`` call.

    ``params`` describes the arguments of the compiled callable — for a raw
    function this is the function's parameters, for an ``nn.Module`` this
    is the parameters of ``forward`` (excluding ``self``).

    Currently only ``params`` is supported::

        ShapesSpec(params=ParamsSpec({"x": TensorSpec(3)}))

    ``globals`` and ``assumptions`` are reserved for future use and will
    raise ``NotImplementedError`` if set.
    """

    def __init__(
        self,
        params: ParamsSpec | None = None,
        globals: Any = None,
        assumptions: Any = None,
    ) -> None:
        if globals is not None:
            raise NotImplementedError("ShapesSpec.globals is not supported yet")
        if assumptions is not None:
            raise NotImplementedError("ShapesSpec.assumptions is not supported yet")
        self._params = params

    @property
    def params(self) -> ParamsSpec | None:
        return self._params

    def __repr__(self) -> str:
        return f"ShapesSpec(params={self._params!r})"


def lookup_spec_from_dynamo_source(source, shapes_spec: ShapesSpec | None) -> LeafSpec:
    """Look up the spec for a function input arg from the shapes_spec.

    Only supports LocalSource with is_input=True (direct function args).
    Returns TensorSpec, IntSpec, or None.
    """
    if shapes_spec is None or shapes_spec.params is None:
        return None
    # Only top-level function input args are supported for now.
    #  Module attributes (self.x), globals, and values computed
    #  during execution are not covered by shapes_spe yet.
    if not isinstance(source, LocalSource) or not source.is_input:
        return None
    return shapes_spec.params._named_args.get(source.local_name)
