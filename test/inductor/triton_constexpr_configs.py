import dataclasses
from enum import IntEnum
from types import SimpleNamespace
from typing import ClassVar, NamedTuple


class UserDefinedTritonKernelConfigMode(IntEnum):
    FAST = 1


@dataclasses.dataclass(frozen=True)
class UserDefinedTritonKernelEnumConfig:
    mode: UserDefinedTritonKernelConfigMode


class UserDefinedTritonKernelConfigNamespace:
    @dataclasses.dataclass(frozen=True)
    class Nested:
        offset: int

    @dataclasses.dataclass(frozen=True)
    class Sibling:
        offset: int

    @dataclasses.dataclass(frozen=True)
    class BareNested:
        offset: int

        def __repr__(self):
            return f"{type(self).__name__}(offset={self.offset!r})"

    class Point(NamedTuple):
        offset: int


@dataclasses.dataclass(frozen=True)
class UserDefinedTritonKernelNestedConfig:
    nested: UserDefinedTritonKernelConfigNamespace.Nested


@dataclasses.dataclass(frozen=True)
class UserDefinedTritonKernelHiddenConfig:
    offset: int
    hidden: object = dataclasses.field(repr=False)


@dataclasses.dataclass(frozen=True)
class UserDefinedTritonKernelNonInitConfig:
    offset: int
    derived: int = dataclasses.field(init=False, default=0)


@dataclasses.dataclass(frozen=True)
class UserDefinedTritonKernelHiddenDefaultConfig:
    offset: int
    scale: int = dataclasses.field(default=1, repr=False)


@dataclasses.dataclass
class UserDefinedTritonKernelCoercingConfig:
    offset: int

    def __post_init__(self):
        self.offset *= 2


@dataclasses.dataclass
class UserDefinedTritonKernelCountingConfig:
    child: object = None
    constructed: ClassVar[int] = 0

    def __post_init__(self):
        type(self).constructed += 1


@dataclasses.dataclass
class UserDefinedTritonKernelSelfReferentialConfig:
    child: object = None


@dataclasses.dataclass(frozen=True)
class UserDefinedTritonKernelDefaultArgConfig:
    offset: int = 1


# This root name would overwrite the conventional triton.language import alias.
tl = dataclasses.make_dataclass("tl", [("offset", int)], frozen=True)


# This root name collides with a binding in the generated launcher's exec scope.
class runner(NamedTuple):
    offset: int


class UserDefinedAttrsLikeConfig:
    # attrs publishes this metadata for the fields used by its generated repr.
    __attrs_attrs__ = (
        SimpleNamespace(name="nested", repr=True),
        SimpleNamespace(name="hidden", repr=False),
    )

    def __init__(self, nested, hidden=None):
        self.nested = nested
        self.hidden = hidden

    def __repr__(self):
        return f"{type(self).__name__}(nested={self.nested!r})"


class UserDefinedAttrsPrivateFieldConfig:
    # attrs exposes a private field `_nested` as the __init__ parameter `nested`
    # (Attribute.alias); the generated source must use the alias.
    __attrs_attrs__ = (
        SimpleNamespace(name="_nested", alias="nested", repr=True, init=True),
    )

    def __init__(self, nested):
        self._nested = nested

    def __repr__(self):
        return f"{type(self).__name__}(_nested={self._nested!r})"

    def __eq__(self, other):
        if type(other) is not type(self):
            return NotImplemented
        return self._nested == other._nested

    def __hash__(self):
        return hash((type(self), self._nested))


class UserDefinedPydanticLikeNoEqConfig:
    # __repr_args__ without an __eq__: rendering must verify field-wise.
    def __init__(self, nested):
        self.nested = nested

    def __repr_args__(self):
        return (("nested", self.nested),)

    def __repr__(self):
        return f"{type(self).__name__}(nested={self.nested!r})"


class UserDefinedPydanticLikeConfig:
    def __init__(self, nested, hidden=None):
        self.nested = nested
        self.hidden = hidden

    def __repr_args__(self):
        return (("nested", self.nested),)

    def __repr__(self):
        return f"{type(self).__name__}(nested={self.nested!r})"

    # Pydantic models compare (and hash frozen models) over all fields.
    def __eq__(self, other):
        if type(other) is not type(self):
            return NotImplemented
        return (self.nested, self.hidden) == (other.nested, other.hidden)

    def __hash__(self):
        return hash((type(self), self.nested, self.hidden))
