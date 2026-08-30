import dataclasses
from enum import IntEnum
from types import SimpleNamespace
from typing import NamedTuple


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


class UserDefinedPydanticLikeConfig:
    def __init__(self, nested, hidden=None):
        self.nested = nested
        self.hidden = hidden

    def __repr_args__(self):
        return (("nested", self.nested),)

    def __repr__(self):
        return f"{type(self).__name__}(nested={self.nested!r})"
