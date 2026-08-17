import dataclasses


@dataclasses.dataclass(frozen=True)
class UserDefinedTritonKernelConfig:
    offset: int


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


@dataclasses.dataclass(frozen=True)
class UserDefinedTritonKernelNestedConfig:
    nested: UserDefinedTritonKernelConfigNamespace.Nested


@dataclasses.dataclass(frozen=True)
class UserDefinedTritonKernelHiddenConfig:
    offset: int
    hidden: object = dataclasses.field(repr=False)


class _AttrsField:
    def __init__(self, name, repr_enabled):
        self.name = name
        self.repr = repr_enabled


class UserDefinedAttrsLikeConfig:
    # attrs publishes this metadata for the fields used by its generated repr.
    __attrs_attrs__ = (_AttrsField("nested", True), _AttrsField("hidden", False))

    def __init__(self, nested, hidden):
        self.nested = nested
        self.hidden = hidden

    def __repr__(self):
        return f"{type(self).__name__}(nested={self.nested!r})"


class UserDefinedPydanticLikeConfig:
    def __init__(self, nested, hidden):
        self.nested = nested
        self.hidden = hidden

    def __repr_args__(self):
        return (("nested", self.nested),)

    def __repr__(self):
        return f"{type(self).__name__}(nested={self.nested!r})"
