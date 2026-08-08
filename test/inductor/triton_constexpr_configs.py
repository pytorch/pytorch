import dataclasses


@dataclasses.dataclass(frozen=True)
class UserDefinedTritonKernelConfig:
    offset: int


class UserDefinedTritonKernelConfigNamespace:
    @dataclasses.dataclass(frozen=True)
    class Nested:
        offset: int


@dataclasses.dataclass(frozen=True)
class UserDefinedTritonKernelNestedConfig:
    nested: UserDefinedTritonKernelConfigNamespace.Nested
