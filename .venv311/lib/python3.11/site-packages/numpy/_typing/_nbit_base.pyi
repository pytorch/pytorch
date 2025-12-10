# pyright: reportDeprecated=false
# pyright: reportGeneralTypeIssues=false
# mypy: disable-error-code=misc

from typing import final

from typing_extensions import deprecated

# Deprecated in NumPy 2.3, 2025-05-01
@deprecated(
    "`NBitBase` is deprecated and will be removed from numpy.typing in the "
    "future. Use `@typing.overload` or a `TypeVar` with a scalar-type as upper "
    "bound, instead. (deprecated in NumPy 2.3)",
)
@final
class NBitBase: ...

@final
class _256Bit(NBitBase): ...

@final
class _128Bit(_256Bit): ...

@final
class _96Bit(_128Bit): ...

@final
class _80Bit(_96Bit): ...

@final
class _64Bit(_80Bit): ...

@final
class _32Bit(_64Bit): ...

@final
class _16Bit(_32Bit): ...

@final
class _8Bit(_16Bit): ...
