from typing import overload

__all__ = [
    "ComplexWarning",
    "VisibleDeprecationWarning",
    "ModuleDeprecationWarning",
    "TooHardError",
    "AxisError",
    "DTypePromotionError",
]

class ComplexWarning(RuntimeWarning): ...
class ModuleDeprecationWarning(DeprecationWarning): ...
class VisibleDeprecationWarning(UserWarning): ...
class RankWarning(RuntimeWarning): ...
class TooHardError(RuntimeError): ...
class DTypePromotionError(TypeError): ...

class AxisError(ValueError, IndexError):
    __slots__ = "_msg", "axis", "ndim"

    axis: int | None
    ndim: int | None
    @overload
    def __init__(self, axis: str, ndim: None = ..., msg_prefix: None = ...) -> None: ...
    @overload
    def __init__(self, axis: int, ndim: int, msg_prefix: str | None = ...) -> None: ...
