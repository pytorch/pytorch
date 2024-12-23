from typing import (
    Any,
    TypeVar,
    Protocol,
)

from numpy._core.numerictypes import (
    issubdtype as issubdtype,
)

_T_contra = TypeVar("_T_contra", contravariant=True)

# A file-like object opened in `w` mode
class _SupportsWrite(Protocol[_T_contra]):
    def write(self, s: _T_contra, /) -> Any: ...

__all__: list[str]

def get_include() -> str: ...

def info(
    object: object = ...,
    maxwidth: int = ...,
    output: None | _SupportsWrite[str] = ...,
    toplevel: str = ...,
) -> None: ...

def source(
    object: object,
    output: None | _SupportsWrite[str] = ...,
) -> None: ...

def show_runtime() -> None: ...
