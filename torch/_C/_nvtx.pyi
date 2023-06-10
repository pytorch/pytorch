from typing import Optional

# Defined in torch/csrc/cuda/shared/nvtx.cpp
def rangePush(
    message: str,
    domain: Optional[str] = None,
    category: Optional[str] = None,
    color: Optional[int] = None,
) -> int: ...
def rangePop(domain: Optional[str] = None) -> int: ...
def rangeStart(
    message: Optional[str],
    domain: Optional[str] = None,
    category: Optional[str] = None,
    color: Optional[int] = None,
) -> int: ...
def rangeEnd(id: int, domain: Optional[str] = None) -> None: ...
def mark(
    message: str,
    domain: Optional[str] = None,
    category: Optional[str] = None,
    color: Optional[int] = None,
) -> None: ...
