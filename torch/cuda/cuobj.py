from collections.abc import Callable

import torch
from torch.types import Storage


__all__: list[str] = [
    "is_available",
    "register_buffer",
    "deregister_buffer",
    "get_rdma_token",
    "put_rdma_token",
]


def _dummy_fn(name: str) -> Callable:
    def fn(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError(f"torch._C.{name} is not supported on this platform")

    return fn


_CUOBJ_SYMBOLS = (
    "_cuobj_available",
    "_cuobj_register_buffer",
    "_cuobj_deregister_buffer",
    "_cuobj_get_rdma_token",
    "_cuobj_put_rdma_token",
)

if not hasattr(torch._C, "_cuobj_available"):
    for _name in _CUOBJ_SYMBOLS:
        if _name != "_cuobj_available" and hasattr(torch._C, _name):
            raise AssertionError(f"{_name} exists but _cuobj_available does not")
    for _name in _CUOBJ_SYMBOLS:
        torch._C.__dict__[_name] = _dummy_fn(_name)


def is_available() -> bool:
    """Return whether NVIDIA cuObject (S3-over-RDMA) support is available.

    This is ``True`` only when PyTorch was built with cuObject
    (``USE_CUOBJ=1``) and a cuObject client connection can be established
    (RDMA-capable NIC and a reachable cuObjServer-backed S3 endpoint).
    """
    if not hasattr(torch._C, "_cuobj_available"):
        return False
    try:
        return bool(torch._C._cuobj_available())
    except RuntimeError:
        return False


def register_buffer(s: Storage) -> None:
    """Register a storage with cuObject for RDMA transfers.

    The storage may live on a CUDA device or in host memory. Registration is
    required before requesting an RDMA token for the buffer.

    Args:
        s (Storage): Buffer to register.
    """
    torch._C._cuobj_register_buffer(s)


def deregister_buffer(s: Storage) -> None:
    """Deregister a storage previously registered with :func:`register_buffer`.

    Args:
        s (Storage): Buffer to deregister.
    """
    torch._C._cuobj_deregister_buffer(s)


def get_rdma_token(s: Storage, size: int, offset: int = 0, is_put: bool = True) -> str:
    """Return an RDMA descriptor (token) for a region of a registered storage.

    The token is passed to the S3 endpoint as the ``x-amz-rdma-token`` header;
    the server then transfers the payload directly into or out of the buffer
    over RDMA. Release it with :func:`put_rdma_token` once the request finishes.

    Args:
        s (Storage): A storage previously passed to :func:`register_buffer`.
        size (int): Number of bytes the token covers.
        offset (int, optional): Byte offset into the buffer. (Default: 0)
        is_put (bool, optional): ``True`` for a PUT (the server reads from the
            buffer), ``False`` for a GET (the server writes into it).
            (Default: ``True``)
    """
    return torch._C._cuobj_get_rdma_token(s, size, offset, is_put)


def put_rdma_token(token: str) -> None:
    """Release an RDMA token returned by :func:`get_rdma_token`.

    Args:
        token (str): The token to release.
    """
    torch._C._cuobj_put_rdma_token(token)
