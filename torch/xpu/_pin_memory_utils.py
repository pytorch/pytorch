import torch


def pin_memory(data_ptr: int, size: int) -> None:
    """
    Registers an existing host memory region as pinned (page-locked) memory on XPU.

    Args:
        data_ptr (int): Pointer to the host memory region. Must be aligned to
            the system's page size.
        size (int): Size of the host memory region in bytes. Must be a multiple of
            the system's page size.
    """
    torch._C._xpu_pinMemory(data_ptr, size)


def unpin_memory(data_ptr: int) -> None:
    """
    Unregisters a previously registered host memory region.

    Args:
        data_ptr (int): Pointer to the host memory region. Must be exactly the pointer
            passed to a previous successful call to pin_memory.
    """
    torch._C._xpu_unpinMemory(data_ptr)
