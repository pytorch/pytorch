import torch


def is_supported() -> bool:
    """
    Whether :func:`pin_memory` accepts any host memory region on this build,
    device and GPU driver.
    """
    return torch.xpu.is_available() and torch._C._xpu_isPinMemorySupported()


def pin_memory(data_ptr: int, size: int) -> None:
    """
    Registers an existing host memory region as pinned (page-locked) memory on XPU.

    Args:
        data_ptr (int): Pointer to the host memory region. Unless
            :func:`is_supported`, must be aligned to the system's page size.
        size (int): Size of the host memory region in bytes. Unless
            :func:`is_supported`, must be a multiple of the system's page size.
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
