# mypy: allow-untyped-defs
import functools
import hashlib
from typing import Optional


@functools.lru_cache(None)
def has_triton_package() -> bool:
    try:
        from triton.compiler.compiler import triton_key

        return triton_key is not None
    except ImportError:
        return False
    except RuntimeError:
        return False


@functools.lru_cache(None)
def has_triton_tma():
    if has_triton_package():
        import torch

        if (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability() >= (9, 0)
            and not torch.version.hip
        ):
            try:
                from triton.tools.experimental_descriptor import (  # noqa: F401
                    create_1d_tma_descriptor,
                    create_2d_tma_descriptor,
                )

                return True
            except ImportError:
                pass

    return False


@functools.lru_cache(None)
def has_triton_tma_device():
    if has_triton_package():
        import torch

        if (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability() >= (9, 0)
            and not torch.version.hip
        ):
            try:
                from triton.language import (  # noqa: F401
                    _experimental_make_tensor_descriptor,
                )

                return True
            except ImportError:
                pass

    return False


@functools.lru_cache(None)
def has_triton() -> bool:
    if not has_triton_package():
        return False

    from torch._dynamo.device_interface import get_interface_for_device

    def cuda_extra_check(device_interface):
        return device_interface.Worker.get_device_properties().major >= 7

    def cpu_extra_check(device_interface):
        import triton.backends

        return "cpu" in triton.backends.backends

    def _return_true(device_interface):
        return True

    triton_supported_devices = {
        "cuda": cuda_extra_check,
        "xpu": _return_true,
        "cpu": cpu_extra_check,
    }

    def is_device_compatible_with_triton():
        for device, extra_check in triton_supported_devices.items():
            device_interface = get_interface_for_device(device)
            if device_interface.is_available() and extra_check(device_interface):
                return True
        return False

    return is_device_compatible_with_triton()


@functools.lru_cache(None)
def triton_backend():
    from triton.compiler.compiler import make_backend
    from triton.runtime.driver import driver

    target = driver.active.get_current_target()
    return make_backend(target)


@functools.lru_cache(None)
def triton_hash_with_backend():
    from triton.compiler.compiler import triton_key

    backend = triton_backend()
    key = f"{triton_key()}-{backend.hash()}"

    # Hash is upper case so that it can't contain any Python keywords.
    return hashlib.sha256(key.encode("utf-8")).hexdigest().upper()


@functools.lru_cache(None)
def triton_set_allocator(device):
    if has_triton_tma_device():
        import torch

        assert torch.cuda.current_device() == device

        def alloc_fn(size: int, alignment: int, stream: Optional[int]):
            return torch.empty(size, device=device, dtype=torch.int8)

        import triton

        triton.set_allocator(alloc_fn)

    return None
