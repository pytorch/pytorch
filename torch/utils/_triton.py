import functools
import hashlib

from torch._dynamo.device_interface import get_interface_for_device


@functools.lru_cache(None)
def has_triton_package() -> bool:
    try:
        import triton

        return triton is not None
    except ImportError:
        return False


@functools.lru_cache(None)
def has_triton() -> bool:
    def cuda_extra_check(device_interface):
        return device_interface.Worker.get_device_properties().major >= 7

    triton_supported_devices = {"cuda": cuda_extra_check}

    def is_device_compatible_with_triton():
        for device, extra_check in triton_supported_devices.items():
            device_interface = get_interface_for_device(device)
            if device_interface.is_available() and extra_check(device_interface):
                return True
        return False

    return is_device_compatible_with_triton() and has_triton_package()


@functools.lru_cache(None)
def triton_backend():
    import torch

    if torch.version.hip:
        # Does not work with ROCm
        return None

    from triton.compiler.compiler import make_backend
    from triton.runtime.driver import driver

    target = driver.active.get_current_target()
    return make_backend(target)


@functools.lru_cache(None)
def triton_hash_with_backend():
    import torch

    if torch.version.hip:
        # Does not work with ROCm
        return None

    from triton.compiler.compiler import triton_key

    backend = triton_backend()
    key = f"{triton_key()}-{backend.hash()}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()
