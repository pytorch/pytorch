import functools
import hashlib
from typing import Any


@functools.cache
def has_triton_package() -> bool:
    try:
        import triton  # noqa: F401

        return True
    except ImportError:
        return False


@functools.cache
def _device_supports_tma() -> bool:
    import torch

    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() >= (9, 0)
        and not torch.version.hip
    )


@functools.cache
def has_triton_experimental_host_tma() -> bool:
    if has_triton_package():
        if _device_supports_tma():
            try:
                from triton.tools.experimental_descriptor import (  # noqa: F401
                    create_1d_tma_descriptor,
                    create_2d_tma_descriptor,
                )

                return True
            except ImportError:
                pass

    return False


@functools.cache
def has_triton_tensor_descriptor_host_tma() -> bool:
    if has_triton_package():
        if _device_supports_tma():
            try:
                from triton.tools.tensor_descriptor import (  # noqa: F401
                    TensorDescriptor,
                )

                return True
            except ImportError:
                pass

    return False


@functools.cache
def has_triton_tma() -> bool:
    return has_triton_tensor_descriptor_host_tma() or has_triton_experimental_host_tma()


@functools.cache
def has_triton_tma_device() -> bool:
    if has_triton_package():
        import torch

        if (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability() >= (9, 0)
            and not torch.version.hip
        ):
            # old API
            try:
                from triton.language.extra.cuda import (  # noqa: F401
                    experimental_device_tensormap_create1d,
                    experimental_device_tensormap_create2d,
                )

                return True
            except ImportError:
                pass

            # new API
            try:
                from triton.language import make_tensor_descriptor  # noqa: F401

                return True
            except ImportError:
                pass

    return False


@functools.lru_cache(None)
def has_triton_stable_tma_api() -> bool:
    if has_triton_package():
        import torch

        if (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability() >= (9, 0)
            and not torch.version.hip
        ):
            try:
                from triton.language import make_tensor_descriptor  # noqa: F401

                return True
            except ImportError:
                pass
    return False


@functools.cache
def has_triton() -> bool:
    if not has_triton_package():
        return False

    from torch._dynamo.device_interface import get_interface_for_device

    def cuda_extra_check(device_interface: Any) -> bool:
        return device_interface.Worker.get_device_properties().major >= 7

    def cpu_extra_check(device_interface: Any) -> bool:
        import triton.backends

        return "cpu" in triton.backends.backends

    def _return_true(device_interface: Any) -> bool:
        return True

    triton_supported_devices = {
        "cuda": cuda_extra_check,
        "xpu": _return_true,
        "cpu": cpu_extra_check,
    }

    def is_device_compatible_with_triton() -> bool:
        for device, extra_check in triton_supported_devices.items():
            device_interface = get_interface_for_device(device)
            if device_interface.is_available() and extra_check(device_interface):
                return True
        return False

    return is_device_compatible_with_triton()


@functools.cache
def triton_backend() -> Any:
    from triton.compiler.compiler import make_backend
    from triton.runtime.driver import driver

    target = driver.active.get_current_target()
    return make_backend(target)


@functools.cache
def triton_hash_with_backend() -> str:
    from torch._inductor.runtime.triton_compat import triton_key

    backend = triton_backend()
    key = f"{triton_key()}-{backend.hash()}"

    # Hash is upper case so that it can't contain any Python keywords.
    return hashlib.sha256(key.encode("utf-8")).hexdigest().upper()
