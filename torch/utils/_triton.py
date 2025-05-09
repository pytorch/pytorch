# mypy: allow-untyped-defs
import functools
import hashlib
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from torch.types import Device


@functools.lru_cache(None)
def has_triton_package() -> bool:
    try:
        from triton.compiler.compiler import triton_key

        return triton_key is not None
    except (ImportError, RuntimeError):
        return False


@functools.lru_cache(None)
def has_triton(device: "Device" = None) -> bool:
    """
    Determine if Triton is available for use on this system for a given device
    (if device is not None) or any available device type if no device is given.
    """
    import torch
    from torch._dynamo.device_interface import (
        DeviceInterface,
        get_interface_for_device,
        get_registered_device_interfaces,
    )

    if not has_triton_package():
        return False

    def device_has_triton(di: type[DeviceInterface]) -> bool:
        if not di.is_available():
            return False

        try:
            di.raise_if_triton_unavailable(device)
        except RuntimeError:
            return False

        return True

    if device is None:
        return any(
            device_has_triton(di) for _, di in get_registered_device_interfaces()
        )

    if not isinstance(device, (str, torch.device)):
        device = torch.device(device)

    return device_has_triton(get_interface_for_device(device))


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
                from triton.language.extra.cuda import (  # noqa: F401
                    experimental_device_tensormap_create1d,
                    experimental_device_tensormap_create2d,
                )

                return True
            except ImportError:
                pass

    return False


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
