import functools

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
    def is_cuda_compatible_with_triton():
        device_interface = get_interface_for_device("cuda")
        return (
            device_interface.is_available()
            and device_interface.Worker.get_device_properties().major >= 7
        )

    return is_cuda_compatible_with_triton() and has_triton_package()
