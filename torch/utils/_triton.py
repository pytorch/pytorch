# mypy: allow-untyped-defs
import functools
import hashlib


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
def has_triton() -> bool:
    from torch._dynamo.device_interface import get_interface_for_device

    def cuda_extra_check(device_interface):
        return device_interface.Worker.get_device_properties().major >= 7

    def _return_true(device_interface):
        return True

    triton_supported_devices = {"cuda": cuda_extra_check, "xpu": _return_true}

    def is_device_compatible_with_triton():
        for device, extra_check in triton_supported_devices.items():
            device_interface = get_interface_for_device(device)
            if device_interface.is_available() and extra_check(device_interface):
                return True
        return False

    return is_device_compatible_with_triton() and has_triton_package()


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


def dtype_to_string(dtype):
    if dtype.name.startswith("fp"):
        suffix = "float" + dtype.name[2:]
    elif dtype.name.startswith("bf"):
        suffix = "bfloat" + dtype.name[2:]
    else:
        suffix = dtype.name
    return "triton.language." + suffix


def patch_triton_dtype_repr():
    import triton

    # Hack to get triton dtype repr to produce an evaluatable expression
    # triton.language.float32 emits triton.language.fp32 which does not
    # exist
    # REMOVE when https://github.com/openai/triton/pull/3342 lands
    triton.language.dtype.__repr__ = lambda self: dtype_to_string(self)
