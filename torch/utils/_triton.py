import functools
import hashlib
import os

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
def triton_backend_hash():
    from triton.common.backend import get_backend, get_cuda_version_key

    import torch

    if torch.version.hip:
        # Does not work with ROCm
        return None

    if not torch.cuda.is_available():
        return None

    backend = get_backend("cuda")
    if backend is None:
        return get_cuda_version_key()
    else:
        return backend.get_version_key()


@functools.lru_cache
def triton_key():
    import pkgutil

    import triton

    TRITON_PATH = os.path.dirname(os.path.abspath(triton.__file__))
    contents = []
    # This is redundant. Doing it to be consistent with upstream.
    # frontend
    with open(os.path.join(TRITON_PATH, "compiler", "compiler.py"), "rb") as f:
        contents += [hashlib.sha256(f.read()).hexdigest()]

    # compiler
    compiler_path = os.path.join(TRITON_PATH, "compiler")
    backends_path = os.path.join(TRITON_PATH, "compiler", "backends")
    for lib in pkgutil.iter_modules([compiler_path, backends_path]):
        with open(lib.module_finder.find_spec(lib.name).origin, "rb") as f:  # type: ignore[call-arg, union-attr, arg-type]
            contents += [hashlib.sha256(f.read()).hexdigest()]
    # backend
    libtriton_hash = hashlib.sha256()
    with open(os.path.join(TRITON_PATH, "_C/libtriton.so"), "rb") as f:
        while True:
            chunk = f.read(1024**2)
            if not chunk:
                break
            libtriton_hash.update(chunk)
    contents.append(libtriton_hash.hexdigest())
    # language
    language_path = os.path.join(TRITON_PATH, "language")
    for lib in pkgutil.iter_modules([language_path]):
        with open(lib.module_finder.find_spec(lib.name).origin, "rb") as f:  # type: ignore[call-arg, union-attr, arg-type]
            contents += [hashlib.sha256(f.read()).hexdigest()]
    from triton import __version__

    return f"{__version__}" + "-".join(contents)


@functools.lru_cache(None)
def triton_hash_with_backend():
    import torch

    if torch.version.hip:
        # Does not work with ROCm
        return None

    backend_hash = triton_backend_hash()
    key = f"{triton_key()}-{backend_hash}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()
